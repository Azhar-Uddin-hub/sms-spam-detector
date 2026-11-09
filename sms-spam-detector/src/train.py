"""Compare two text pipelines using training folds; evaluate on a held-out test set."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import ROOT, load_dataset
from src.preprocess import normalize_text

MODEL_PATH = ROOT / "models" / "spam_detector.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.json"
SEED = 42


def make_pipelines() -> dict[str, Pipeline]:
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=4.0, max_iter=1000, random_state=SEED),
    }
    return {
        name: Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=normalize_text, lowercase=False,
                ngram_range=(1, 2), sublinear_tf=True,
            )),
            ("classifier", classifier),
        ])
        for name, classifier in classifiers.items()
    }


def train(model_path: Path = MODEL_PATH, report_path: Path = REPORT_PATH) -> dict:
    frame, dataset = load_dataset()
    # Split before fitting vocabulary or weights. Duplicates were removed upstream.
    train_rows, test_rows = train_test_split(
        frame, test_size=0.2, stratify=frame["label"], random_state=SEED,
    )
    x_train, y_train = train_rows["message"], train_rows["label"]
    x_test, y_test = test_rows["message"], test_rows["label"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipelines = make_pipelines()
    comparisons = {}
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, x_train, y_train, cv=folds, scoring="f1_macro")
        comparisons[name] = {
            "cv_macro_f1_mean": float(scores.mean()),
            "cv_macro_f1_std": float(scores.std()),
        }
    # Select by training CV, never by the held-out test score.
    selected = max(comparisons, key=lambda name: comparisons[name]["cv_macro_f1_mean"])
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="binary", pos_label="spam", zero_division=0,
        )
        comparisons[name].update({
            "accuracy": float(accuracy_score(y_test, predicted)),
            "spam_precision": float(precision),
            "spam_recall": float(recall),
            "spam_f1": float(f1),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=["ham", "spam"]).tolist(),
        })
    # Frequency charts describe training messages only, excluding common English words.
    words = {}
    for label in ["ham", "spam"]:
        counter = CountVectorizer(stop_words="english", preprocessor=normalize_text)
        counts = counter.fit_transform(train_rows.loc[train_rows["label"] == label, "message"])
        totals = counts.sum(axis=0).A1
        terms = counter.get_feature_names_out()
        top = totals.argsort()[-15:][::-1]
        words[label] = [{"word": str(terms[i]), "count": int(totals[i])} for i in top]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "random_seed": SEED,
        "dataset": dataset,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "selection_method": "Highest mean macro F1 across 5 stratified training folds",
        "selected_model": selected,
        "models": comparisons,
        "top_words": words,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the report beside the model in the same bundle so the UI cannot mix runs.
    joblib.dump({"pipeline": pipelines[selected], "report": report}, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_or_train() -> dict:
    if not MODEL_PATH.exists():
        train()
    # This file is generated locally by this project. Never load untrusted joblib files.
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    results = train()
    print(json.dumps(results, indent=2))
