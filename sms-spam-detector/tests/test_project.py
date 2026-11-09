"""Check data leakage boundaries, input handling and the actual Streamlit flow."""

from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data import ROOT, download_dataset, prepare_data
from src.predict import predict_message
from src.preprocess import normalize_text
from src.train import load_or_train, make_pipelines


@pytest.fixture(scope="session")
def bundle():
    return load_or_train()


def test_normalization_preserves_useful_signals():
    assert normalize_text("  FREE\n prize １２３! ") == "free prize 123!"


def test_deduplication_removes_label_conflicts_and_empty_messages():
    frame = pd.DataFrame([
        ("ham", "Hello friend"), ("ham", " HELLO  friend "),
        ("ham", "same message"), ("spam", "SAME MESSAGE"),
        ("spam", "Claim your prize"), ("ham", "   "), ("ham", None),
    ], columns=["label", "message"])
    clean, summary = prepare_data(frame)
    assert set(clean["normalized"]) == {"hello friend", "claim your prize"}
    assert summary["duplicate_rows_removed"] == 1
    assert summary["conflicting_rows_removed"] == 2
    assert summary["invalid_or_empty_rows_removed"] == 2


def test_downloader_preserves_quotes_and_downloads_only_once(tmp_path):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("SMSSpamCollection", 'ham\t"Meet me later"\nspam\tFree prize\tcall now\n')
    with patch("src.data.urlopen", return_value=BytesIO(buffer.getvalue())) as fetch:
        destination = tmp_path / "spam.csv"
        download_dataset(destination)
        download_dataset(destination)
        fetch.assert_called_once()
    rows = pd.read_csv(destination)
    assert rows.loc[0, "message"] == '"Meet me later"'
    assert rows.loc[1, "message"] == "Free prize\tcall now"


def test_vocabulary_does_not_learn_from_prediction_messages():
    pipeline = make_pipelines()["Multinomial Naive Bayes"]
    pipeline.fit(["meet friend tomorrow", "free cash prize"], ["ham", "spam"])
    pipeline.predict(["heldoutuniquetoken prize"])
    assert "heldoutuniquetoken" not in pipeline.named_steps["tfidf"].vocabulary_


@pytest.mark.parametrize("message", ["", " \n ", "a" * 5001, "🛸🛸🛸", "zxqvzzqxxzz"])
def test_invalid_or_unfamiliar_input_is_not_classified(bundle, message):
    with pytest.raises(ValueError):
        predict_message(bundle["pipeline"], message)


def test_prediction_probability_matches_returned_label(bundle):
    result = predict_message(bundle["pipeline"], "Hey, are we still meeting for lunch tomorrow?")
    assert result["label"] in {"ham", "spam"}
    assert 0 <= result["confidence"] <= 1
    expected = result["spam_probability"] if result["label"] == "spam" else 1 - result["spam_probability"]
    assert result["confidence"] == pytest.approx(expected)


def test_evaluation_report_counts_and_selection_are_consistent(bundle):
    report = bundle["report"]
    assert report["train_rows"] + report["test_rows"] == report["dataset"]["clean_rows"]
    winner = max(report["models"], key=lambda name: report["models"][name]["cv_macro_f1_mean"])
    assert report["selected_model"] == winner
    for metrics in report["models"].values():
        assert sum(map(sum, metrics["confusion_matrix"])) == report["test_rows"]


def test_app_handles_empty_input_and_classifies_a_sample(bundle):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    app.button(key="FormSubmitter:message_form-Analyze message").click().run()
    assert any("Please enter" in warning.value for warning in app.warning)
    next(button for button in app.button if button.label == "Everyday message").click().run()
    assert "meeting for lunch" in app.text_area[0].value
    app.button(key="FormSubmitter:message_form-Analyze message").click().run()
    assert not app.exception
    assert any("Prediction:" in item.value for item in [*app.success, *app.error])
    app.text_area[0].input("🛸🛸🛸")
    app.button(key="FormSubmitter:message_form-Analyze message").click().run()
    assert not app.success
    assert any("No familiar words" in warning.value for warning in app.warning)
