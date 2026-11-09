"""Validate one message and return a prediction without storing its text."""

from sklearn.pipeline import Pipeline

from src.preprocess import normalize_text

MAX_MESSAGE_LENGTH = 5000


def predict_message(pipeline: Pipeline, message: str) -> dict:
    if not isinstance(message, str) or not normalize_text(message):
        raise ValueError("Please enter a message containing text.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Please keep your message under {MAX_MESSAGE_LENGTH:,} characters.")
    vector = pipeline.named_steps["tfidf"].transform([message])
    if vector.nnz == 0:
        raise ValueError("No familiar words were found. Try a longer English message.")
    probabilities = pipeline.predict_proba([message])[0]
    classes = list(pipeline.classes_)
    label = str(pipeline.predict([message])[0])
    return {
        "label": label,
        "confidence": float(probabilities[classes.index(label)]),
        "spam_probability": float(probabilities[classes.index("spam")]),
    }
