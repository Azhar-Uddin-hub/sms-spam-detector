"""Keep normalization identical during training and prediction."""

import unicodedata


def normalize_text(message: str) -> str:
    """Normalize Unicode, case and whitespace without stripping useful numbers."""
    return " ".join(unicodedata.normalize("NFKC", message).lower().split())
