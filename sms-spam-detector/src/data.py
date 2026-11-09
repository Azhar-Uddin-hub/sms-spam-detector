"""Download the public UCI archive and prepare an honest evaluation dataset."""

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from src.preprocess import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "spam.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_dataset(path: Path = DATA_PATH) -> Path:
    """Fetch once. Read the expected archive member without extracting files."""
    if path.exists():
        return path
    request = Request(DATASET_URL, headers={"User-Agent": "SMS-Spam-Portfolio/1.0"})
    with urlopen(request, timeout=60) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as zipped:
        content = zipped.read("SMSSpamCollection").decode("utf-8-sig")
    rows = []
    for line in content.splitlines():
        if not line.strip():
            continue
        label, separator, message = line.partition("\t")
        if not separator or label not in {"ham", "spam"} or not message.strip():
            raise ValueError("Unexpected UCI dataset format; download was not saved.")
        rows.append((label, message))
    if not rows:
        raise ValueError("The downloaded dataset is empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["label", "message"]).to_csv(path, index=False)
    return path


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop empty rows, conflicting labels and duplicate normalized messages."""
    if not {"label", "message"}.issubset(frame.columns):
        raise ValueError("The dataset needs label and message columns.")
    raw_count = len(frame)
    clean = frame[["label", "message"]].dropna().copy()
    if not clean["label"].isin(["ham", "spam"]).all():
        raise ValueError("Labels must be ham or spam.")
    clean["message"] = clean["message"].astype(str)
    clean["normalized"] = clean["message"].map(normalize_text)
    clean = clean[clean["normalized"].str.len() > 0]
    valid_count = len(clean)
    conflicts = clean.groupby("normalized")["label"].nunique()
    conflicting_keys = conflicts[conflicts > 1].index
    conflicting_rows = int(clean["normalized"].isin(conflicting_keys).sum())
    clean = clean[~clean["normalized"].isin(conflicting_keys)]
    before_deduplication = len(clean)
    clean = clean.drop_duplicates("normalized").reset_index(drop=True)
    if set(clean["label"]) != {"ham", "spam"}:
        raise ValueError("The cleaned dataset must contain both ham and spam.")
    return clean, {
        "raw_rows": raw_count,
        "invalid_or_empty_rows_removed": raw_count - valid_count,
        "conflicting_rows_removed": conflicting_rows,
        "duplicate_rows_removed": before_deduplication - len(clean),
        "clean_rows": len(clean),
        "class_counts": {str(k): int(v) for k, v in clean["label"].value_counts().items()},
    }


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    return prepare_data(pd.read_csv(download_dataset(path)))


if __name__ == "__main__":
    _, summary = load_dataset()
    print(summary)
