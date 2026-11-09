# Dataset

`python -m src.data` downloads the UCI SMS Spam Collection to `spam.csv`.
The raw CSV is ignored by Git and is not needed in the repository.

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Creators: Tiago Almeida and Jos Hidalgo. DOI: https://doi.org/10.24432/C5CC84

Dataset license: Creative Commons Attribution 4.0 (CC BY 4.0), as listed by UCI.
The downloader converts the tab-separated source into a two-column CSV. Training
normalizes message text and removes duplicates and conflicting labels; actual
row counts are recorded in `reports/metrics.json` rather than assuming the count
displayed on the dataset webpage matches the archive exactly.
