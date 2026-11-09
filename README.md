# SMS Spam Detector

A beginner-friendly machine learning project with a Streamlit web app. Paste an
English SMS to classify it as spam or legitimate, and explore real evaluation
results for Logistic Regression and Multinomial Naive Bayes.

## Run locally

```powershell
cd sms-spam-detector
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first run downloads the public UCI dataset and trains the models. No API key
is required. Python 3.13 is recommended.

## Explore the project

- [Setup, learning guide, evaluation method and limitations](sms-spam-detector/README.md)
- [Model results](sms-spam-detector/reports/metrics.json)
- [Dataset source and attribution](sms-spam-detector/data/README.md)
- [Source code](sms-spam-detector/src)

On the held-out test set, the selected Logistic Regression model achieved a spam
F1 score of 0.906 and caught 106 of 128 spam messages. It missed 22 spam messages;
these historical-data results do not guarantee real-world performance. English
messages only; Bangla and Banglish performance has not been evaluated.

## Deploy on Streamlit Community Cloud

Select this repository, branch `main`, and entrypoint
`sms-spam-detector/app.py`. In Advanced settings, choose Python 3.13. No secrets
are needed. The app trains a fresh model on the host at first startup.

The source dataset is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
by Tiago Almeida and Jos Hidalgo, provided under CC BY 4.0. See the dataset
attribution file for details.
