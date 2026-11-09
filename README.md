# SMS Spam Detector

A beginner-friendly machine learning project with a Streamlit web app. Paste an
English SMS to classify it as spam or legitimate, and explore real evaluation
results for Logistic Regression and Multinomial Naive Bayes.

## Try the app

**[Open SMS Spam Detector](https://azhar-sms-spam-detector.streamlit.app/)**

Open the link on your phone or computer, paste an English SMS, and click
**Analyze message**. You do not need Python, a GitHub account, or an API key.

## Run locally

```powershell
cd sms-spam-detector
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The repository includes the trained model, so the app can start without a
dataset download or training. No API key is required. Python 3.13 is recommended.
To reproduce training, run `python -m src.train` in the project folder.

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
are needed. The included model keeps startup quick on the shared hosting CPU.

The source dataset is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
by Tiago Almeida and Jos Hidalgo, provided under CC BY 4.0. See the dataset
attribution file for details.
