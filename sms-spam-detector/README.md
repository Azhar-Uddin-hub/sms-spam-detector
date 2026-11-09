# SMS Spam Detector

A beginner-friendly machine learning app that classifies English SMS messages as
**spam** or **not spam**. Compare two real models, inspect their held-out results,
and try new messages in a Streamlit interface.

**[Use the live app](https://azhar-sms-spam-detector.streamlit.app/)** in your
browser without installing anything.

This is the first project in a three-project learning path: spam detection,
student grade prediction, and movie recommendations.

## Run on Windows

Open PowerShell in this folder. Python 3.13 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If your default Python is newer and dependencies cannot install, use an installed
Python 3.13, or `uv venv --python 3.13 .venv` if you have uv.

Open the local URL printed by Streamlit, usually http://localhost:8501.
No activation command or API key is required. The repository includes a trained
model, so you can skip the explicit training command to start the app immediately.
Reproducing training needs internet access for the initial dataset download.
Starting the app without the included model automatically downloads and trains it.

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## What you can explore

- Enter a message or try two example messages.
- See a prediction and the model's estimated confidence.
- Compare Multinomial Naive Bayes and Logistic Regression.
- Inspect spam precision, recall, F1, accuracy and the confusion matrix.
- Explore frequent words in spam and legitimate training messages.
- Read how the pipeline works and where its predictions can fail.

## Learn the project in this order

1. **`src/data.py`**: download the data, understand labels, and remove duplicate
   messages. `ham` means a legitimate message.
2. **`src/preprocess.py`**: normalize Unicode, case and whitespace. Numbers and
   punctuation are preserved in normalization; TF-IDF uses its default word
   tokenizer to select tokens.
3. **`src/train.py`**: split the data, turn words into TF-IDF features, compare
   models and save the winning pipeline. A pipeline keeps preprocessing and
   prediction together.
4. **`src/predict.py`**: validate user input and produce predictions. Messages with
   no vocabulary overlap receive a helpful warning rather than a guess.
5. **`app.py`**: connect your trained model to a small web interface.

NLTK is deliberately unnecessary for this first version: scikit-learn supplies
the text features we need, keeping setup and the learning path smaller.

## Evaluation method

Empty messages and exact duplicates after normalization are removed **before** the
split. Every message with conflicting labels is excluded. This prevents identical
normalized messages appearing in both training and test data.

An 80/20 stratified split uses seed 42. Both candidates are compared using
five-fold stratified cross-validation **inside the training set**, with vocabulary
and TF-IDF weights fitted separately for each fold. The model with the highest
mean macro F1 is selected. Each model is then fitted on the training set and
evaluated on the held-out test set for reporting. The saved winner stays fitted
on the training set, so the displayed results describe the actual saved model.

The model comparison uses fixed hyperparameters. It does not include a broader
hyperparameter search or claim statistical significance between the models.

Actual results and data-cleaning counts are in [`reports/metrics.json`](reports/metrics.json)
after training, and in the app's Model results tab. Scores are computed from the
data; they are never hard-coded into the UI.

## Check the project

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover normalization, duplicate/conflict removal, archive parsing, vocabulary
isolation, input validation, prediction probability mapping, evaluation counts,
and the Streamlit input-to-result flow. They also verify that the included model
loads without starting training. Tests use the included model and mock archive
downloads, so they do not need dataset network access.

`requirements-lock.txt` records the exact runtime packages used for this local
build. Use it instead of `requirements.txt` when recreating this environment.

## Files

```text
sms-spam-detector/
  app.py
  src/               # Data, preprocessing, training and prediction
  data/README.md     # Source and attribution; downloaded CSV is ignored
  models/            # Included trained model; runtime logs are ignored
  reports/           # Real evaluation results
  tests/             # Data, inference and app checks
  .streamlit/        # Theme and telemetry setting
  requirements.txt
  requirements-dev.txt
  requirements-lock.txt
```

## Dataset and attribution

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
by Tiago Almeida and Jos Hidalgo, DOI
[10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The downloader converts the original tab-separated archive member to CSV;
normalization and filtering happen during training.

The dataset page lists 5,574 instances; actual archive and cleaned row counts are
measured during training and recorded in the report. See also the introductory
paper, *Contributions to the study of SMS spam filtering: new collection and
results* (Almeida, Hidalgo and Yamakami, 2011).

## Limitations

- Historical English SMS data does not establish performance on today's scams,
  email, Bangla, Banglish, or other languages.
- Spam is the minority class. Accuracy alone can conceal missed spam, which is
  why the app also shows spam precision, recall and F1.
- Confidence is an **uncalibrated model probability**, not proof of safety.
- Normalized deduplication does not remove every near-duplicate or template.
- A random held-out split is not an independent real-world evaluation.
- The app code does not save entered messages. Hosting infrastructure may have
  its own data handling policies.
- Load only model files you generated or trust; joblib uses Python serialization.

## Portfolio next steps

Before presenting this as a finished portfolio project, add your own explanation
of the results and screenshots from your running app. Then publish the repository
and optionally deploy it to Streamlit Community Cloud with `app.py` as the entry
point and Python 3.13. The included model avoids CPU-heavy startup training on
shared hosting. scikit-learn, Joblib, NumPy and SciPy versions are fixed to the
versions used to produce it. PyArrow is restricted below version 25 to match
the hosting platform's compatibility workaround.
The source is published at
[Azhar-Uddin-hub/sms-spam-detector](https://github.com/Azhar-Uddin-hub/sms-spam-detector).

Good later experiments: analyze false positives, compare word and character
features using training validation, evaluate calibration, add near-duplicate
grouping, and test on a separately collected modern dataset.

## Implementation references

- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: text feature pipelines](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html)
- [Streamlit: app testing](https://docs.streamlit.io/develop/api-reference/app-testing)
