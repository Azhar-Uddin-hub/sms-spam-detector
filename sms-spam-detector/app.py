"""Run with: python -m streamlit run app.py"""

import pandas as pd
import streamlit as st

from src.predict import MAX_MESSAGE_LENGTH, predict_message
from src.train import load_or_train

st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_or_train()


st.caption("PROJECT 01 / NATURAL LANGUAGE PROCESSING")
st.title("A second opinion for your inbox.")
st.write("Paste an English SMS to see whether a machine learning model thinks it is spam.")

try:
    with st.spinner("Loading the trained model…"):
        bundle = get_artifacts()
except Exception:
    st.error("The model could not be loaded. Run `python -m src.train` in the project folder to see the error and rebuild it. The first download needs internet access.")
    st.stop()

pipeline, report = bundle["pipeline"], bundle["report"]
chosen = report["models"][report["selected_model"]]

with st.sidebar:
    st.header("SMS Spam Detector")
    st.write("A beginner's end-to-end machine learning project.")
    st.divider()
    st.caption("MODEL IN USE")
    st.write(report["selected_model"])
    st.caption("DATASET")
    st.write(f"{report['dataset']['clean_rows']:,} unique messages")
    st.write("80% training · 20% held-out testing")
    st.divider()
    st.markdown("[Dataset and attribution](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)")
    st.caption("Educational demo. A prediction is not a guarantee that a message or link is safe.")

check_tab, results_tab, learn_tab = st.tabs(["Check a message", "Model results", "How it works"])

with check_tab:
    entry, guide = st.columns([3, 2], gap="large")
    with entry:
        st.subheader("What did you receive?")
        st.caption("Try an example, or enter your own message.")
        samples = {
            "Promotional message": "Congratulations! You have won a free cash prize. Call now to claim your reward!",
            "Everyday message": "Hey, are we still meeting for lunch tomorrow?",
        }
        example_columns = st.columns(2)
        for column, (name, text) in zip(example_columns, samples.items()):
            if column.button(name, width="stretch"):
                st.session_state["message"] = text
                st.session_state.pop("prediction", None)
        with st.form("message_form"):
            message = st.text_area("SMS message", key="message", height=180,
                                   max_chars=MAX_MESSAGE_LENGTH,
                                   placeholder="Type or paste a message here…")
            submitted = st.form_submit_button("Analyze message", type="primary", width="stretch")
        if submitted:
            st.session_state.pop("prediction", None)
            try:
                st.session_state["prediction"] = predict_message(pipeline, message)
            except ValueError as error:
                st.warning(str(error))
        if "prediction" in st.session_state:
            result = st.session_state["prediction"]
            if result["label"] == "spam":
                st.error("Prediction: Spam")
            else:
                st.success("Prediction: Not spam")
            st.metric("Model confidence in this prediction", f"{result['confidence']:.1%}")
            st.caption("This probability is an uncalibrated model estimate. It may be overconfident or wrong.")
    with guide:
        st.subheader("Look beyond the prediction")
        st.write("Spam often includes unexpected rewards, urgent demands, or requests to call unfamiliar numbers.")
        st.info("This model recognizes patterns in older English SMS messages. It may miss new scams, and it has not been evaluated on Bangla or Banglish.")
        st.caption("The app code does not save the messages you enter. Model training uses the public UCI dataset.")

with results_tab:
    st.subheader("Measured on messages held out of training")
    columns = st.columns(3)
    columns[0].metric("Spam precision", f"{chosen['spam_precision']:.1%}")
    columns[1].metric("Spam recall", f"{chosen['spam_recall']:.1%}")
    columns[2].metric("Spam F1 score", f"{chosen['spam_f1']:.1%}")
    st.caption("Precision: how often a spam flag is correct. Recall: how much spam is caught. F1 balances both.")
    comparison = pd.DataFrame(report["models"]).T
    comparison = comparison[["cv_macro_f1_mean", "spam_precision", "spam_recall", "spam_f1", "accuracy"]].astype(float)
    comparison.columns = ["Training CV macro F1", "Test spam precision", "Test spam recall", "Test spam F1", "Test accuracy"]
    st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
    st.caption(f"Selection: {report['selection_method']}. The test set was used only for final reporting.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        matrix = pd.DataFrame(chosen["confusion_matrix"],
                              index=["Actual not spam", "Actual spam"],
                              columns=["Predicted not spam", "Predicted spam"])
        st.dataframe(matrix, width="stretch")
        st.caption(f"Test set: {report['test_rows']:,} messages. False positives appear in the top-right cell.")
    with right:
        st.markdown("**Frequent words in training messages**")
        label = st.radio("Message category", ["spam", "ham"], horizontal=True,
                         format_func=lambda value: "Not spam" if value == "ham" else "Spam")
        words = pd.DataFrame(report["top_words"][label]).set_index("word")
        st.bar_chart(words, horizontal=True)
        st.caption("Raw word counts, excluding common English words; these are not explanations for an individual prediction.")
    with st.expander("Dataset preparation and reproducibility"):
        st.json({"dataset": report["dataset"], "seed": report["random_seed"],
                 "trained_at_utc": report["created_at_utc"], "scikit_learn": report["sklearn_version"]})

with learn_tab:
    st.subheader("From a message to a prediction")
    for title, description in [
        ("1. Prepare the data", "Load labeled SMS messages. Remove empty messages, conflicting labels and duplicates before splitting."),
        ("2. Turn text into numbers", "TF-IDF represents individual words and pairs of words. Fit the vocabulary on training messages only."),
        ("3. Compare two models", "Train Multinomial Naive Bayes and Logistic Regression. Choose using five-fold cross-validation within the training set."),
        ("4. Evaluate honestly", "Report precision, recall, F1 and a confusion matrix on the separate test set."),
    ]:
        st.markdown(f"**{title}**")
        st.write(description)
    st.warning("Limitations: historical English data, class imbalance, no verified probability calibration, and no testing on current real-world scam traffic. Near-duplicate messages may still exist after exact normalized deduplication.")
