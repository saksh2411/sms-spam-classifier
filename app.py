import streamlit as st
import joblib
import re

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="centered")

# -----------------------------
# Fixed thresholds in code
# -----------------------------
CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.20

# -----------------------------
# Session state defaults
# -----------------------------
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "sample_choice" not in st.session_state:
    st.session_state.sample_choice = "Select a sample message"

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "result_label" not in st.session_state:
    st.session_state.result_label = ""

if "ham_prob" not in st.session_state:
    st.session_state.ham_prob = 0.0

if "spam_prob" not in st.session_state:
    st.session_state.spam_prob = 0.0

if "margin" not in st.session_state:
    st.session_state.margin = 0.0

# -----------------------------
# Load model files
# -----------------------------
try:
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    model = joblib.load("models/spam_classifier.joblib")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text

def predict_message(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prob = model.predict_proba(vector)[0]

    ham_prob = float(prob[0])
    spam_prob = float(prob[1])

    if spam_prob > 0.60:
        label = "spam"
    elif ham_prob > 0.70:
        label = "ham"
    else:
        label = "suspicious"

    margin = abs(spam_prob - ham_prob)

    return label, ham_prob, spam_prob, margin

def load_sample():
    samples = {
        "Select a sample message": "",
        "Sample 1 - Spam": "Congratulations! You have won a free iPhone. Click the link now.",
        "Sample 2 - Ham": "Hey, are we still meeting at 5 pm today?",
        "Sample 3 - Spam": "URGENT! Your account has been suspended. Verify immediately.",
        "Sample 4 - Ham": "Please send me the notes after class.",
        "Sample 5 - Borderline": "You have a pending account notification. Please review it soon."
    }
    st.session_state.message_input = samples[st.session_state.sample_choice]
    st.session_state.show_result = False

def clear_all():
    st.session_state.message_input = ""
    st.session_state.sample_choice = "Select a sample message"
    st.session_state.show_result = False
    st.session_state.result_label = ""
    st.session_state.ham_prob = 0.0
    st.session_state.spam_prob = 0.0
    st.session_state.margin = 0.0

def run_prediction():
    text = st.session_state.message_input.strip()
    if text:
        label, ham_prob, spam_prob, margin = predict_message(text)
        st.session_state.result_label = label
        st.session_state.ham_prob = ham_prob
        st.session_state.spam_prob = spam_prob
        st.session_state.margin = margin
        st.session_state.show_result = True
    else:
        st.session_state.show_result = False
        st.warning("Please enter a message first.")

# -----------------------------
# UI
# -----------------------------
st.title("SMS Spam Classifier")
st.write("Classify a message as **ham**, **spam**, or **suspicious**.")

st.selectbox(
    "Sample messages",
    [
        "Select a sample message",
        "Sample 1 - Spam",
        "Sample 2 - Ham",
        "Sample 3 - Spam",
        "Sample 4 - Ham",
        "Sample 5 - Borderline"
    ],
    key="sample_choice",
    on_change=load_sample
)

st.text_area(
    "Message",
    key="message_input",
    height=180,
    placeholder="Type or paste the message here..."
)

col1, col2 = st.columns(2)

with col1:
    st.button("Predict", use_container_width=True, on_click=run_prediction)

with col2:
    st.button("Clear / Reset", use_container_width=True, on_click=clear_all)

# -----------------------------
# Result box
# -----------------------------
if st.session_state.show_result:
    with st.container(border=True):
        st.subheader("Result")

        if st.session_state.result_label == "spam":
            st.badge("SPAM", color="red")
            st.error("This message looks like spam.")
        elif st.session_state.result_label == "ham":
            st.badge("HAM", color="green")
            st.success("This message looks like ham.")
        else:
            st.badge("SUSPICIOUS", color="orange")
            st.warning("This message is borderline and should be reviewed manually.")

        st.write(f"**Ham probability:** {st.session_state.ham_prob:.2f}")
        st.progress(st.session_state.ham_prob)

        st.write(f"**Spam probability:** {st.session_state.spam_prob:.2f}")
        st.progress(st.session_state.spam_prob)

        st.write(f"**Confidence gap:** {st.session_state.margin:.2f}")
        st.caption("Rule used: Spam if spam probability > 0.60, Ham if ham probability > 0.70, otherwise Suspicious.")