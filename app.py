import streamlit as st
import nltk
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = [ps.stem(w) for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

def rule_based_score(text):
    risky_phrases = [
        "verify your account",
        "confirm your",
        "avoid interruption",
        "expires today",
        "update your details",
        "action required"
    ]
    score = 0
    for phrase in risky_phrases:
        if phrase in text.lower():
            score += 0.15
    return min(score, 0.6)

@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.joblib")
    tfidf = joblib.load("tfidf.joblib")
    return model, tfidf

model, tfidf = load_artifacts()

st.set_page_config(page_title="Spam Detection", layout="centered")
st.title("📧 Spam Message Detection")

user_input = st.text_area("Enter SMS / Email text:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        ml_prob = model.predict_proba(vector)[0][1]
        final_prob = min(ml_prob + rule_based_score(user_input), 1.0)
        confidence = round(final_prob * 100, 2)

        if final_prob > 0.7:
            st.error(f"🚨 SPAM  \nConfidence: {confidence}%")
        elif final_prob > 0.3:
            st.warning(f"⚠️ SUSPICIOUS  \nConfidence: {confidence}%")
        else:
            st.success(f"✅ NOT SPAM  \nConfidence: {round(100-confidence,2)}%")
