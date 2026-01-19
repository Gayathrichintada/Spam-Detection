import pandas as pd
import nltk
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = [ps.stem(w) for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
df.columns = ['label','message']
df['label'] = df['label'].map({'ham':0,'spam':1})

df['clean'] = df['message'].apply(clean_text)

# TF-IDF with ngrams
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save artifacts
joblib.dump(model, "spam_model.joblib")
joblib.dump(tfidf, "tfidf.joblib")

print("✅ Model and TF-IDF saved successfully")
