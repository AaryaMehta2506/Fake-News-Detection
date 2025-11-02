import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')


# Text Cleaning

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)


# Load and Train Model

@st.cache_resource
def load_or_train_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        # st.info("Training model for the first time...")

        # Load datasets
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")

        # Add labels
        true_df["label"] = 1
        fake_df["label"] = 0

        # Combine and shuffle
        df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        df["content"] = (df["title"].fillna('') + " " + df["text"].fillna('')).apply(clean_text)

        X = df["content"]
        y = df["label"]

        # Split and vectorize
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        # Evaluate
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        # st.success(f"Model trained successfully with Accuracy: {acc:.4f}")

        # Save for reuse
        pickle.dump(model, open("model.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

        return model, vectorizer


# Prediction Function

def predict_news(news_text, model, vectorizer):
    clean_news = clean_text(news_text)
    vec = vectorizer.transform([clean_news])
    probs = model.predict_proba(vec)[0]
    pred = np.argmax(probs)
    confidence = probs[pred] * 100
    return pred, confidence


# Streamlit App

st.set_page_config(page_title="Fake News Detection")

st.title("Fake News Detection System")
st.write("Check whether a given news statement is **real or fake** using a trained NLP model.")
st.write("Model was trained on political/news articles, not general statements.")

# st.markdown("---")

# Load or train model
model, vectorizer = load_or_train_model()

# Input text
news_text = st.text_area("Enter News Content:", height=200, placeholder="Type or paste a news article here...")

if st.button("Analyze"):
    if not news_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        label, confidence = predict_news(news_text, model, vectorizer)

        # st.markdown("---")
        if confidence < 60:
            st.info(f"Uncertain — please verify with reliable sources (Confidence: {confidence:.2f}%)")
        elif label == 1:
            st.success(f"Real News (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"Fake News (Confidence: {confidence:.2f}%)")

# st.markdown("---")
# st.caption("Model: Logistic Regression | Features: TF-IDF (10k) | Dataset: True.csv + Fake.csv")
