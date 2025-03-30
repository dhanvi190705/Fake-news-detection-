import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the datasets
@st.cache_data
def load_data():
    train_df = pd.read_csv("/Users/yogitadoshi/Downloads/fake-news/train2.csv")
    test_df = pd.read_csv("/Users/yogitadoshi/Downloads/fake-news/test1.csv")
    return train_df, test_df

train_df, test_df = load_data()

# Data Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply text cleaning
train_df['text'] = train_df['text'].astype(str).apply(clean_text)
test_df['text'] = test_df['text'].astype(str).apply(clean_text)

# Features and Labels
X_train = train_df['text']
y_train = train_df['label']

# Convert text data to TF-IDF features
@st.cache_data
def vectorize_data():
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    joblib.dump(vectorizer, "vectorizer.pkl")
    return X_train_tfidf, vectorizer

X_train_tfidf, vectorizer = vectorize_data()

# Train and Save Models
@st.cache_resource
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(kernel="linear"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    model_accuracies = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        accuracy = accuracy_score(y_train, model.predict(X_train_tfidf))
        model_accuracies[name] = accuracy
        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
    return model_accuracies

model_accuracies = train_models()

# Streamlit App
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Comparison", "Fake News Prediction"])

if page == "Home":
    st.title("Fake News Detection System")
    st.write("This application detects fake news using different machine learning models.")
    st.write("### Models Used:")
    st.write("- Logistic Regression")
    st.write("- Support Vector Machine (SVM)")
    st.write("- Random Forest")
    st.write("- Decision Tree")

elif page == "Model Comparison":
    st.title("Model Comparison")
    
    st.write("### Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'red', 'green', 'orange'])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Models")
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig)
    
    for model_name in model_accuracies.keys():
        st.subheader(model_name)
        model = joblib.load(f"{model_name.replace(' ', '_').lower()}_model.pkl")
        accuracy = model_accuracies[model_name]
        
        # Classification Report
        st.write("### Classification Report")
        st.text(classification_report(y_train, model.predict(X_train_tfidf)))
        
        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_train, model.predict(X_train_tfidf))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

elif page == "Fake News Prediction":
    st.title("Fake News Prediction")
    st.write("Enter text below to check if it's fake or real news:")
    
    user_input = st.text_area("Enter News Text")
    
    if st.button("Predict"):
        if user_input:
            user_input_cleaned = clean_text(user_input)
            vectorizer = joblib.load("vectorizer.pkl")
            user_input_tfidf = vectorizer.transform([user_input_cleaned])
            
            model = joblib.load("random_forest_model.pkl")  # Default model for prediction
            prediction = model.predict(user_input_tfidf)[0]
            result = "Real News" if prediction == 1 else "Fake News"
            
            st.write(f"Prediction: **{result}**")
