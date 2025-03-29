import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train_df = pd.read_csv("/Users/yogitadoshi/Downloads/fake-news/train2.csv")
test_df = pd.read_csv("/Users/yogitadoshi/Downloads/fake-news/test1.csv")

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
X_test = test_df['text']
test_ids = test_df['id']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Models Separately
log_model = LogisticRegression()
log_model.fit(X_train_tfidf, y_train)
log_accuracy = accuracy_score(y_train, log_model.predict(X_train_tfidf))

svm_model = SVC(kernel="linear")
svm_model.fit(X_train_tfidf, y_train)
svm_accuracy = accuracy_score(y_train, svm_model.predict(X_train_tfidf))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_accuracy = accuracy_score(y_train, rf_model.predict(X_train_tfidf))

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_tfidf, y_train)
dt_accuracy = accuracy_score(y_train, dt_model.predict(X_train_tfidf))

# Streamlit App
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Logistic Regression", "SVM", "Random Forest", "Decision Tree", "Model Comparison"])

if page == "Home":
    st.title("Fake News Detection System")
    st.write("This application detects fake news using different machine learning models.")
    st.write("### Models Used:")
    st.write("- Logistic Regression")
    st.write("- Support Vector Machine (SVM)")
    st.write("- Random Forest")
    st.write("- Decision Tree")
    st.write("Navigate to different pages to see the model outputs.")

else:
    st.title(f"{page} Model")
    
    if page == "Logistic Regression":
        accuracy = log_accuracy
        model = log_model
    elif page == "SVM":
        accuracy = svm_accuracy
        model = svm_model
    elif page == "Random Forest":
        accuracy = rf_accuracy
        model = rf_model
    elif page == "Decision Tree":
        accuracy = dt_accuracy
        model = dt_model
    
    # Accuracy Bar Graph
    fig, ax = plt.subplots()
    ax.bar([page], [accuracy], color='blue')
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)
    
    # Classification Report
    st.write("### Classification Report")
    st.text(classification_report(y_train, model.predict(X_train_tfidf)))

# Model Comparison Page
if page == "Model Comparison":
    st.title("Model Comparison")
    model_accuracies = {
        "Logistic Regression": log_accuracy,
        "SVM": svm_accuracy,
        "Random Forest": rf_accuracy,
        "Decision Tree": dt_accuracy
    }
    
    # Accuracy Comparison Bar Chart
    fig, ax = plt.subplots()
    ax.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'red', 'green', 'orange'])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Models")
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig)

    st.write("### Accuracy Scores")
    for model_name, acc in model_accuracies.items():
        st.write(f"{model_name}: {acc:.4f}")

# Create submission file using the best model (Random Forest in this case)
submission_df = pd.DataFrame({"id": test_ids, "label": rf_model.predict(X_test_tfidf)})
submission_df.to_csv("submission.csv", index=False)