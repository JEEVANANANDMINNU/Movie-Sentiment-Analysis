import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression


# Website title
st.title("Movie Review Sentiment Analysis")


# Load dataset
data = pd.read_csv("IMDB Dataset.csv")


# Features and labels
X = data['review']
y = data['sentiment']


# Convert text into numbers
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)


# User input box
review = st.text_area("Enter Movie Review")


# Predict button
if st.button("Predict"):

    # Convert review
    review_vector = vectorizer.transform([review])

    # Prediction
    result = model.predict(review_vector)

    # Show result
    st.success(f"Prediction: {result[0]}")