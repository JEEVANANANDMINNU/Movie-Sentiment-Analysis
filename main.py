import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("IMDB Dataset.csv")


# Input reviews
X = data['review']

# Output labels
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


# Predict test data
y_pred = model.predict(X_test)


# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Test your own review
review = [input("Enter movie review: ")]


# Convert review
review_vector = vectorizer.transform(review)


# Predict sentiment
result = model.predict(review_vector)


print("Prediction:", result[0])