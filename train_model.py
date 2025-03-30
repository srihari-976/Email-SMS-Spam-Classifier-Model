import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk_resources = ['punkt', 'stopwords']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Initialize NLTK components
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def extract_features(text):
    features = {}
    
    # Basic counts
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Special character counts
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['currency_symbol_count'] = len(re.findall(r'[$€£₹]', text))
    
    # Number counts
    features['number_count'] = len(re.findall(r'\d+', text))
    
    # Uppercase ratio
    if len(text) > 0:
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
    else:
        features['uppercase_ratio'] = 0
        
    return features

def transform_text(text):
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4}|\(\d{3}\)\s*\d{3}[-.\s]\d{4}', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove special characters and numbers but keep certain spam indicators
    words = [word for word in words if any(c.isalpha() for c in word) or word in ['$', '£', '€', '₹', '%']]
    
    # Remove stopwords but keep certain important ones
    important_words = {'free', 'win', 'winner', 'won', 'cash', 'prize', 'offer', 'limited', 'urgent', 'now'}
    words = [word for word in words if word not in stop_words or word in important_words]
    
    # Apply stemming
    words = [ps.stem(word) for word in words]
    
    return " ".join(words)

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df = df[['text', 'label']]

# Extract additional features
feature_df = pd.DataFrame([extract_features(text) for text in df['text']])

# Preprocess the text
df['processed_text'] = df['text'].apply(transform_text)

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['label'], test_size=0.2, random_state=42
)

# Create and fit the TF-IDF vectorizer with custom parameters
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Include phrases up to 3 words
    stop_words='english',
    token_pattern=r'\b\w+\b',  # Keep single characters like $
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the model with adjusted class weights
model = MultinomialNB(alpha=0.1)  # Reduced smoothing for better spam detection
model.fit(X_train_tfidf, y_train)

# Print model performance
train_score = model.score(X_train_tfidf, y_train)
test_score = model.score(X_test_tfidf, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Print detailed classification report
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Cross-validation score
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f) 