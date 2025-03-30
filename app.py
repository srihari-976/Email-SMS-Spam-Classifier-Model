from flask import Flask, render_template, request, jsonify
import pickle
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

app = Flask(__name__)

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

def create_feature_indicators(features):
    indicators = []
    
    # Create indicators for suspicious features
    if features['currency_symbol_count'] > 0:
        indicators.append("💰 Contains currency symbols")
    if features['exclamation_count'] > 2:
        indicators.append("❗ Multiple exclamation marks")
    if features['uppercase_ratio'] > 0.3:
        indicators.append("📢 Excessive uppercase")
    if features['number_count'] > 3:
        indicators.append("🔢 Multiple numbers")
        
    return indicators

# Load the model and vectorizer
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Extract features
        features = extract_features(message)
        
        # Preprocess
        transformed_sms = transform_text(message)
        
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict
        prediction = model.predict(vector_input)[0]
        confidence = float(np.max(model.predict_proba(vector_input)))
        
        # Get indicators
        indicators = create_feature_indicators(features)
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': confidence,
            'processed_text': transformed_sms,
            'indicators': indicators
        })
