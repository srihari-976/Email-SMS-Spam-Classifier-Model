import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="wide"
)

import pickle
import string
import nltk
import re
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

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

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(height=250, margin={'l': 10, 'r': 10, 't': 10, 'b': 10})
    return fig

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
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea {
        font-size: 16px;
    }
    .result-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .spam {
        background-color: #FF4B4B;
        border: 2px solid #FF0000;
        color: white;
    }
    .spam h2 {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    .ham {
        background-color: #00CC66;
        border: 2px solid #00994C;
        color: white;
    }
    .ham h2 {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    .indicator {
        padding: 0.75rem 1rem;
        margin: 0.5rem;
        border-radius: 8px;
        background-color: #2E303E;
        color: white;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .stTextArea textarea {
        background-color: #2E303E;
        color: white;
        border: 1px solid #404040;
    }
    .stTextArea textarea:focus {
        border-color: #666666;
        box-shadow: 0 0 0 2px rgba(102, 102, 102, 0.2);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    h1, h2, h3, p {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
st.title("📧 Email/SMS Spam Classifier")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Enter your message below to check if it's spam or not
    </div>
""", unsafe_allow_html=True)

# Input area
input_sms = st.text_area("Enter your message:", height=150)

# Prediction button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button('🔍 Predict', use_container_width=True)

if predict_button and input_sms:
    with st.spinner('Analyzing message...'):
        # Extract features
        features = extract_features(input_sms)
        
        # Preprocess
        transformed_sms = transform_text(input_sms)
        
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict
        prediction = model.predict(vector_input)[0]
        confidence = np.max(model.predict_proba(vector_input))
        
        # Display results
        st.markdown("### Results")
        
        # Create result box with appropriate styling
        result_class = "spam" if prediction == 1 else "ham"
        result_text = "⚠️ SPAM" if prediction == 1 else "✅ Not Spam"
        
        st.markdown(f"""
            <div class='result-box {result_class}'>
                <h2 style='text-align: center; margin: 0;'>{result_text}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Display confidence gauge
        st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
        
        # Display confidence percentage
        st.markdown(f"""
            <div style='text-align: center; color: #666;'>
                Confidence: {confidence:.2%}
            </div>
        """, unsafe_allow_html=True)
        
        # Display suspicious features
        indicators = create_feature_indicators(features)
        if indicators:
            st.markdown("### Suspicious Features Detected")
            for indicator in indicators:
                st.markdown(f"<div class='indicator'>{indicator}</div>", unsafe_allow_html=True)
        
        # Display processed text
        with st.expander("View processed text"):
            st.text(transformed_sms)