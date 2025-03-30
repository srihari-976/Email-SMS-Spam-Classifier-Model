# 📧 Email/SMS Spam Classifier

An intelligent spam classification system that uses machine learning to identify spam messages with high accuracy. The system features a modern dark-themed UI and provides detailed analysis of message characteristics.

## Deployed Application

🚀 **Check out the live version of Email SMS Spam Classifier Model!** 🚀

[![Live Application](https://img.shields.io/badge/Live%20Application-Click%20Here-brightgreen)](https://spam-mail-detector.streamlit.app/)

## 🚀 Features

- Real-time spam detection
- Confidence score visualization
- Suspicious feature detection
- Dark mode interface
- Detailed text analysis
- 97.67% accuracy on test data

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📝 Example Messages to Try

### Spam Examples

1. **Lottery Scam**:
```
CONGRATULATIONS! You've won £1,000,000 in the UK lottery! To claim your prize, contact us at +44-XXX-XXXX or reply NOW!
```

2. **Prize Winner**:
```
You are the lucky winner of 2 lakh rupees! Contact immediately to claim your cash prize! Urgent - respond within 24hrs!!!
```

3. **Banking Scam**:
```
Dear Customer, your account will be suspended! Update your KYC by clicking http://fakebank.com immediately!
```

4. **Marketing Spam**:
```
FREE GIFT! Buy one get THREE free! Limited time offer - 90% OFF on all items! Shop now at www.fakeshop.com
```

5. **Investment Scam**:
```
INVEST NOW! 1000% guaranteed returns in crypto! Don't miss this opportunity. Contact our expert: +1-XXX-XXXX
```

### Ham (Non-Spam) Examples

1. **Regular Meeting**:
```
Hi Team, reminder about our weekly meeting tomorrow at 10 AM. Please prepare your updates.
```

2. **Friend's Message**:
```
Hey! Are we still on for dinner tonight at 7? Let me know if you need directions to the restaurant.
```

3. **Delivery Update**:
```
Your package has been delivered. Thank you for shopping with us!
```

4. **Birthday Wish**:
```
Happy Birthday! Hope you have a wonderful day filled with joy and laughter. 🎂
```

5. **Work Related**:
```
Please review the attached document and send your feedback by end of day.
```

## 🔍 Features to Notice

The classifier looks for several suspicious patterns:
- Multiple exclamation marks (!!!)
- Currency symbols (₹, $, £, €)
- Excessive uppercase text
- Multiple numbers
- Common spam keywords (free, win, prize, urgent)
- URLs and phone numbers
- Unusual punctuation patterns

## 📊 Performance

- Training Accuracy: 99.17%
- Testing Accuracy: 97.67%
- Cross-validation Score: 98.50%

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request


## 🙏 Acknowledgments

- Dataset source: UCI Machine Learning Repository
- NLTK for text processing
- Streamlit for the web interface
- scikit-learn for machine learning
