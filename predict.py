import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load vectorizer and model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/xgb_fake_job_model.pkl")

# Download stopwords if not already
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Clean text the same way as in training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Main prediction function
def predict_fake_job(title, company_profile, description, requirements):
    # Merge all inputs just like in training
    full_text = f"{title} {company_profile} {description} {requirements}"
    clean_input = clean_text(full_text)
    X_input = vectorizer.transform([clean_input])
    
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]
    
    label = "Fake" if pred == 1 else "Real"
    confidence = round(prob * 100, 2)
    
    return label, confidence
