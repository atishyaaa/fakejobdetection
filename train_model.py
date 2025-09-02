import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# Download stopwords if not already present
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

# Load original dataset
df = pd.read_csv("data/fake_job_postings.csv")
print("Original dataset shape:", df.shape)
print(df.head())

# Add 30 realistic fake jobs
modern_fake_jobs = [
    {
        'title': "Remote Data Entry Operator",
        'location': "Remote",
        'department': "Operations",
        'description': "Earn ₹40,000/month from home. Just type PDF to Word. No experience needed.",
        'requirements': "Laptop or phone. ₹399 registration fee. No interviews required.",
        'fraudulent': 1
    },
    {
        'title': "Instagram Brand Collaborator",
        'location': "Mumbai, India",
        'department': "Marketing",
        'description': "Get ₹2,000/post by promoting our fashion products on Instagram.",
        'requirements': "Public Instagram. ₹699 onboarding kit. Must tag 3 pages to start.",
        'fraudulent': 1
    },
    {
        'title': "Crypto Investment Trainee",
        'location': "Bangalore",
        'department': "Finance",
        'description': "Learn crypto trading. Double your investment in 15 days. Mentorship provided.",
        'requirements': "₹1,000 investment. Join Telegram group. No formalities.",
        'fraudulent': 1
    },
    {
        'title': "WhatsApp Survey Responder",
        'location': "Remote",
        'department': "Support",
        'description': "Fill 100+ surveys daily on WhatsApp. Earn ₹25 per form. Paid every evening.",
        'requirements': "Pay ₹299 activation. Aadhar & UPI needed. Task login shared post-payment.",
        'fraudulent': 1
    },
    {
        'title': "Work-from-Home YouTube Commenter",
        'location': "Remote",
        'department': "Content",
        'description': "Get ₹10/comment on trending videos. Daily payout guaranteed.",
        'requirements': "Complete paid trial of ₹250 for account setup. No interview.",
        'fraudulent': 1
    },
    {
        'title': "Online Internship - Marketing Executive",
        'location': "Delhi",
        'department': "Marketing",
        'description': "Virtual internship with offer letter. Great for freshers and students.",
        'requirements': "₹599 internship fee. No interview. Offer letter on payment.",
        'fraudulent': 1
    },
    {
        'title': "Telegram Influencer Program",
        'location': "Online",
        'department': "Branding",
        'description': "Earn ₹1,000 daily sharing content via Telegram. Weekly bonuses.",
        'requirements': "Pay ₹499 toolkit fee. Referral mandatory. Telegram ID needed.",
        'fraudulent': 1
    },
    {
        'title': "Freelance Captcha Solver",
        'location': "Remote",
        'department': "Tech",
        'description': "Earn ₹5,000/week solving captchas. Unlimited work available.",
        'requirements': "₹300 deposit required. Task ID shared via WhatsApp.",
        'fraudulent': 1
    },
    {
        'title': "Amazon Package Receiver",
        'location': "Hyderabad",
        'department': "Logistics",
        'description': "Receive packages and forward them. ₹2,000 per package shipped.",
        'requirements': "Provide home address. ₹500 refundable verification fee.",
        'fraudulent': 1
    },
    {
        'title': "Work-from-Home Typist",
        'location': "Remote",
        'department': "Admin",
        'description': "Simple form filling from PDF. No interview. Start immediately.",
        'requirements': "₹499 software charge. Training included. Start today.",
        'fraudulent': 1
    },
    {
        'title': "ML Intern – Certificate Provided",
        'location': "Chennai",
        'department': "AI Research",
        'description': "Work on exciting ML projects remotely. Earn a completion certificate.",
        'requirements': "₹799 internship processing fee. No coding test. Quick onboarding.",
        'fraudulent': 1
    },
    {
        'title': "Quick Loan Processing Officer",
        'location': "Pune",
        'department': "Finance",
        'description': "Help customers with instant loan approvals. ₹25,000 fixed pay.",
        'requirements': "Submit ₹499 for onboarding. No job history required.",
        'fraudulent': 1
    },
    {
        'title': "Remote Review Writer",
        'location': "Anywhere in India",
        'department': "Content",
        'description': "Write fake reviews for Amazon & Google. ₹50 per review.",
        'requirements': "₹299 login setup. Mobile required. No approval needed.",
        'fraudulent': 1
    },
    {
        'title': "Voice Artist (Chat Roleplay)",
        'location': "Remote",
        'department': "Entertainment",
        'description': "Roleplay using pre-written scripts. Earn ₹1,500/hr.",
        'requirements': "Pay ₹499 for script access. NDA mandatory.",
        'fraudulent': 1
    },
    {
        'title': "Student Internship – Digital Marketing",
        'location': "Online",
        'department': "Marketing",
        'description': "Learn and earn internship. Live projects + letter.",
        'requirements': "₹999 enrollment. No selection process. Start next day.",
        'fraudulent': 1
    },
    {
        'title': "Gaming Tester Job",
        'location': "Remote",
        'department': "Tech",
        'description': "Test online games and earn ₹700/day. Flexible hours.",
        'requirements': "₹349 entry pass. Start same day. No tests.",
        'fraudulent': 1
    },
    {
        'title': "Telegram Trading Job",
        'location': "Remote",
        'department': "Finance",
        'description': "Follow trade tips and earn commissions. Fast ROI guaranteed.",
        'requirements': "Deposit ₹1,000. WhatsApp updates daily. No formal job letter.",
        'fraudulent': 1
    },
    {
        'title': "Work-from-Home WhatsApp Recruiter",
        'location': "India",
        'department': "HR",
        'description': "Recruit candidates for ₹100/referral. Daily payouts.",
        'requirements': "₹299 admin charge. UPI and ID proof required.",
        'fraudulent': 1
    },
    {
        'title': "AI Tool Tester – Remote",
        'location': "Anywhere",
        'department': "AI Tools",
        'description': "Test our AI voice tool. ₹100 per task. 10 tasks daily.",
        'requirements': "₹500 access fee. Start same day.",
        'fraudulent': 1
    },
    {
        'title': "Facebook Page Commenter",
        'location': "Remote",
        'department': "PR",
        'description': "Engage with our posts daily. Earn per like & comment.",
        'requirements': "₹199 setup. Social media required.",
        'fraudulent': 1
    },
    {
        'title': "Internship – UI/UX Remote",
        'location': "Online",
        'department': "Design",
        'description': "Intern with startup. No interview. Free tools shared.",
        'requirements': "₹399 onboarding. Guaranteed internship.",
        'fraudulent': 1
    },
    {
        'title': "Earn from Reels – No Followers Required",
        'location': "Remote",
        'department': "Social Media",
        'description': "Earn ₹500/reel. Just upload from templates.",
        'requirements': "Pay ₹349 content access. No approval needed.",
        'fraudulent': 1
    },
    {
        'title': "Online Tutor – Quick Hiring",
        'location': "Remote",
        'department': "Education",
        'description': "Teach basic English. ₹1,200/day fixed. No degree needed.",
        'requirements': "₹599 verification charge. Start next day.",
        'fraudulent': 1
    },
    {
        'title': "Remote Editor Intern (No Interview)",
        'location': "India",
        'department': "Content",
        'description': "Work on blog editing. No skills needed. Training provided.",
        'requirements': "₹299 platform entry. Letter on completion.",
        'fraudulent': 1
    },
    {
        'title': "Photo Editing Assistant",
        'location': "Online",
        'department': "Design",
        'description': "Edit 10 images/day. Earn ₹100/image. No tools required.",
        'requirements': "₹349 Photoshop ID. Start today.",
        'fraudulent': 1
    },
    {
        'title': "Data Validation Intern (Paid)",
        'location': "Remote",
        'department': "Data",
        'description': "Review Google Sheets and approve entries. Easy work.",
        'requirements': "₹499 training + access. No onboarding delay.",
        'fraudulent': 1
    },
    {
        'title': "Form Filling Agent",
        'location': "Remote",
        'department': "Support",
        'description': "Submit student forms daily. ₹800/day.",
        'requirements': "₹399 login ID charge. Mobile only.",
        'fraudulent': 1
    },
    {
        'title': "Internship – SEO Trainee",
        'location': "Remote",
        'department': "Digital",
        'description': "Learn SEO in 4 weeks. Placement support guaranteed.",
        'requirements': "₹599 registration. Offer letter upfront.",
        'fraudulent': 1
    },
    {
        'title': "Remote BPO Voice Support – Hindi",
        'location': "Remote",
        'department': "Customer Care",
        'description': "Take voice calls and earn ₹1,000/day. No training needed.",
        'requirements': "₹399 setup fee. WhatsApp based login.",
        'fraudulent': 1
    },
    {
        'title': "WhatsApp Group Manager",
        'location': "Remote",
        'department': "Marketing",
        'description': "Manage 20 groups. Share posts. ₹100/post payout.",
        'requirements': "Pay ₹499 for slot booking. Paid via UPI.",
        'fraudulent': 1
    }
]


modern_fake_df = pd.DataFrame(modern_fake_jobs)

# Fill missing columns if any
for col in df.columns:
    if col not in modern_fake_df.columns:
        modern_fake_df[col] = ""

# Match column order and append
modern_fake_df = modern_fake_df[df.columns]
df = pd.concat([df, modern_fake_df], ignore_index=True)
print("✅ Updated dataset shape:", df.shape)


print("\n🔍 Missing values per column:\n", df.isnull().sum())
print("\n📊 Class distribution:\n", df['fraudulent'].value_counts())


# Drop irrelevant columns
df.drop(columns=[
    'job_id', 'salary_range', 'benefits', 'telecommuting',
    'has_company_logo', 'has_questions', 'employment_type',
    'required_experience', 'required_education', 'industry',
    'function'
], inplace=True, errors='ignore')

# Merge text columns
df['text'] = (
    df['title'].fillna('') + ' ' +
    df['company_profile'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['requirements'].fillna('')
)
df = df[df['text'].str.strip() != '']

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['fraudulent']
print("TF-IDF matrix shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance - Calculate scale_pos_weight
scale = y_train.value_counts()[0] / y_train.value_counts()[1]

# XGBoost Model
xgb_model = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("✅ [XGBoost] Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# SHAP Explanation (optional in scripts)
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

explainer = shap.Explainer(xgb_model, X_train_dense)
shap_values = explainer(X_test_dense)

# Optional visualization (uncomment in notebook use)
# shap.plots.beeswarm(shap_values)
# shap.plots.waterfall(shap_values[0])

# Save model and vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(xgb_model, "models/xgb_fake_job_model.pkl")
print("✅ Vectorizer and model saved.")
