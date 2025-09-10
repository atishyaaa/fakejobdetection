# Honest Hire – Fake Job Detection 🔍

## 📌 Overview
The internet is flooded with fraudulent job postings that mislead applicants and waste valuable time. **Honest Hire** is a machine learning–powered solution that identifies whether a job posting is real or fake, helping job seekers stay safe.

This project applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to detect fraudulent job ads based on text patterns and metadata.

---

## ⚙️ Features
- Detects **real vs fake** job postings  
- Uses **XGBoost classifier** with **TF-IDF vectorization**  
- Simple **Streamlit app** for making predictions  
- End-to-end pipeline: preprocessing → training → prediction  

---

## 🛠️ Tech Stack

- **Programming Language**: Python 🐍  
- **Data Handling**: Pandas  
- **Text Preprocessing**: NLTK (stopwords, regex cleaning)  
- **Feature Extraction**: Scikit-learn (TF-IDF Vectorizer)  
- **Modeling**: XGBoost (XGBClassifier)  
- **Evaluation**: Scikit-learn (accuracy, classification report, confusion matrix)  
- **Explainability**: SHAP (model interpretability)  
- **Serialization**: Joblib (saving models & vectorizers)  
- **App Framework**: Streamlit (frontend)  

---

## 📂 Project Structure

* `fakejobdetection/`
    * `models/`: Pre-trained models
    * `notebooks/`: Jupyter notebooks
    * `.gitignore`: Files to ignore in Git
    * `app.py`: Streamlit application
    * `predict.py`: Script for predictions
    * `README.md`: Project documentation
    * `requirements.txt`: Dependencies
    * `train_model.py`: Training script
---

### 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/atishyaaa/fakejobdetection.git](https://github.com/atishyaaa/fakejobdetection.git)
    cd fakejobdetection
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

---

### 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

### 📜 License

This project is licensed under the MIT License.
