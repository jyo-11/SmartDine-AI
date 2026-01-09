# 🍽 SmartDine AI – Restaurant Review Sentiment Analysis

SmartDine AI is an end-to-end Natural Language Processing (NLP) application that predicts whether a restaurant review expresses **positive** or **negative** sentiment.  
It uses **TF-IDF vectorization** with a **Logistic Regression** classifier and is deployed as a **real-time web application** using Streamlit.

🔗 **Live Application**  
https://smartdine-ai-tyvv5zuqgkuwvijou88wek.streamlit.app/

---

## 📌 Problem Statement

Online restaurant reviews strongly influence customer decisions and restaurant reputation.  
Reading thousands of reviews manually is inefficient and subjective. SmartDine AI automates this process by classifying reviews into positive or negative sentiment, allowing businesses and customers to quickly understand overall feedback.

---

## 🧠 Machine Learning Approach

### Dataset  
The project uses real Zomato restaurant reviews from Hyderabad.  
Each record contains:
- Review text  
- Rating (1–5)

Ratings are converted into sentiment labels:
- **Rating ≥ 4 → Positive**
- **Rating ≤ 2 → Negative**
- **Rating = 3 → Neutral (removed)**

---

### Text Processing  
Reviews are transformed into numerical features using **TF-IDF (Term Frequency – Inverse Document Frequency)** with:
- Stopword removal  
- Unigrams and bigrams  
- Maximum 5000 features  

This allows the model to identify which words and phrases are most important for sentiment prediction.

---

### Model  
A **Logistic Regression** classifier is trained on the TF-IDF vectors to predict sentiment.

The model achieved approximately **92% accuracy** on unseen test data.

---

### Model Persistence  
To ensure consistency between training and deployment:
- The trained **TF-IDF vectorizer**
- The trained **Logistic Regression model**

are serialized and saved as `.pkl` files and reused during inference.

---

### Deployment  
The model is deployed using **Streamlit**, allowing users to:
- Enter any restaurant review
- Receive:
  - Sentiment prediction (Positive / Negative)
  - Confidence score
  - Key words influencing the prediction  

This demonstrates how machine learning models transition from research notebooks to real-world applications.

---

## 🖥 Application Features

- Real-time sentiment prediction  
- Prediction confidence score  
- Displays influential words affecting the model  
- Interactive interface to experiment with text changes  

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF  
- Logistic Regression  
- Streamlit  

---

## ▶ Run Locally

```bash
git clone <your-repository-url>
cd SmartDine-AI
pip install -r requirements.txt
streamlit run app.py
