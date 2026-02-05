# 📝 NLP Project — Yelp Review Sentiment Classification

A Natural Language Processing (NLP) project that classifies Yelp reviews into **1-star vs 5-star categories** using text vectorization and machine learning models.

This project demonstrates a complete NLP pipeline including text preprocessing, feature extraction, model training, and evaluation.

---

## 📌 Overview

This project uses the Yelp Review dataset to build a binary text classifier that predicts whether a review is highly negative (1 star) or highly positive (5 stars).

The workflow includes:
- Text cleaning
- Exploratory text analysis
- Feature extraction using Bag-of-Words / TF-IDF
- Model training and evaluation

---

## 🎯 Objective

Classify Yelp reviews into:
- 1 Star → Negative Review
- 5 Star → Positive Review

Using NLP feature engineering and supervised learning.

---

## 🧩 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK / text preprocessing tools
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📊 Dataset

**Yelp Review Dataset (Kaggle)**

Main fields used:
- Review text
- Star rating

Filtering applied:
- Only 1-star and 5-star reviews used for clear sentiment separation.

---

## 🔬 NLP Pipeline

### 1️⃣ Text Preprocessing
- Lowercasing
- Removing punctuation
- Stopword removal
- Tokenization
- Optional stemming/lemmatization (if applied)

---

### 2️⃣ Exploratory Analysis
- Review length distribution
- Word frequency patterns
- Star rating distribution
- Text length vs rating trends

---

### 3️⃣ Feature Extraction

Text converted into numeric features using:

- Count Vectorizer (Bag of Words)
- TF-IDF transformation

Example:

- Raw Text → Token Counts → TF-IDF Features

---

### 4️⃣ Model Training

Typical models used for this task:

- Naive Bayes
- Logistic Regression
- Linear classifiers (depending on your notebook flow)

Training steps:
- Train/test split
- Fit model on vectorized text
- Predict sentiment class

---

### 5️⃣ Evaluation

## Model evaluated using:
- Accuracy
- Confusion Matrix
- Precision / Recall
- F1 Score
- Classification Report

## 📈 Results

- Using Naive Bayes MultinomialNB model:
- `classification_report:`
```
              precision    recall  f1-score   support

           1       0.88      0.70      0.78       228
           5       0.93      0.98      0.96       998

    accuracy                           0.93      1226
   macro avg       0.91      0.84      0.87      1226
weighted avg       0.92      0.93      0.92      1226
```

- Using Naive TF-IDF Transformer:
- `classification_report:`
```
              precision    recall  f1-score   support

           1       0.00      0.00      0.00       228
           5       0.81      1.00      0.90       998

    accuracy                           0.81      1226
   macro avg       0.41      0.50      0.45      1226
weighted avg       0.66      0.81      0.73      1226
```
- The main takeaway from this project is that the Naive Bayes MultinomialNB model outperforms the Naive TF-IDF Transformer model by a huge margin.

---

## ▶️ How to Run

### Clone repository

```bash
git clone https://github.com/rohitb281/nlp-project-yelp_reviews.git
cd nlp-project-yelp_reviews
```

### Install dependencies
```
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### Launch notebook
```
jupyter notebook
```

### Open:
```
NLP Project.ipynb
```
- Run all cells.

---

## 🧠 Concepts Demonstrated
- Text preprocessing pipeline
- NLP feature engineering
- Bag-of-Words modeling
- TF-IDF vectorization
- Binary text classification
- Model evaluation metrics
- Handling labeled text datasets

---

## 🚀 Possible Improvements
- Add bigrams/trigrams
- Try deep learning (LSTM / Transformers)
- Hyperparameter tuning
- Cross-validation
- Handle neutral (3-star) reviews
- Deploy as sentiment API
- Use pre-trained embeddings

---

## ⚠️ Limitations
- Only extreme ratings used (1 & 5)
- Simple preprocessing pipeline
- No sarcasm/context understanding
- Vocabulary limited to training data

----

## 📄 License
- Open for educational and portfolio use.

---

## 👤 Author
- Rohit Bollapragada
- GitHub: https://github.com/rohitb281
