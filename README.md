# Quora-Ques-Similarity-Challenge-


# Quora Question Similarity Detection

This project focuses on identifying whether two questions on Quora are semantically similar. The goal is to reduce duplicate question entries by leveraging Natural Language Processing (NLP) and deep learning techniques.

## Problem Statement

Given a pair of questions, predict if they are duplicates or not. This is a binary classification task where the model must determine if two input questions convey the same meaning.

## Dataset

- Dataset: Quora Question Pairs
- Size: 400,000+ pairs of questions with binary labels (1 for duplicates, 0 otherwise)
- Preprocessing steps included: lowercasing, punctuation removal, stopword removal, tokenization

## Technologies Used

- Python
- NumPy, Pandas
- Scikit-Learn
- Word2Vec (Google pre-trained vectors)
- TensorFlow / Keras (LSTM Model)
- TF-IDF, cosine similarity, and handcrafted features

## Model Development

Two approaches were developed:
1. **Traditional ML**: Using engineered features with Logistic Regression, Random Forest, and XGBoost.
2. **Deep Learning**: Using LSTM neural networks with pre-trained word embeddings.

### Performance

- LSTM model achieved an F1-score of ~0.88.
- Feature engineering and hyperparameter tuning were key to improving performance.

## Key Learnings

- Practical application of NLP preprocessing techniques
- Training LSTM models for text similarity
- Importance of feature engineering and model evaluation metrics

## How to Run

1. Clone the repository.
2. Open the Jupyter Notebook `Quora_Challenge.ipynb`.
3. Run each cell sequentially after installing dependencies.

## Author

Amit Kushwaha  
[GitHub - AmitS1009](https://github.com/AmitS1009)
