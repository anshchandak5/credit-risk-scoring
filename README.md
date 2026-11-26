# Credit Risk Intelligence

I built this project to understand how machine-learning models can be used to estimate credit-card default risk. It is an educational Streamlit dashboard where you can compare models, inspect sample customer profiles, and test a new customer profile yourself.

## Live demo

[Open the app here](https://credit-risk-scoring-gikgxwbp7edsnyxvp7yq9w.streamlit.app/)

## What the project does

- Compares XGBoost, Random Forest, and Logistic Regression models.
- Shows accuracy, recall, ROC-AUC, F1 score, and other evaluation metrics.
- Lets you select from the included customer records and view a prediction.
- Lets you enter a new customer profile and generate a risk estimate.
- Shows the main factors behind an XGBoost prediction.
- Includes a simple expected-credit-loss calculator for learning purposes.

## Dataset

I used the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients). It contains anonymized historical credit-card records from 30,000 clients and is included here for reproducibility and demonstration.

## Tools used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- SQLite

## Run it locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Note

This is a portfolio and learning project. The predictions, risk categories, and loss calculations are illustrative only and should not be used for real lending decisions.

