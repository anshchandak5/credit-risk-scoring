# Credit Risk Intelligence

An educational Streamlit dashboard that compares three trained classification models and estimates the probability of credit-card default from an entered customer profile and six months of payment history.

## Live Demo

[Open the live Credit Risk Intelligence dashboard](https://credit-risk-scoring-gikgxwbp7edsnyxvp7yq9w.streamlit.app/)

## Dataset

This project includes the [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients), a public, anonymized research dataset of 30,000 historical credit-card client records. The included SQLite database powers the Existing Customer view; the original UCI spreadsheet is retained for reproducibility.

## Public demo safeguards

- The included records are from the public UCI research dataset, not this project's own customers.
- The dashboard is an educational demonstration and is not a lending or production credit-decision system.
- Included model artifacts support the dashboard's XGBoost, Random Forest, and Logistic Regression choices. `reports/model_evaluation.pkl` supplies the displayed evaluation metrics and feature order.
- This demonstration is not financial advice, a production credit-decision system, or a validated lending model.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select the branch containing this project, set the entry point to `app.py`, and choose Python 3.11 in Advanced settings.
4. Deploy and use the generated `streamlit.app` URL.

## Files intentionally excluded

The `.gitignore` prevents publication of virtual environments, environment files, and Streamlit secrets.

