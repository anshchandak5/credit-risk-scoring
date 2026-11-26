import streamlit as st
import pandas as pd
import numpy as np
import re
import sqlite3
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

def project_path(relative_path):
    return BASE_DIR / relative_path

# ============================================================
# 2. DARK THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0E1117;
    color: #F5F7FA;
}
.block-container {
    max-width: 1320px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
    background: #1A202B;
    border: 1px solid #303846;
    border-radius: 11px;
    padding: 13px 15px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #171D27;
    border: 1px solid #303846;
    border-radius: 12px;
}
div.stButton > button, div.stFormSubmitButton > button {
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
}
div[data-testid="stSelectbox"] > div > div {
    border-radius: 8px;
}
.profile-label {
    color: #AAB4C3;
    font-size: 0.88rem;
    margin-bottom: 4px;
}
.profile-value {
    color: #F5F7FA;
    font-size: 1.45rem;
    font-weight: 650;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. FEATURE LABELS
# ============================================================

FEATURE_LABELS = {
    "LIMIT_BAL": "Available credit limit",
    "SEX": "Customer gender",
    "EDUCATION": "Education category",
    "MARRIAGE": "Marital status",
    "AGE": "Customer age",
    "PAY_0": "Most recent payment status",
    "PAY_2": "Payment status 2 months ago",
    "PAY_3": "Payment status 3 months ago",
    "PAY_4": "Payment status 4 months ago",
    "PAY_5": "Payment status 5 months ago",
    "PAY_6": "Payment status 6 months ago",
    "BILL_AMT1": "Most recent credit card bill",
    "BILL_AMT2": "Previous credit card bill",
    "BILL_AMT3": "Credit card bill 3 months ago",
    "BILL_AMT4": "Credit card bill 4 months ago",
    "BILL_AMT5": "Credit card bill 5 months ago",
    "BILL_AMT6": "Credit card bill 6 months ago",
    "PAY_AMT1": "Most recent payment amount",
    "PAY_AMT2": "Previous payment amount",
    "PAY_AMT3": "Payment amount 3 months ago",
    "PAY_AMT4": "Payment amount 4 months ago",
    "PAY_AMT5": "Payment amount 5 months ago",
    "PAY_AMT6": "Payment amount 6 months ago",
    "avg_utilization": "Average credit usage",
    "delayed_months": "Number of delayed payment periods",
    "repayment_ratio": "Payment-to-bill ratio",
    "bill_growth": "Change in outstanding bill amount"
}

MODEL_FILES = {
    "XGBoost": "models/xgboost_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "Logistic Regression": "models/logistic_regression_model.pkl"
}

# ============================================================
# 4. LOAD PROJECT DATA
# ============================================================

@st.cache_resource
def load_models():
    loaded = {}
    for model_name, relative_path in MODEL_FILES.items():
        path = project_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        loaded[model_name] = joblib.load(path)
    return loaded

@st.cache_resource
def load_evaluation():
    path = project_path("reports/model_evaluation.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    return joblib.load(path)

@st.cache_data
def load_customer_data():
    db_path = project_path("database/credit_risk.db")
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as connection:
        original = pd.read_sql_query(
            "SELECT * FROM credit_customers", connection
        )
        engineered = pd.read_sql_query(
            """
            SELECT ID, avg_utilization, delayed_months,
                   repayment_ratio, bill_growth
            FROM customer_risk_features
            """,
            connection
        )

    return original.merge(engineered, on="ID", how="inner")

try:
    models = load_models()
    evaluation = load_evaluation()
    df = load_customer_data()

    results = evaluation["results"]
    best_model_name = evaluation.get("best_model", "XGBoost")
    feature_columns = list(evaluation["feature_columns"])

except Exception as exc:
    st.error(f"Unable to load the project files: {exc}")
    st.info(
        "Make sure app.py is in the project folder and that the "
        "models, reports and database folders are alongside it."
    )
    st.stop()

required_columns = feature_columns + [
    "ID", "default payment next month", "AGE", "LIMIT_BAL",
    "BILL_AMT1", "PAY_AMT1", "PAY_0", "delayed_months",
    "avg_utilization", "repayment_ratio", "bill_growth"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Required database columns are missing: {missing_columns}")
    st.stop()

df = df.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def format_currency(value):
    if pd.isna(value):
        return "Not available"
    # The source dataset uses New Taiwan Dollar (NT$).
    return f"NT$ {float(value):,.0f}"

def risk_category(probability):
    if probability < 0.20:
        return "Low Risk"
    if probability < 0.40:
        return "Moderate Risk"
    if probability < 0.60:
        return "High Risk"
    return "Very High Risk"

def display_value(value):
    if pd.isna(value):
        return "Not available"
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):,.2f}"
    return str(value)

def original_feature_name(transformed_name):
    name = str(transformed_name).split("__", 1)[-1]
    for feature in sorted(feature_columns, key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    return name

def shap_for_customer(model, customer_row):
    """Return SHAP contributions aggregated to original input columns."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    number_pattern = re.compile(
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    )

    def safe_float(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            flat = np.asarray(item, dtype=object).reshape(-1)
            parsed = [safe_float(part) for part in flat]
            if not parsed:
                raise ValueError("SHAP returned an empty contribution.")
            return float(np.mean(parsed))

        if isinstance(item, (int, float, np.number)):
            return float(item)

        matches = number_pattern.findall(str(item).strip())
        if not matches:
            raise ValueError(
                f"SHAP returned a non-numeric contribution: {item!r}"
            )
        return float(np.mean([float(match) for match in matches]))

    def as_float_array(value):
        if hasattr(value, "toarray"):
            value = value.toarray()
        object_values = np.asarray(value, dtype=object)
        return np.asarray(
            [safe_float(item) for item in object_values.reshape(-1)],
            dtype=float
        ).reshape(object_values.shape)

    transformed_row = preprocessor.transform(customer_row)
    transformed_row = as_float_array(transformed_row)

    names = list(preprocessor.get_feature_names_out())

    background_raw = df[feature_columns].sample(
        n=min(80, len(df)), random_state=42
    )
    background = preprocessor.transform(background_raw)
    background = as_float_array(background)

    if hasattr(classifier, "get_booster"):
        contributions = classifier.get_booster().predict(
            xgb.DMatrix(transformed_row),
            pred_contribs=True
        )
        raw = np.asarray(contributions)[:, :-1]
    elif hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        raw = explainer.shap_values(transformed_row)
    elif hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, background)
        raw = explainer.shap_values(transformed_row)
    else:
        explainer = shap.Explainer(classifier, background)
        raw = explainer(transformed_row).values

    if hasattr(raw, "values"):
        raw = raw.values
    if isinstance(raw, list):
        raw = raw[1] if len(raw) > 1 else raw[0]

    values = as_float_array(raw)

    # Some SHAP / XGBoost version combinations return object arrays
    # containing strings such as "['2.2120833E-1']". Convert each
    # scalar robustly, including scientific notation and nested arrays.
    if not np.issubdtype(values.dtype, np.number):
        number_pattern = re.compile(
            r"[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?"
        )

        def safe_float(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                flat = np.asarray(item, dtype=object).reshape(-1)
                parsed = [safe_float(part) for part in flat]
                if not parsed:
                    raise ValueError("SHAP returned an empty contribution.")
                return float(np.mean(parsed))

            if isinstance(item, (int, float, np.number)):
                return float(item)

            text_value = str(item).strip()
            matches = number_pattern.findall(text_value)
            if not matches:
                raise ValueError(
                    f"SHAP returned a non-numeric contribution: {text_value!r}"
                )

            parsed_numbers = [float(match) for match in matches]
            if len(parsed_numbers) == 1:
                return parsed_numbers[0]

            # A scalar SHAP cell should contain one number. If a library
            # version wraps a scalar in a textual array, average the
            # contained numeric values rather than crashing the dashboard.
            return float(np.mean(parsed_numbers))

        values = np.asarray(
            [safe_float(item) for item in values.reshape(-1)],
            dtype=float
        ).reshape(values.shape)
    else:
        values = values.astype(float)

    # Normalize common SHAP binary-classification output shapes.
    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
        else:
            values = values.reshape(values.shape[0], -1)

    values = values.reshape(1, -1)[0]

    if len(names) != len(values):
        raise ValueError(
            "SHAP feature count does not match the preprocessed model input "
            f"({len(values)} contributions vs {len(names)} features)."
        )

    grouped = {}
    for transformed_name, contribution in zip(names, values):
        original_name = original_feature_name(transformed_name)
        grouped[original_name] = (
            grouped.get(original_name, 0.0) + float(contribution)
        )
    return grouped

# ============================================================
# 6. SESSION STATE / NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "overview"
if "assessment" not in st.session_state:
    st.session_state.assessment = None

# ============================================================
# PAGE 1 — MODEL PERFORMANCE OVERVIEW
# ============================================================

if st.session_state.page == "overview":
    st.title("💳 Credit Risk Intelligence")
    st.write(
        "Compare the trained models before assessing an individual customer."
    )
    st.caption("MODEL PERFORMANCE OVERVIEW")
    st.divider()

    st.subheader("Compare Machine Learning Models")
    st.caption(
        "All metrics below are calculated on the same held-out test dataset."
    )

    model_order = ["XGBoost", "Random Forest", "Logistic Regression"]
    columns = st.columns(3)

    for index, name in enumerate(model_order):
        metric_values = results[name]["metrics"]
        is_best = name == best_model_name
        with columns[index]:
            with st.container(border=True):
                st.caption("⭐ BEST RESULT" if is_best else "MODEL")
                st.markdown(f"### {name}")
                st.metric("Accuracy", f"{metric_values['Accuracy'] * 100:.2f}%")
                st.caption(f"Recall: {metric_values['Recall'] * 100:.2f}%")
                st.caption(f"ROC-AUC: {metric_values['ROC-AUC'] * 100:.2f}%")
                st.caption(f"F1 score: {metric_values['F1 Score'] * 100:.2f}%")

    st.subheader("Compare a Performance Measure")
    metric_options = [
        "Accuracy", "Precision", "Recall", "F1 Score",
        "ROC-AUC", "Average Precision", "Balanced Accuracy"
    ]
    selected_metric = st.selectbox(
        "Choose a measure",
        metric_options,
        key="overview_metric"
    )

    chart_values = [
        results[name]["metrics"][selected_metric] * 100
        for name in model_order
    ]

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bars = ax.barh(model_order, chart_values, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    ax.set_title(selected_metric)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption(
        "**Accuracy** = overall correct predictions. "
        "**Recall** = actual defaulters correctly identified. "
        "**ROC-AUC** = how well the model separates defaulters from non-defaulters."
    )

    best_accuracy = max(
        model_order, key=lambda n: results[n]["metrics"]["Accuracy"]
    )
    best_recall = max(
        model_order, key=lambda n: results[n]["metrics"]["Recall"]
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            f"**Highest accuracy: {best_accuracy}** "
            f"({results[best_accuracy]['metrics']['Accuracy'] * 100:.2f}%)."
        )
    with col_b:
        st.success(
            f"**Highest recall: {best_recall}** "
            f"({results[best_recall]['metrics']['Recall'] * 100:.2f}%)."
        )

    st.caption(
        "The model with the highest accuracy is not automatically the best "
        "lending choice. Recall is important because it measures how many "
        "actual defaulters the model catches."
    )

    st.divider()
    st.subheader("Individual Customer Analysis")
    st.write(
        "Open the customer assessment page to view a customer's financial "
        "profile, choose a model and estimate potential credit loss."
    )
    if st.button(
        "Open Customer Risk Assessment →",
        type="primary",
        use_container_width=True
    ):
        st.session_state.page = "customer"
        st.session_state.assessment = None
        st.rerun()

    st.divider()
    st.caption(
        "Educational project using historical credit card data. "
        "Predictions are not validated lending decisions."
    )

# ============================================================
# PAGE 2 — CUSTOMER ASSESSMENT
# ============================================================

elif st.session_state.page == "customer":
    if st.button("← Back to Model Performance"):
        st.session_state.page = "overview"
        st.rerun()

    st.title("👤 Customer Risk Assessment")
    st.write(
        "Assess an existing customer or test a new customer using the "
        "same trained machine-learning models."
    )
    st.divider()

    model_order = ["XGBoost", "Random Forest", "Logistic Regression"]

    def model_option_label(name):
        return (
            f"{name} (Best result)"
            if name == best_model_name else name
        )

    existing_tab, new_tab = st.tabs(
        ["Existing Customer", "Test New Customer"]
    )

    # ========================================================
    # TAB 1 — EXISTING CUSTOMER
    # ========================================================
    with existing_tab:
        st.subheader("1. Select a prediction model")

        model_order = ["XGBoost", "Random Forest", "Logistic Regression"]

        def model_option_label(name):
            return (
                f"{name} (Best result)"
                if name == best_model_name else name
            )

        selected_model_name = st.selectbox(
            "Prediction model",
            model_order,
            index=(
                model_order.index(best_model_name)
                if best_model_name in model_order else 0
            ),
            format_func=model_option_label,
            key="existing_model"
        )
        selected_model = models[selected_model_name]

        st.subheader("2. Select a customer")
        customer_id = st.selectbox(
            "Customer ID",
            df["ID"].tolist(),
            key="existing_customer_id"
        )

        customer = df.loc[df["ID"] == customer_id].iloc[0]
        customer_data = customer[feature_columns].to_frame().T

        # The model verdict is calculated immediately. No lending
        # assumptions are needed to get the credit-risk prediction.
        try:
            probability = float(
                selected_model.predict_proba(customer_data)[0][1]
            )
            category = risk_category(probability)
            st.subheader("Model verdict")

            verdict_col, probability_col = st.columns([1, 2])
            with verdict_col:
                if category == "Low Risk":
                    st.success(f"**{category}**")
                elif category == "Moderate Risk":
                    st.warning(f"**{category}**")
                else:
                    st.error(f"**{category}**")

            with probability_col:
                st.metric(
                    "Probability of default",
                    f"{probability * 100:.2f}%"
                )
                st.progress(float(np.clip(probability, 0.0, 1.0)))

            st.caption(
                f"Prediction from {selected_model_name}. "
                "Risk bands are illustrative project thresholds, "
                "not validated lending or regulatory cut-offs."
            )
        except Exception as exc:
            probability = None
            st.error(f"Could not generate the customer prediction: {exc}")

        st.divider()
        st.subheader("3. Customer financial profile")
        st.caption(
            "These are the customer's recorded database values. "
            "They update when you choose a different customer."
        )

        profile_items = [
            ("Age", f"{int(customer['AGE'])} years"),
            ("Credit limit", format_currency(customer["LIMIT_BAL"])),
            ("Average credit usage", f"{float(customer['avg_utilization']) * 100:.2f}%"),
            ("Most recent bill", format_currency(customer["BILL_AMT1"])),
            ("Most recent payment", format_currency(customer["PAY_AMT1"])),
            ("Delayed payment periods", str(int(customer["delayed_months"]))),
            ("Payment-to-bill ratio", f"{float(customer['repayment_ratio']) * 100:.2f}%"),
            ("Change in bill amount", f"{float(customer['bill_growth']) * 100:.2f}%"),
            ("Most recent payment status", str(int(customer["PAY_0"])))
        ]

        for start in range(0, len(profile_items), 3):
            group = profile_items[start:start + 3]
            profile_cols = st.columns(3)
            for col, (label, value) in zip(profile_cols, group):
                with col:
                    with st.container(border=True):
                        st.caption(label)
                        st.markdown(f"### {value}")

        with st.expander("View complete six-month payment history"):
            history = pd.DataFrame({
                "Period": [
                    "Most recent", "Previous month", "3 months ago",
                    "4 months ago", "5 months ago", "6 months ago"
                ],
                "Bill amount": [
                    customer[f"BILL_AMT{i}"] for i in range(1, 7)
                ],
                "Payment amount": [
                    customer[f"PAY_AMT{i}"] for i in range(1, 7)
                ],
                "Payment status": [
                    customer[f"PAY_{i}"] for i in [0, 2, 3, 4, 5, 6]
                ]
            })
            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

        # SHAP explanation is separate from the verdict and lending
        # calculator. It uses the same selected model and customer.
        st.divider()
        st.subheader("4. Why did the model give this verdict?")

        if probability is not None:
            with st.expander("Show the main factors behind this prediction"):
                try:
                    contributions = shap_for_customer(
                        selected_model, customer_data
                    )
                    explanation = pd.DataFrame({
                        "Feature": list(contributions.keys()),
                        "Contribution": list(contributions.values())
                    })
                    explanation["Customer factor"] = explanation["Feature"].map(
                        lambda feature: FEATURE_LABELS.get(feature, feature)
                    )
                    explanation["Absolute impact"] = (
                        explanation["Contribution"].abs()
                    )
                    top = explanation.sort_values(
                        "Absolute impact", ascending=False
                    ).head(10).copy()

                    display_rows = []
                    for _, row in top.iterrows():
                        impact = float(row["Contribution"])
                        direction = (
                            "Pushes risk higher" if impact > 0
                            else "Pushes risk lower" if impact < 0
                            else "No clear direction"
                        )
                        display_rows.append({
                            "Customer factor": row["Customer factor"],
                            "Recorded value": display_value(
                                customer[row["Feature"]]
                            ),
                            "Influence": direction
                        })

                    st.dataframe(
                        pd.DataFrame(display_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                    chart_data = top.sort_values("Contribution")
                    chart_colors = [
                        "#E45757" if value > 0 else "#4A9BE8"
                        for value in chart_data["Contribution"]
                    ]
                    fig, ax = plt.subplots(figsize=(8, 3.4))
                    ax.barh(
                        chart_data["Customer factor"],
                        chart_data["Contribution"],
                        color=chart_colors,
                        height=0.62
                    )
                    ax.axvline(0, color="#9AA4B2", linewidth=0.8)
                    ax.set_xlabel("Contribution to model output")
                    ax.set_title("Top factors influencing this prediction")
                    ax.grid(axis="x", alpha=0.18)
                    ax.set_axisbelow(True)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

                    st.caption(
                        "Positive contributions push the model towards higher "
                        "default risk; negative contributions push it towards "
                        "lower risk. SHAP does not prove causation."
                    )
                except Exception as exc:
                    st.warning(
                        "The prediction is available, but SHAP could not "
                        "generate an explanation for this model."
                    )
                    st.caption(f"Technical detail: {exc}")

        # Lending calculations are intentionally separate from the
        # machine-learning verdict above.
        st.divider()
        st.subheader("5. Lending situation")
        st.write(
            "Use this section only when you want to estimate the bank's "
            "potential financial loss. It does not change the model verdict."
        )

        latest_bill = max(0.0, float(customer["BILL_AMT1"]))
        with st.form("existing_lending_form"):
            lending_col1, lending_col2 = st.columns(2)

            with lending_col1:
                outstanding_balance = st.number_input(
                    "Outstanding balance / amount owed (NT$)",
                    min_value=0.0,
                    value=latest_bill,
                    step=500.0,
                    key="existing_ead",
                    help=(
                        "Starts with the customer's latest bill. Change it "
                        "if the actual current outstanding balance is different."
                    )
                )
                st.caption(
                    f"Suggested value from latest bill: "
                    f"{format_currency(latest_bill)}"
                )

            with lending_col2:
                recovery_rate = st.slider(
                    "How much does the bank expect to recover?",
                    min_value=0,
                    max_value=100,
                    value=55,
                    step=1,
                    format="%d%%",
                    key="existing_recovery"
                )
                st.caption(
                    f"If default occurs, the estimated unrecovered share is "
                    f"{100 - recovery_rate}%."
                )

            with st.expander("What does the recovery percentage mean?"):
                st.markdown(
                    """
                    If the customer owes **NT$100,000** and the bank expects
                    to recover **55%**, the bank may collect NT$55,000.
                    The remaining NT$45,000 is the loss if default occurs.

                    Use a bank-approved recovery estimate when available.
                    The 55% starting value is only an example.
                    """
                )

            calculate_loss = st.form_submit_button(
                "Calculate potential credit loss",
                type="primary",
                use_container_width=True
            )

        if calculate_loss:
            if probability is None:
                st.error(
                    "The model prediction is unavailable, so expected loss "
                    "cannot be calculated."
                )
            else:
                lgd = 1.0 - recovery_rate / 100.0
                expected_loss = (
                    probability * lgd * float(outstanding_balance)
                )

                st.subheader("Potential credit loss")
                loss_cols = st.columns(3)
                loss_cols[0].metric(
                    "Outstanding balance",
                    format_currency(outstanding_balance)
                )
                loss_cols[1].metric(
                    "Loss if default occurs",
                    f"{lgd * 100:.1f}%"
                )
                loss_cols[2].metric(
                    "Expected credit loss",
                    format_currency(expected_loss)
                )

                with st.expander("See how this amount was calculated"):
                    st.markdown(
                        f"""
                        | Component | Value |
                        |---|---:|
                        | Probability of default | {probability * 100:.2f}% |
                        | Outstanding balance | {format_currency(outstanding_balance)} |
                        | Expected recovery | {recovery_rate}% |
                        | Loss if default occurs | {lgd * 100:.2f}% |
                        | **Expected credit loss** | **{format_currency(expected_loss)}** |
                        """
                    )
                    st.latex(r"ECL = PD \\times LGD \\times EAD")

                st.caption(
                    "Illustrative expected-loss estimate. It is separate "
                    "from the model's risk verdict."
                )

    # ========================================================
    # TAB 2 — TEST A NEW CUSTOMER
    # ========================================================
    with new_tab:
        st.subheader("Test a new customer")
        st.write(
            "Enter the customer's details and six months of payment history. "
            "The app will calculate the additional model features and run "
            "the selected trained model."
        )

        new_model_name = st.selectbox(
            "Prediction model for new customer",
            model_order,
            index=(
                model_order.index(best_model_name)
                if best_model_name in model_order else 0
            ),
            format_func=model_option_label,
            key="new_customer_model"
        )

        with st.form("new_customer_form"):
            st.markdown("#### Personal and credit details")
            personal1, personal2, personal3 = st.columns(3)

            with personal1:
                new_age = st.number_input(
                    "Age (years)", min_value=18, max_value=100,
                    value=30, step=1
                )
            with personal2:
                new_limit = st.number_input(
                    "Credit limit (NT$)", min_value=0.0,
                    value=50000.0, step=1000.0
                )
            with personal3:
                new_sex = st.selectbox(
                    "Gender code",
                    [1, 2],
                    format_func=lambda value: (
                        "Male" if value == 1 else "Female"
                    )
                )

            personal4, personal5 = st.columns(2)
            with personal4:
                education_labels = {
                    1: "Graduate school",
                    2: "University",
                    3: "High school",
                    4: "Other",
                    5: "Unknown (code 5)",
                    6: "Unknown (code 6)",
                    0: "Unknown (code 0)"
                }

                new_education = st.selectbox(
                    "Education",
                    [1, 2, 3, 4, 5, 6, 0],
                    format_func=lambda value: education_labels[value],
                    help=(
                        "The original dataset contains three separate "
                        "unknown education codes (0, 5 and 6). They are "
                        "shown separately so you can identify the exact "
                        "category code used by the model."
                    )
                )
            with personal5:
                new_marriage = st.selectbox(
                    "Marital status",
                    [1, 2, 3, 0],
                    format_func=lambda value: {
                        1: "Married",
                        2: "Single",
                        3: "Other",
                        0: "Unknown"
                    }[value]
                )

            st.markdown("#### Six-month payment history")
            st.caption(
                "Enter the bill amount, payment amount and payment status "
                "for each month. Status 0 means on time; positive values "
                "indicate delayed payment periods; negative values represent "
                "paid duly or no consumption categories in the source dataset."
            )

            bill_values = {}
            payment_values = {}
            payment_status_values = {}

            for month_index in range(1, 7):
                if month_index == 1:
                    period_label = "Most recent month"
                    status_key = 0
                else:
                    period_label = (
                        "Previous month" if month_index == 2
                        else f"{month_index} months ago"
                    )
                    status_key = month_index

                with st.expander(period_label, expanded=(month_index == 1)):
                    bill_col, payment_col, status_col = st.columns(3)

                    with bill_col:
                        bill_values[month_index] = st.number_input(
                            f"Bill amount {month_index} (NT$)",
                            min_value=0.0,
                            value=10000.0 if month_index == 1 else 9000.0,
                            step=500.0,
                            key=f"new_bill_{month_index}"
                        )

                    with payment_col:
                        payment_values[month_index] = st.number_input(
                            f"Payment amount {month_index} (NT$)",
                            min_value=0.0,
                            value=2000.0 if month_index == 1 else 1500.0,
                            step=500.0,
                            key=f"new_payment_{month_index}"
                        )

                    with status_col:
                        payment_status_values[status_key] = st.selectbox(
                            f"Payment status {month_index}",
                            [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                            index=2,
                            format_func=lambda value: (
                                f"{value} — On time" if value == 0
                                else f"{value} — Delayed" if value > 0
                                else f"{value} — No delay / special status"
                            ),
                            key=f"new_status_{status_key}"
                        )

            st.caption(
                "The model also uses four engineered fields: average credit "
                "usage, delayed payment periods, payment-to-bill ratio and "
                "change in bill amount. These are calculated from the values "
                "entered above."
            )

            submit_new_customer = st.form_submit_button(
                "Predict New Customer Risk",
                type="primary",
                use_container_width=True
            )

        if submit_new_customer:
            try:
                # Match the feature names and order used during training.
                new_record = {
                    "LIMIT_BAL": float(new_limit),
                    "SEX": int(new_sex),
                    "EDUCATION": int(new_education),
                    "MARRIAGE": int(new_marriage),
                    "AGE": int(new_age),
                    "PAY_0": int(payment_status_values[0]),
                    "PAY_2": int(payment_status_values[2]),
                    "PAY_3": int(payment_status_values[3]),
                    "PAY_4": int(payment_status_values[4]),
                    "PAY_5": int(payment_status_values[5]),
                    "PAY_6": int(payment_status_values[6]),
                    "BILL_AMT1": float(bill_values[1]),
                    "BILL_AMT2": float(bill_values[2]),
                    "BILL_AMT3": float(bill_values[3]),
                    "BILL_AMT4": float(bill_values[4]),
                    "BILL_AMT5": float(bill_values[5]),
                    "BILL_AMT6": float(bill_values[6]),
                    "PAY_AMT1": float(payment_values[1]),
                    "PAY_AMT2": float(payment_values[2]),
                    "PAY_AMT3": float(payment_values[3]),
                    "PAY_AMT4": float(payment_values[4]),
                    "PAY_AMT5": float(payment_values[5]),
                    "PAY_AMT6": float(payment_values[6])
                }

                bill_total = sum(bill_values.values())
                payment_total = sum(payment_values.values())
                average_bill = bill_total / 6.0

                # Transparent default feature engineering for a new record.
                # Confirm these formulas against the project's run_sql.py
                # before using the outputs for real lending decisions.
                new_record["avg_utilization"] = (
                    average_bill / float(new_limit)
                    if float(new_limit) > 0 else 0.0
                )
                new_record["delayed_months"] = int(
                    sum(1 for status in payment_status_values.values()
                        if status > 0)
                )
                new_record["repayment_ratio"] = (
                    payment_total / bill_total
                    if bill_total > 0 else 0.0
                )
                oldest_bill = abs(float(bill_values[6]))
                new_record["bill_growth"] = (
                    (float(bill_values[1]) - float(bill_values[6]))
                    / oldest_bill
                    if oldest_bill > 0 else 0.0
                )

                new_customer_data = pd.DataFrame(
                    [[new_record[column] for column in feature_columns]],
                    columns=feature_columns
                )

                new_probability = float(
                    models[new_model_name].predict_proba(
                        new_customer_data
                    )[0][1]
                )
                new_category = risk_category(new_probability)

                st.divider()
                st.subheader("New customer prediction")

                result_col1, result_col2 = st.columns([1, 2])
                with result_col1:
                    if new_category == "Low Risk":
                        st.success(f"**{new_category}**")
                    elif new_category == "Moderate Risk":
                        st.warning(f"**{new_category}**")
                    else:
                        st.error(f"**{new_category}**")

                with result_col2:
                    st.metric(
                        "Probability of default",
                        f"{new_probability * 100:.2f}%"
                    )
                    st.progress(
                        float(np.clip(new_probability, 0.0, 1.0))
                    )

                st.subheader("Details used for this prediction")
                new_profile_cols = st.columns(3)
                new_profile = [
                    ("Age", f"{new_age} years"),
                    ("Credit limit", format_currency(new_limit)),
                    ("Most recent bill", format_currency(bill_values[1])),
                    ("Most recent payment", format_currency(payment_values[1])),
                    ("Delayed payment periods", str(new_record["delayed_months"])),
                    ("Average credit usage", f"{new_record['avg_utilization'] * 100:.2f}%"),
                    ("Payment-to-bill ratio", f"{new_record['repayment_ratio'] * 100:.2f}%"),
                    ("Change in bill amount", f"{new_record['bill_growth'] * 100:.2f}%"),
                    ("Prediction model", new_model_name)
                ]
                for index, (label, value) in enumerate(new_profile):
                    with new_profile_cols[index % 3]:
                        st.metric(label, value)


            except Exception as exc:
                st.error(f"New customer prediction failed: {exc}")

    st.divider()
    st.caption(
        "Credit Risk Intelligence | Machine Learning | Financial Risk Analysis"
    )
    st.caption(
        "Educational project using historical credit card data. "
        "Predictions are not validated lending decisions."
    )

