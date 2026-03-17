import sys
import os

# Garante que src/ é encontrado independente do working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import subprocess

if not os.path.exists(os.path.join(ROOT, 'data', 'telco_churn.csv')):
    subprocess.run(['bash', os.path.join(ROOT, 'setup.sh')], check=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from src.preprocess import load_and_clean, encode, split

st.set_page_config(
    page_title="Churn Risk Detector",
    page_icon="📉",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(ROOT, 'models', 'xgb_model.pkl'))
    feature_cols = joblib.load(os.path.join(ROOT, 'models', 'feature_columns.pkl'))
    explainer = shap.TreeExplainer(model, data=get_test_data(feature_cols))
    return model, feature_cols, explainer

@st.cache_data
def get_test_data(feature_cols):
    df = load_and_clean(os.path.join(ROOT, 'data', 'telco_churn.csv'))
    df = encode(df)
    _, X_test, _, _ = split(df)
    return X_test

def build_customer(tenure, monthly_charges, contract, internet_service,
                   tech_support, online_security, multiple_lines,
                   paperless_billing, payment_method, senior_citizen,
                   partner, dependents, phone_service, online_backup,
                   device_protection, streaming_tv, streaming_movies,
                   feature_cols):
    """Monta o DataFrame do cliente com todas as colunas do modelo."""

    # Base com zeros
    customer = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Numéricas diretas
    customer['tenure'] = tenure
    customer['MonthlyCharges'] = monthly_charges
    customer['TotalCharges'] = tenure * monthly_charges

    # Binárias
    customer['SeniorCitizen'] = 1 if senior_citizen == 'Sim' else 0
    customer['Partner'] = 1 if partner == 'Sim' else 0
    customer['Dependents'] = 1 if dependents == 'Sim' else 0
    customer['PhoneService'] = 1 if phone_service == 'Sim' else 0
    customer['MultipleLines'] = 1 if multiple_lines == 'Sim' else 0
    customer['OnlineSecurity'] = 1 if online_security == 'Sim' else 0
    customer['OnlineBackup'] = 1 if online_backup == 'Sim' else 0
    customer['DeviceProtection'] = 1 if device_protection == 'Sim' else 0
    customer['TechSupport'] = 1 if tech_support == 'Sim' else 0
    customer['StreamingTV'] = 1 if streaming_tv == 'Sim' else 0
    customer['StreamingMovies'] = 1 if streaming_movies == 'Sim' else 0
    customer['PaperlessBilling'] = 1 if paperless_billing == 'Sim' else 0

    # One-hot InternetService
    if internet_service == 'Fiber optic':
        customer['InternetService_Fiber optic'] = 1
    elif internet_service == 'No':
        customer['InternetService_No'] = 1

    # One-hot Contract
    if contract == 'One year':
        customer['Contract_One year'] = 1
    elif contract == 'Two year':
        customer['Contract_Two year'] = 1

    # One-hot PaymentMethod
    if payment_method == 'Credit card (automatic)':
        customer['PaymentMethod_Credit card (automatic)'] = 1
    elif payment_method == 'Electronic check':
        customer['PaymentMethod_Electronic check'] = 1
    elif payment_method == 'Mailed check':
        customer['PaymentMethod_Mailed check'] = 1

    return customer


def render_risk_badge(nivel):
    colors = {
        'ALTO': ('#FCEBEB', '#A32D2D'),
        'MÉDIO': ('#FAEEDA', '#854F0B'),
        'BAIXO': ('#EAF3DE', '#3B6D11')
    }
    bg, fg = colors[nivel]
    st.markdown(
        f'<div style="background:{bg};color:{fg};padding:8px 18px;'
        f'border-radius:8px;font-weight:500;font-size:18px;display:inline-block">'
        f'Risco {nivel}</div>',
        unsafe_allow_html=True
    )


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("Churn Risk Detector")
st.caption("Preveja o risco de cancelamento de clientes com explicação individual")

model, feature_cols, explainer = load_model()

with st.sidebar:
    st.header("Dados do cliente")

    st.subheader("Conta")
    tenure = st.slider("Tempo como cliente (meses)", 0, 72, 12)
    monthly_charges = st.number_input("Cobrança mensal ($)", 18.0, 120.0, 65.0, step=1.0)
    contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Fatura digital", ["Sim", "Não"])
    payment_method = st.selectbox("Método de pagamento", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.subheader("Serviços")
    internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Suporte técnico", ["Sim", "Não"])
    online_security = st.selectbox("Segurança online", ["Sim", "Não"])
    online_backup = st.selectbox("Backup online", ["Sim", "Não"])
    device_protection = st.selectbox("Proteção de dispositivo", ["Sim", "Não"])
    streaming_tv = st.selectbox("Streaming TV", ["Sim", "Não"])
    streaming_movies = st.selectbox("Streaming filmes", ["Sim", "Não"])
    multiple_lines = st.selectbox("Múltiplas linhas", ["Sim", "Não"])
    phone_service = st.selectbox("Serviço de telefone", ["Sim", "Não"])

    st.subheader("Perfil")
    senior_citizen = st.selectbox("Idoso (65+)", ["Não", "Sim"])
    partner = st.selectbox("Tem parceiro(a)", ["Não", "Sim"])
    dependents = st.selectbox("Tem dependentes", ["Não", "Sim"])

    calcular = st.button("Calcular risco", use_container_width=True)

if calcular:
    customer = build_customer(
        tenure, monthly_charges, contract, internet_service,
        tech_support, online_security, multiple_lines,
        paperless_billing, payment_method, senior_citizen,
        partner, dependents, phone_service, online_backup,
        device_protection, streaming_tv, streaming_movies,
        feature_cols
    )

    prob = model.predict_proba(customer)[0][1]
    nivel = 'ALTO' if prob > 0.6 else 'MÉDIO' if prob > 0.3 else 'BAIXO'

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilidade de churn", f"{prob:.1%}")
    col2.metric("Tempo como cliente", f"{tenure} meses")
    col3.metric("Cobrança mensal", f"${monthly_charges:.0f}")

    st.markdown("---")
    render_risk_badge(nivel)
    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall plot
    st.subheader("Por que esse risco?")
    shap_vals = explainer.shap_values(customer)

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
        data=customer.iloc[0].values,
        feature_names=feature_cols
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig)
    plt.close()

    # Tabela de fatores
    st.subheader("Top fatores de risco")
    contributions = pd.Series(shap_vals[0], index=feature_cols)
    top = contributions.abs().nlargest(8).index
    df_factors = pd.DataFrame({
        'Feature': top,
        'Contribuição': contributions[top].round(4),
        'Direção': ['Aumenta risco' if contributions[f] > 0 else 'Reduz risco' for f in top]
    }).reset_index(drop=True)

    st.dataframe(df_factors, use_container_width=True)

else:
    st.info("Configure os dados do cliente na barra lateral e clique em **Calcular risco**.")