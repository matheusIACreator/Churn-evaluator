import shap
from PIL.ImageOps import pad
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os


def load_explainer(model, X_test=None)->shap.TreeExplainer:
    explainer = shap.TreeExplainer(model,X_test)
    return explainer


def plot_summary(explainer, X_test:pd.DataFrame, save_path:str = 'assets/shap_summary.png'):
    """
    Summary plot global - mostra quas features mais impactam o modelo
    e em qual direção (vermelho= aumenta charn, azul = reduz charn)
    """

    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10,7))
    shap.summary_plot(shap_values,X_test,plot_type='dot',show=False)
    plt.title('Impacto global das features no risco de churn', pad=15)
    plt.tight_layout()
    plt.savefig(save_path,dpi=150,bbox_inches='tight')
    plt.show()
    print(f"Summary plot salvo em {save_path}")

    return shap_values

def plot_bar(explainer, X_test : pd.DataFrame, save_path:str = 'assets/shap_bar.png'):
    """
    Bar plot - ranking simples de impotância média absoluta
    Mais fácil de explicar para não-técnicos
    
    """

    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        plot_type='bar',
        show=False
    )
    plt.title('Importância média das features (SHAP)', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Bar plot salvo em {save_path}")

def explain_customer(
    customer: pd.DataFrame,
    model,
    explainer: shap.TreeExplainer,
    feature_cols: list
) -> dict:
    """
    Explica a predição para um cliente individual.
    Retorna dicionário com probabilidade e top fatores de risco.
    """
    prob = model.predict_proba(customer)[0][1]

    shap_vals = explainer.shap_values(customer)
    contributions = pd.Series(shap_vals[0], index=feature_cols)

    # Top 5 features que mais contribuíram (positivo = aumenta churn)
    top_risk = contributions.nlargest(5)
    top_protect = contributions.nsmallest(3)

    factors = []
    for feat, val in top_risk.items():
        valor_real = customer[feat].values[0]
        direcao = "aumenta" if val > 0 else "reduz"
        factors.append({
            'feature': feat,
            'valor': valor_real,
            'contribuicao': round(float(val), 4),
            'direcao': direcao
        })

    return {
        'probabilidade_churn': round(float(prob), 4),
        'nivel_risco': 'ALTO' if prob > 0.6 else 'MÉDIO' if prob > 0.3 else 'BAIXO',
        'top_fatores': factors
    }


def plot_waterfall(
    customer: pd.DataFrame,
    explainer: shap.TreeExplainer,
    feature_cols: list,
    save_path: str = 'assets/shap_waterfall.png'
):
    """
    Waterfall plot para um cliente individual —
    o gráfico mais visual para mostrar no portfólio
    """
    shap_vals = explainer.shap_values(customer)

    explanation = shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=customer.iloc[0].values,
        feature_names=feature_cols
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title('Explicação individual do risco de churn', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Waterfall plot salvo em {save_path}")