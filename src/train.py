import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
from src.preprocess import load_and_clean, encode, split, save_columns

def get_scale_pos_weight(y_train):
    """
    calcula o peso para compensar desbalanceamento entre classes
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    return neg/pos

def train_baseline(X_train,y_train ):
    """Regressão logística como baseline de comparação"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train,y_train)
    return model

def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=get_scale_pos_weight(y_train),
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        verbosity=1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    return model
def evaluate(model, X_test, y_test, model_name:str):
    y_pred= model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print(f"\n{'='*60}")
    print(f"Avaliação do modelo: {model_name}")
    print(f"{'='*60}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ficou', 'Churnou']))

    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    return y_pred, y_proba

def plot_results(models_data: list, X_test, y_test):
    os.makedirs('assets', exist_ok=True)
    n = len(models_data)
    fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n))

    for i, (name, model) in enumerate(models_data):
        # Confusion matrix
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=['Ficou', 'Churnou'],
            ax=axes[i][0],          # ax separado
            colorbar=False
        )
        axes[i][0].set_title(f'Matriz de Confusão — {name}')

        # Curva ROC
        RocCurveDisplay.from_estimator(
            model, X_test, y_test,
            ax=axes[i][1]           # ax separado
        )
        axes[i][1].set_title(f'Curva ROC — {name}')

    plt.tight_layout()
    plt.savefig('assets/model_evaluation.png', dpi=150, bbox_inches='tight')
    print("\nGráficos salvos em assets/model_evaluation.png")
    plt.show()

if __name__ =='__main__':
    df = load_and_clean('data/telco_churn.csv')
    df = encode(df)
    X_train, X_test, y_train, y_test = split(df)

    save_columns(X_train)

    #Treinando os modelos
    print("Treinando Baseline (Logistic Regression)....")
    baseline = train_baseline(X_train,y_train)

    print("\nTreinando XGBoost...")
    xgb_model= train_xgboost(X_train,y_train)

    #Avalia
    evaluate(baseline,X_test,y_test,'Baseline - Logistic Regression')
    evaluate(xgb_model,X_test,y_test,'XGBoost')

    plot_results([
        ('Baseline - Logistic Regression', baseline),
        ('XGBoost', xgb_model)
    ], X_test, y_test)

    #Salvar melhor modelo
    os.makedirs('models',exist_ok=True)
    joblib.dump(xgb_model,'models/xgb_model.pkl')
    print("\nMelhor modelo salvo em models/xgb_model.pkl")