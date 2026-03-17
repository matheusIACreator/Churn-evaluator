import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean(path:str)->pd.DataFrame:
    df=pd.read_csv(path)

    #TotalCharges vem como string com espaços em branco
    df['TotalCharges']= pd.to_numeric(df['TotalCharges'],errors='coerce')

    #Clientes com Ternure=0 na tem total charges ainda - preenche com 0
    df['TotalCharges']=df['TotalCharges'].fillna(0)

    #remove coluna de ID - nao é feature

    df=df.drop(columns=['customerID'])

    return df

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binárias diretas
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'Churn'
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # Normaliza variações de "No" ANTES de mapear
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col] = df[col].replace({
            'No internet service': 'No',
            'No phone service': 'No'
        })
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        # Garante int — map pode deixar NaN se sobrou valor inesperado
        df[col] = df[col].fillna(0).astype(int)

    # One-hot nas categóricas restantes
    df = pd.get_dummies(
        df,
        columns=['InternetService', 'Contract', 'PaymentMethod'],
        drop_first=True,
        dtype=int   # força int em vez de bool — evita problemas no sklearn
    )

    return df
    
def split(df:pd.DataFrame,target:str = 'Churn',test_size:float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]

    x_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=42,stratify=y)

    return x_train, X_test, y_train, y_test

def save_columns(X_train: pd.DataFrame,path:str='models/feature_columns.pk1'):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    joblib.dump(list(X_train.columns),path)
    print(f"Colunas salvas em {path} - total: {len(X_train.columns)} features")