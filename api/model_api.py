import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


class ClientFeatures(BaseModel):
    """
    Schéma d'entrée aligné sur les colonnes brutes du CSV,
    SANS la cible `default.payment.next.month` ni l'ID.
    """

    # Caractéristiques socio-démographiques
    LIMIT_BAL: float
    SEX: str
    EDUCATION: str
    MARRIAGE: str
    AGE: int

    # Historique de paiement (statuts de paiement, sous forme de texte, comme dans le CSV)
    PAY_0: str
    PAY_2: str
    PAY_3: str
    PAY_4: str
    PAY_5: str
    PAY_6: str

    # Montants facturés
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    # Montants remboursés
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


app = FastAPI(title="API défaut de paiement", version="1.0")


@app.on_event("startup")
def load_model() -> None:
    """
    Chargement du modèle, du scaler et de la liste de features
    entraînés dans le notebook `eda.ipynb`.
    """
    global model, scaler, feature_columns
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("features.pkl")


@app.get("/")
def read_root() -> dict:
    return {"message": "API de prédiction de défaut de paiement en ligne."}


@app.post("/predict")
def predict_default(client: ClientFeatures) -> dict:
    """
    1. Reçoit les features brutes (comme dans le CSV).
    2. Applique le même encodage que dans le notebook (get_dummies).
    3. Aligne les colonnes sur `feature_columns`.
    4. Applique le scaler et le modèle.
    """
    # 1) Passage en DataFrame
    df_input = pd.DataFrame([client.dict()])

    # 2) Encodage one-hot des variables catégorielles (comme dans le notebook)
    df_encoded = pd.get_dummies(df_input)

    # 3) Alignement sur les colonnes utilisées à l'entraînement
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4) Mise à l'échelle + prédiction
    X_scaled = scaler.transform(df_aligned.values)
    proba_default = model.predict_proba(X_scaled)[:, 1][0]
    prediction = int(proba_default >= 0.5)

    return {
        "prediction": prediction,
        "probability_default": float(proba_default),
    }

