# Projet TP – Data Science avec Python : défaut de paiement carte de crédit

## 1. Contexte métier

Une institution financière accorde des cartes de crédit à plusieurs dizaines de milliers de clients.
Elle subit des pertes importantes liées aux clients qui **ne remboursent pas leurs dettes** (défaut de paiement).

L'objectif business est de :
- **Mieux identifier les clients à risque de défaut**, avant d'accorder ou de renouveler une ligne de crédit.
- Adapter les décisions : ajustement de la limite de crédit, conditions plus strictes, suivi spécifique, voire refus.

Le problème est formulé comme une **classification supervisée binaire** :
> prédire si un client sera en défaut de paiement le mois suivant.

Une partie non supervisée (clustering) permet de **segmenter les clients en profils de risque**.

---

## 2. Données

- Fichier : `credit_default_labeled2.csv`
- Chaque ligne correspond à un client.
- Colonnes principales :
  - `LIMIT_BAL` : limite de crédit.
  - `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` : informations socio-démographiques.
  - `PAY_0`, `PAY_2`, ..., `PAY_6` : historique de statut de paiement / retard.
  - `BILL_AMT1` ... `BILL_AMT6` : montants facturés sur plusieurs mois.
  - `PAY_AMT1` ... `PAY_AMT6` : montants effectivement remboursés.
  - `default.payment.next.month` : **cible** (1 = défaut, 0 = pas de défaut).

---

## 3. Structure du projet

- `eda.ipynb`  
  Notebook Jupyter principal contenant :
  - **EDA** : structure, statistiques descriptives, valeurs manquantes, visualisations, corrélations.
  - **Nettoyage / preprocessing** :
    - Suppression de l'ID (si présent).
    - Encodage one-hot (`pd.get_dummies`) des variables catégorielles.
    - Séparation `X` / `y` (`default.payment.next.month`).
    - Split train / test avec `train_test_split` et `stratify=y`.
    - Normalisation avec `StandardScaler`.
  - **Modèle supervisé** : `LogisticRegression` avec classification report, matrice de confusion, AUC, courbe ROC.
  - **Modèle non supervisé** : `KMeans` (segmentation clients) + taux de défaut par cluster.
  - **Sauvegarde** des artefacts :
    - `model.pkl` : modèle de régression logistique.
    - `scaler.pkl` : scaler.
    - `features.pkl` : liste des colonnes utilisées comme features.
  - **Analyse business** : impact, limites, perspectives.

- `api/model_api.py`  
  API **FastAPI** exposant le modèle via un endpoint `/predict`.

- `credit_default_labeled2.csv`  
  Jeu de données brut.

- `requirements.txt`  
  Dépendances Python du projet.

- `Dockerfile`  
  Image Docker pour lancer l'API en production ou en démonstration.

---

## 4. Installation des dépendances

Créer (optionnellement) un environnement virtuel, puis installer les dépendances principales :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib fastapi uvicorn
```

---

## 5. Utilisation du notebook `eda.ipynb`

1. Ouvrir le notebook dans Jupyter / VS Code.
2. Exécuter les cellules dans l'ordre (de haut en bas).
3. À la fin, vérifier que les fichiers suivants sont générés dans le même dossier :
   - `model.pkl`
   - `scaler.pkl`
   - `features.pkl`

Ces fichiers sont utilisés ensuite par l'API.

---

## 6. Lancer l'API FastAPI

### 6.1. En local (sans Docker)

Depuis le dossier du projet (contenant `api/model_api.py` et les `.pkl`) :

```bash
uvicorn api.model_api:app --reload
```

L'API sera accessible sur `http://127.0.0.1:8000`.

- `GET /` : endpoint de test (message simple).
- `POST /predict` : endpoint de prédiction.

### 6.2. Avec Docker (optionnel / bonus)

Construire l'image :

```bash
docker build -t credit-default-api .
```

Lancer le conteneur :

```bash
docker run -p 8000:8000 credit-default-api
```

L'API sera également accessible sur `http://127.0.0.1:8000`.

### 6.3. Schéma d'entrée `/predict`

Le corps de la requête doit contenir un JSON avec les **colonnes brutes** (sans l'ID ni la cible), par exemple :

```json
{
  "LIMIT_BAL": 20000,
  "SEX": "Female",
  "EDUCATION": "University",
  "MARRIAGE": "Married",
  "AGE": 30,
  "PAY_0": "Paid duly",
  "PAY_2": "Paid duly",
  "PAY_3": "Paid duly",
  "PAY_4": "Paid duly",
  "PAY_5": "Paid duly",
  "PAY_6": "Paid duly",
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0
}
```

### 6.2. Traitement côté API

Dans `model_api.py` :
- Les données reçues sont converties en `DataFrame`.
- `pd.get_dummies` est appliqué pour reproduire l'encodage du notebook.
- Les colonnes sont **réalignées** sur `features.pkl` (reindex avec `fill_value=0`).
- Le scaler est appliqué.
- Le modèle prédit une **probabilité de défaut** et une **classe** (0 ou 1).

Réponse typique :

```json
{
  "prediction": 0,
  "probability_default": 0.23
}
```

---


## 7. Auteurs / Contributeurs

- Mouradi Iliasse
- Zbiri Salah-Eddine
