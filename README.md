# Démo : régression linéaire en application (FastAPI + Streamlit)

Application pédagogique qui montre comment un modèle de machine learning entraîné dans un notebook devient un **service** (API) consommé par une **interface** (formulaire web).

Jeu de données : `Advertising.csv` (200 observations).
- Entrées : `TV`, `radio`, `newspaper` (budgets publicitaires, en milliers de $)
- Cible : `sales` (ventes, en milliers de $)

```
Streamlit (formulaire) --HTTP/JSON--> FastAPI --> pipeline.predict (scaler + modèle) --> JSON --> Streamlit
```

## Structure finale du projet

```
demo-regression/
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env                  # réglages locaux (non versionné)
├── .env.example          # modèle de .env (versionné)
├── README.md
├── train_model.py        # découpe, entraîne et sauvegarde le pipeline
├── test_api.py           # test de l'API en Python
├── data/
│   └── Advertising.csv
├── models/
│   └── pipeline.joblib   # scaler + régression, créé par train_model.py
├── api/
│   ├── __init__.py
│   ├── main.py           # routes de l'API
│   └── schemas/
│       ├── __init__.py   # ré-exporte les schémas
│       └── budget.py     # classe Budget
└── app/
    └── streamlit_app.py
```

## Prérequis

- Python 3.10 ou plus récent
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets et d'environnements)
- Le fichier `Advertising.csv`

Installation de uv :

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Étape 1 : créer le projet

```bash
uv init demo-regression
cd demo-regression
```

## Étape 2 : dépendances et dossiers

```bash
uv add fastapi "uvicorn[standard]" streamlit requests joblib pandas python-dotenv "scikit-learn==1.6.1"
mkdir api app models data
mkdir api/schemas
```

Supprimez le `main.py` généré par défaut (`rm main.py` sous Linux/macOS, `del main.py` sous Windows).

> **Pourquoi `scikit-learn==1.6.1` ?** Un fichier `.joblib` doit être relu avec la version de scikit-learn qui l'a créé, sinon des avertissements (voire des erreurs) apparaissent. Épingler le numéro garantit que tous les postes, et plus tard le serveur de déploiement, utilisent la même version. Pour connaître la vôtre : `import sklearn; print(sklearn.__version__)`.

> **Pourquoi `uvicorn[standard]` ?** Le suffixe ajoute des dépendances optionnelles, dont `watchfiles`, qui permet le rechargement automatique (`--reload`) en développement.

> **Pourquoi `python-dotenv` ?** Il lit le fichier `.env` (voir l'étape 7). Il est déjà installé indirectement par `uvicorn[standard]`, mais on le déclare explicitement puisque notre code l'utilise.

## Étape 3 : entraîner et sauvegarder le pipeline

Placez `Advertising.csv` dans `data/`, puis créez `train_model.py` à la racine du projet :

```python
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Chargement des données
df = pd.read_csv("data/Advertising.csv")
X = df[["TV", "radio", "newspaper"]]   # l'ordre des colonnes est important
y = df["sales"]

# 2. Découpage : 80 % pour apprendre, 20 % pour évaluer sur des données jamais vues
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Pipeline : standardisation puis régression linéaire, enchaînées dans UN objet
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression()),
])

# 4. Entraînement sur le jeu d'apprentissage uniquement
pipeline.fit(X_train, y_train)

# 5. Évaluation sur le jeu de test
y_pred = pipeline.predict(X_test)
print(f"R²   : {r2_score(y_test, y_pred):.3f}")
print(f"RMSE : {mean_squared_error(y_test, y_pred) ** 0.5:.3f}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.3f}")

# 6. Sauvegarde d'un seul fichier : le pipeline complet
joblib.dump(pipeline, "models/pipeline.joblib")

# 7. Contrôle : un coefficient par variable (sur données standardisées)
print("Coefficients :", pipeline.named_steps["regression"].coef_)
```

```bash
uv run python train_model.py
```

Résultat attendu (relevé en exécutant ce script sur le jeu de données du projet ; ces valeurs correspondent à celles du notebook d'origine) :

```
R²   : 0.899
RMSE : 1.782
MAE  : 1.461
Coefficients : [3.76419618 2.79230665 0.05597551]
```

### Pourquoi `train_test_split` ?

Évaluer un modèle sur les données qui ont servi à l'entraîner est trompeur : il peut simplement les « retenir ». On met donc de côté 20 % des lignes (`test_size=0.2`) que le modèle ne voit jamais pendant l'apprentissage, puis on mesure sa qualité sur elles. `random_state=42` fixe le tirage au sort : le découpage est identique à chaque exécution, les résultats sont reproductibles.

### Pourquoi un `Pipeline` ?

Un `Pipeline` enchaîne plusieurs étapes (ici `StandardScaler` puis `LinearRegression`) dans **un seul objet** :

- **Un seul fichier** à sauvegarder et à charger (`pipeline.joblib`), au lieu de deux (modèle et scaler) qu'il faudrait garder synchronisés.
- **`predict` standardise automatiquement** avant de prédire : l'API n'a plus à appeler `scaler.transform` elle-même, donc ne peut plus l'oublier.
- **Pas de fuite de données** : `fit` calcule la moyenne et l'écart-type sur le jeu d'apprentissage seulement. Standardiser *avant* le découpage laisserait des informations du jeu de test dans le scaler.

### Lire les résultats

- **R² ≈ 0,90** : le modèle explique environ 90 % de la variation des ventes sur le jeu de test.
- **RMSE ≈ 1,78** et **MAE ≈ 1,46** : erreurs en milliers de $, soit environ 1 780 $ (écart typique) et 1 460 $ (erreur absolue moyenne).
- **Coefficients** (dans l'ordre TV, radio, newspaper), calculés sur des données standardisées, donc comparables : la TV (3,76) et la radio (2,79) pèsent fortement sur les ventes, les journaux (0,056) presque pas.

> Ce pipeline est entraîné sur 80 % des données. En production, on réentraîne parfois sur la totalité une fois l'évaluation faite ; ici on garde le pipeline évalué, pour que les chiffres affichés correspondent au modèle réellement utilisé.

**Vérification :**

```bash
uv run python -c "import joblib; p = joblib.load('models/pipeline.joblib'); print(p.named_steps['regression'].coef_)"
```

## Étape 4 : l'API FastAPI (squelette)

Créez un fichier vide `api/__init__.py` (il indique à Python que `api` est un paquet), puis `api/main.py` :

```python
from pathlib import Path

import joblib
from fastapi import FastAPI

# Chemin du dossier models/, calculé depuis l'emplacement de ce fichier
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Pipeline complet (standardisation + régression), chargé une seule fois au démarrage
pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")

app = FastAPI(title="API de prédiction des ventes")


@app.get("/")
def accueil():
    return {"message": "API ok"}
```

**Lancement** (depuis la racine du projet) :

```bash
uv run uvicorn api.main:app --reload
```

- http://127.0.0.1:8000 : message d'accueil
- http://127.0.0.1:8000/docs : documentation interactive (Swagger)

## Étape 5 : le schéma et la route de prédiction

### 5a. Le schéma de données, dans un fichier séparé

`api/schemas/budget.py` :

```python
from pydantic import BaseModel, Field


class Budget(BaseModel):
    """Corps JSON attendu par /predict (validé automatiquement)."""

    TV: float = Field(ge=0, description="Budget TV (k$)")
    radio: float = Field(ge=0, description="Budget radio (k$)")
    newspaper: float = Field(ge=0, description="Budget newspaper (k$)")
```

`api/schemas/__init__.py` (il **ré-exporte** la classe, ce qui permet l'import court `from .schemas import Budget`) :

```python
from .budget import Budget
```

> **Pourquoi `float` ?** Les budgets du jeu de données ont des décimales (230,1 ; 37,8...). Un `int` rejetterait `230.1`. Un `float` accepte aussi les entiers : `100` devient `100.0`.

### 5b. La route `/predict`

Complétez `api/main.py` : nouvel import, puis la route.

```python
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from .schemas import Budget

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")

app = FastAPI(title="API de prédiction des ventes")


@app.get("/")
def accueil():
    return {"message": "API ok"}


@app.post("/predict")
def predict(budgets: Budget):
    # Mêmes noms et même ordre de colonnes qu'à l'entraînement
    X = pd.DataFrame(
        [[budgets.TV, budgets.radio, budgets.newspaper]],
        columns=["TV", "radio", "newspaper"],
    )
    sales = pipeline.predict(X)[0]   # standardise PUIS prédit, en un seul appel
    return {"sales": round(float(sales), 2)}
```

Le point dans `from .schemas import Budget` signifie « dans le même paquet que ce fichier » (le dossier `api/`). L'API se recharge seule : vérifiez que `POST /predict` apparaît dans `/docs`.

> **Un seul appel, plus de scaler dans l'API.** `pipeline.predict(X)` enchaîne la standardisation puis la prédiction. Le pipeline mémorise aussi les noms de colonnes vus à l'entraînement : si le DataFrame porte d'autres noms ou un autre ordre, scikit-learn lève une `ValueError` au lieu de produire un résultat faux en silence (constaté avec scikit-learn 1.9.1). Ce contrôle n'existe pas si l'on passe un simple tableau numpy sans noms de colonnes : d'où le DataFrame.

## Étape 6 : tester l'API

**Méthode universelle (Python)**, avec `test_api.py` à la racine :

```python
import requests

reponse = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"TV": 100, "radio": 20, "newspaper": 10},
)
print(reponse.status_code, reponse.json())
```

```bash
uv run python test_api.py
```

Résultat attendu : `200 {'sales': 11.26}`.

**Avec curl** (les guillemets dépendent du terminal) :

```bash
# Linux / macOS (bash, zsh)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TV": 100, "radio": 20, "newspaper": 10}'
```

```bat
:: Windows, invite de commandes (cmd)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"TV\": 100, \"radio\": 20, \"newspaper\": 10}"
```

> Sous **PowerShell**, `curl` est un alias de `Invoke-WebRequest` et les guillemets du JSON sont mal transmis à `curl.exe`. Utilisez `cmd`, le script Python, ou la page `/docs`.

## Étape 7 : l'interface Streamlit

### 7a. Version avec variable d'environnement

Créez `app/streamlit_app.py` :

```python
import os

import requests
import streamlit as st

# Adresse du serveur de l'API, SANS la route.
# os.environ.get("API_URL", ...) cherche une variable d'environnement nommée API_URL ;
# si elle n'existe pas, la valeur par défaut (l'API locale) est utilisée.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"

st.title("Prédiction des ventes")
st.write("Veuillez saisir les budgets publicitaires")

# st.form : la page ne se relance qu'au clic sur le bouton
with st.form("formulaire"):
    tv = st.number_input("TV", min_value=0.0, value=100.0, step=10.0)
    radio = st.number_input("Radio", min_value=0.0, value=20.0, step=1.0)
    newspaper = st.number_input("Newspaper", min_value=0.0, value=10.0, step=1.0)
    submit = st.form_submit_button("Calculer les ventes")

if submit:
    donnees = {"TV": tv, "radio": radio, "newspaper": newspaper}
    try:
        response = requests.post(PREDICT_URL, json=donnees, timeout=5)
        response.raise_for_status()
        sales = response.json()["sales"]
        st.success(f"Ventes prévues : {sales} k$")
    except requests.exceptions.RequestException:
        st.error("Impossible de joindre l'API.")
```

**Lancement** : l'API tourne dans un premier terminal, puis dans un second :

```bash
uv run streamlit run app/streamlit_app.py
```

Ouvrez http://localhost:8501. Avec les valeurs par défaut (100 / 20 / 10), vous devez obtenir **11,26 k$**.

> **Pourquoi l'URL sans `/predict` ?** L'adresse du serveur change selon l'environnement (local, déployé), alors que la route fait partie du code. Sans slash final : `https://mon-api.onrender.com`.

### 7b. Charger un fichier `.env` avec `load_dotenv()`

Python ne lit **pas** automatiquement un fichier `.env` : sans appel explicite, il est ignoré et la valeur par défaut s'applique en silence. Ajoutez l'import et l'appel **avant** la lecture de `API_URL` :

```python
import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Lit le fichier .env et ajoute son contenu aux variables d'environnement
load_dotenv()

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"
```

Créez à la racine du projet le fichier `.env` :

```
API_URL=http://127.0.0.1:8000
```

Règles de syntaxe : pas d'espace autour du `=`, pas de guillemets nécessaires, pas de `/predict` ni de slash final, et un nom identique au caractère près à celui du code.

Ordre de priorité : variable déjà définie dans le terminal, puis `.env`, puis valeur par défaut du code.

**Vérification :** remplacez le port par `9999` dans `.env`, relancez Streamlit (une seule instance !) : « Impossible de joindre l'API. » doit apparaître. Remettez ensuite `8000`.

Le `.env` n'est pas versionné. Versionnez à la place un modèle `.env.example` contenant la même ligne : chacun le copie en `.env` et l'adapte.

## Étape 8 : `.gitignore`

```gitignore
# Environnement virtuel (recréé avec `uv sync`)
.venv/

# Caches Python
__pycache__/
*.pyc

# Secrets et réglages locaux
.env
.streamlit/secrets.toml
```

À **versionner** : `pyproject.toml`, `uv.lock` (dépendances exactes), `.env.example`, et `models/pipeline.joblib` (indispensable à la prédiction, et léger ici).

## Cloner et relancer le projet

```bash
git clone <url-du-depot>
cd demo-regression
uv sync                  # recrée l'environnement à partir de uv.lock
cp .env.example .env     # Windows : copy .env.example .env
uv run uvicorn api.main:app --reload          # terminal 1
uv run streamlit run app/streamlit_app.py     # terminal 2
```

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `{"detail":"Not Found"}` sur `/predict` | Route non rechargée ou fichier non sauvegardé | Sauvegarder `api/main.py`, vérifier `/docs`, relancer uvicorn |
| `unmatched close brace` avec curl | Guillemets mal transmis par PowerShell | Utiliser `cmd` ou `test_api.py` |
| Erreur 422 | JSON invalide ou budget négatif | Vérifier les noms `TV`, `radio`, `newspaper` et les valeurs |
| « Impossible de joindre l'API » | API non lancée, ou `API_URL` à `None` / erronée | Démarrer uvicorn ; vérifier la valeur par défaut et le `.env` |
| `ImportError: cannot import name 'Budget'` | `schemas/__init__.py` vide, ou nom de classe différent de l'import | Y mettre `from .budget import Budget` ; vérifier l'orthographe |
| `ImportError: attempted relative import...` | `main.py` lancé directement | Utiliser `uv run uvicorn api.main:app` |
| Le `.env` semble ignoré | `load_dotenv()` absent ou appelé après la lecture de `API_URL` | L'appeler avant `os.environ.get` |
| Les modifications n'ont aucun effet | Ancienne instance de Streamlit toujours active (autre port) | Tout arrêter (Ctrl+C), relancer une seule fois |
| `FileNotFoundError` sur `pipeline.joblib` | Le pipeline n'a pas encore été créé | Lancer `uv run python train_model.py` |
| `ValueError: The feature names should match...` | Colonnes du DataFrame différentes de l'entraînement (noms ou ordre) | Utiliser `["TV", "radio", "newspaper"]`, dans cet ordre |
| Avertissement scikit-learn à la lecture du `.joblib` | Versions différentes | Épingler la version utilisée à l'entraînement |

## Pistes d'évolution

- Un schéma de réponse (`PredictionResponse`) pour documenter aussi la sortie dans `/docs`.
- Déploiement : Streamlit Community Cloud ne héberge que l'interface ; l'API doit l'être ailleurs (par exemple Render), puis `API_URL` doit pointer vers cette adresse.
- Ajouter des tests automatisés (pytest) pour `/predict`.
- Validation croisée (`cross_val_score`) et recherche d'hyperparamètres (`GridSearchCV`) : ils s'appliquent directement au pipeline, sans fuite de données.
