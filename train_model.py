import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. Chargement des données
df = pd.read_csv("data/Advertising.csv")
X = df[["TV", "radio", "newspaper"]]   # l'ordre des colonnes est important
y = df["sales"]

# 2. Standardisation : moyenne 0, écart-type 1 pour chaque variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Entraînement de la régression linéaire multiple
model = LinearRegression()
model.fit(X_scaled, y)

# 4. Sauvegarde des deux objets
joblib.dump(model, "models/model_scaled.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("Coefficients :", model.coef_)