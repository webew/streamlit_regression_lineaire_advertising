import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Lit le fichier .env et ajoute son contenu aux variables d'environnement
load_dotenv()

# Lit la variable d'environnement API_URL. Si elle n'existe pas, on utilise l'adresse locale par défaut.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"

# st.write("API_URL lue :", API_URL)

st.title("Prédiction des ventes")
st.write("Veuillez saisir les budgets publicitaires")

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