import requests
import streamlit as st

# Interface Streamlit pour la saisie.
st.title("Démo de l'API de Proposition de Tags")

# Saisie du titre et du corps de la question.
title_input = st.text_input("Titre")
body_input = st.text_area("Corps de la question")

# Bouton pour obtenir une suggestion de tags.
if st.button("Obtenir des Tags"):
    if title_input and body_input:
        # Combine le titre et le corps de la question.
        input_text = title_input + " " + body_input
        
        # Envoie de la requête à l'API.
        response = requests.post("http://127.0.0.1:8000/predict", json={"title": title_input, "body": body_input})
        
        # Traitement de la réponse de l'API.
        if response.status_code == 200:
            response_data = response.json()
            tags = response_data.get("tags", [])

            # Vérifiez si `tags` est une liste de chaînes.
            if all(isinstance(tag, str) for tag in tags):
                st.write("Tags proposés:", tags)
            else:
                st.error("Un ou plusieurs tags ne sont pas des chaînes de caractères.")
        else:
            st.error("Erreur API : {}".format(response.status_code))
            st.write("Réponse brute de l'API:", response.json())
    else:
        st.error("Veuillez remplir le titre et le corps de la question.")

# demarrer en executant : streamlit run interface.py