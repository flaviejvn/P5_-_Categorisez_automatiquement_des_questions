import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialisation de l'application FastAPI.
application = FastAPI()

# Charger le modèle de SentenceTransformer pour l'encodage des phrases.
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

# Chargement du modèle enregistré dans MLflow.
model_uri = "models:/SVM_BERT_BestParams/1" # ok
model = mlflow.sklearn.load_model(model_uri) # ok

# Chargement des tags binarisés.
with open('mlb_classes.pkl', 'rb') as f:
    #all_tags = pickle.load(f).tolist()
    all_tags = pickle.load(f)

# Définition de la structure de la requête.
class Query(BaseModel):
    title: str
    body: str

# Endpoint pour la prédiction.
@app.post("/predict")
async def predict_tags(query: Query):
    
    # Concaténation du titre et du corps de la question.
    input_text = query.title + " " + query.body # ok

    # Transformation de l'entrée avec le modèle SentenceTransformer.
    vectorized_input = sentence_transformer_model.encode([input_text])

    # Prédiction des tags.
    prediction = model.predict(vectorized_input)

    # Conversion des indices prédits en tags
    predicted_tags = [str(all_tags[i]) for i, val in enumerate(prediction[0]) if val >= 0.5]
    
    # Retourne les tags sous forme de liste de chaînes.
    return {"tags": predicted_tags}
            
@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Lancer l'application en local.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="127.0.0.1", port=8000)
    
# demarrer en executant : uvicorn application:app --reload