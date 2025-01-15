# Importation des bibliothèques nécessaires
import numpy as np  # Pour les opérations numériques, notamment la manipulation de tableaux/matrices
import pandas as pd  # Pour la manipulation de données structurées (comme les fichiers CSV)
from sentence_transformers import models, SentenceTransformer  # Pour le traitement de phrases avec des modèles de type Transformer
from functools import lru_cache  # Pour mettre en cache les résultats des fonctions
from flask import Flask, request, jsonify  # Pour créer une API REST avec Flask
from sklearn.preprocessing import normalize  # Pour normaliser les vecteurs

# Chargement du dataset contenant les questions et réponses
df = pd.read_csv("medical_qa_dataset.csv")  # Lire un fichier CSV contenant des questions et réponses médicales
questions = df['question'].tolist()  # Extraire la colonne contenant les questions et la convertir en liste
answers = df['answer'].tolist()  # Extraire la colonne contenant les réponses et la convertir en liste

# Chargement des embeddings pré-calculés pour les questions
print("Loading precomputed question embeddings...")  # Indique que les embeddings sont en cours de chargement
question_embeddings = np.load("combined_question_embeddings.npy")  # Charger les embeddings à partir d'un fichier NumPy

# Chargement des modèles pour encoder les questions saisies
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')  
# Modèle MiniLM pour générer des embeddings rapides et précis

# Chargement du modèle BioBERT spécialisé pour les données biomédicales
biobert_transformer = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=256)  

# Ajout d'un module de pooling pour créer des embeddings de phrases
biobert_pooling = models.Pooling(biobert_transformer.get_word_embedding_dimension())  

# Création d'un modèle SentenceTransformer combinant BioBERT et pooling
biobert_model = SentenceTransformer(modules=[biobert_transformer, biobert_pooling])  

# Utilisation d'un cache pour optimiser le calcul des embeddings des questions saisies
@lru_cache(maxsize=100)  # Stocke jusqu'à 100 résultats pour éviter de recalculer des embeddings identiques
def encode_input_question(input_question):
    """
    Génère les embeddings combinés (MiniLM + BioBERT) pour une question donnée.
    """
    # Générer les embeddings pour la question avec MiniLM
    minilm_embedding = minilm_model.encode(input_question, normalize_embeddings=True)
    # Générer les embeddings pour la question avec BioBERT
    biobert_embedding = biobert_model.encode(input_question, normalize_embeddings=True)
    # Combiner les deux embeddings (concaténation)
    combined_embedding = np.concatenate([minilm_embedding, biobert_embedding])
    return combined_embedding

# Fonction pour trouver la réponse la plus pertinente
def get_answer(input_question):
    """
    Trouve la réponse la plus pertinente en fonction de la question saisie.
    """
    # Calculer l'embedding de la question saisie
    input_embedding = encode_input_question(input_question)

    # Normaliser les embeddings des questions pré-calculées et de la question saisie
    normalized_question_embeddings = normalize(question_embeddings, axis=1)
    normalized_input_embedding = normalize(input_embedding.reshape(1, -1), axis=1)

    # Calculer les similarités cosinus entre la question saisie et les questions du dataset
    similarities = np.dot(normalized_question_embeddings, normalized_input_embedding.T).flatten()

    # Trouver l'indice de la question la plus similaire
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[most_similar_index]  # Récupérer le score de similarité
    return answers[most_similar_index], similarity_score  # Retourner la réponse associée et le score

# Initialisation de l'application Flask
app = Flask(__name__)

# Définition d'une route pour traiter les questions
@app.route('/get_answer', methods=['POST'])
def answer_question():
    """
    Route API pour recevoir une question et retourner la réponse la plus pertinente.
    """
    # Récupérer les données JSON envoyées dans la requête
    data = request.json
    user_question = data.get('question')  # Extraire la question de l'utilisateur

    if not user_question:  # Vérifier si une question est fournie
        return jsonify({"error": "Question is required"}), 400  # Retourner une erreur si la question est absente

    try:
        # Trouver la réponse et le score de similarité
        answer, similarity = get_answer(user_question)
        return jsonify({
            "question": user_question,  # La question saisie
            "answer": answer,  # La réponse trouvée
            "similarity": float(similarity)  # Le score de similarité (converti en flottant pour JSON)
        })
    except Exception as e:  # Gestion des erreurs imprévues
        return jsonify({"error": str(e)}), 500  # Retourner une erreur interne avec le message correspondant

# Exécution du serveur Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  
    # L'application est disponible sur toutes les interfaces réseau, au port 5000, avec le mode debug activé
