# Importation des bibliothèques nécessaires
import numpy as np  # Bibliothèque pour les calculs numériques et manipulation de tableaux/matrices
from sentence_transformers import SentenceTransformer, models  # Bibliothèque pour le traitement de phrases avec des modèles de type Transformer
import pandas as pd  # Bibliothèque pour la manipulation de données structurées comme les fichiers CSV

# Chargement du dataset contenant des questions médicales
df = pd.read_csv("medical_qa_dataset.csv")  # Lire le fichier CSV contenant les données (par exemple, des questions médicales et leurs réponses)
questions = df['question'].tolist()  # Extraire uniquement la colonne contenant les questions et la convertir en une liste Python

# Chargement du premier modèle de génération d'embeddings (MiniLM)
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')  
# MiniLM est un modèle compact et rapide qui génère des représentations vectorielles (embeddings) pour des phrases.

# Chargement du deuxième modèle, BioBERT, conçu pour des textes biomédicaux
biobert_transformer = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=256)  
# BioBERT est un modèle optimisé pour les données biomédicales, basé sur le modèle BERT.

# Ajout d'un module de pooling pour générer des embeddings de phrases
biobert_pooling = models.Pooling(biobert_transformer.get_word_embedding_dimension())  
# Le module de pooling combine les représentations de chaque mot dans une phrase pour produire un vecteur unique représentant toute la phrase.

# Création d'un modèle SentenceTransformer avec les composants de BioBERT
biobert_model = SentenceTransformer(modules=[biobert_transformer, biobert_pooling])  
# Ce modèle combine le transformeur BioBERT avec un module de pooling pour générer des embeddings de phrases.

# Pré-calcul des embeddings pour les questions du dataset
print("Pré-calcul des embeddings des questions...")  # Affiche un message pour indiquer le début du traitement
minilm_embeddings = minilm_model.encode(questions, normalize_embeddings=True)  
# Génère des embeddings normalisés (vecteurs unitaires) pour les questions en utilisant MiniLM.
biobert_embeddings = biobert_model.encode(questions, normalize_embeddings=True)  
# Génère des embeddings normalisés pour les mêmes questions en utilisant le modèle BioBERT.

# Combinaison des deux ensembles d'embeddings
combined_embeddings = np.concatenate((minilm_embeddings, biobert_embeddings), axis=1)  
# Concatène les embeddings générés par MiniLM et BioBERT pour chaque question. 
# La taille du vecteur résultant est la somme des dimensions des deux modèles.

# Sauvegarde des embeddings combinés dans un fichier pour une utilisation future
np.save("combined_question_embeddings.npy", combined_embeddings)  
# Enregistre les embeddings sous forme d'un fichier NumPy (.npy) pour les réutiliser sans recalculer.
print("Les embeddings combinés ont été sauvegardés dans 'combined_question_embeddings.npy'")  
# Message pour indiquer que les embeddings combinés ont été sauvegardés avec succès.
