Serveur de Questions-Réponses Médicales

  Un serveur basé sur Flask qui fournit des réponses à des questions médicales en utilisant des embeddings combinés de deux modèles Sentence-BERT : all-MiniLM-L6-v2 (usage général) et BioBERT (spécifique au domaine biomédical). Cette approche améliore la précision de la recherche sémantique en exploitant les forces complémentaires des modèles.

Fonctionnalités :

  Combine les embeddings de all-MiniLM-L6-v2 et BioBERT pour des représentations sémantiques enrichies.
  Pré-calcul et sauvegarde des embeddings combinés pour des questions prédéfinies afin de garantir des temps de réponse rapides.
  API Flask permettant de répondre dynamiquement aux questions.
  Idéal pour les systèmes de questions-réponses liés au domaine médical.

Installation :

  Prérequis : 
  Python 3.7 ou une version ultérieure
  pip (gestionnaire de paquets Python)

Étapes :

  git clone https://github.com/AyaDouaaElfahimi/Question_Answer_HealthCare_ML.git

Installer les dépendances :

  pip install -r requirements.txt

Préparer le jeu de données Placez votre fichier medical_qa_dataset.csv dans le répertoire racine. Assurez-vous qu'il contient les colonnes suivantes :

  question : Questions médicales prédéfinies.
  answer : Réponses correspondantes aux questions.

Pré-calculer et sauvegarder les embeddings combinés Exécutez le script save_embd.py pour pré-calculer et sauvegarder les embeddings combinés :

  python save_embd.py

Lancer le serveur Démarrez le serveur Flask en exécutant main.py :
  python main.py

 - Le serveur sera disponible à l'adresse http://127.0.0.1:5000.

Utilisation de l'API :
  Point d'accès : /get_answer
  Méthode : POST

  Payload :
  {
      "question": "Votre question médicale ici"
  }

Réponse:

  {
      "question": "Votre question",
      "answer": "Réponse pertinente extraite du jeu de données",
      "similarity": 0.9235
  }

Exemple de commande CURL : 

  curl -X POST http://127.0.0.1:5000/get_answer -H "Content-Type: application/json" -d '{"question": "What is the treatment for fever?"}'


Fonctionnement :

Embeddings combinés :
Pré-calcul :
Pendant le pré-calcul, les embeddings sont générés à l'aide de deux modèles :
  all-MiniLM-L6-v2 : Modèle Sentence-BERT d'usage général.
  BioBERT : Modèle spécifique au domaine biomédical.
  Les embeddings des deux modèles sont concaténés pour former une représentation enrichie de chaque question.
  
Recherche de similarité :

  Lors de l'inférence, la question saisie est encodée à l'aide des deux modèles, et les embeddings sont concaténés.
  La similarité est calculée à l'aide du produit scalaire entre l'embedding de la question et les embeddings combinés pré-calculés.
  
Pourquoi combiner les modèles ?

  all-MiniLM-L6-v2 capture les nuances linguistiques générales.
  BioBERT est spécialisé dans le vocabulaire biomédical, le rendant plus efficace pour les requêtes spécifiques au domaine.
  Leur combinaison garantit la prise en compte des informations générales et spécifiques au domaine.

Dépendances :

Les packages Python requis sont listés dans requirements.txt. Installez-les en exécutant :

  pip install -r requirements.txt

Example : 

<img width="960" alt="CapturePy" src="https://github.com/user-attachments/assets/0eeff3b2-11af-44a4-8dd8-1094b72a0a97" />

