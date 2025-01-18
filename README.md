# Text Mining Repository

Ce repository contient une collection de notebooks Python qui explorent et appliquent des techniques avancées de **text mining** et de **traitement du langage naturel (NLP)**. Les notebooks couvrent une variété de méthodes pour la représentation de texte, la classification, la modélisation de sujets, et l'analyse de sentiment.

## Contenu du Repository

### 1. Représentation de Texte avec des Méthodes Classiques
- **Notebook** : `Embeddings_Classique.ipynb`
- **Description** : Ce notebook explore les techniques classiques de représentation de texte, telles que :
  - **One-Hot Encoding**
  - **Bag of Words (BoW)**
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - **N-grams**
- **Objectif** : Comprendre comment ces méthodes transforment le texte en représentations numériques et analyser leurs forces et limites.

### 2. Word Embedding avec des Méthodes Statiques
- **Notebook** : `Word Embeddings avancé.ipynb`
- **Description** : Ce notebook implémente des méthodes avancées de word embedding, notamment :
  - **Word2Vec** (CBOW et Skip-gram)
  - **GloVe**
  - **FastText**
- **Objectif** : Comparer ces techniques pour capturer les relations sémantiques entre les mots et analyser leur efficacité sur différents corpus.

### 3. Word Embedding Contextuel
- **Notebook** : `Word Embeddings contextuels.ipynb`
- **Description** : Ce notebook explore les techniques de word embedding contextuel, telles que :
  - **ELMo**
  - **BERT**
- **Objectif** : Comprendre comment ces modèles capturent le contexte des mots et comparer leurs performances avec les méthodes statiques.

### 4. Modélisation de Sujets (Topic Modeling)
- **Notebook** : `Topic_Modeling.ipynb`
- **Description** : Ce notebook applique différentes techniques de modélisation de sujets, notamment :
  - **LDA (Latent Dirichlet Allocation)**
  - **LSA (Latent Semantic Analysis)**
  - **BERTopic**
- **Objectif** : Identifier les sujets dominants dans un corpus de texte et analyser les résultats obtenus avec chaque méthode.

### 5. Classification de Texte avec SVM et CNN
- **Notebook** : `SVM_CNN.ipynb`
- **Description** : Ce notebook implémente des modèles de classification de texte en utilisant :
  - **SVM (Support Vector Machine)**
  - **CNN (Convolutional Neural Network)**
- **Objectif** : Comparer les performances de ces modèles pour l'analyse de sentiment sur un ensemble de données de critiques de films.

### 6. Classification de Texte avec RNN
- **Notebook** : `RNN.ipynb`
- **Description** : Ce notebook explore l'utilisation des réseaux de neurones récurrents (RNN) pour la classification de texte.
- **Objectif** : Implémenter un modèle RNN pour l'analyse de sentiment et analyser son efficacité.

### 7. Analyse de Sentiment avec LSTM et Bi-LSTM
- **Notebook** : `LSTM_BiLSTM.ipynb`
- **Description** : Ce notebook implémente des modèles LSTM (Long Short-Term Memory) et Bi-LSTM (Bidirectional LSTM) pour l'analyse de sentiment.
- **Objectif** : Comparer les performances des modèles LSTM et Bi-LSTM sur une tâche de classification binaire.

## Bibliothèques Utilisées
- **scikit-learn** : Pour les méthodes classiques de text mining (TF-IDF, BoW, etc.).
- **gensim** : Pour l'implémentation de Word2Vec et LDA.
- **transformers** : Pour les modèles contextuels comme BERT.
- **tensorflow/keras** : Pour les modèles de deep learning (CNN, RNN, LSTM).
- **nltk** : Pour le prétraitement des textes (tokenization, suppression des stop words, etc.).
- **pandas/numpy** : Pour la manipulation des données.

## Comment Utiliser ce Repository
1. Clonez le repository sur votre machine locale :
   ```bash
   git clone https://github.com/votre-username/text-mining.git
