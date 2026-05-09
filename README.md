# Système RAG de Recommandation de Films

## Qu'est-ce que c'est ?

J'ai construit un système capable de recommander des films en répondant à vos questions en langage naturel. Plutôt que de chercher des mots clés, le système comprend le *sens* de votre question et trouve les films les plus pertinents.

Par exemple, si vous demandez "Je cherche un thriller psychologique avec un retournement inattendu", le système ne cherche pas juste ces mots clés. Il comprend que vous voulez quelque chose de suspensful, psychologique, et avec une bonne histoire. C'est ça le magic du RAG.

## Comment ça marche ?

Le système fonctionne en deux phases :

**Phase 1 : Préparation**

Je charge 4800 films du dataset TMDB, je crée un texte riche pour chaque film (titre, genres, synopsis, note), puis je transforme ce texte en vecteurs numériques (grâce à un modèle d'IA). Ces vecteurs sont ensuite stockés dans une base vectorielle appelée FAISS, qui permet une recherche ultra-rapide.

**Phase 2 : Répondre à vos questions (à chaque utilisation)**

Quand vous posez une question, je la transforme aussi en vecteur , je cherche les 4 films les plus similaires dans FAISS, puis j'envoie ces films + votre question à un LLM (Groq) qui génère une réponse intelligente avec des recommandations argumentées.

Résultat : vous obtenez des recommandations pertinentes avec des explications.

## Installation et utilisation

### Prérequis

- Python 3.10+
- Une clé API Groq (gratuite sur https://console.groq.com)

### Mise en place

1. Clonez ou téléchargez ce projet
2. Créez un environnement virtuel :
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # ou: source venv/bin/activate  # Mac/Linux
```
3. Installez les dépendances :
```bash
   pip install -r requirements.txt
```
4. Créez un fichier `.env` à la racine et mettez votre clé API :

### Lancer le système

**D'abord, on indexe les films** :
```bash
python indexation.py
```

Ça prend environ 2 minutes. Le système charge les 4800 films, génère les vecteurs, et crée l'index FAISS. À la fin, vous aurez deux fichiers dans le dossier `output/` : l'index FAISS et les métadonnées.

**Ensuite, posez vos questions** :
```bash
python rag.py
```

Le système vous demande votre question, puis optionnellement une langue (en/fr/autre). Il cherche les meilleurs films et vous propose des recommandations avec explications.

## Les améliorations que j'ai apportées

### Filtre par langue

J'ai ajouté la possibilité de filtrer par langue. C'est pratique si vous cherchez uniquement des films en anglais ou en français. Si vous appuyez juste sur Entrée, il cherche dans tous les films.

### Prompt amélioré

J'ai beaucoup travaillé sur le prompt que j'envoie au LLM. Le système demande explicitement :
- De citer uniquement les films trouvés (pas d'hallucination)
- De justifier chaque recommandation
- De structurer la réponse clairement
- D'avertir si aucun film vraiment pertinent n'a été trouvé

Résultat : les recommandations sont beaucoup plus utiles et argumentées.

## Pourquoi ces choix techniques ?

**FAISS plutôt qu'une API cloud** : FAISS est gratuit et fonctionne localement. Pas de latence réseau, pas de limite de requêtes, données qui restent sur votre machine. C'est parfait pour un prototype.

**sentence-transformers multilingue** : Ce modèle comprend à la fois l'anglais et le français. 768 dimensions, c'est un bon équilibre entre précision et vitesse.

**Groq LLM** : Groq offre un tier gratuit très généreux, c'est ultra rapide, et la qualité est bonne pour un modèle 8B.

**Persistance FAISS** : Au lieu de réindexer les 4800 films à chaque lancement, j'ai sauvegardé l'index sur disque. Ça fait gagner énormément de temps.

## Résultats

L'indexation prend ~90 secondes, ensuite chaque question reçoit une réponse en 10-15 secondes. Les recommandations sont pertinentes et bien justifiées. 

J'ai testé avec des questions comme "Je cherche un thriller psychologique" et le système trouve bien Avatar, Inception, Black Swan, Memento - des choix logiques.

## Limitations et améliorations futures

Le système fonctionne bien pour les films populaires et bien documentés. Pour les films obscurs, les résultats peuvent être moins pertinents.

Une amélioration intéressante serait le "chunking intelligent" - au lieu d'avoir 1 film = 1 vecteur, découper chaque synopsis en paragraphes pour une recherche plus granulaire.

Un autre bonus serait un historique de conversation - garder les questions précédentes en mémoire pour que le LLM puisse répondre "Compare Avatar et Inception" en se souvenant des films mentionnés avant.

## Comment utiliser ce code

Vous pouvez modifier :
- Le `system_prompt` dans `rag.py` pour changer le ton des recommandations
- La taille des chunks d'embedding si vous voulez plus/moins de contexte
- Le modèle LLM (j'utilise llama-3.1-8b mais Groq en propose d'autres)

Le code est relativement simple et bien commenté. N'hésitez pas à l'adapter à vos besoins.

## Remerciements

Ce projet utilise :
- FAISS (Facebook AI Similarity Search)
- sentence-transformers (Hugging Face)
- Groq API
- Dataset TMDB de Kaggle


---

**Made by Franck Kevin TAMBOU**