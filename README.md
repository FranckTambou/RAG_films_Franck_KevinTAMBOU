# 🎬 Système RAG de Recommandation de Films

## Description

Système RAG (Retrieval-Augmented Generation) capable de recommander des films en répondant à des questions en langage naturel.

**Exemples de questions :**
- "Je cherche un thriller psychologique"
- "Recommande-moi un film d'animation familial"
- "Un film comme Inception?"

## Architecture

- *Données* : 4800 films du dataset TMDB
- *Indexation* : Embeddings + FAISS (base vectorielle)
- *Recherche* : Similarité cosinus avec FAISS
- *LLM* : Groq (llama-3.1-8b-instant)

## Comment utiliser

## 1. Configuration
```bash
# Activer l'environnement
venv\Scripts\activate

# Remplir .env avec notre clé Groq
GROQ_API_KEY=votre_clé_ici
```

## 2. Indexation
```bash
python indexation.py
```

## 3. Poser des questions
```bash
python rag.py
```

Posez vos questions et le système recommande des films !

# Données

- Source : TMDB 5000 Movie Dataset
- Total : 4800 films (après nettoyage)
- Langue : Anglais

# Choix techniques

- **FAISS** : Rapide, local, gratuit
- **sentence-transformers** : Embeddings 768 dimensions
- **Groq** : LLM gratuit et rapide

# Résultats

- Films indexés : 4800
- Temps indexation : 1-2 minutes
- Temps par requête : 10-15 secondes
- Qualité : Recommandations pertinentes ✅
