# Compte-rendu TP RAG

## Difficultés rencontrées

1. **Modèle Groq déprécié** 
   - Problème : llama3-8b-8192 n'existe plus
   - Solution : Utilisation de llama-3.1-8b-instant

## Décisions de conception

1. **FAISS au lieu d'API cloud**
   - Avantage : Gratuit, rapide, local
   - Inconvénient : Pas scalable en production

2. **1 film = 1 chunk entier**
   - Avantage : Simple, chaque film est une unité logique
   - Inconvénient : Moins de granularité

## Résultats

✅ Le système marche ! 4800 films indexés
✅ Recommandations pertinentes
✅ Temps acceptable (~10-15 sec par question)

## Conclusion

Le RAG fonctionne correctement et démontre les concepts clés de :
- Indexation vectorielle (FAISS)
- Recherche sémantique
- Augmentation LLM avec contexte