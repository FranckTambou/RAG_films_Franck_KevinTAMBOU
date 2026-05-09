import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from dotenv import load_dotenv


#CONFIGURATION INITIALE
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Erreur : GROQ_API_KEY non trouvée dans .env")
    exit(1)

client = Groq(api_key=api_key)

print("=" * 80)
print("🎬 SYSTÈME RAG DE RECOMMANDATION DE FILMS")
print("=" * 80)


#Charger l'index FAISS et les métadonnées
print("\n📂 Chargement de l'index FAISS...")

try:
    index = faiss.read_index("output/films_index.faiss")
    print(f"✅ Index FAISS chargé : {index.ntotal} films")
except FileNotFoundError:
    print("❌ Erreur : output/films_index.faiss non trouvé")
    print("   Lancez d'abord : python indexation.py")
    exit(1)

print("📂 Chargement des métadonnées...")
try:
    with open("output/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    films = metadata["films"]
    print(f"✅ {len(films)} films chargés")
except FileNotFoundError:
    print("❌ Erreur : output/metadata.pkl non trouvé")
    exit(1)


#Chargement du modèle d'embedding
print("\n🧠 Chargement du modèle d'embedding...")
modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
print("✅ Modèle chargé")


#Recherche des films pertinents
def rechercher_films(question, k=4):
    """
    Recherche les k films les plus pertinents pour une question.
    
    Args:
        question: la question de l'utilisateur
        k: nombre de résultats à retourner
    
    Returns:
        Liste de films avec scores de pertinence
    """
    #Encodage de la question
    question_vector = modele.encode([question])
    question_vector = np.array(question_vector, dtype=np.float32)
    
    #Recherche dans FAISS
    distances, indices = index.search(question_vector, k)
    
    #Récupération des films correspondants
    resultats = []
    for idx, distance in zip(indices[0], distances[0]):
        film = films[idx]
        score = 1 / (1 + distance)
        resultats.append({
            "film": film,
            "score": score,
            "index": idx
        })
    
    return resultats


#Recherche avec filtre de langue
def rechercher_films_avec_langue(question, langue=None, k=4):
    """
    Recherche les k films les plus pertinents avec filtre optionnel par langue.
    
    Args:
        question: la question de l'utilisateur
        langue: code langue ('en', 'fr', etc.) ou None pour pas de filtre
        k: nombre de résultats à retourner
    
    Returns:
        Liste de films filtrés par langue
    """
    #1-Recherche normale
    resultats = rechercher_films(question, k=10)
    
    #2-Filtre par langue si demandé
    if langue and langue.strip():
        resultats_filtres = []
        for r in resultats:
            film = r["film"]
            langue_film = film.get('release_date', '')[:4]  # Utiliser release_date comme proxy
            
            #Vérification de la langue originale
            if 'original_language' in film:
                langue_film = film.get('original_language', '').lower()
                if langue_film == langue.lower():
                    resultats_filtres.append(r)
        
        resultats = resultats_filtres
    
    #3-Retourner les k meilleurs
    return resultats[:k]


#Génération de la réponse avec Groq

def generer_reponse(question, resultats):
    """
    Génère une réponse en utilisant les films trouvés comme contexte.
    
    Args:
        question: la question de l'utilisateur
        resultats: résultats de recherche (films pertinents)
    
    Returns:
        Réponse du LLM
    """
    #Construction du contexte
    contexte = "Voici les films pertinents trouvés dans la base de données :\n\n"
    for i, res in enumerate(resultats, 1):
        film = res["film"]
        contexte += f"{i}. {film['title']} ({film['release_date'][:4]})\n"
        contexte += f"   Note: {film['vote_average']}/10\n"
        contexte += f"   Genres: {', '.join(film['genres_list'])}\n"
        contexte += f"   Synopsis: {film['overview'][:150]}...\n\n"
    
    #Construction du prompt
    system_prompt ="""Tu es un expert passionné en recommandation de films avec des années d'expérience.

INSTRUCTIONS CRITIQUES :
1. Tu recommandes UNIQUEMENT les films fournis dans le contexte
2. Ne jamais inventer ou ajouter des films non listés
3. Cite TOUJOURS le titre, l'année et la note pour chaque recommandation
4. Explique POURQUOI chaque film correspond à la demande
5. Structure ta réponse de manière claire avec numérotation
6. Si aucun film pertinent n'est trouvé, dis-le explicitement

STYLE DE RÉPONSE :
- Enthousiaste mais professionnel
- Concis mais informatif (max 150 mots par film)
- Cite des détails du synopsis pour justifier
- Mentionne les genres principaux

FORMAT ATTENDU :
"Je recommande [nombre] film(s) :

1. **[TITRE]** ([ANNÉE]) — Note: [NOTE]/10
   Genres: [genre1], [genre2]
   Pourquoi: [explication détaillée en 2-3 phrases]

2. [film suivant...]"

AVERTISSEMENT :
Si ta confiance est basse (films peu pertinents), ajoute :
"⚠️ Remarque : Les films trouvés ne correspondent que partiellement à votre recherche."
"""

    user_prompt = f"""Contexte des films trouvés :
{contexte}

Question de l'utilisateur : {question}

Basé sur les films trouvés ci-dessus, fais une recommandation pertinente et argumentée."""

    #Appel de Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )
    
    return response.choices[0].message.content


#Interface de questions-réponses
print("\n" + "=" * 80)
print("✅ Système prêt ! Posez vos questions de recommandation.")
print("=" * 80)
print("💡 Exemples de questions :")
print("  - 'Je cherche un thriller psychologique avec un retournement inattendu'")
print("  - 'Recommande-moi un film d'animation familial sorti après 2010'")
print("  - 'Un film comme Inception mais plus accessible ?'")
print("  - Tapez 'quit' pour quitter")
print("=" * 80)

while True:
    print()
    question = input("🎬 Votre question : ").strip()
    
    if question.lower() in ["quit", "exit", "q"]:
        print("\n👋 Au revoir !")
        break
    
    if not question:
        print("⚠️ Veuillez poser une question.")
        continue
    
    print("\n🔍 Recherche en cours...")
    
    #Demander la langue (c'est optionnel)
    langue_input = input("🌍 Langue (en/fr/autre ou appuyez Entrée pour tous) : ").strip()
    
    #Rechercher avec filtre
    resultats = rechercher_films_avec_langue(question, langue=langue_input, k=4)
    
    #rechercher sans filtre si pas de réponses avec filtre
    if not resultats and langue_input:
        print("⚠️ Aucun film trouvé pour cette langue.")
        print("🔄 Recherche sans filtre...")
        resultats = rechercher_films(question, k=4)
    
    print(f"✅ {len(resultats)} films trouvés !")
    
    #Générer la réponse
    print("\n💭 Génération de la réponse...")
    reponse = generer_reponse(question, resultats)
    
    print("\n" + "=" * 80)
    print("🎬 RECOMMANDATIONS :")
    print("=" * 80)
    print(reponse)
    print("=" * 80)
    
    #Affichage des sources
    print("\n📌 Sources utilisées :")
    for i, res in enumerate(resultats, 1):
        film = res["film"]
        print(f"  {i}. {film['title']} (score: {res['score']:.2f})")