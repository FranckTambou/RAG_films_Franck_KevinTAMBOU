import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from tqdm import tqdm
import pickle

print("=" * 80)
print("PHASE 1 : INDEXATION DES FILMS")
print("=" * 80)


#Charger et nettoyer les données
print("\n📂 Chargement des données...")
df = pd.read_csv("data/tmdb_5000_movies.csv")
print(f"✅ {len(df)} films chargés")

#Fonction pour parser les genres JSON
def parse_genres(genres_str):
    """Transforme le JSON des genres en liste de noms"""
    if pd.isna(genres_str) or genres_str == "":
        return []
    try:
        genres_list = json.loads(genres_str)
        return [g["name"] for g in genres_list]
    except:
        return []

#Nettoyage des données
df["genres_list"] = df["genres"].apply(parse_genres)
df = df.dropna(subset=["overview"])  # Enlever les 3 films sans synopsis

print(f"✅ Données nettoyées : {len(df)} films restants")


#Création d'un texte riche pour chaque film
print("\n📝Création de textes enrichis...")

def creer_texte_film(row):
    """Crée un texte riche combinant titre, synopsis, genres, note, année"""
    titre = row["title"]
    synopsis = row["overview"]
    genres = ", ".join(row["genres_list"]) if row["genres_list"] else "Unknown"
    note = row["vote_average"]
    annee = row["release_date"][:4] if pd.notna(row["release_date"]) else "Unknown"
    
    texte = f"""
Film: {titre}
Année: {annee}
Genres: {genres}
Note IMDb: {note}/10
Synopsis: {synopsis}
"""
    return texte.strip()

df["texte_complet"] = df.apply(creer_texte_film, axis=1)

#Exemple de texte
print(f"\n📌 Exemple de texte créé pour le film #1 :")
print("-" * 80)
print(df["texte_complet"].iloc[0][:300] + "...")
print("-" * 80)


#Création des embeddings
print("\nGénération des embeddings...")
print("(Chargement du modèle sentence-transformers - peut prendre 30-60 secondes)")

#Chargement du modèle multilingue
modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
print(f"✅ Modèle chargé: dimension = 768")

#Encodage de tous les films
print(f"\n🔄 Encoding des {len(df)} films en cours...")
embeddings = modele.encode(df["texte_complet"].tolist(), show_progress_bar=True)
print(f"✅ {len(embeddings)} embeddings créés")
print(f"   Dimension de chaque embedding: {embeddings.shape[1]}")


#Création de l'index FAISS

print("\nCréation de l'index FAISS...")


embeddings_float32 = np.array(embeddings, dtype=np.float32)

#Création de l'index FAISS
dimension = embeddings_float32.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_float32)

print(f"✅ Index FAISS créé")
print(f"   Nombre de vecteurs : {index.ntotal}")
print(f"   Dimension : {dimension}")


#Sauvegarder l'index et les métadonnées

print("\n Sauvegarde de l'index et métadonnées...")

#Si le dossier output n'existe pas on le crée
os.makedirs("output", exist_ok=True)

#Sauvegarde de l'index FAISS
faiss.write_index(index, "output/films_index.faiss")
print("✅ Index FAISS sauvegardé : output/films_index.faiss")

#Sauvegarde des métadonnées (films + embeddings pour vérification)
metadata = {
    "films": df[["id", "title", "overview", "genres_list", "vote_average", "release_date"]].to_dict(orient="records"),
    "textes": df["texte_complet"].tolist(),
    "embeddings_shape": embeddings_float32.shape,
}

with open("output/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print("✅ Métadonnées sauvegardées : output/metadata.pkl")

 
print("\n" + "=" * 80)
print("✅ INDEXATION TERMINÉE AVEC SUCCÈS !")
print("=" * 80)
print(f"Films indexés : {index.ntotal}")
print(f"Dimension des embeddings : {dimension}")
print(f"Fichiers créés :")
print(f"  - output/films_index.faiss (index FAISS)")
print(f"  - output/metadata.pkl (métadonnées)")
print("\nVous pouvez maintenant lancer rag.py pour poser des questions ! 🎬")