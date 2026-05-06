import pandas as pd
import json

#chargement du csv
df = pd.read_csv("data/tmdb_5000_movies.csv")

print("=" * 80)
print("EXPLORATION DU DATASET TMDB")
print("=" * 80)

#Info générale
print(f"\n📊 Nombre total de films : {len(df)}")
print(f"📊 Colonnes disponibles : {len(df.columns)}")

#Affichons quelques colonnes importantes
print("\n" + "=" * 80)
print("COLONNES IMPORTANTES POUR NOTRE RAG :")
print("=" * 80)
print(df[["id", "title", "overview", "genres", "vote_average", "release_date"]].head(2))

#Un seul film en détail
print("\n" + "=" * 80)
print("EXEMPLE DÉTAILLÉ : Film #1")
print("=" * 80)
film = df.iloc[0]
print(f"Titre : {film['title']}")
print(f"Synopsis : {film['overview'][:200]}...")
print(f"Note : {film['vote_average']}/10")
print(f"Genres (RAW) : {film['genres']}")
print(f"Année : {film['release_date'][:4] if pd.notna(film['release_date']) else 'N/A'}")

#Compréhension du format des genres
print("\n" + "=" * 80)
print("STRUCTURE DES GENRES (JSON) :")
print("=" * 80)
if pd.notna(film['genres']) and film['genres'] != "":
    try:
        genres_json = json.loads(film['genres'])
        print(f"Genres parsés : {[g['name'] for g in genres_json]}")
    except:
        print("Erreur parsing genres")

#Vérification des données manquantes
print("\n" + "=" * 80)
print("DONNÉES MANQUANTES :")
print("=" * 80)
print(df[["title", "overview", "genres", "vote_average"]].isnull().sum())

#Statistiques sur les notes
print("\n" + "=" * 80)
print("STATISTIQUES DES NOTES :")
print("=" * 80)
print(df["vote_average"].describe())