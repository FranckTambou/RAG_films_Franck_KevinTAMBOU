import pandas as pd

#Chargement du CSV
df = pd.read_csv("data/tmdb_5000_movies.csv")

#Affichage des infos
print(f"✅ CSV chargé avec succès!")
print(f"Nombre de films : {len(df)}")
print(f"\nColonnes du dataset :")
print(df.columns.tolist())
print(f"\nPremiers films :")
print(df[["title", "overview", "vote_average"]].head(3))