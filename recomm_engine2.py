import pandas as pd
from pathlib import Path

# Get the folder of this Python file
folder = Path(__file__).parent

# Read CSV files
movies = pd.read_csv(folder / "movies.csv")
ratings = pd.read_csv(folder / "ratings.csv")

# Choose user
user_id = 1

# User ratings
user_ratings = ratings[ratings["userId"] == user_id]

# Movies rated 4 or more
liked = user_ratings[user_ratings["rating"] >= 4]

# Add movie information
liked = pd.merge(liked, movies, on="movieId")

print("\nMovies liked by User 1:")
print(liked[["title", "rating"]].to_string(index=False))

# Find genres
genres = []

for g in liked["genres"]:
    genres.extend(g.split("|"))

# Favourite genre
favorite_genre = pd.Series(genres).value_counts().index[0]

print("\nFavourite Genre:", favorite_genre)

# Recommend movies
recommendations = movies[
    movies["genres"].str.contains(favorite_genre, na=False)
]

# Remove already rated movies
recommendations = recommendations[
    ~recommendations["movieId"].isin(user_ratings["movieId"])
]

print("\nRecommended Movies:")
print(recommendations[["title", "genres"]].head(10).to_string(index=False))
recommendations = recommendations[
~recommendations["movieId"].isin(user_ratings["movieId"])
]

print("\nRecommended Movies:")
print(recommendations[["title", "genres"]].head(10))
