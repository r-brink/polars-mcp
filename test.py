import polars as pl

# Sample data with list-type column
movies = pl.DataFrame(
    {
        "movie_id": [1, 2, 3, 4],
        "title": ["The Dark Knight", "Inception", "Interstellar", "Joker"],
        "genres": [
            ["Action", "Drama"],
            ["Action", "Sci-Fi"],
            ["Drama", "Sci-Fi", "Adventure"],
            ["Drama", "Thriller"],
        ],
        "rating": [9.0, 8.8, 8.6, 8.5],
        "box_office_millions": [1004, 836, 677, 1074],
    }
)

# Explode the genres list into separate rows — useful for one-to-many expansion
df_exploded = movies.explode("genres")

# Use to_dummies on the exploded genre column for multi-hot encoding
genre_dummies = df_exploded.select(["movie_id", "genres"]).to_dummies(
    "genres", separator="_"
)

# Join back and compute a custom metric
df_analysis = (
    df_exploded.select(["movie_id", "title", "rating", "box_office_millions"])
    .unique()
    .join(
        genre_dummies.group_by("movie_id").agg(pl.col("^genres_.*$").sum()),
        on="movie_id",
        how="left",
    )
)

# Use with_columns to dynamically compute a financial efficiency score
# rating * 10 + log(box_office_millions + 1) - 2 * (box_office_millions > 1000)
df_analysis = df_analysis.with_columns(
    [
        (
            pl.col("rating") * 10
            + (pl.col("box_office_millions") + 1).log()
            - 2 * (pl.col("box_office_millions") > 1000).cast(pl.Int32)
        ).alias("efficiency_score")
    ]
)

print(
    df_analysis.select(["title", "efficiency_score"]).sort(
        "efficiency_score", descending=True
    )
)
