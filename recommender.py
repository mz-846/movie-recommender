"""
Shared feature-engineering and recommendation logic.

`analysis.ipynb` uses these functions to build the model files, and `app.py` uses the same
functions to serve recommendations, so what you evaluate is exactly what you deploy.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer

_ps = PorterStemmer()


# ---------------------------------------------------------------------------
# Parsing the TMDB JSON-in-CSV columns
# ---------------------------------------------------------------------------
def names(obj: str, limit: int | None = None) -> list[str]:
    """'[{"id": 28, "name": "Action"}, ...]' -> ['Action', ...]"""
    items = ast.literal_eval(obj) if isinstance(obj, str) else []
    out = [d["name"] for d in items]
    return out[:limit] if limit else out


def director(crew: str) -> list[str]:
    for d in ast.literal_eval(crew) if isinstance(crew, str) else []:
        if d.get("job") == "Director":
            return [d["name"]]
    return []


def collapse(tokens: list[str]) -> list[str]:
    """'Sam Worthington' -> 'samworthington'.

    Multi-word names must become ONE token, otherwise every actor called "Sam" or
    every keyword containing "war" would count as a match.
    """
    return [re.sub(r"[^a-z0-9]", "", t.lower()) for t in tokens]


def stem_text(text: str) -> str:
    """Lower-case + Porter-stem so 'loved', 'loving', 'love' become the same token."""
    return " ".join(_ps.stem(w) for w in text.split())


TAG_WEIGHTS = {"overview": 1, "genres": 2, "keywords": 2, "cast": 2, "director": 3}


def load_movies(movies_path="data/tmdb_5000_movies.csv",
                credits_path="data/tmdb_5000_credits.csv") -> pd.DataFrame:
    """Load, merge and engineer the `tags` text used for content-based filtering."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    # Merge on the numeric id. Merging on title (as in the starter) duplicates rows,
    # because 'Batman', 'The Host' and 'Out of the Blue' each exist twice.
    df = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")
    df["overview"] = df["overview"].fillna("")
    df["genres_list"] = df["genres"].apply(names)
    df["keywords_list"] = df["keywords"].apply(names)
    df["cast_list"] = df["cast"].apply(lambda x: names(x, limit=3))
    df["director_list"] = df["crew"].apply(director)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype("Int64")

    # Field weights: repeating a token raises its weight in the vector. Weights were chosen
    # in analysis.ipynb by offline evaluation (overview x1, genres/keywords/cast x2,
    # director x3) - they beat equal weights and put e.g. Nolan films next to Inception.
    w = TAG_WEIGHTS
    df["tags"] = (
        df["overview"].str.split() * w["overview"]
        + df["genres_list"].apply(collapse) * w["genres"]
        + df["keywords_list"].apply(collapse) * w["keywords"]
        + df["cast_list"].apply(collapse) * w["cast"]
        + df["director_list"].apply(collapse) * w["director"]
    ).apply(lambda toks: stem_text(" ".join(toks).lower()))

    keep = ["id", "title", "year", "overview", "genres_list", "cast_list", "director_list",
            "vote_average", "vote_count", "popularity", "tags"]
    return df[keep].rename(columns={"genres_list": "genres", "cast_list": "cast",
                                    "director_list": "director"}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------
def top_k_neighbours(sim: np.ndarray, k: int = 50) -> dict:
    """Keep only each movie's k most similar movies.

    The full 4,800 x 4,800 float64 matrix is ~185 MB - too large for GitHub (100 MB
    file limit) and wasteful, because the app only ever shows the top 10-20. Storing the
    top 50 neighbours per movie is ~2 MB and gives identical recommendations.
    """
    sim = sim.copy()
    np.fill_diagonal(sim, -1)  # a movie is not its own recommendation
    idx = np.argsort(-sim, axis=1)[:, :k]
    scores = np.take_along_axis(sim, idx, axis=1)
    return {"indices": idx.astype(np.int32), "scores": scores.astype(np.float32)}


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def cosine_to_profile(matrix, rows: list[int]) -> np.ndarray:
    """Cosine similarity of every row of `matrix` to the mean of `rows` (a taste profile).

    Works for both a scipy sparse matrix (TF-IDF) and a dense numpy matrix (SVD factors).
    """
    profile = np.asarray(matrix[rows].mean(axis=0)).ravel()
    dots = np.asarray(matrix @ profile).ravel()
    if hasattr(matrix, "multiply"):  # sparse
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    else:
        norms = np.linalg.norm(matrix, axis=1)
    return dots / (norms * np.linalg.norm(profile) + 1e-12)


def item_item_similarity(interactions, k: int = 50):
    """Item-based collaborative filtering matrix from a (users x items) sparse 0/1 matrix.

    sim(i, j) = cosine between the columns of i and j = "how much the *audiences* of the
    two films overlap". Only the k strongest neighbours per film are kept (sparse), which
    keeps the file small and (checked in the notebook) loses no accuracy.
    """
    from scipy import sparse

    X = sparse.csc_matrix(interactions, dtype=np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()) + 1e-9
    sim = (X.T @ X).toarray() / np.outer(norms, norms)
    np.fill_diagonal(sim, 0)
    n = sim.shape[0]
    idx = np.argsort(-sim, axis=1)[:, :k]
    vals = np.take_along_axis(sim, idx, axis=1)
    out = sparse.csr_matrix((vals.ravel(), (np.repeat(np.arange(n), k), idx.ravel())),
                            shape=(n, n), dtype=np.float32)
    out.eliminate_zeros()
    return out


def collaborative_scores(item_sim, liked_rows: list[int]) -> np.ndarray:
    """Item-based CF: score = sum of each film's similarity to the films you liked."""
    return np.asarray(item_sim[liked_rows].sum(axis=0)).ravel()


def hybrid_scores(liked_rows: list[int], content_matrix, item_sim, weight: float) -> np.ndarray:
    """Blend content-based and collaborative similarity to a set of liked movies.

    content : cosine similarity of TF-IDF tag vectors to the liked films' average vector
    collab  : item-based CF score from MovieLens ratings ("people who watched these
              also watched...")
    hybrid  = weight * collab + (1 - weight) * content, both min-max scaled to [0, 1].

    Films nobody rated in MovieLens get collab = 0, so they can still be recommended on
    content alone (the cold-start case) - but only if their content match is strong.
    """
    content = minmax(cosine_to_profile(content_matrix, liked_rows))
    if weight == 0:
        return content
    collab = minmax(collaborative_scores(item_sim, liked_rows))
    return weight * collab + (1 - weight) * content
