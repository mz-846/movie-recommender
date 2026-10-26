"""
Flask web app for the Movie Recommender.

Run locally:  python app.py          -> http://localhost:5000
Docker:       docker build -t movie-recommender . && docker run -p 5000:5000 movie-recommender

Two modes:
  /recommend  (POST)  content-based "films similar to X"       -> model.pkl + similarity.pkl
  /hybrid     (POST)  hybrid "pick films you love"              -> hybrid.pkl
  /api/recommend?title=...   JSON version of /recommend (handy for testing)

Posters: set the environment variable TMDB_API_KEY (free key from themoviedb.org) to show
real posters. Without it the app still works and shows styled title cards instead.
"""
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

import recommender as rec

BASE = Path(__file__).parent
app = Flask(__name__)

# ---------------------------------------------------------------- load models once
movies = pickle.load(open(BASE / "model.pkl", "rb"))
similarity = pickle.load(open(BASE / "similarity.pkl", "rb"))      # top-50 neighbours
hybrid = pickle.load(open(BASE / "hybrid.pkl", "rb"))
TITLES = movies["title"].tolist()
TITLE_TO_IDX = {t: i for i, t in reversed(list(enumerate(TITLES)))}  # first occurrence wins
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
N_RESULTS = 12


@lru_cache(maxsize=5000)
def fetch_poster(movie_id: int) -> str | None:
    """Poster URL from the TMDB API, or None if no key / request fails (never crashes the page)."""
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                         params={"api_key": TMDB_API_KEY, "language": "en-US"}, timeout=4)
        path = r.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except Exception:
        return None


def to_cards(indices) -> list[dict]:
    rows = movies.iloc[list(indices)]
    with ThreadPoolExecutor(max_workers=8) as pool:            # fetch posters in parallel
        posters = list(pool.map(fetch_poster, rows["id"].astype(int)))
    return [{
        "title": r.title,
        "year": "" if r.year is None or str(r.year) == "<NA>" else int(r.year),
        "genres": ", ".join(r.genres[:3]),
        "rating": f"{r.vote_average:.1f}",
        "overview": (r.overview[:180] + "…") if len(r.overview) > 180 else r.overview,
        "poster": p,
        "initials": "".join(w[0] for w in r.title.split()[:2]).upper(),
    } for r, p in zip(rows.itertuples(), posters)]


def get_recommendations(title: str, n: int = N_RESULTS) -> list[int]:
    """Content-based: the n most similar films to `title`."""
    idx = TITLE_TO_IDX[title]
    return similarity["indices"][idx, :n].tolist()


def get_hybrid_recommendations(liked_titles: list[str], n: int = N_RESULTS) -> list[int]:
    """Hybrid: blend content + item-based collaborative similarity to all liked films."""
    liked = [TITLE_TO_IDX[t] for t in liked_titles]
    scores = rec.hybrid_scores(liked, hybrid["tfidf"], hybrid["item_sim"], hybrid["weight"])
    scores[liked] = -np.inf
    return np.argsort(-scores)[:n].tolist()


def page(**ctx):
    ctx.setdefault("mode", "similar")
    return render_template("index.html", movie_list=TITLES, weight=hybrid["weight"],
                           posters_enabled=bool(TMDB_API_KEY), **ctx)


# ---------------------------------------------------------------- routes
@app.route("/")
def home():
    return page()


@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form.get("selected_movie", "").strip()
    if title not in TITLE_TO_IDX:
        return page(error=f"Couldn't find “{title}”. Pick a title from the suggestions.")
    return page(selected=title, results=to_cards(get_recommendations(title)),
                heading=f"Because you liked {title}")


@app.route("/hybrid", methods=["POST"])
def hybrid_recommend():
    raw = [request.form.get(f"liked_{i}", "").strip() for i in range(5)]
    liked = [t for t in dict.fromkeys(raw) if t]
    unknown = [t for t in liked if t not in TITLE_TO_IDX]
    if not liked or unknown:
        msg = "Pick at least one film." if not liked else f"Couldn't find: {', '.join(unknown)}"
        return page(mode="hybrid", liked=raw, error=msg)
    return page(mode="hybrid", liked=raw, results=to_cards(get_hybrid_recommendations(liked)),
                heading="Picked for someone who loves " + ", ".join(liked))


@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "")
    if title not in TITLE_TO_IDX:
        return jsonify(error="unknown title"), 404
    return jsonify(title=title, recommendations=[TITLES[i] for i in get_recommendations(title)])


@app.route("/health")
def health():
    return {"status": "ok", "movies": len(TITLES)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
