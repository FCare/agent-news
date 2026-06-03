"""
TF-IDF + DBSCAN clustering for news articles.
Replaces the LLM clustering step — instant, deterministic, no API call.
"""

import logging
import re
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crawler import RawArticle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category rules — keyword → category (checked in order, first match wins)
# ---------------------------------------------------------------------------

CATEGORIES = [
    "International",
    "Europe",
    "France",
    "Économie & Finance",
    "Géopolitique & Défense",
    "Informatique & IA",
    "Science & Technologie",
    "Société & Environnement",
    "Littérature & BD",
]

_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Informatique & IA", [
        "ia", "intelligence artificielle", "artificial intelligence", "llm", "gpt", "chatgpt",
        "openai", "anthropic", "gemini", "mistral", "deepseek", "copilot", "claude",
        "cybersécurité", "cybersecurity", "hack", "malware", "ransomware", "faille",
        "logiciel", "software", "algorithme", "cloud", "numérique", "digital",
        "robot", "robotique", "data", "machine learning", "deep learning",
        "stackoverflow", "github", "open source", "linux", "windows", "android",
        "chip", "nvidia", "processeur", "semiconduteur",
    ]),
    ("Science & Technologie", [
        "espace", "nasa", "spacex", "satellite", "fusée", "rocket", "mars", "lune",
        "recherche", "scientifique", "découverte", "étude", "physique", "chimie",
        "biologie", "génétique", "médecine", "vaccin", "cancer", "santé", "hôpital",
        "climate", "climat", "énergie", "nucléaire", "solaire", "hydrogène",
    ]),
    ("Géopolitique & Défense", [
        "guerre", "war", "militaire", "military", "conflit", "conflict", "otan", "nato",
        "armée", "army", "missile", "drone", "arme", "weapon", "sanctions",
        "ukraine", "russie", "russia", "israel", "gaza", "iran", "corée",
        "diplomatie", "ambassadeur", "traité", "treaty", "sommet", "summit",
    ]),
    ("Économie & Finance", [
        "économie", "economy", "bourse", "stock", "marché", "market", "inflation",
        "banque", "bank", "investissement", "investment", "pib", "gdp",
        "emploi", "chômage", "unemployment", "commerce", "trade", "budget",
        "entreprise", "company", "startup", "acquisition", "fusion", "merger",
        "bitcoin", "crypto", "ethereum", "finance",
    ]),
    ("France", [
        "france", "paris", "assemblée nationale", "sénat", "élysée",
        "gouvernement", "ministre", "premier ministre", "président", "présidente",
        "loi", "réforme", "grève", "manifestation", "police", "gendarmerie",
        "célébrité", "people", "personnalité", "médias français",
    ]),
    ("Europe", [
        "europe", "european", "ue", "union européenne", "bruxelles", "brussels",
        "commission européenne", "parlement européen", "eurozone",
        "allemagne", "germany", "italie", "italy", "espagne", "spain",
        "royaume-uni", "uk", "britain", "brexit", "pologne", "hongrie",
    ]),
    ("Littérature & BD", [
        "livre", "book", "roman", "novel", "auteur", "author", "écrivain",
        "bd", "bande dessinée", "manga", "comic", "comics", "graphic novel",
        "littérature", "literature", "prix littéraire", "booker", "goncourt",
        "film", "cinéma", "série", "netflix", "disney", "marvel", "dc",
        "musique", "music", "album", "concert",
    ]),
    ("Société & Environnement", [
        "climat", "climate", "environnement", "environment", "écologie", "ecology",
        "réchauffement", "co2", "émissions", "pollution", "biodiversité",
        "migration", "immigratio", "réfugié", "refugee",
        "justice", "tribunal", "procès", "condamné", "prison",
        "inégalité", "pauvreté", "poverty", "éducation", "école",
    ]),
    ("International", []),  # default catch-all
]


def _assign_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            return category
    return "International"


# ---------------------------------------------------------------------------
# Stop words (FR + EN combined, minimal)
# ---------------------------------------------------------------------------

_STOP_WORDS = set("""
a au aux avec ce ces cet cette dans de des du elle elles en et eux il ils je
la le les leur leurs lui ma mais me mes meme mon même ni nos notre nous on ou
par pas pour qu que qui sa se ses si son sur ta te tes ton tu un une vos votre
vous y a about after all also an and any are as at be been being by can could
did do does doing done during each few for from get got had has have he her
here him his how in into is it its just know like me might more most my no
not now of off on once only or other our out over own said same see she so
some such than that the their them then there these they this those through
to too under until up very was we were what when where which while who will
with would you your
""".split())


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def _recency_score(article: RawArticle) -> float:
    """Higher score for more recent articles."""
    if not article.published_at:
        return 0.5
    try:
        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
        return max(0.0, 1.0 - age_h / 48)
    except Exception:
        return 0.5


def cluster_articles(articles: list[RawArticle],
                     eps: float = 0.25,
                     min_samples: int = 2) -> list[dict]:
    """
    Cluster articles by title similarity using TF-IDF + DBSCAN.
    Returns topics sorted by importance (cluster size × recency).
    """
    if not articles:
        return []

    titles = [a.title for a in articles]

    # TF-IDF vectorisation (bilingual: FR + EN titles together)
    vec = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words=list(_STOP_WORDS),
    )
    try:
        X = vec.fit_transform(titles)
    except Exception as e:
        logger.error(f"TF-IDF failed: {e}")
        return []

    feature_names = vec.get_feature_names_out()

    # Cosine distance matrix for DBSCAN
    sim = cosine_similarity(X)
    np.clip(sim, 0, 1, out=sim)
    dist = 1.0 - sim

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    labels = db.fit_predict(dist)

    # Build cluster → article indices mapping
    clusters: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(i)

    topics = []

    # ── Named clusters (label ≥ 0) ─────────────────────────────────────
    for lbl, indices in clusters.items():
        if lbl == -1:
            continue  # singletons handled below

        cluster_articles = [articles[i] for i in indices]

        # Representative title: article closest to cluster centroid
        # .mean() on sparse returns np.matrix — convert to array for sklearn
        centroid = np.asarray(X[indices].mean(axis=0))
        sims_to_centroid = cosine_similarity(centroid, X[indices])[0]
        rep_idx = indices[int(np.argmax(sims_to_centroid))]
        title = articles[rep_idx].title

        # Top keywords for category assignment
        cluster_tfidf = np.asarray(X[indices].mean(axis=0)).ravel()
        top_kw_idx = cluster_tfidf.argsort()[-8:][::-1]
        keywords = [feature_names[i] for i in top_kw_idx if cluster_tfidf[i] > 0]

        # Category from title + keywords
        cat_text = title + " " + " ".join(keywords)
        category = _assign_category(cat_text)

        # Importance = cluster size × mean recency
        recency = np.mean([_recency_score(a) for a in cluster_articles])
        importance = len(indices) * (0.5 + 0.5 * recency)

        topics.append({
            "title":         title,
            "category":      category,
            "importance":    importance,
            "keywords":      keywords[:5],
            "article_count": len(indices),
            "_article_indices": indices,
        })

    # ── Singletons (label == -1) ────────────────────────────────────────
    for i in clusters.get(-1, []):
        a = articles[i]
        kw_vec = X[i].toarray()[0]
        top_kw_idx = kw_vec.argsort()[-5:][::-1]
        keywords = [feature_names[j] for j in top_kw_idx if kw_vec[j] > 0]
        category = _assign_category(a.title + " " + " ".join(keywords))
        recency = _recency_score(a)
        topics.append({
            "title":         a.title,
            "category":      category,
            "importance":    0.5 * recency,
            "keywords":      keywords[:5],
            "article_count": 1,
            "_article_indices": [i],
        })

    topics.sort(key=lambda t: -t["importance"])
    logger.info(
        f"Clustering TF-IDF: {len(articles)} articles → "
        f"{len(topics)} sujets "
        f"({len(clusters)-(-1 in clusters)} groupes, "
        f"{len(clusters.get(-1, []))} singletons)"
    )
    return topics
