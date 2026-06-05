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
    "Astronomie & Espace",
    "Science & Technologie",
    "Médecine & Santé",
    "Sport",
    "Automobile & Mobilité",
    "Immobilier & Logement",
    "Voyages & Tourisme",
    "Droit & Justice",
    "Éducation & Recherche",
    "Société & Environnement",
    "Culture & Médias",
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
    ("Astronomie & Espace", [
        "espace", "nasa", "spacex", "esa", "satellite", "fusée", "rocket", "lanceur",
        "mars", "lune", "moon", "jupiter", "vénus", "astéroïde", "comète",
        "astronaute", "cosmonaute", "iss", "station spatiale", "télescope",
        "james webb", "hubble", "exoplanète", "trou noir", "supernova", "galaxie",
        "exploration spatiale", "ariane", "starship", "falcon",
    ]),
    ("Science & Technologie", [
        "recherche", "scientifique", "découverte", "étude", "physique", "chimie",
        "biologie", "génétique", "laboratoire", "expérience",
        "énergie", "nucléaire", "solaire", "hydrogène", "fusion nucléaire",
        "nanotechnologie", "quantum", "quantique",
    ]),
    ("Médecine & Santé", [
        "médecine", "médical", "santé", "hôpital", "clinique", "chirurgie",
        "vaccin", "vaccination", "épidémie", "pandémie", "virus", "pathogène",
        "cancer", "tumeur", "thérapie", "traitement", "médicament", "pharmaceutique",
        "maladie", "symptôme", "diagnostic", "patient", "urgences", "soin",
        "oms", "who", "hôpitaux", "infirmier", "infirmière", "médecin", "docteur",
        "psychiatrie", "neurologie", "cardiologie", "oncologie", "pédiatrie",
        "santé mentale", "mental health", "alzheimer", "diabète", "obésité",
    ]),
    ("Sport", [
        "football", "foot", "soccer", "tennis", "rugby", "basketball", "cyclisme",
        "tour de france", "formule 1", "f1", "moto gp", "athlétisme", "natation",
        "jeux olympiques", "olympic", "coupe du monde", "world cup", "ligue des champions",
        "champions league", "nba", "nfl", "mls", "premier league", "ligue 1",
        "boxe", "mma", "ufc", "golf", "ski", "snowboard", "handball", "volleyball",
        "sport", "sportif", "sportive", "entraîneur", "coach", "transfert", "joueur",
        "joueuse", "champion", "championnat", "tournoi", "match", "score", "but", "essai",
    ]),
    ("Automobile & Mobilité", [
        "voiture", "automobile", "auto", "car", "véhicule", "véhicule électrique",
        "vé", "ev", "tesla", "renault", "peugeot", "stellantis", "volkswagen", "toyota",
        "bmw", "mercedes", "audi", "ford", "gm", "general motors",
        "moteur", "carburant", "essence", "diesel", "hybride", "recharge", "borne",
        "permis de conduire", "sécurité routière", "accident de la route",
        "transport", "autoroute", "embouteillage", "trafic",
        "moto", "scooter", "trottinette", "vélo électrique", "mobilité douce",
        "sncf", "ratp", "train", "tgv", "métro", "tramway", "bus",
    ]),
    ("Immobilier & Logement", [
        "immobilier", "logement", "appartement", "maison", "loyer", "location",
        "achat immobilier", "vente immobilier", "prix immobilier", "marché immobilier",
        "propriétaire", "locataire", "bailleur", "agence immobilière",
        "construction", "promoteur", "permis de construire", "rénovation",
        "hlm", "logement social", "crise du logement", "squat", "expulsion",
        "taux hypothécaire", "crédit immobilier", "prêt immobilier",
    ]),
    ("Voyages & Tourisme", [
        "voyage", "tourisme", "touriste", "destination", "vacances", "séjour",
        "hôtel", "airbnb", "booking", "croisière", "charter",
        "aéroport", "compagnie aérienne", "airline", "vol", "grève aérien",
        "passeport", "visa", "frontière", "douane",
        "patrimoine", "unesco", "site touristique", "musée", "monument",
        "île", "plage", "montagne", "randonnée", "camping",
    ]),
    ("Droit & Justice", [
        "tribunal", "cour", "jugement", "verdict", "condamné", "condamnation",
        "acquittement", "procès", "audience", "plainte", "inculpé", "mis en examen",
        "prison", "incarcération", "peine", "liberté conditionnelle",
        "avocat", "magistrat", "juge", "procureur", "parquet",
        "loi", "législation", "jurisprudence", "constitution", "décret",
        "droits humains", "human rights", "amnesty", "discrimination", "harcèlement",
        "viol", "agression", "meurtre", "homicide", "fraude", "corruption",
        "cour suprême", "conseil constitutionnel", "cour de cassation",
    ]),
    ("Éducation & Recherche", [
        "éducation", "école", "collège", "lycée", "université", "fac", "campus",
        "élève", "étudiant", "enseignant", "professeur", "recteur",
        "bac", "baccalauréat", "brevet", "parcoursup", "orientation",
        "réforme scolaire", "programme scolaire", "ministre de l'éducation",
        "recherche scientifique", "laboratoire", "publication", "thèse", "doctorat",
        "cnrs", "inserm", "inrae", "bourse", "master",
        "alphabétisation", "illettrisme", "formation professionnelle",
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
    ("Culture & Médias", [
        # Cinéma & séries
        "film", "cinéma", "cinema", "movie", "série", "series", "netflix", "amazon prime",
        "disney", "hbo", "streaming", "oscar", "césar", "festival", "cannes",
        "réalisateur", "acteur", "actrice", "box office",
        # Musique
        "musique", "music", "album", "concert", "tournée", "tour", "chanteur",
        "chanteuse", "groupe", "grammy", "victoires de la musique", "spotify",
        # People & célébrités
        "célébrité", "people", "personnalité", "show business", "showbiz",
        "people", "star", "vedette", "influenceur", "influencer",
        # Jeux vidéo
        "jeu vidéo", "video game", "gaming", "playstation", "xbox", "nintendo",
        "steam", "e-sport", "esport",
        # Littérature & BD
        "livre", "book", "roman", "novel", "auteur", "author", "écrivain",
        "bd", "bande dessinée", "manga", "comic", "comics", "graphic novel",
        "littérature", "literature", "prix littéraire", "booker", "goncourt",
        # Médias
        "presse", "journal", "media", "médias", "journaliste", "télévision", "radio",
    ]),
    ("Société & Environnement", [
        "climat", "climate", "environnement", "environment", "écologie", "ecology",
        "réchauffement", "co2", "émissions", "pollution", "biodiversité",
        "migration", "immigration", "réfugié", "refugee",
        "inégalité", "pauvreté", "poverty", "précarité",
        "féminisme", "sexisme", "racisme", "lgbtq", "diversité",
        "société", "communauté", "association", "bénévolat",
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
