"""
ChromaDB vector store for semantic search over articles and bulletin topics.

Four collections:
- articles       : crawled article bodies (TTL = crawler TTL)
- bulletin_topics: deep dives from generated bulletins (kept 3 months)
- publishers     : distinct publisher names for fuzzy/semantic lookup
- subjects       : vue CONSOLIDÉE d'un sujet à travers plusieurs jours (voir
                   subject_consolidation.py/storage.py) — un document par sujet,
                   ré-upserté (même id) à chaque mise à jour de son résumé courant,
                   contrairement à bulletin_topics qui garde un document PAR ÉDITION.
"""

import hashlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("/data/chroma")
# Multilingual model — good for FR/EN mixed news content (~2.2 GB, cached after first run)
EMBEDDING_MODEL = "BAAI/bge-m3"

_client = None
_articles_col = None
_topics_col = None
_publishers_col = None
_subjects_col = None
# Le backfill au démarrage tourne en tâche de fond pendant que d'autres appels
# (seed_publishers_if_empty, etc.) peuvent initialiser ces singletons depuis un
# autre thread d'executor — sans verrou, deux threads créent le client/une
# collection en même temps et corrompent l'objet Rust sous-jacent.
_init_lock = threading.RLock()


def _get_client():
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                import chromadb
                CHROMA_PATH.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _get_ef(device: str = "cuda:0"):
    import torch
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    if torch.cuda.is_available():
        try:
            # dtype en string ("float16"), pas l'objet torch.dtype — Chroma sérialise
            # en JSON la config de l'embedding function pour la persister dans la collection.
            return SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL, device=device,
                model_kwargs={"torch_dtype": "float16"},
            )
        except Exception as e:
            logger.warning(f"Chargement du modèle sur {device} échoué ({e}), repli CPU")
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL, device="cpu")


def warmup_model() -> None:
    """Force le chargement du modèle d'embedding dès le démarrage de l'agent (appelé
    une fois dans main()), plutôt que de le laisser se charger paresseusement au
    premier accès — évite un cold-start de ~20-30s (chargement + vérifications
    HuggingFace Hub) sur la toute première recherche de wiki-agent après un
    redémarrage. Le modèle reste ensuite chargé en permanence (pas de unload_model
    entre les cycles du pipeline, contrairement à avant — marge VRAM désormais
    suffisante, voir docker-compose.yml de nanovllm-voxcpm)."""
    _subjects_collection()
    logger.info("ChromaDB: modèle d'embedding chargé (warmup)")


def _articles_collection():
    global _articles_col
    if _articles_col is None:
        with _init_lock:
            if _articles_col is None:
                _articles_col = _get_client().get_or_create_collection(
                    name="articles",
                    embedding_function=_get_ef(),
                    metadata={"hnsw:space": "cosine"},
                )
    return _articles_col


def _topics_collection():
    global _topics_col
    if _topics_col is None:
        with _init_lock:
            if _topics_col is None:
                _topics_col = _get_client().get_or_create_collection(
                    name="bulletin_topics",
                    embedding_function=_get_ef(),
                    metadata={"hnsw:space": "cosine"},
                )
    return _topics_col


def _publishers_collection():
    global _publishers_col
    if _publishers_col is None:
        with _init_lock:
            if _publishers_col is None:
                _publishers_col = _get_client().get_or_create_collection(
                    name="publishers",
                    embedding_function=_get_ef(),
                    metadata={"hnsw:space": "cosine"},
                )
    return _publishers_col


def _subjects_collection():
    global _subjects_col
    if _subjects_col is None:
        with _init_lock:
            if _subjects_col is None:
                _subjects_col = _get_client().get_or_create_collection(
                    name="subjects",
                    embedding_function=_get_ef(),
                    metadata={"hnsw:space": "cosine"},
                )
    return _subjects_col


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _topic_id(date: str, title: str) -> str:
    return hashlib.md5(f"{date}:{title}".encode()).hexdigest()


def _subject_id(subject_id: int) -> str:
    # Clé SQLite déjà unique et stable — pas besoin de hasher comme pour les autres
    # collections, dont la clé naturelle est une chaîne arbitraire (url, date:title).
    return f"subject-{subject_id}"


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_articles(articles: list[dict]) -> None:
    """Store/update articles in the vector store. Called after each crawl.
    Utilise une instance transitoire du modèle d'embedding sur cuda:1 (déchargée
    juste après, voir finally) plutôt que le singleton persistant de
    _articles_collection() (cuda:0, utilisé par search_articles) : ce batch de
    ~1800 articles est le seul point du pipeline avec un besoin ponctuel de VRAM
    important, à isoler du modèle qui reste chargé en continu sur cuda:0 à côté
    de voxcpm2/unmute (voir warmup_model).
    Embeddings calculés à la main puis passés via `embeddings=` plutôt que de
    confier `ef` à get_or_create_collection : la collection "articles" est
    persistée depuis des runs antérieurs (EF stockée = cuda:0), et ChromaDB
    reconstruit l'embedding function depuis cette config au lieu d'utiliser
    l'instance cuda:1 passée ici — le device demandé était silencieusement
    ignoré, provoquant le même OOM sur cuda:0 qu'on cherchait à éviter."""
    if not articles:
        return
    ef = None
    try:
        ef = _get_ef(device="cuda:1")
        col = _get_client().get_or_create_collection(
            name="articles",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        ids, docs, metas = [], [], []
        for a in articles:
            doc = f"{a['title']}\n{(a.get('body') or '')}"
            ids.append(_article_id(a["url"]))
            docs.append(doc)
            crawled_at_str = a.get("crawled_at", datetime.now(timezone.utc).isoformat())
            try:
                crawled_at_ts = int(datetime.fromisoformat(crawled_at_str).timestamp())
            except Exception:
                crawled_at_ts = int(datetime.now(timezone.utc).timestamp())
            metas.append({
                "url":        a["url"],
                "title":      a["title"],
                "publisher":  a.get("publisher", ""),
                "country":    a.get("country", ""),
                "crawled_at": crawled_at_str,
                "crawled_ts": crawled_at_ts,
            })
        # Batch upsert (ChromaDB handles duplicates) — embeddings calculés via `ef`
        # (cuda:1) puis fournis directement, voir docstring
        BATCH = 100
        for i in range(0, len(ids), BATCH):
            batch_docs = docs[i:i+BATCH]
            col.upsert(
                ids=ids[i:i+BATCH],
                embeddings=ef(batch_docs),
                documents=batch_docs,
                metadatas=metas[i:i+BATCH],
            )
        logger.info(f"ChromaDB: {len(ids)} articles upsertés (cuda:1)")
    except Exception as e:
        logger.error(f"ChromaDB upsert articles failed: {e}")
    finally:
        if ef is not None:
            del ef
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def upsert_publishers(articles: list[dict]) -> None:
    """Keep the publishers collection in sync after each crawl."""
    publishers = {a["publisher"] for a in articles if a.get("publisher")}
    if not publishers:
        return
    try:
        col = _publishers_collection()
        ids = [hashlib.md5(p.encode()).hexdigest() for p in publishers]
        col.upsert(ids=ids, documents=list(publishers))
    except Exception as e:
        logger.error(f"ChromaDB upsert publishers failed: {e}")


def _collection_count(name: str) -> int:
    """Compte les entrées d'une collection sans construire l'embedding function —
    .count() n'a besoin d'aucun modèle, seulement upsert/query en ont besoin."""
    try:
        return _get_client().get_collection(name=name, embedding_function=None).count()
    except Exception:
        return 0  # collection pas encore créée


def seed_publishers_if_empty(articles: list[dict]) -> None:
    """Populate the publishers collection from existing articles if it is empty."""
    try:
        if _collection_count("publishers") == 0:
            upsert_publishers(articles)
            logger.info(f"ChromaDB: publishers seedés depuis {len(articles)} articles existants")
    except Exception as e:
        logger.error(f"ChromaDB seed publishers failed: {e}")


def find_similar_publishers(query: str, n_results: int = 5) -> list[str]:
    """Return publisher names semantically close to the query (handles typos/voice)."""
    try:
        col = _publishers_collection()
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents"],
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        logger.error(f"ChromaDB find_similar_publishers failed: {e}")
        return []


def upsert_bulletin_topics(bulletin_json: dict, date: str) -> None:
    """Store deep dives from a bulletin. Called after each bulletin generation."""
    try:
        col = _topics_collection()
        ids, docs, metas = [], [], []
        for stories in bulletin_json.get("categories", {}).values():
            for s in stories:
                title = s.get("title", "")
                doc = f"{title}\n{s.get('summary', '')}\n{s.get('deep_dive', '')}"
                ids.append(_topic_id(date, title))
                docs.append(doc[:2000])
                metas.append({
                    "title":        title,
                    "category":     s.get("category", ""),
                    "date":         date,
                    "date_int":     int(date.replace("-", "")),  # YYYYMMDD for numeric filtering
                    "what_to_watch": s.get("what_to_watch", ""),
                    "sources":      ", ".join(
                        src["name"] if isinstance(src, dict) else src
                        for src in s.get("sources", [])
                    ),
                })
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"ChromaDB: {len(ids)} topics upsertés (bulletin {date})")
    except Exception as e:
        logger.error(f"ChromaDB upsert topics failed: {e}")


def upsert_subject(subject_id: int, title: str, category: str, summary: str,
                    last_updated_date: str) -> None:
    """Ré-upsert (même id, voir _subject_id) à chaque création ou mise à jour d'un
    sujet — remplace intégralement le document précédent, la recherche ne porte
    donc toujours que sur le résumé COURANT, jamais un résumé périmé d'une édition
    antérieure (voir storage.create_subject/update_subject)."""
    try:
        col = _subjects_collection()
        col.upsert(
            ids=[_subject_id(subject_id)],
            documents=[f"{title}\n{summary}"],
            metadatas=[{
                "subject_id": subject_id,
                "category": category,
                "last_updated_date": last_updated_date,
                "last_updated_date_int": int(last_updated_date.replace("-", "")),
            }],
        )
    except Exception as e:
        logger.error(f"ChromaDB upsert subject failed (id {subject_id}): {e}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_topics(query: str, n_results: int = 8,
                  date_filter: str | None = None) -> list[dict]:
    """Semantic search in bulletin deep dives."""
    try:
        col = _topics_collection()
        where = {"date_int": {"$gte": int(date_filter.replace("-", ""))}} if date_filter else None
        kwargs = dict(query_texts=[query], n_results=min(n_results, col.count()))
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search topics failed: {e}")
        return []


def search_subjects(query: str, n_results: int = 8, category: str | None = None) -> list[dict]:
    """Double usage : (1) recherche sémantique utilisateur (wiki, sans filtre), et
    (2) recherche de candidats de continuité par subject_consolidation.py (filtrée
    sur `category` — voir storage.get_subjects_by_category, même principe)."""
    try:
        col = _subjects_collection()
        count = col.count()
        if count == 0:
            return []
        kwargs = dict(query_texts=[query], n_results=min(n_results, count))
        if category is not None:
            kwargs["where"] = {"category": category}
        results = col.query(**kwargs)
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search subjects failed: {e}")
        return []


def search_articles(query: str, n_results: int = 8,
                    min_crawled_ts: int | None = None) -> list[dict]:
    """Semantic search in raw articles."""
    try:
        col = _articles_collection()
        count = col.count()
        if count == 0:
            return []
        kwargs = dict(query_texts=[query], n_results=min(n_results, count))
        if min_crawled_ts is not None:
            kwargs["where"] = {"crawled_ts": {"$gte": min_crawled_ts}}
        results = col.query(**kwargs)
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search articles failed: {e}")
        return []


def purge_old_articles(retention_days: int = 3) -> None:
    """Delete raw articles older than retention_days from the vector store."""
    from datetime import timedelta
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=retention_days)).timestamp())
    try:
        col = _articles_collection()
        before = col.count()
        col.delete(where={"crawled_ts": {"$lt": cutoff_ts}})
        deleted = before - col.count()
        if deleted:
            logger.info(f"ChromaDB: {deleted} articles supprimés (antérieurs à J-{retention_days})")
    except Exception as e:
        logger.error(f"ChromaDB purge articles failed: {e}")


def purge_old_topics(retention_days: int = 90) -> None:
    """Delete bulletin topics older than retention_days. Mirrors SQLite bulletin retention."""
    from datetime import timedelta
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_int = int(cutoff_dt.strftime("%Y%m%d"))
    try:
        col = _topics_collection()
        before = col.count()
        col.delete(where={"date_int": {"$lt": cutoff_int}})
        deleted = before - col.count()
        if deleted:
            logger.info(f"ChromaDB: {deleted} topics supprimés (antérieurs au {cutoff_dt.strftime('%Y-%m-%d')})")
    except Exception as e:
        logger.error(f"ChromaDB purge topics failed: {e}")


def purge_old_subjects(retention_days: int = 365) -> None:
    """Miroir de purge_old_topics, mais sur last_updated_date_int et une rétention
    bien plus longue (voir storage.SUBJECT_RETENTION_DAYS) — un sujet actif ne doit
    jamais être purgé juste parce qu'il est ancien, seulement s'il est inactif."""
    from datetime import timedelta
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_int = int(cutoff_dt.strftime("%Y%m%d"))
    try:
        col = _subjects_collection()
        before = col.count()
        col.delete(where={"last_updated_date_int": {"$lt": cutoff_int}})
        deleted = before - col.count()
        if deleted:
            logger.info(f"ChromaDB: {deleted} sujets supprimés (inactifs depuis avant le {cutoff_dt.strftime('%Y-%m-%d')})")
    except Exception as e:
        logger.error(f"ChromaDB purge subjects failed: {e}")


def _format_results(results: dict) -> list[dict]:
    out = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({
            "content":  doc,
            "metadata": meta,
            "score":    round(1 - dist, 3),  # cosine similarity
        })
    return out
