"""
ChromaDB vector store for semantic search over articles and bulletin topics.

Three collections:
- articles       : crawled article bodies (TTL = crawler TTL)
- bulletin_topics: deep dives from generated bulletins (kept 3 months)
- publishers     : distinct publisher names for fuzzy/semantic lookup
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("/data/chroma")
# Multilingual model — good for FR/EN mixed news content (~470 MB, cached after first run)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_client = None
_articles_col = None
_topics_col = None
_publishers_col = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


def _get_ef():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _articles_collection():
    global _articles_col
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
        _topics_col = _get_client().get_or_create_collection(
            name="bulletin_topics",
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _topics_col


def _publishers_collection():
    global _publishers_col
    if _publishers_col is None:
        _publishers_col = _get_client().get_or_create_collection(
            name="publishers",
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _publishers_col


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _topic_id(date: str, title: str) -> str:
    return hashlib.md5(f"{date}:{title}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_articles(articles: list[dict]) -> None:
    """Store/update articles in the vector store. Called after each crawl."""
    if not articles:
        return
    try:
        col = _articles_collection()
        ids, docs, metas = [], [], []
        for a in articles:
            doc = f"{a['title']}\n{(a.get('body') or '')[:800]}"
            ids.append(_article_id(a["url"]))
            docs.append(doc)
            metas.append({
                "url":       a["url"],
                "title":     a["title"],
                "publisher": a.get("publisher", ""),
                "country":   a.get("country", ""),
                "crawled_at": a.get("crawled_at", datetime.now(timezone.utc).isoformat()),
            })
        # Batch upsert (ChromaDB handles duplicates)
        BATCH = 100
        for i in range(0, len(ids), BATCH):
            col.upsert(
                ids=ids[i:i+BATCH],
                documents=docs[i:i+BATCH],
                metadatas=metas[i:i+BATCH],
            )
        logger.info(f"ChromaDB: {len(ids)} articles upsertés")
    except Exception as e:
        logger.error(f"ChromaDB upsert articles failed: {e}")


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


def seed_publishers_if_empty(articles: list[dict]) -> None:
    """Populate the publishers collection from existing articles if it is empty."""
    try:
        if _publishers_collection().count() == 0:
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
                    "sources":      ", ".join(s.get("sources", [])),
                })
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"ChromaDB: {len(ids)} topics upsertés (bulletin {date})")
    except Exception as e:
        logger.error(f"ChromaDB upsert topics failed: {e}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_topics(query: str, n_results: int = 8,
                  date_filter: str | None = None) -> list[dict]:
    """Semantic search in bulletin deep dives."""
    try:
        col = _topics_collection()
        where = {"date": {"$gte": date_filter}} if date_filter else None
        kwargs = dict(query_texts=[query], n_results=min(n_results, col.count()))
        if where:
            kwargs["where"] = where
        results = col.query(**kwargs)
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search topics failed: {e}")
        return []


def search_articles(query: str, n_results: int = 8) -> list[dict]:
    """Semantic search in raw articles."""
    try:
        col = _articles_collection()
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        return _format_results(results)
    except Exception as e:
        logger.error(f"ChromaDB search articles failed: {e}")
        return []


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
