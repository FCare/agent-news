"""Recherche sémantique pour le wiki statique (voir wiki_build.py) : réutilise
vector_store.py tel quel, aucun nouvel index — même principe que côté contes-agent."""

import asyncio
import logging

from fastapi import APIRouter

import storage
import vector_store
from wiki_build import _subject_slug

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_LIMIT = 8


@router.get("/api/wiki/search")
async def search(q: str = "", limit: int = DEFAULT_LIMIT) -> dict:
    query = q.strip()
    if not query:
        return {"subjects": []}

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, vector_store.search_subjects, query, limit)

    subjects = []
    for r in results:
        subject_id = r["metadata"]["subject_id"]
        # Titre/catégorie relus depuis SQLite (pas les métadonnées Chroma, qui peuvent
        # être figées depuis le dernier upsert) : le slug généré ici doit correspondre
        # EXACTEMENT à celui produit par wiki_build.py, qui part toujours du titre
        # SQLite actuel.
        subject = await storage.get_subject(subject_id)
        if not subject:
            continue  # sujet purgé depuis (voir storage.purge_old_subjects) — lien mort évité
        subjects.append({
            "id": subject_id,
            "title": subject["title"],
            "category": subject["category"],
            "summary": r["content"],
            "score": r["score"],
            "url": f"/wiki/sujets/{_subject_slug(subject['title'], subject_id)}/",
        })

    return {"subjects": subjects}
