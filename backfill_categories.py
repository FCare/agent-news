"""Recorrige la catégorie des sujets déjà en base (subjects.category) avec un appel LLM
léger par sujet (titre + résumé consolidé), à la place du classifieur par mots-clés
défaillant (clustering.py::_assign_category — voir bulletin_gen.py::_DEEP_DIVE_TOOL pour
le correctif appliqué aux futurs sujets). Ne touche ni le titre, ni le résumé, ni les
éditions — uniquement subjects.category (et son miroir dans la collection Chroma
"subjects", via vector_store.upsert_subject).

Usage (dans le conteneur) : python3 backfill_categories.py
"""

import asyncio
import logging
import os
import sys

import aiosqlite
import openai

import storage
import vector_store
from bulletin_gen import CATEGORIES, _llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]

_CLASSIFY_TOOL = [{
    "type": "function",
    "function": {
        "name": "classify_subject",
        "description": "Détermine la catégorie la plus appropriée pour un sujet d'actualité",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": CATEGORIES,
                    "description": "Catégorie la plus appropriée, choisie à partir du titre et du résumé.",
                },
            },
            "required": ["category"],
        },
    },
}]

_SYSTEM_PROMPT = (
    "Tu catégorises des sujets d'actualité. On te donne le titre et le résumé d'un sujet. "
    "Choisis la catégorie la plus appropriée parmi la liste donnée, à partir du contenu réel "
    "du sujet — pas d'une catégorie générique par défaut. Appelle classify_subject."
)


async def _set_category(subject_id: int, category: str) -> None:
    async with aiosqlite.connect(storage.DB_PATH) as db:
        await db.execute("UPDATE subjects SET category = ? WHERE id = ?", (category, subject_id))
        await db.commit()


async def main() -> None:
    await storage.init_db()
    llm_client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)

    subjects = await storage.get_all_subjects()
    logger.info(f"backfill_categories: {len(subjects)} sujets à reclasser")

    n_changed = 0
    loop = asyncio.get_event_loop()
    for i, s in enumerate(subjects):
        result = await _llm(
            llm_client, LLM_MODEL,
            system=_SYSTEM_PROMPT,
            user=f"TITRE: {s['title']}\nRÉSUMÉ: {s['summary']}",
            tool=_CLASSIFY_TOOL,
        )
        new_category = result.get("category")
        if not new_category or new_category not in CATEGORIES:
            logger.warning(f"  [{i+1}/{len(subjects)}] #{s['id']} '{s['title'][:60]}' — réponse invalide, ignoré")
            continue
        if new_category != s["category"]:
            logger.info(f"  [{i+1}/{len(subjects)}] #{s['id']} '{s['title'][:60]}': {s['category']} -> {new_category}")
            await _set_category(s["id"], new_category)
            await loop.run_in_executor(
                None, vector_store.upsert_subject,
                s["id"], s["title"], new_category, s["summary"], s["last_updated_date"],
            )
            n_changed += 1
        elif (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(subjects)}] ...")

    logger.info(f"backfill_categories: terminé — {n_changed}/{len(subjects)} catégories corrigées")


if __name__ == "__main__":
    asyncio.run(main())
