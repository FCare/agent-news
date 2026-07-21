"""Génère bulletin_json["category_summaries"] pour les bulletins déjà en base (créés
avant l'ajout de cette fonctionnalité, voir bulletin_gen.py::_generate_category_summaries) —
un paragraphe de synthèse par catégorie couvrant tous ses sujets du jour, affiché sur
les pages wiki dates/<date>.md entre le titre de la catégorie et la liste des sujets.

Usage (dans le conteneur) : python3 backfill_category_summaries.py
"""

import asyncio
import logging
import os
import sys

import openai

import storage
from bulletin_gen import _generate_category_summaries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]


async def main() -> None:
    await storage.init_db()
    llm_client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)

    history = await storage.get_history_list(limit=9999)
    logger.info(f"backfill_category_summaries: {len(history)} bulletins à traiter")

    for h in history:
        date = h["date"]
        row = await storage.get_bulletin_by_date(date)
        if not row:
            continue
        bulletin_json = row["bulletin_json"]
        if bulletin_json.get("category_summaries"):
            logger.info(f"  {date}: déjà présent, ignoré")
            continue
        logger.info(f"  {date}: génération ({len(bulletin_json.get('categories', {}))} catégories)...")
        bulletin_json["category_summaries"] = await _generate_category_summaries(
            bulletin_json.get("categories", {}), llm_client, LLM_MODEL
        )
        await storage.save_bulletin(
            date=date,
            flash=bulletin_json.get("flash", ""),
            headline=bulletin_json.get("headline", ""),
            bulletin_json=bulletin_json,
            n_articles=row.get("n_articles", 0),
            n_topics=row.get("n_topics", 0),
        )
        logger.info(f"  {date}: OK ({len(bulletin_json['category_summaries'])} synthèses)")

    logger.info("backfill_category_summaries: terminé")


if __name__ == "__main__":
    asyncio.run(main())
