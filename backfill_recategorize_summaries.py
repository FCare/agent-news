"""Resynchronise bulletin_json["category_summaries"] avec la catégorie ACTUELLE de
chaque sujet (subjects.category, qui s'auto-corrige au fil du temps par vote
majoritaire — voir storage.py::update_subject) plutôt que la catégorie figée au
moment de la génération du bulletin.

Sans ça, la page wiki dates/<date>.md peut afficher un paragraphe de synthèse qui
mentionne un sujet sous une catégorie ("International" contient un ministre
français) alors que ce même sujet est maintenant listé sous sa bonne catégorie
("France") juste en dessous — la liste de liens lit déjà subjects.category (à jour),
seul le texte de synthèse restait figé sur le regroupement d'origine.

Usage (dans le conteneur) : python3 backfill_recategorize_summaries.py
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
    logger.info(f"backfill_recategorize_summaries: {len(history)} bulletins à traiter")

    for h in history:
        date = h["date"]
        row = await storage.get_bulletin_by_date(date)
        if not row:
            continue
        bulletin_json = row["bulletin_json"]
        old_categories = bulletin_json.get("categories", {})
        if not old_categories:
            continue

        # Catégorie actuelle par titre d'édition (subjects.category, à jour)
        members = await storage.get_subjects_by_date(date)
        current_category_by_title = {m["edition_title"]: m["category"] for m in members}

        new_categories: dict[str, list[dict]] = {}
        n_moved = 0
        for old_cat, stories in old_categories.items():
            for story in stories:
                cat = current_category_by_title.get(story["title"], old_cat)
                if cat != old_cat:
                    n_moved += 1
                new_categories.setdefault(cat, []).append(story)

        if n_moved == 0:
            logger.info(f"  {date}: déjà à jour, ignoré")
            continue

        logger.info(f"  {date}: {n_moved} sujet(s) reclassé(s) depuis la génération — régénération des synthèses...")
        bulletin_json["categories"] = new_categories
        bulletin_json["category_summaries"] = await _generate_category_summaries(
            new_categories, llm_client, LLM_MODEL
        )
        await storage.save_bulletin(
            date=date,
            flash=bulletin_json.get("flash", ""),
            headline=bulletin_json.get("headline", ""),
            bulletin_json=bulletin_json,
            n_articles=row.get("n_articles", 0),
            n_topics=row.get("n_topics", 0),
        )
        logger.info(f"  {date}: OK")

    logger.info("backfill_recategorize_summaries: terminé")


if __name__ == "__main__":
    asyncio.run(main())
