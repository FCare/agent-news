"""Rejoue la consolidation de sujets (subject_consolidation.py) sur les bulletins
déjà en base, du plus ancien au plus récent — la consolidation est par nature
séquentielle (chaque bulletin est comparé à l'état déjà accumulé des sujets), donc
l'ordre chronologique est important, contrairement au reste du pipeline qui ne
traite qu'un bulletin à la fois.

Usage (dans le conteneur) : python3 backfill_subjects.py
"""

import asyncio
import logging
import os
import sys

import openai

import storage
import subject_consolidation

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
    dates = sorted(h["date"] for h in history)  # plus ancien -> plus récent
    logger.info(f"backfill_subjects: {len(dates)} bulletins à traiter ({dates[0]} -> {dates[-1] if dates else '-'})")

    for date in dates:
        bulletin = await storage.get_bulletin_by_date(date)
        if not bulletin:
            continue
        logger.info(f"--- {date} ---")
        await subject_consolidation.consolidate_bulletin(
            bulletin["bulletin_json"], date, llm_client, LLM_MODEL
        )

    subjects = await storage.get_all_subjects()
    logger.info(f"backfill_subjects: terminé — {len(subjects)} sujets consolidés")
    for s in subjects:
        editions = await storage.get_subject_editions(s["id"])
        if len(editions) > 1:
            logger.info(f"  [{s['category']}] '{s['title']}' — {len(editions)} éditions ({[e['date'] for e in editions]})")


if __name__ == "__main__":
    asyncio.run(main())
