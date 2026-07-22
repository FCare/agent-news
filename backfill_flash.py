"""Régénère bulletin_json["headline"]/["flash"] ("L'essentiel du jour" sur le wiki)
pour les bulletins déjà en base, avec la consigne assouplie (6 phrases, répartition
libre entre sujets — voir bulletin_gen.py::_FLASH_TOOL) au lieu de l'ancienne
contrainte fixe (3 phrases, un sujet par phrase).

Usage (dans le conteneur) : python3 backfill_flash.py
"""

import asyncio
import logging
import os
import sys

import openai

import storage
from bulletin_gen import _generate_flash

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
    logger.info(f"backfill_flash: {len(history)} bulletins à traiter")

    for h in history:
        date = h["date"]
        row = await storage.get_bulletin_by_date(date)
        if not row:
            continue
        bulletin_json = row["bulletin_json"]
        all_topics = [s for stories in bulletin_json.get("categories", {}).values() for s in stories]
        if not all_topics:
            continue
        all_topics.sort(key=lambda t: -t.get("importance", 0))

        headline, flash = await _generate_flash(all_topics, llm_client, LLM_MODEL)
        bulletin_json["headline"] = headline
        bulletin_json["flash"] = flash
        await storage.save_bulletin(
            date=date,
            flash=flash,
            headline=headline,
            bulletin_json=bulletin_json,
            n_articles=row.get("n_articles", 0),
            n_topics=row.get("n_topics", 0),
        )
        logger.info(f"  {date}: OK — {headline}")

    logger.info("backfill_flash: terminé")


if __name__ == "__main__":
    asyncio.run(main())
