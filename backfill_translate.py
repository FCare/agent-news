"""Traduit en français le titre/résumé des sujets déjà en base qui ne le sont pas
(subjects.title / subjects.summary) — un topic dont toutes les sources sont en anglais
produit parfois un titre non traduit malgré la consigne du prompt (voir le correctif
appliqué dans bulletin_gen.py::_DEEP_DIVE_TOOL et subject_consolidation.py::_MATCH_TOOL
pour les futurs sujets). Une base mêlant français et anglais dégrade la similarité
sémantique utilisée pour rapprocher les sujets d'un jour à l'autre (vector_store.
search_subjects) — c'est le résumé consolidé, pas les éditions individuelles
(volontairement gardées telles quelles, voir storage.py::update_subject), qui alimente
cet embedding et doit donc être en français.

Détection de langue via langdetect (rapide, local — pas d'appel LLM pour les ~700
sujets déjà en français) ; traduction via LLM uniquement pour les sujets détectés non-fr.

Usage (dans le conteneur) : python3 backfill_translate.py
"""

import asyncio
import logging
import os
import sys

import aiosqlite
import openai
from langdetect import LangDetectException, detect

import storage
import vector_store
from bulletin_gen import _TRANSLATE_TOOL, _llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY = os.environ["LLAMACPP_API_KEY"]

_TRANSLATE_SYSTEM_PROMPT = (
    "Tu traduis des extraits d'actualité en français. Traduis fidèlement, garde le "
    "ton factuel et les chiffres/noms propres intacts. Appelle translate_excerpts avec "
    "exactement 2 traductions, dans l'ordre : [titre, résumé]."
)


def _is_french(text: str) -> bool:
    try:
        return detect(text) == "fr"
    except LangDetectException:
        return True  # texte trop court/ambigu pour trancher — on ne touche pas


async def _set_title_summary(subject_id: int, title: str, summary: str) -> None:
    async with aiosqlite.connect(storage.DB_PATH) as db:
        await db.execute(
            "UPDATE subjects SET title = ?, summary = ? WHERE id = ?",
            (title, summary, subject_id),
        )
        await db.commit()


async def main() -> None:
    await storage.init_db()
    llm_client = openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)

    subjects = await storage.get_all_subjects()
    logger.info(f"backfill_translate: {len(subjects)} sujets à vérifier")

    n_translated = 0
    loop = asyncio.get_event_loop()
    for i, s in enumerate(subjects):
        # Testés séparément : un résumé déjà en français (ex: traduit lors d'une
        # consolidation précédente) peut masquer un titre resté en anglais si on ne
        # teste que la concaténation des deux (constaté sur données réelles — titre
        # anglais + résumé français détectés "fr" dans l'ensemble).
        if _is_french(s["title"]) and _is_french(s["summary"]):
            if (i + 1) % 100 == 0:
                logger.info(f"  [{i+1}/{len(subjects)}] ...")
            continue

        result = await _llm(
            llm_client, LLM_MODEL,
            system=_TRANSLATE_SYSTEM_PROMPT,
            user=f"1) {s['title']}\n2) {s['summary']}",
            tool=_TRANSLATE_TOOL,
        )
        translations = result.get("translations") or []
        if len(translations) != 2 or not all(translations):
            logger.warning(f"  [{i+1}/{len(subjects)}] #{s['id']} '{s['title'][:60]}' — traduction invalide, ignoré")
            continue

        new_title, new_summary = translations
        logger.info(f"  [{i+1}/{len(subjects)}] #{s['id']}: '{s['title'][:60]}' -> '{new_title[:60]}'")
        await _set_title_summary(s["id"], new_title, new_summary)
        await loop.run_in_executor(
            None, vector_store.upsert_subject,
            s["id"], new_title, s["category"], new_summary, s["last_updated_date"],
        )
        n_translated += 1

    logger.info(f"backfill_translate: terminé — {n_translated}/{len(subjects)} sujets traduits")


if __name__ == "__main__":
    asyncio.run(main())
