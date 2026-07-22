"""Resynchronise bulletin_json["categories"]/["category_summaries"] avec l'état
ACTUEL des sujets (subjects.category, qui s'auto-corrige au fil du temps par vote
majoritaire — voir storage.py::update_subject) plutôt que le regroupement figé au
moment de la génération du bulletin. Deux corrections :

1. Reclasse un topic dans sa catégorie à jour quand elle a changé depuis la
   génération (ex: "International" contenait un ministre français, maintenant
   listé sous "France" partout ailleurs dans le wiki).
2. Retire les topics ORPHELINS : le clustering du jour crée parfois 2-3 topics
   quasi-identiques pour le même événement réel (constaté à grande échelle sur
   données réelles — ~200 cas sur 12 jours), que la consolidation fusionne ensuite
   en UN SEUL sujet. Le(s) topic(s) perdant(s) de cette fusion restent référencés
   dans bulletin_json mais plus nulle part ailleurs (aucune édition ne les
   pointe) — sans ce nettoyage, la synthèse par catégorie peut leur emprunter des
   détails (ex: mauvais pays/continent) qui ne correspondent plus à rien
   d'affiché sur la page.

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
        n_dropped = 0
        for old_cat, stories in old_categories.items():
            for story in stories:
                current_cat = current_category_by_title.get(story["title"])
                if current_cat is None:
                    # Orpheline : doublon de clustering fusionné ailleurs par la
                    # consolidation (voir docstring) — aucun sujet ne la référence
                    # plus, on la retire plutôt que de la laisser sous old_cat.
                    n_dropped += 1
                    continue
                if current_cat != old_cat:
                    n_moved += 1
                new_categories.setdefault(current_cat, []).append(story)

        if n_moved == 0 and n_dropped == 0:
            logger.info(f"  {date}: déjà à jour, ignoré")
            continue

        logger.info(
            f"  {date}: {n_moved} sujet(s) reclassé(s), {n_dropped} orpheline(s) "
            "retirée(s) — régénération des synthèses..."
        )
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
