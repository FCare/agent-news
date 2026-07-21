"""Consolide les topics générés chaque jour (bulletin_gen.py) en "sujets" pérennes
qui accumulent leurs éditions au fil du temps (voir storage.py: subjects/
subject_editions, vector_store.py: collection "subjects") — un sujet réel comme
"Guerre en Ukraine" reste UNE entrée qui évolue, plutôt que de réapparaître comme N
topics disjoints à chaque bulletin où il est à nouveau question de lui.

Appelé par main.py::run_bulletin_pipeline, entre generate_bulletin et save_bulletin.
"""

import asyncio
import logging
import os

import openai

import storage
import vector_store
from bulletin_gen import _llm

logger = logging.getLogger(__name__)

# En dessous de ce score de similarité cosinus, on ne considère même pas le candidat
# (évite un appel LLM pour un sujet clairement sans rapport) — point de départ à
# calibrer sur le backfill des bulletins réels, pas une valeur figée a priori (voir
# le plan : aucune donnée de continuité n'existe encore pour la calibrer autrement).
MATCH_SCORE_THRESHOLD = float(os.environ.get("SUBJECT_MATCH_THRESHOLD", "0.55"))
MAX_CANDIDATES = 3

_MATCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "confirm_subject_match",
        "description": (
            "Détermine si un topic d'actualité du jour est la continuation d'un sujet "
            "déjà suivi, et si oui produit son résumé consolidé mis à jour"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "is_same_subject": {
                    "type": "boolean",
                    "description": (
                        "true SEULEMENT si c'est réellement le même événement/développement "
                        "réel qui se poursuit — pas juste un thème proche. Ex: 'Guerre en "
                        "Ukraine' et 'Guerre en Irak' sont deux sujets DISTINCTS malgré le mot "
                        "commun 'guerre'. En cas de doute, false."
                    ),
                },
                "updated_title": {
                    "type": "string",
                    "description": (
                        "Si is_same_subject=true : titre à jour du sujet, TOUJOURS EN FRANÇAIS "
                        "(même si le topic du jour a un titre anglais — traduis, ne le recopie "
                        "jamais tel quel). Chaîne vide sinon."
                    ),
                },
                "updated_summary": {
                    "type": "string",
                    "description": (
                        "Si is_same_subject=true : résumé consolidé en 3-5 phrases, TOUJOURS EN "
                        "FRANÇAIS, fusionnant l'historique du sujet et la nouvelle édition du "
                        "jour (traduis si le topic du jour est dans une autre langue). Chaîne "
                        "vide sinon."
                    ),
                },
            },
            "required": ["is_same_subject", "updated_title", "updated_summary"],
        },
    },
}]

_MATCH_SYSTEM_PROMPT = (
    "Tu suis l'évolution de sujets d'actualité dans le temps. On te donne le résumé "
    "consolidé d'un sujet déjà suivi, et un nouveau topic paru aujourd'hui dans la même "
    "catégorie. RÈGLE STRICTE : is_same_subject doit être true UNIQUEMENT s'il s'agit "
    "objectivement du MÊME événement/développement réel qui se poursuit (la même guerre, "
    "la même élection, la même affaire judiciaire) — jamais deux événements distincts qui "
    "partagent seulement un thème ou des mots-clés. updated_title et updated_summary "
    "doivent TOUJOURS être rédigés en français, même si le topic du jour est dans une "
    "autre langue — traduis, ne recopie jamais un texte source tel quel. "
    "Appelle confirm_subject_match."
)


async def _find_candidate(
    topic: dict, llm_client: openai.OpenAI, model: str
) -> tuple[int, str, str] | None:
    """Retourne (subject_id, titre_à_jour, résumé_consolidé) si un sujet existant a
    été confirmé par le LLM comme continuation de ce topic, sinon None.

    PAS de filtre par catégorie ici (contrairement à une première version) : constaté
    sur le backfill réel que bulletin_gen.py catégorise parfois différemment le même
    événement réel d'un jour à l'autre (ex: la finale du Mondial 2026 vue tour à tour
    "Sport", "Astronomie & Espace", "Informatique & IA") — un filtre strict fragmentait
    donc un même sujet en une dizaine d'entrées disjointes. La confirmation LLM
    ci-dessous (règle explicite anti-faux-positif) reste le seul garde-fou nécessaire."""
    query = f"{topic.get('title', '')}\n{topic.get('summary', '')}"
    loop = asyncio.get_event_loop()
    candidates = await loop.run_in_executor(
        None, vector_store.search_subjects, query, MAX_CANDIDATES
    )
    for c in candidates:
        if c["score"] < MATCH_SCORE_THRESHOLD:
            continue
        subject_id = c["metadata"]["subject_id"]
        subject = await storage.get_subject(subject_id)
        if not subject:
            continue  # incohérence Chroma/SQLite (ex: sujet purgé récemment) — on ignore
        result = await _llm(
            llm_client, model,
            system=_MATCH_SYSTEM_PROMPT,
            user=(
                f"SUJET DÉJÀ SUIVI (résumé consolidé actuel):\n"
                f"{subject['title']}\n{subject['summary']}\n\n"
                f"NOUVEAU TOPIC DU JOUR:\n"
                f"{topic.get('title', '')}\n{topic.get('summary', '')}"
            ),
            tool=_MATCH_TOOL,
        )
        if result.get("is_same_subject") and result.get("updated_summary"):
            return subject_id, result.get("updated_title") or subject["title"], result["updated_summary"]
    return None


async def consolidate_topic(topic: dict, date: str, llm_client: openai.OpenAI, model: str) -> None:
    """Une entrée de bulletin_json["categories"][cat][i] -> crée ou met à jour le
    subjects/subject_editions correspondant. N'échoue jamais bruyamment vers
    l'appelant : le bulletin brut est déjà sauvegardé indépendamment (voir
    main.py::run_bulletin_pipeline), un raté de consolidation ne doit jamais faire
    échouer tout le pipeline."""
    title = topic.get("title", "")
    summary = topic.get("summary", "")
    category = topic.get("category", "")
    if not title or not summary:
        return

    loop = asyncio.get_event_loop()
    try:
        match = await _find_candidate(topic, llm_client, model)
        if match:
            subject_id, updated_title, updated_summary = match
            await storage.update_subject(
                subject_id, updated_title, updated_summary, date,
                edition_title=title, edition_summary=summary,
                deep_dive=topic.get("deep_dive"), what_to_watch=topic.get("what_to_watch"),
                sources=topic.get("sources"), date_range=topic.get("date_range"),
                category=category,
            )
            # Catégorie relue depuis SQLite (vote majoritaire dans update_subject,
            # potentiellement différente de `category` — la classification bruitée du
            # jour) : c'est elle qui doit alimenter la recherche/l'affichage, pas la
            # valeur brute du topic d'aujourd'hui.
            updated_subject = await storage.get_subject(subject_id)
            voted_category = updated_subject["category"] if updated_subject else category
            await loop.run_in_executor(
                None, vector_store.upsert_subject,
                subject_id, updated_title, voted_category, updated_summary, date,
            )
            logger.info(f"subject_consolidation: '{title}' rapproché du sujet #{subject_id} ('{updated_title}')")
        else:
            subject_id = await storage.create_subject(
                title, category, summary, date,
                deep_dive=topic.get("deep_dive"), what_to_watch=topic.get("what_to_watch"),
                sources=topic.get("sources"), date_range=topic.get("date_range"),
            )
            await loop.run_in_executor(
                None, vector_store.upsert_subject, subject_id, title, category, summary, date,
            )
            logger.info(f"subject_consolidation: '{title}' -> nouveau sujet #{subject_id}")
    except Exception as e:
        logger.error(f"subject_consolidation: échec pour '{title}': {e}")


async def consolidate_bulletin(bulletin_json: dict, date: str, llm_client: openai.OpenAI, model: str) -> None:
    """Point d'entrée appelé par run_bulletin_pipeline, après generate_bulletin et
    avant save_bulletin — un topic à la fois (le sémaphore LLM est déjà partagé via
    bulletin_gen._llm, pas de concurrence supplémentaire à gérer ici)."""
    n = sum(len(v) for v in bulletin_json.get("categories", {}).values())
    logger.info(f"subject_consolidation: {n} topics à consolider pour le {date}")
    for stories in bulletin_json.get("categories", {}).values():
        for topic in stories:
            await consolidate_topic(topic, date, llm_client, model)
