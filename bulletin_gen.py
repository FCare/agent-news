import asyncio
import json
import logging
from typing import Any

import openai

from crawler import RawArticle
from search_client import SearchClient

logger = logging.getLogger(__name__)

CATEGORIES = [
    "International",
    "Europe",
    "France",
    "Économie & Finance",
    "Géopolitique & Défense",
    "Informatique & IA",
    "Science & Technologie",
    "Société & Environnement",
    "Littérature & BD",
]

# Context budget: 128k tokens available.
# Worst-case deep dive: 8 articles × 3 000 chars + 10 searches × 2 000 chars
# ≈ (24 000 + 20 000) / 4 ≈ 11 000 tokens — well within limit.
MAX_ARTICLES_FOR_CLUSTER = 80        # titles only — beyond this, topics repeat
MAX_BODY_IN_DEEP_DIVE = 3000         # full article body (matches crawler MAX_BODY_CHARS)
MAX_ARTICLES_IN_DEEP_DIVE = 8        # max articles per topic in deep dive
MAX_SEARCH_REPORT_IN_DEEP_DIVE = 2000  # richer search context
SEARCH_QUERIES_PER_TOPIC = 10

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_CLUSTER_TOOL = [{
    "type": "function",
    "function": {
        "name": "identify_topics",
        "description": (
            "Identifie les sujets d'actualité distincts parmi les articles fournis. "
            "Regroupe les articles couvrant le même événement en un seul sujet. "
            "Classe par importance décroissante."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "description": "Liste des sujets identifiés, du plus au moins important",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Titre court et accrocheur en français"
                            },
                            "category": {
                                "type": "string",
                                "enum": CATEGORIES,
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Score 1-10, 10 = sujet majeur du jour"
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "3-5 mots-clés en français et anglais pour retrouver les articles liés"
                            },
                        },
                        "required": ["title", "category", "importance", "keywords"],
                    }
                }
            },
            "required": ["topics"],
        }
    }
}]

_SEARCH_QUERIES_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_search_queries",
        "description": "Génère exactement 10 requêtes web pour approfondir un sujet d'actualité",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "10 requêtes couvrant: (1) faits récents précis, (2) contexte historique, "
                        "(3) réactions officielles, (4) implications économiques, "
                        "(5) implications géopolitiques, (6) analyses d'experts, "
                        "(7) perspective américaine, (8) perspective européenne, "
                        "(9) chiffres et données clés, (10) prochains développements attendus"
                    ),
                    "minItems": 10,
                    "maxItems": 10,
                }
            },
            "required": ["queries"],
        }
    }
}]

_DEEP_DIVE_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_analysis",
        "description": "Génère l'analyse complète d'un sujet pour le journal télévisé",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "Résumé factuel en 3-5 phrases, style JT 20h. "
                        "Faits précis, chiffres si disponibles, ton oral."
                    )
                },
                "deep_dive": {
                    "type": "string",
                    "description": (
                        "Analyse approfondie: contexte historique, enjeux stratégiques, "
                        "chiffres clés, perspectives EU et US, déclarations importantes, "
                        "conséquences potentielles. Ton oral sérieux, 6-10 phrases, sans liste."
                    )
                },
                "what_to_watch": {
                    "type": "string",
                    "description": "Ce qu'il faut surveiller / prochaines étapes. 1-2 phrases."
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Noms des médias sources (max 6)"
                },
            },
            "required": ["summary", "deep_dive", "what_to_watch", "sources"],
        }
    }
}]

_FLASH_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_flash",
        "description": "Génère le flash info et le titre du JT du jour",
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "Titre principal du jour, une phrase percutante"
                },
                "flash": {
                    "type": "string",
                    "description": (
                        "3 phrases résumant les 3 sujets les plus importants du jour. "
                        "Commence par 'Ce soir au journal...'. Style présentateur TV."
                    )
                },
            },
            "required": ["headline", "flash"],
        }
    }
}]

_ANSWER_TOOL = [{
    "type": "function",
    "function": {
        "name": "answer_question",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Réponse complète à la question en français, style oral. "
                        "Basée sur les informations disponibles dans le bulletin."
                    )
                },
                "needs_web_search": {
                    "type": "boolean",
                    "description": "True si le bulletin ne couvre pas ce sujet"
                },
            },
            "required": ["answer", "needs_web_search"],
        }
    }
}]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _recover_partial_json(raw: str) -> dict:
    """Try to extract complete objects from truncated tool-call JSON."""
    # Find last complete object before truncation
    last = raw.rfind("},")
    if last == -1:
        last = raw.rfind("}")
    if last == -1:
        return {}
    fragment = raw[:last + 1]
    # Locate the opening of the topics array
    topics_pos = fragment.find('"topics"')
    if topics_pos == -1:
        return {}
    array_start = fragment.find("[", topics_pos)
    if array_start == -1:
        return {}
    try:
        return json.loads(fragment[: last + 1] + "]}")
    except Exception:
        return {}


def _llm_call(llm_client: openai.OpenAI, model: str, system: str,
              user: str, tool: list, max_tokens: int | None = None) -> dict:
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        tools=tool,
        tool_choice="required",
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    resp = llm_client.chat.completions.create(**kwargs)
    calls = resp.choices[0].message.tool_calls
    if not calls:
        return {}

    raw = calls[0].function.arguments
    finish = resp.choices[0].finish_reason
    if finish == "length":
        logger.warning(f"LLM output truncated (finish=length), attempting JSON recovery "
                       f"({len(raw)} chars)")
        logger.debug(f"  raw[:500]: {raw[:500]}")
        logger.debug(f"  raw[-200]: {raw[-200:]}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON parse error: {e} — raw length={len(raw)}")
        logger.warning(f"  first 300 chars: {raw[:300]}")
        logger.warning(f"  last  200 chars: {raw[-200:]}")
        recovered = _recover_partial_json(raw)
        if recovered:
            n = len(recovered.get("topics", []))
            logger.warning(f"  JSON recovery: {n} sujets récupérés")
        return recovered


async def _llm(llm_client: openai.OpenAI, model: str, system: str,
               user: str, tool: list, max_tokens: int | None = None) -> dict:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None, _llm_call, llm_client, model, system, user, tool, max_tokens
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Article matching
# ---------------------------------------------------------------------------

def _find_articles_for_topic(topic: dict, articles: list[RawArticle],
                              n: int = MAX_ARTICLES_IN_DEEP_DIVE) -> list[RawArticle]:
    """Keyword-based matching: score each article against topic title + keywords."""
    keywords = [w.lower() for w in topic.get("keywords", [])]
    title_words = [w.lower() for w in topic["title"].split() if len(w) > 3]
    all_terms = set(keywords + title_words)

    def score(a: RawArticle) -> int:
        text = (a.title + " " + a.body[:200]).lower()
        return sum(1 for term in all_terms if term in text)

    ranked = sorted(articles, key=score, reverse=True)
    return [a for a in ranked[:n] if score(a) > 0] or ranked[:3]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

async def _cluster_topics(articles: list[RawArticle],
                           llm_client: openai.OpenAI, model: str) -> list[dict]:
    sample = articles[:MAX_ARTICLES_FOR_CLUSTER]
    articles_text = "\n".join(
        f"[{i}] [{a.publisher} / {a.country}] {a.title}"
        for i, a in enumerate(sample)
    )
    logger.info(f"  → appel LLM clustering ({len(sample)} titres)...")
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es éditeur d'un journal télévisé. Identifie EXACTEMENT 15 sujets d'actualité "
            "distincts parmi les titres fournis. Pas plus, pas moins. "
            "Regroupe les articles couvrant le même événement. Classe par importance décroissante. "
            f"Catégories: {', '.join(CATEGORIES)}. "
            "Les sujets tech/IA → 'Informatique & IA'. "
            "keywords: 3 mots courts max. Appelle identify_topics."
        ),
        user=f"Articles ({len(sample)} titres):\n\n{articles_text}",
        tool=_CLUSTER_TOOL,
        max_tokens=2000,
    )
    MAX_TOPICS = 25
    topics = result.get("topics", [])
    topics.sort(key=lambda t: -t.get("importance", 0))

    # Deduplicate by title (LLM sometimes generates the same generic topic multiple times)
    seen_titles: set[str] = set()
    unique_topics = []
    for t in topics:
        key = t["title"].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_topics.append(t)
    n_dupes = len(topics) - len(unique_topics)
    topics = unique_topics

    if len(topics) > MAX_TOPICS:
        logger.info(f"Clustering: {len(topics)} sujets ({n_dupes} doublons supprimés) → tronqué à {MAX_TOPICS}")
        topics = topics[:MAX_TOPICS]
    else:
        logger.info(f"Clustering: {len(topics)} sujets ({n_dupes} doublons supprimés)")
    for t in topics:
        logger.info(f"  [{t.get('importance',0):2d}] [{t['category']}] {t['title']}")
    return topics


async def _generate_search_queries(topic: dict,
                                    llm_client: openai.OpenAI, model: str) -> list[str]:
    logger.info(f"    → appel LLM requêtes...")
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es journaliste d'investigation. Génère exactement 10 requêtes de recherche web "
            "pour approfondir un sujet d'actualité selon les 10 angles imposés. "
            "Formule des requêtes efficaces pour un moteur d'actualités. "
            "Réponds en appelant generate_search_queries."
        ),
        user=(
            f"Sujet: {topic['title']}\n"
            f"Catégorie: {topic['category']}"
        ),
        tool=_SEARCH_QUERIES_TOOL,
    )
    queries = result.get("queries", [])
    if not queries:
        queries = [topic["title"]]
    queries = queries[:SEARCH_QUERIES_PER_TOPIC]
    for i, q in enumerate(queries, 1):
        logger.info(f"    requête {i:2d}: {q}")
    return queries


async def _generate_deep_dive(topic: dict, articles: list[RawArticle],
                               search_results: list[dict],
                               llm_client: openai.OpenAI, model: str) -> dict:
    # Find relevant articles via keyword matching (no LLM indices needed)
    matched = _find_articles_for_topic(topic, articles)
    article_excerpts = [
        f"[{a.publisher}] {a.title}\n{a.body[:MAX_BODY_IN_DEEP_DIVE]}"
        for a in matched
    ]

    # Gather search reports
    search_excerpts = []
    for r in search_results:
        report = (r.get("report") or "").strip()
        if report:
            search_excerpts.append(report[:MAX_SEARCH_REPORT_IN_DEEP_DIVE])

    articles_block = "\n\n---\n\n".join(article_excerpts) or "(aucun extrait)"
    searches_block = "\n\n---\n\n".join(search_excerpts) or "(aucun résultat)"

    user_content = (
        f"SUJET: {topic['title']}\n"
        f"CATÉGORIE: {topic['category']}\n"
        f"FAITS INITIAUX: (voir les articles ci-dessous)\n\n"
        f"=== EXTRAITS D'ARTICLES ({len(article_excerpts)}) ===\n{articles_block}\n\n"
        f"=== RÉSULTATS DE RECHERCHE ({len(search_excerpts)}) ===\n{searches_block}"
    )

    logger.info(f"    → appel LLM deep dive ({len(article_excerpts)} articles, {len(search_excerpts)} recherches)...")
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es un expert journaliste du Monde diplomatique. Tu reçois des extraits d'articles "
            "et des résultats de recherches approfondies sur un sujet d'actualité. "
            "Rédige une analyse complète en français: faits précis avec chiffres, contexte historique, "
            "enjeux, perspectives EU et US, ce qu'il faut retenir. "
            "Ton oral, sérieux, factuel, sans liste ni tirets. "
            "Appelle generate_analysis."
        ),
        user=user_content,
        tool=_DEEP_DIVE_TOOL,
    )

    summary   = result.get("summary",       topic["title"])
    deep_dive = result.get("deep_dive",     "")
    watch     = result.get("what_to_watch", "")
    sources   = result.get("sources",       [])
    logger.info(f"  résumé   : {summary}")
    logger.info(f"  analyse  : {deep_dive[:200]}{'…' if len(deep_dive) > 200 else ''}")
    if watch:
        logger.info(f"  à suivre : {watch}")
    if sources:
        logger.info(f"  sources  : {', '.join(sources)}")
    return {
        "title":       topic["title"],
        "category":    topic["category"],
        "importance":  topic.get("importance", 5),
        "summary":     summary,
        "deep_dive":   deep_dive,
        "what_to_watch": watch,
        "sources":     sources,
    }


async def _generate_flash(topics: list[dict],
                           llm_client: openai.OpenAI, model: str) -> tuple[str, str]:
    top = topics[:10]
    topics_text = "\n".join(
        f"- [{t['category']}] {t['title']}: {t.get('summary', '')[:200]}"
        for t in top
    )
    logger.info(f"  → appel LLM flash ({len(top)} sujets)...")
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es présentateur du journal de 20h. Rédige le titre et le flash info du jour "
            "à partir des sujets principaux. Style TV, oral, percutant, factuel. "
            "Appelle generate_flash."
        ),
        user=f"Sujets du jour:\n\n{topics_text}",
        tool=_FLASH_TOOL,
    )
    headline = result.get("headline", "L'actualité du jour")
    flash    = result.get("flash",    "")
    logger.info(f"[TITRE]  {headline}")
    logger.info(f"[FLASH]  {flash}")
    return headline, flash


def _assemble_bulletin(flash: str, headline: str, topics: list[dict]) -> dict:
    categories: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for t in topics:
        cat = t.get("category", "International")
        if cat not in categories:
            cat = "International"
        categories[cat].append(t)
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    return {
        "flash": flash,
        "headline": headline,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_bulletin(articles: list[RawArticle], search_client: SearchClient,
                             llm_client: openai.OpenAI, model: str) -> dict:
    logger.info(f"Génération bulletin: {len(articles)} articles")

    # Step 1: Cluster into topics
    topics = await _cluster_topics(articles, llm_client, model)
    if not topics:
        logger.error("Aucun sujet identifié")
        return {}

    # Step 2-4: For each topic, generate queries, search, build deep dive
    total_searches = len(topics) * SEARCH_QUERIES_PER_TOPIC
    logger.info(f"[2/4] Enrichissement: {len(topics)} sujets × {SEARCH_QUERIES_PER_TOPIC} requêtes = {total_searches} recherches MQTT")
    enriched_topics = []
    for i, topic in enumerate(topics):
        logger.info(f"  sujet [{i+1}/{len(topics)}] {topic['title']!r} ({topic['category']})")

        queries = await _generate_search_queries(topic, llm_client, model)
        search_results = await search_client.search_many(queries, "news")
        n_results = sum(1 for r in search_results if r.get("report"))
        logger.info(f"  sujet [{i+1}/{len(topics)}] {n_results}/{len(queries)} résultats → deep dive en cours")

        deep = await _generate_deep_dive(topic, articles, search_results, llm_client, model)
        enriched_topics.append(deep)
        logger.info(f"  sujet [{i+1}/{len(topics)}] OK")

    logger.info(f"[2/4] Enrichissement terminé")
    logger.info(f"[3/4] Génération du flash...")
    # Step 5: Flash
    headline, flash = await _generate_flash(enriched_topics, llm_client, model)

    # Step 6: Assemble
    bulletin = _assemble_bulletin(flash, headline, enriched_topics)
    logger.info(
        f"Bulletin généré: {sum(len(v) for v in bulletin['categories'].values())} sujets "
        f"en {len(bulletin['categories'])} catégories"
    )
    return bulletin


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

async def answer_question(query: str, bulletin: dict, search_client: SearchClient,
                           llm_client: openai.OpenAI, model: str) -> str:
    # Flatten all topics from bulletin
    all_topics = []
    for stories in bulletin.get("categories", {}).values():
        all_topics.extend(stories)

    if not all_topics:
        return "Aucun bulletin disponible pour répondre à cette question."

    topics_summary = "\n".join(
        f"[{t['category']}] {t['title']}: {t.get('summary', '')[:300]}"
        for t in all_topics
    )

    # Find relevant topics + decide if web search needed
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es un expert en actualités. Tu as accès au bulletin du jour. "
            "Réponds à la question posée en utilisant les informations disponibles. "
            "Si le sujet n'est pas couvert dans le bulletin, indique needs_web_search=true. "
            "Appelle answer_question."
        ),
        user=(
            f"Question: {query}\n\n"
            f"Bulletin du jour:\n{topics_summary}"
        ),
        tool=_ANSWER_TOOL,
    )

    answer = result.get("answer", "")
    needs_search = result.get("needs_web_search", False)

    if needs_search or not answer:
        logger.info(f"Question '{query[:50]}' → recherche web complémentaire")
        search_results = await search_client.search_many(
            [query, f"{query} actualité récente", f"{query} news"], "news"
        )
        reports = [r.get("report", "") for r in search_results if r.get("report")]
        if reports:
            search_block = "\n\n".join(r[:600] for r in reports[:5])
            result2 = await _llm(
                llm_client, model,
                system=(
                    "Tu es expert en actualités. Réponds à la question en français "
                    "à partir des résultats de recherche fournis. Ton oral, factuel. "
                    "Appelle answer_question avec needs_web_search=false."
                ),
                user=f"Question: {query}\n\nRecherches:\n{search_block}",
                tool=_ANSWER_TOOL,
            )
            answer = result2.get("answer", answer)

    return answer or "Je n'ai pas trouvé d'information sur ce sujet dans l'actualité récente."
