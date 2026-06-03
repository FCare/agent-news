import asyncio
import json
import logging
import os

import openai

from clustering import cluster_articles
from crawler import RawArticle
from search_client import SearchClient

logger = logging.getLogger(__name__)

_LLM_SEMAPHORE = asyncio.Semaphore(2)

CATEGORIES = [
    "International",
    "Europe",
    "France",
    "Économie & Finance",
    "Géopolitique & Défense",
    "Informatique & IA",
    "Science & Technologie",
    "Société & Environnement",
    "Culture & Médias",
]

# Context budget: 128k tokens available.
# Worst-case deep dive: 8 articles × 3 000 chars + 10 searches × 2 000 chars
# ≈ (24 000 + 20 000) / 4 ≈ 11 000 tokens — well within limit.
MAX_TOPICS = int(os.environ.get("MAX_TOPICS", 25))  # cap after TF-IDF clustering
MAX_BODY_IN_DEEP_DIVE = 3000         # full article body (matches crawler MAX_BODY_CHARS)
MAX_ARTICLES_IN_DEEP_DIVE = 8        # max articles per topic in deep dive
MAX_SEARCH_REPORT_IN_DEEP_DIVE = 2000  # richer search context
SEARCH_QUERIES_PER_TOPIC = int(os.environ.get("SEARCH_QUERIES_PER_TOPIC", 5))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


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
                        "5 requêtes couvrant: (1) faits récents, (2) contexte historique, "
                        "(3) réactions officielles, (4) implications économiques/géopolitiques, "
                        "(5) analyses d'experts et perspectives"
                    ),
                    "minItems": 5,
                    "maxItems": 5,
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
                "title_fr": {
                    "type": "string",
                    "description": (
                        "Titre du sujet en français, court et accrocheur. "
                        "Traduis si nécessaire depuis l'anglais ou l'allemand."
                    )
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Résumé factuel en 3-5 phrases, style JT 20h. "
                        "Faits précis, chiffres si disponibles, ton oral. En français."
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
            "required": ["title_fr", "summary", "deep_dive", "what_to_watch", "sources"],
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
                        "Style factuel et direct, sans formule d'introduction."
                    )
                },
            },
            "required": ["headline", "flash"],
        }
    }
}]

_TRANSLATE_TOOL = [{
    "type": "function",
    "function": {
        "name": "translate_excerpts",
        "parameters": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extraits traduits en français, dans le même ordre que l'entrée",
                }
            },
            "required": ["translations"],
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
    async with _LLM_SEMAPHORE:
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
    """
    Use cluster indices from TF-IDF clustering when available,
    fall back to keyword matching otherwise.
    """
    indices = topic.get("_article_indices")
    if indices:
        return [articles[i] for i in indices[:n] if i < len(articles)]

    # Fallback: keyword matching
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

def _do_cluster_topics(articles: list[RawArticle]) -> list[dict]:
    """TF-IDF + DBSCAN clustering — no LLM, instant."""
    topics = cluster_articles(articles)
    topics = topics[:MAX_TOPICS]
    for t in topics:
        logger.info(
            f"  [{t['article_count']:3d} art.] [{t['category']}] {t['title']}"
        )
    return topics


async def _generate_search_queries(topic: dict,
                                    llm_client: openai.OpenAI, model: str) -> list[str]:
    logger.info(f"    → appel LLM requêtes...")
    result = await _llm(
        llm_client, model,
        system=(
            f"Tu es journaliste d'investigation. Génère exactement {SEARCH_QUERIES_PER_TOPIC} requêtes "
            "de recherche web pour enrichir un sujet d'actualité. "
            "Couvre: (1) faits récents, (2) contexte/historique, (3) réactions officielles, "
            "(4) implications économiques ou géopolitiques, (5) perspectives d'experts. "
            "Requêtes courtes et efficaces pour un moteur d'actualités. "
            "Appelle generate_search_queries."
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


def _fmt_date(dt) -> str:
    if not dt:
        return ""
    try:
        from zoneinfo import ZoneInfo
        if dt.tzinfo is None:
            from datetime import timezone as _tz
            dt = dt.replace(tzinfo=_tz.utc)
        dt = dt.astimezone(ZoneInfo("Europe/Paris"))
        return dt.strftime("%-d %b %Y %Hh%M")
    except Exception:
        return str(dt)[:16]


async def _generate_deep_dive(topic: dict, articles: list[RawArticle],
                               search_results: list[dict],
                               llm_client: openai.OpenAI, model: str) -> dict:
    # Find relevant articles — include publication date in excerpts
    matched = _find_articles_for_topic(topic, articles)
    article_excerpts = []
    article_dates = []
    for a in matched:
        date_tag = _fmt_date(a.published_at)
        article_dates.append(a.published_at)
        header = f"[{a.publisher}{' — ' + date_tag if date_tag else ''}] {a.title}"
        article_excerpts.append(f"{header}\n{a.body[:MAX_BODY_IN_DEEP_DIVE]}")

    # Date range of sources
    valid_dates = [d for d in article_dates if d]
    if valid_dates:
        from datetime import timezone as _tz
        def _to_utc(d):
            return d.replace(tzinfo=_tz.utc) if d.tzinfo is None else d
        oldest = min(valid_dates, key=_to_utc)
        newest = max(valid_dates, key=_to_utc)
        date_range = f"{_fmt_date(oldest)} → {_fmt_date(newest)}"
    else:
        date_range = ""

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
        + (f"PÉRIODE DES SOURCES: {date_range}\n" if date_range else "")
        + f"\n=== EXTRAITS D'ARTICLES ({len(article_excerpts)}) ===\n{articles_block}\n\n"
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

    title_fr  = result.get("title_fr",      topic["title"])
    summary   = result.get("summary",       topic["title"])
    deep_dive = result.get("deep_dive",     "")
    watch     = result.get("what_to_watch", "")
    sources   = result.get("sources",       [])
    logger.info(f"  titre_fr : {title_fr}")
    logger.info(f"  résumé   : {summary}")
    logger.info(f"  analyse  : {deep_dive[:200]}{'…' if len(deep_dive) > 200 else ''}")
    if watch:
        logger.info(f"  à suivre : {watch}")
    if sources:
        logger.info(f"  sources  : {', '.join(sources)}")
    return {
        "title":        title_fr,
        "category":     topic["category"],
        "importance":   topic.get("importance", 5),
        "date_range":   date_range,
        "summary":      summary,
        "deep_dive":    deep_dive,
        "what_to_watch": watch,
        "sources":      sources,
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
            "Rédige un titre accrocheur et un flash info du jour à partir des sujets principaux. "
            "Style factuel et direct. "
            "INTERDIT : toute formule d'introduction ('Ce soir', 'Bonsoir', 'Au programme'), "
            "toute formule de transition ('à suivre', 'restez avec nous', 'on en parle'). "
            "Commence directement par les faits. Le flash se termine sur un fait, pas une promesse. "
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

    # Step 1: Cluster into topics (TF-IDF + DBSCAN — no LLM, instant)
    loop = asyncio.get_event_loop()
    topics = await loop.run_in_executor(None, _do_cluster_topics, articles)
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
# Question answering (ChromaDB semantic search)
# ---------------------------------------------------------------------------

async def answer_question(query: str, bulletin: dict, search_client: SearchClient,
                           llm_client: openai.OpenAI, model: str) -> str:
    import vector_store

    loop = asyncio.get_event_loop()

    # 1. Semantic search in bulletin topics (deep dives)
    topic_hits = await loop.run_in_executor(
        None, vector_store.search_topics, query, 6
    )
    # 2. Semantic search in raw articles
    article_hits = await loop.run_in_executor(
        None, vector_store.search_articles, query, 6
    )

    logger.info(
        f"ChromaDB '{query[:50]}': {len(topic_hits)} topics, {len(article_hits)} articles"
    )

    # Translate raw article excerpts to French before building context
    if article_hits:
        raw_excerpts = [h["content"][:400] for h in article_hits]
        result = await _llm(
            llm_client, model,
            system=(
                "Traduis chaque extrait numéroté en français. "
                "Conserve le même ordre et la même numérotation. "
                "Appelle translate_excerpts."
            ),
            user="\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(raw_excerpts)),
            tool=_TRANSLATE_TOOL,
        )
        translated = result.get("translations", [])
        if len(translated) == len(article_hits):
            for i, h in enumerate(article_hits):
                h["content"] = translated[i]

    # Build context from semantic hits (all in French)
    context_parts = []
    for h in topic_hits:
        meta = h["metadata"]
        context_parts.append(
            f"[{meta.get('category','')} — {meta.get('date','')} — score {h['score']}]\n"
            f"{h['content'][:600]}"
        )
    for h in article_hits:
        meta = h["metadata"]
        context_parts.append(
            f"[{meta.get('publisher','')} — score {h['score']}]\n"
            f"{h['content'][:400]}"
        )

    has_context = bool(context_parts)
    context_block = "\n\n---\n\n".join(context_parts) if context_parts else ""

    # 3. LLM answer from semantic context
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es un expert en actualités. Réponds à la question en français, ton oral, factuel. "
            "Base-toi uniquement sur le contexte fourni. "
            "Si le contexte ne couvre pas le sujet, indique needs_web_search=true. "
            "Appelle answer_question."
        ),
        user=(
            f"Question: {query}\n\n"
            f"Contexte (résultats sémantiques):\n{context_block or '(aucun résultat)'}"
        ),
        tool=_ANSWER_TOOL,
    )

    answer = result.get("answer", "")
    needs_search = result.get("needs_web_search", False) or not has_context

    # 4. Web search fallback si pas de contexte ou LLM non satisfait
    if needs_search or not answer:
        logger.info(f"Question '{query[:50]}' → recherche web complémentaire")
        search_results = await search_client.search_many(
            [query, f"{query} actualité récente"], "news"
        )
        reports = [r.get("report", "") for r in search_results if r.get("report")]
        if reports:
            search_block = "\n\n".join(r[:800] for r in reports[:4])
            result2 = await _llm(
                llm_client, model,
                system=(
                    "Tu es expert en actualités. Réponds à la question en français, "
                    "ton oral, factuel, à partir des résultats de recherche. "
                    "Appelle answer_question avec needs_web_search=false."
                ),
                user=f"Question: {query}\n\nRecherches:\n{search_block}",
                tool=_ANSWER_TOOL,
            )
            answer = result2.get("answer", answer)

    return answer or "Je n'ai pas trouvé d'information sur ce sujet dans l'actualité récente."
