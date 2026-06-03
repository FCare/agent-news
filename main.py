import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import openai
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from nexus_client import NexusClient

import bulletin_gen
import storage
import vector_store
from crawler import crawl_articles
from search_client import SearchClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VK_URL              = os.environ["VK_URL"]
MQTT_HOST           = os.environ["MQTT_HOST"]
MQTT_PORT           = int(os.environ.get("MQTT_PORT", "1883"))
SERVICE_USERNAME    = os.environ["MQTT_SERVICE_USERNAME"]
SERVICE_API_KEY     = os.environ["MQTT_SERVICE_API_KEY"]
LLM_BASE_URL        = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL           = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY    = os.environ["LLAMACPP_API_KEY"]
BULLETIN_HOURS      = os.environ.get("BULLETIN_HOURS", "8,20")

AGENT_NAME = "news"
_subscribed_users: set[str] = set()
_is_generating = False
_search_client = SearchClient()

# ---------------------------------------------------------------------------
# LLM client (shared)
# ---------------------------------------------------------------------------

def _get_llm_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)

# ---------------------------------------------------------------------------
# Bulletin pipeline
# ---------------------------------------------------------------------------

async def run_bulletin_pipeline() -> None:
    global _is_generating
    if _is_generating:
        logger.warning("Pipeline déjà en cours, skip")
        return

    _is_generating = True
    logger.info("=== Début pipeline bulletin ===")

    try:
        # 1. Crawl
        logger.info("[1/4] Crawl fundus en cours...")
        articles = await crawl_articles()
        if not articles:
            logger.error("[1/4] Aucun article crawlé, pipeline annulé")
            return
        logger.info(f"[1/4] Crawl terminé: {len(articles)} articles")

        # 2. Save articles (SQLite + ChromaDB)
        rows = [
            {
                "url": a.url, "title": a.title, "body": a.body,
                "publisher": a.publisher, "country": a.country,
                "published_at": a.published_at.isoformat() if a.published_at else None,
            }
            for a in articles
        ]
        await storage.save_articles(rows)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.upsert_articles, rows)

        # 3. Generate bulletin
        logger.info("[2/4] Clustering et identification des sujets...")
        llm = _get_llm_client()
        bulletin = await bulletin_gen.generate_bulletin(articles, _search_client, llm, LLM_MODEL)
        if not bulletin:
            logger.error("[2/4] Bulletin vide, pipeline annulé")
            return

        # 4. Save (SQLite + ChromaDB) — only if bulletin is valid
        logger.info("[3/4] Sauvegarde du bulletin...")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        n_topics = sum(len(v) for v in bulletin.get("categories", {}).values())
        has_flash = bool(bulletin.get("flash", "").strip())
        has_headline = bulletin.get("headline", "") not in ("", "L'actualité du jour")
        if not has_flash or not has_headline:
            logger.warning("[3/4] Bulletin incomplet (flash ou headline manquant) — sauvegarde ignorée pour ne pas écraser un bulletin valide")
        else:
            await storage.save_bulletin(
                date=today,
                flash=bulletin["flash"],
                headline=bulletin["headline"],
                bulletin_json=bulletin,
                n_articles=len(articles),
                n_topics=n_topics,
            )
            await loop.run_in_executor(
                None, vector_store.upsert_bulletin_topics, bulletin, today
            )

        # 5. Purge old data
        logger.info("[4/4] Purge des données anciennes...")
        await storage.purge_old_data()
        await loop.run_in_executor(None, vector_store.purge_old_topics)

        logger.info(f"=== Pipeline terminé: {n_topics} sujets pour le {today} ===")

    except Exception as e:
        logger.exception(f"Erreur pipeline: {e}")
    finally:
        _is_generating = False

# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _format_flash(bulletin_row: dict) -> str:
    b = bulletin_row["bulletin_json"]
    date = bulletin_row["date"]
    return f"[{date}] {b.get('headline', '')}\n\n{b.get('flash', '')}"


def _format_bulletin(bulletin_row: dict) -> str:
    b = bulletin_row["bulletin_json"]
    date = bulletin_row["date"]
    parts = [f"JOURNAL DU {date.upper()}", f"{b.get('headline', '')}", "", b.get("flash", ""), ""]

    for cat, stories in b.get("categories", {}).items():
        parts.append(f"── {cat.upper()} ──")
        for story in stories:
            parts.append(f"\n• {story['title']}")
            parts.append(story.get("summary", ""))
        parts.append("")

    return "\n".join(parts)


def _format_category(bulletin_row: dict, category: str) -> str:
    b = bulletin_row["bulletin_json"]
    date = bulletin_row["date"]
    categories = b.get("categories", {})

    # Fuzzy match
    matched = None
    for cat in categories:
        if category.lower() in cat.lower() or cat.lower() in category.lower():
            matched = cat
            break
    if not matched:
        avail = ", ".join(categories.keys())
        return f"Catégorie '{category}' introuvable. Disponibles: {avail}"

    stories = categories[matched]
    parts = [f"[{date}] {matched.upper()}", ""]
    for story in stories:
        parts.append(f"• {story['title']}")
        parts.append(story.get("summary", ""))
        parts.append("")
    return "\n".join(parts)


def _format_deep_dive(bulletin_row: dict, topic_query: str) -> str:
    b = bulletin_row["bulletin_json"]
    date = bulletin_row["date"]

    best: dict | None = None
    best_score = -1
    query_lower = topic_query.lower()

    for stories in b.get("categories", {}).values():
        for story in stories:
            title_lower = story["title"].lower()
            score = sum(1 for word in query_lower.split() if word in title_lower)
            if score > best_score:
                best_score = score
                best = story

    if not best:
        return "Sujet introuvable dans le bulletin."

    date_range = best.get("date_range", "")
    parts = [
        f"[{date}] {best['title']}",
        f"Catégorie: {best.get('category', '?')}"
        + (f"  ·  Sources du {date_range}" if date_range else ""),
        "",
        best.get("summary", ""),
        "",
        "── ANALYSE APPROFONDIE ──",
        "",
        best.get("deep_dive", ""),
        "",
    ]
    if best.get("what_to_watch"):
        parts += ["── À SUIVRE ──", "", best["what_to_watch"], ""]
    if best.get("sources"):
        parts.append(f"Sources: {', '.join(best['sources'])}")

    return "\n".join(parts)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "Aucun bulletin en archive."
    parts = ["HISTORIQUE DES BULLETINS", ""]
    for row in history:
        parts.append(
            f"• {row['date']} — {row['headline']} "
            f"({row['n_topics']} sujets, {row['n_articles']} articles)"
        )
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# MQTT handlers
# ---------------------------------------------------------------------------

async def on_user_connected(topic: str, payload) -> None:
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    private_topics = payload.get("private_topics", [])

    if not username or not password:
        return

    agent_topics_topic = None
    for entry in private_topics:
        for t in entry.get("topics", []):
            if t["topic"].endswith("/agent_topics"):
                agent_topics_topic = t["topic"]
                break

    if not agent_topics_topic:
        logger.warning(f"[{username}] agent_topics introuvable, skip")
        return

    request_topic = f"users/{username}/news/request"
    result_topic  = f"users/{username}/news/result"

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    await nexus.publish(agent_topics_topic, [{
        "agent": AGENT_NAME,
        "topics": [
            {
                "topic": request_topic,
                "description": (
                    "Source EXCLUSIVE d'actualités en temps réel. "
                    "OBLIGATOIRE pour toute question sur l'actualité, les nouvelles du jour, "
                    "les événements récents, la politique, l'économie, le sport, la science, la culture. "
                    "Ne jamais répondre à une question d'actualité sans consulter ce service en premier. "
                    "Types disponibles : "
                    "flash (résumé des 3 sujets du jour en 3 phrases), "
                    "bulletin (journal complet par catégories), "
                    "category (sujets d'une catégorie : International, Europe, France, Économie, Géopolitique, Informatique, Science, Société, Culture), "
                    "deep_dive (analyse approfondie d'un sujet précis, avec champ 'topic'), "
                    "question (question libre en langage naturel sur l'actu du jour, avec champ 'query'), "
                    "history (liste des bulletins archivés)."
                ),
                "access": "write",
                "response_topic": result_topic,
                "format": {
                    "type": "flash | bulletin | category | deep_dive | question | history",
                    "category": "(optionnel) ex: 'Europe', 'Informatique & IA'",
                    "topic": "(optionnel) mot-clé ou titre de sujet pour deep_dive",
                    "query": "(optionnel) question libre pour type=question",
                    "date": "(optionnel) YYYY-MM-DD pour type=history",
                },
            },
            {
                "topic": result_topic,
                "description": "Réponse du journal. Champ 'content' contient le texte.",
                "access": "read",
                "format": {
                    "type": "string",
                    "content": "string",
                    "bulletin_date": "YYYY-MM-DD",
                    "is_generating": "bool",
                },
            },
        ],
    }])
    logger.info(f"[{username}] Topics news déclarés")

    if username in _subscribed_users:
        return
    _subscribed_users.add(username)

    async def on_news_request(t: str, p) -> None:
        if not isinstance(p, dict):
            return
        req_type = p.get("type", "bulletin").lower()
        logger.info(f"[{username}] Requête news: type={req_type}")

        bulletin_row = await storage.get_latest_bulletin()

        status_note = ""

        content = ""

        if req_type == "flash":
            if bulletin_row:
                content = _format_flash(bulletin_row) + status_note
            else:
                content = "Bulletin non disponible." + (" Génération en cours..." if _is_generating else "")

        elif req_type == "bulletin":
            if bulletin_row:
                content = _format_bulletin(bulletin_row) + status_note
            else:
                content = "Bulletin non disponible." + (" Génération en cours..." if _is_generating else "")

        elif req_type == "category":
            category = p.get("category", "")
            if not category:
                content = f"Précise une catégorie. Disponibles: {', '.join(bulletin_gen.CATEGORIES)}"
            elif bulletin_row:
                content = _format_category(bulletin_row, category) + status_note
            else:
                content = "Bulletin non disponible."

        elif req_type == "deep_dive":
            topic_query = p.get("topic", "")
            if not topic_query:
                content = "Précise un sujet avec le champ 'topic'."
            elif bulletin_row:
                content = _format_deep_dive(bulletin_row, topic_query) + status_note
            else:
                content = "Bulletin non disponible."

        elif req_type == "question":
            query = p.get("query", "").strip()
            if not query:
                content = "Précise une question avec le champ 'query'."
            elif bulletin_row:
                llm = _get_llm_client()
                content = await bulletin_gen.answer_question(
                    query, bulletin_row["bulletin_json"], _search_client, llm, LLM_MODEL
                )
            else:
                content = "Aucun bulletin disponible pour répondre à cette question."

        elif req_type == "history":
            date = p.get("date", "")
            if date:
                row = await storage.get_bulletin_by_date(date)
                if row:
                    content = _format_bulletin(row)
                else:
                    content = f"Pas de bulletin pour le {date}."
            else:
                history = await storage.get_history_list()
                content = _format_history(history)

        else:
            content = f"Type inconnu: {req_type}. Disponibles: flash, bulletin, category, deep_dive, question, history."

        await nexus.publish(result_topic, {
            "type": req_type,
            "content": content,
            "bulletin_date": bulletin_row["date"] if bulletin_row else "",
            "is_generating": _is_generating,
        })

    nexus.subscribe(request_topic, on_news_request)
    nexus.start_listening()
    logger.info(f"[{username}] Abonné à {request_topic}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    await storage.init_db()

    global _search_client
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    await _search_client.setup(nexus, SERVICE_USERNAME, SERVICE_API_KEY)

    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    hours = BULLETIN_HOURS
    logger.info(f"News service démarré — LLM: {LLM_MODEL} — bulletins à {hours}h")

    # Scheduler: cron aux heures configurées (ex: "8,20" → 8h et 20h)
    scheduler = AsyncIOScheduler(timezone="Europe/Paris")
    scheduler.add_job(
        run_bulletin_pipeline,
        "cron",
        hour=hours,
        minute=0,
        id="bulletin_pipeline",
        max_instances=1,
    )
    scheduler.start()

    # Initial run
    logger.info("Lancement du premier bulletin...")
    asyncio.create_task(run_bulletin_pipeline())

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
