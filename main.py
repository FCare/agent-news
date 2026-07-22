import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import openai
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from nexus_client import NexusClient

import bulletin_gen
import storage
import subject_consolidation
import vector_store
import wiki_api
import wiki_build
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
_subscribed_sessions: set[str] = set()
_is_generating = False
_search_client = SearchClient()

app = FastAPI(title="agent-news")
app.include_router(wiki_api.router)
if wiki_build.WIKI_SITE_DIR.is_dir():
    # Monté seulement si le site a déjà été généré au moins une fois (sinon
    # StaticFiles refuse de démarrer, faisant planter tout l'agent au boot) —
    # voir wiki_build.run(), appelé en fin de run_bulletin_pipeline ou
    # manuellement via `python3 wiki_build.py`.
    app.mount("/wiki", StaticFiles(directory=wiki_build.WIKI_SITE_DIR, html=True), name="wiki")
else:
    logger.warning(
        f"Wiki pas encore généré ({wiki_build.WIKI_SITE_DIR} introuvable) — "
        "/wiki indisponible tant que wiki_build.run() n'a pas tourné au moins une fois"
    )

@app.get("/")
async def root():
    # Le site généré vit sous /wiki (voir le mount StaticFiles ci-dessus) — sans
    # cette redirection, news.caronboulme.fr/ (ce que tape naturellement un
    # utilisateur) renvoie 404, seul /wiki/ répond.
    return RedirectResponse(url="/wiki/")

@app.post("/pipeline/run")
async def trigger_pipeline():
    if _is_generating:
        return {"status": "already_running"}
    asyncio.create_task(run_bulletin_pipeline())
    return {"status": "started"}

@app.get("/bulletin")
async def get_bulletin(date: str | None = None):
    row = await storage.get_bulletin_by_date(date) if date else await storage.get_latest_bulletin()
    if not row:
        return {"error": "Aucun bulletin disponible"}
    b = row["bulletin_json"]
    return {
        "date": row["date"],
        "headline": b.get("headline", ""),
        "flash": b.get("flash", ""),
        "categories": b.get("categories", {}),
        "n_articles": row.get("n_articles"),
        "n_topics": row.get("n_topics"),
    }

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
        await loop.run_in_executor(None, vector_store.upsert_publishers, rows)
        # Déchargé dès maintenant plutôt qu'en fin de pipeline seulement : c'est la
        # seule étape qui a vraiment besoin du modèle d'embedding chargé en continu
        # (upsert de ~1800 articles) — les ~35-40 min d'enrichissement LLM qui
        # suivent n'y touchent pas, sauf la consolidation (subject_consolidation.py,
        # requêtes courtes) qui le rechargera à la volée si besoin (lazy, voir
        # unload_model). Libère la VRAM pendant cette longue fenêtre pour d'autres
        # services GPU partagés (voir run_pipeline_with_gpu.sh).
        await loop.run_in_executor(None, vector_store.unload_model)

        # 3. Generate bulletin
        logger.info("[2/4] Clustering et identification des sujets...")
        llm = _get_llm_client()
        bulletin = await bulletin_gen.generate_bulletin(articles, _search_client, llm, LLM_MODEL)
        if not bulletin:
            logger.error("[2/4] Bulletin vide, pipeline annulé")
            return

        # 3bis. Consolidation des sujets (voir subject_consolidation.py) — AVANT la
        # sauvegarde du bulletin brut, mais un échec ici ne doit jamais empêcher cette
        # sauvegarde (le bulletin du jour reste la donnée de référence, la vue
        # consolidée est dérivée et peut être rejouée plus tard si besoin, voir
        # backfill_subjects.py).
        logger.info("[2b/4] Consolidation des sujets...")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            await subject_consolidation.consolidate_bulletin(bulletin, today, llm, LLM_MODEL)
        except Exception as e:
            logger.error(f"[2b/4] Consolidation des sujets échouée: {e}")

        # 3ter. Dédoublonnage post-consolidation : le clustering (TF-IDF/DBSCAN)
        # crée parfois plusieurs topics quasi-identiques pour le même événement réel
        # dans un même bulletin (constaté à grande échelle sur données réelles,
        # ~15-20/jour) — consolidate_bulletin() vient de les fusionner en UN SEUL
        # sujet/édition, mais bulletin["categories"] garde encore les topics perdants
        # de cette fusion (aucune édition ne les référence plus). Sans ce nettoyage,
        # category_summaries peut leur emprunter des détails (mauvais pays/continent
        # p.ex.) qui ne correspondent plus à rien d'affiché sur le wiki — même
        # logique que backfill_recategorize_summaries.py, appliquée ici en direct
        # pour ne plus dépendre d'un backfill après coup.
        try:
            members_today = await storage.get_subjects_by_date(today)
            current_category_by_title = {m["edition_title"]: m["category"] for m in members_today}
            deduped_categories: dict[str, list[dict]] = {}
            n_dropped = 0
            for old_cat, stories in bulletin.get("categories", {}).items():
                for story in stories:
                    current_cat = current_category_by_title.get(story["title"])
                    if current_cat is None:
                        n_dropped += 1
                        continue
                    deduped_categories.setdefault(current_cat, []).append(story)
            if n_dropped:
                logger.info(f"[2b/4] {n_dropped} topic(s) doublon(s) retiré(s) après consolidation")
                bulletin["categories"] = deduped_categories
                bulletin["category_summaries"] = await bulletin_gen._generate_category_summaries(
                    deduped_categories, llm, LLM_MODEL
                )
        except Exception as e:
            logger.error(f"[2b/4] Dédoublonnage post-consolidation échoué: {e}")

        # 4. Save (SQLite + ChromaDB) — only if bulletin is valid
        logger.info("[3/4] Sauvegarde du bulletin...")
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
        await loop.run_in_executor(None, vector_store.purge_old_articles)
        await loop.run_in_executor(None, vector_store.purge_old_topics)
        # Rétention distincte et bien plus longue (voir storage.SUBJECT_RETENTION_DAYS) :
        # un sujet actif survit tant qu'il est réactualisé, indépendamment des 90 jours
        # de purge_old_data sur les bulletins bruts.
        await storage.purge_old_subjects()
        await loop.run_in_executor(None, vector_store.purge_old_subjects)

        # 6. Régénération du wiki (voir wiki_build.py) — un échec ici ne doit jamais
        # faire échouer le pipeline (bulletin déjà sauvegardé indépendamment) ; le
        # wiki reste simplement à sa dernière version générée avec succès.
        logger.info("Régénération du wiki...")
        try:
            wiki_result = await wiki_build.run()
            logger.info(f"Wiki régénéré: {wiki_result}")
        except Exception as e:
            logger.error(f"Régénération du wiki échouée: {e}")

        logger.info(f"=== Pipeline terminé: {n_topics} sujets pour le {today} ===")

    except Exception as e:
        logger.exception(f"Erreur pipeline: {e}")
    finally:
        _is_generating = False
        # Le modèle d'embedding n'est utile que le temps du pipeline (crawl +
        # sauvegarde du bulletin) — le décharger libère la VRAM jusqu'au prochain
        # cycle (rechargement automatique et transparent au prochain accès).
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.unload_model)

# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

def _format_flash(bulletin_row: dict, category: str | None = None) -> str:
    b = bulletin_row["bulletin_json"]
    date = bulletin_row["date"]
    categories = b.get("categories", {})
    parts = [f"[{date}] {b.get('headline', '')}", ""]
    for cat, stories in categories.items():
        if category and category.lower() not in cat.lower():
            continue
        parts.append(f"── {cat.upper()} ──")
        for s in stories:
            parts.append(f"• {s['title']}")
        parts.append("")
    return "\n".join(parts)



def _format_category(bulletin_row: dict, category: str) -> str | None:
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
        return None

    stories = categories[matched]
    parts = [f"[{date}] {matched.upper()}", ""]
    for story in stories:
        parts.append(f"• {story['title']}")
    parts.append("")
    return "\n".join(parts)


def _format_publisher(articles: list[dict], publisher: str) -> str:
    if not articles:
        return f"Aucun article récent de '{publisher}'."
    parts = [f"ARTICLES RÉCENTS — {publisher.upper()}", ""]
    for a in articles:
        parts.append(f"• {a['title']}")
    return "\n".join(parts)


# En dessous de ce score de similarité cosinus, on considère qu'aucun sujet du
# bulletin ne correspond réellement à la question — mieux vaut laisser la main
# à answer_question (recherche sémantique + LLM) que de renvoyer un sujet
# vaguement proche mais hors-sujet.
_DEEP_DIVE_MIN_SCORE = 0.35


def _format_deep_dive(bulletin_row: dict, topic_query: str) -> str:
    b = bulletin_row["bulletin_json"]
    date = bulletin_row["date"]

    hits = [h for h in vector_store.search_topics(topic_query, n_results=5)
            if h["metadata"].get("date") == date]
    if not hits or hits[0]["score"] < _DEEP_DIVE_MIN_SCORE:
        return None

    best_title = hits[0]["metadata"]["title"]
    best = next(
        (story for stories in b.get("categories", {}).values() for story in stories
         if story["title"] == best_title),
        None,
    )
    if not best:
        return None

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
        # sources = [{"name": ..., "url": ...}] (voir bulletin_gen.py::_generate_deep_dive)
        # depuis la migration ; les anciens bulletins peuvent encore contenir de
        # simples chaînes (noms de médias sans URL) — les deux formes sont acceptées.
        names = [s["name"] if isinstance(s, dict) else s for s in best["sources"]]
        parts.append(f"Sources: {', '.join(names)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MQTT handlers
# ---------------------------------------------------------------------------

async def on_user_connected(topic: str, payload) -> None:
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    password = payload.get("password")
    session_id = payload.get("session_id")
    private_topics = payload.get("private_topics", [])

    if not username or not password or not session_id:
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

    request_topic = f"users/{username}/{session_id}/news/request"
    result_topic  = f"users/{username}/{session_id}/news/result"

    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    publishers = await storage.get_distinct_publishers()

    await nexus.publish(agent_topics_topic, [{
        "agent": AGENT_NAME,
        "topics": [
            {
                "topic": request_topic,
                "description": (
                    "Source EXCLUSIVE d'actualités en temps réel. OBLIGATOIRE pour toute "
                    "question sur l'actualité, les nouvelles, les événements récents. "
                    "news_fetch → sans 'query' : flash du jour (titres) ; avec 'query' "
                    "(sujet/entité SANS référence temporelle, la date est automatique) : "
                    "répond sur un sujet/pays/personne/événement précis. 'category' optionnel "
                    f"(seulement si pas de query) parmi : {', '.join(bulletin_gen.CATEGORIES)} "
                    "— défaut 'France' si le domaine du flash n'est pas précisé. Ex: "
                    "'l\\'actu tech' → category='Informatique & IA' ; 'quoi de neuf avec Trump ?' "
                    "→ query='trump'. "
                    "source → UNIQUEMENT si un média est cité explicitement par son nom (ex: "
                    "'les nouvelles de Korben') ; 'publisher' = nom exact tel qu\\'il apparaît "
                    "dans la liste. "
                    "Tous types acceptent 'date' optionnel (YYYY-MM-DD) pour un bulletin passé."
                ),
                "access": "write",
                "response_topic": result_topic,
                "format": {
                    "type": "news_fetch | source",
                    "query": "(optionnel) sujet/entité pour news_fetch, ex: 'ukraine', 'trump'",
                    "category": "(optionnel, news_fetch sans query) voir description pour la liste",
                    "publisher": "(source uniquement) " + str(publishers),
                    "date": "(optionnel) YYYY-MM-DD, défaut=aujourd'hui",
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
                },
            },
        ],
    }])
    logger.info(f"[{username}/{session_id}] Topics news déclarés")

    if session_id in _subscribed_sessions:
        return
    _subscribed_sessions.add(session_id)

    async def on_news_request(t: str, p) -> None:
        if not isinstance(p, dict):
            logger.warning(f"[{username}] Payload news invalide (non-dict): {p!r}")
            await nexus.publish(result_topic, {"error": "Invalid request payload: expected a JSON object"})
            return

        reply_to = p.get("reply_to", result_topic)
        req_type = p.get("type", "news_fetch").lower()
        logger.info(f"[{username}] Requête news: {p}")

        try:
            await _handle_news_request(p, req_type, reply_to)
        except Exception as e:
            logger.exception(f"[{username}] Erreur traitement requête news: {e}")
            await nexus.publish(reply_to, {
                "type": req_type,
                "error": f"Internal error while processing the request: {e}",
            })

    async def _handle_news_request(p: dict, req_type: str, reply_to: str) -> None:
        date_param = p.get("date", "").strip()
        if date_param:
            bulletin_row = await storage.get_bulletin_by_date(date_param)
        else:
            bulletin_row = await storage.get_latest_bulletin()

        content = ""

        if req_type in ("news_fetch", "flash", "question", "deep_dive"):
            query = (p.get("query") or p.get("topic") or "").strip()

            if query:
                # Subject/entity query → semantic search + LLM answer
                if bulletin_row:
                    result = _format_deep_dive(bulletin_row, query)
                    if result is not None:
                        content = result
                    else:
                        llm = _get_llm_client()
                        content = await bulletin_gen.answer_question(
                            query, bulletin_row["bulletin_json"], _search_client, llm, LLM_MODEL,
                            ref_date=date_param or bulletin_row["date"],
                        )
                else:
                    content = "Aucun bulletin disponible."
            else:
                # No query → flash (full or filtered by category)
                if bulletin_row:
                    category_filter = p.get("category", "").strip()
                    if category_filter:
                        result = _format_category(bulletin_row, category_filter)
                        if result is not None:
                            content = result
                        else:
                            llm = _get_llm_client()
                            content = await bulletin_gen.answer_question(
                                category_filter, bulletin_row["bulletin_json"], _search_client, llm, LLM_MODEL,
                                ref_date=date_param or bulletin_row["date"],
                            )
                    else:
                        content = _format_flash(bulletin_row)
                else:
                    content = "Bulletin non disponible."

        elif req_type == "source":
            publisher = p.get("publisher", "").strip()
            if not publisher:
                content = "Précise un site ou média avec le champ 'publisher'."
            else:
                articles = await storage.get_articles_by_publisher(publisher)
                if articles:
                    content = _format_publisher(articles, publisher)
                else:
                    loop = asyncio.get_event_loop()
                    similar = await loop.run_in_executor(
                        None, vector_store.find_similar_publishers, publisher
                    )
                    if similar:
                        content = f"Aucun article trouvé pour '{publisher}'. Sources proches disponibles : {', '.join(similar)}"
                    else:
                        content = f"Aucun article trouvé pour '{publisher}'."

        else:
            content = f"Type inconnu: {req_type}. Disponibles: news_fetch, source."

        logger.info(f"[{username}] Réponse ({len(content)} chars):\n{content}")
        await nexus.publish(reply_to, {
            "type": req_type,
            "content": content,
            "bulletin_date": bulletin_row["date"] if bulletin_row else "",
        })

    nexus.subscribe(request_topic, on_news_request)
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à {request_topic}")


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

    # Initial run — skip if a bulletin already exists for today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = await storage.get_bulletin_by_date(today)
    if existing:
        logger.info(f"Bulletin du {today} déjà présent, pipeline ignoré au démarrage.")
        articles = await storage.get_recent_articles()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.seed_publishers_if_empty, articles)
    else:
        logger.info("Aucun bulletin pour aujourd'hui, lancement du pipeline...")
        asyncio.create_task(run_bulletin_pipeline())

    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
