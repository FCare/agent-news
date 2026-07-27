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
            logger.error("[3/4] Bulletin incomplet (flash ou headline manquant) — sauvegarde ignorée pour ne pas écraser un bulletin valide")
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
        # Modèle d'embedding gardé chargé en permanence (voir warmup au démarrage,
        # main()) plutôt que déchargé ici entre deux cycles : évite le cold-start de
        # ~20-30s (chargement + vérifications HuggingFace Hub) subi par wiki-agent
        # sur la première recherche après un cycle. Marge VRAM suffisante depuis le
        # plafonnement de voxcpm2 (~10 Go au lieu de ~14,3 Go, voir docker-compose.yml
        # de nanovllm-voxcpm).

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Description exposée sur common/wiki_registry/news (retain=True) pour que wiki-agent sache
# quoi/où chercher — voir agents/wiki-agent. agent-news ne répond plus lui-même aux
# requêtes MQTT de Joshua/Panoramix (uniquement construire/maintenir le wiki) ;
# wiki-agent est désormais le seul point de contact, en consultant ce wiki publié
# plutôt que la base de données interne (search_topics, non consolidée — voir
# l'historique de conversation pour le constat qui a motivé ce changement).
WIKI_REGISTRY_DESCRIPTION = (
    "Sujets d'actualité consolidés au fil du temps (résumé qui évolue à chaque "
    "nouvelle édition), organisés par catégorie, date et source."
)


async def main() -> None:
    await storage.init_db()

    global _search_client
    nexus = NexusClient.from_api_key(VK_URL, MQTT_HOST, SERVICE_USERNAME, SERVICE_API_KEY, MQTT_PORT)

    await _search_client.setup(nexus, SERVICE_USERNAME, SERVICE_API_KEY)
    # Toujours nécessaire même sans abonnement applicatif : _search_client (recherche
    # web SearXNG pendant l'enrichissement des bulletins) utilise nexus.request(), qui
    # a besoin de la connexion persistante pour s'abonner à son topic de réponse.
    nexus.start_listening()

    # Avant l'enregistrement du wiki (ci-dessous) : autant que le modèle soit déjà en
    # cours de chargement avant que wiki-agent puisse recevoir l'annonce et tenter une
    # recherche.
    await asyncio.get_event_loop().run_in_executor(None, vector_store.warmup_model)

    await nexus.publish(f"common/wiki_registry/{AGENT_NAME}", {
        "agent": AGENT_NAME,
        # Adresse interne du réseau Docker "ansible" (pas le domaine public
        # news.caronboulme.fr) : ce dernier passe par Traefik + middleware vk-hybrid,
        # qui rejette (401) les appels d'un service interne sans cookie/API key VK —
        # inutile ici, wiki-agent et agent-news sont sur le même réseau de confiance.
        "base_url": "http://agent-news:8080",
        "search_path": "/api/wiki/search",
        "description": WIKI_REGISTRY_DESCRIPTION,
    }, retain=True)
    logger.info(f"Enregistrement common/wiki_registry/{AGENT_NAME} publié (retain)")

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
