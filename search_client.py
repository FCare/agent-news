import asyncio
import logging
import os

logger = logging.getLogger(__name__)

SERVICE_REQUEST_TOPIC = "service/search/request"

SEARCH_TIMEOUT     = float(os.environ.get("SEARCH_TIMEOUT", "45.0"))
SEARCH_CONCURRENCY = int(os.environ.get("SEARCH_CONCURRENCY", "8"))

# Budget de contexte demandé à agent-web-search, qu'il répartit entre son volet
# fond et son volet actualité en coupant sur une frontière de source. Le fixer
# ici plutôt que de tronquer la réponse à l'arrivée : une coupe en aval sur le
# rapport concaténé supprimerait le volet actualité, qui vient en second.
# 2400 par recherche × SEARCH_QUERIES_PER_TOPIC (5) = 12 000 caractères de
# contexte de recherche par sujet, à côté des 24 000 d'extraits d'articles.
SEARCH_MAX_CHARS = int(os.environ.get("SEARCH_MAX_CHARS", "2400"))


class SearchClient:
    """
    Envoie des requêtes de recherche à agent-web-search via nexus.request().
    Chaque requête obtient son propre topic reply/{uuid}.
    Un semaphore global limite le nombre de recherches simultanées pour ne pas
    saturer les moteurs.

    La réponse contient un volet 'background' (encyclopédie et web) et un volet
    'recent' (presse), déjà concaténés dans 'report'. Ce sont des extraits
    bruts : c'est le deep dive du bulletin qui les met en forme, une synthèse
    intermédiaire ne ferait que lui retirer des dates et des chiffres.
    """

    def __init__(self) -> None:
        self._nexus = None
        self._sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def setup(self, nexus, service_username: str, api_key: str) -> None:
        self._nexus = nexus
        logger.info(f"SearchClient prêt — topic: {SERVICE_REQUEST_TOPIC}")

    async def search(self, query: str, max_chars: int = SEARCH_MAX_CHARS) -> dict:
        if self._nexus is None:
            return {"report": "", "sources": [], "topic": query[:60]}

        async with self._sem:
            result = await self._nexus.request(
                SERVICE_REQUEST_TOPIC,
                {"query": query, "n_results": 10, "max_chars": max_chars},
                timeout=SEARCH_TIMEOUT,
            )
        if result is None:
            logger.warning(f"Timeout recherche: {query[:60]!r}")
            return {"report": "", "sources": [], "topic": query[:60]}
        return result

    async def search_many(self, queries: list[str]) -> list[dict]:
        async def _one(query: str, i: int) -> dict:
            logger.info(f"    recherche [{i+1}/{len(queries)}] → {query[:70]!r}")
            r = await self.search(query)
            logger.info(f"    recherche [{i+1}/{len(queries)}] ← {len(r.get('report') or '')} chars")
            return r

        return list(await asyncio.gather(*[_one(q, i) for i, q in enumerate(queries)]))
