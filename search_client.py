import asyncio
import logging
import os

logger = logging.getLogger(__name__)

SERVICE_REQUEST_TOPIC = "service/search/request"

SEARCH_TIMEOUT     = float(os.environ.get("SEARCH_TIMEOUT", "45.0"))
SEARCH_CONCURRENCY = int(os.environ.get("SEARCH_CONCURRENCY", "8"))


class SearchClient:
    """
    Envoie des requêtes de recherche à agent-web-search via nexus.request().
    Chaque requête obtient son propre topic reply/{uuid}.
    Un semaphore global limite le nombre de recherches simultanées pour ne pas
    saturer les moteurs.

    La réponse contient un volet 'background' (encyclopédie et web) et un volet
    'recent' (presse), déjà concaténés dans 'report'. Le volet presse recoupe
    en partie les flux RSS déjà crawlés, mais le volet fond apporte ce que le
    crawl n'a pas : institutionnels, agences, presse étrangère.
    """

    def __init__(self) -> None:
        self._nexus = None
        self._sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def setup(self, nexus, service_username: str, api_key: str) -> None:
        self._nexus = nexus
        logger.info(f"SearchClient prêt — topic: {SERVICE_REQUEST_TOPIC}")

    async def search(self, query: str, detail_level: int = 3) -> dict:
        if self._nexus is None:
            return {"report": "", "sources": [], "topic": query[:60]}

        async with self._sem:
            result = await self._nexus.request(
                SERVICE_REQUEST_TOPIC,
                {"query": query, "n_results": 10, "detail_level": detail_level},
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
