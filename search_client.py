import asyncio
import logging
import os

logger = logging.getLogger(__name__)

SEARCH_INIT_DELAY = float(os.environ.get("SEARCH_INIT_DELAY", "3.0"))
SEARCH_TIMEOUT = float(os.environ.get("SEARCH_TIMEOUT", "45.0"))
SEARCH_INTER_DELAY = float(os.environ.get("SEARCH_INTER_DELAY", "0.3"))


class SearchClient:
    """
    Sends news search requests to agent-web-search via MQTT.
    Requests are sequential (one at a time) because the search result topic
    carries no correlation ID.
    """

    def __init__(self) -> None:
        self._nexus = None
        self._request_topic: str = ""
        self._result_topic: str = ""
        self._lock = asyncio.Lock()
        self._pending: asyncio.Future | None = None
        self._ready = False

    async def setup(self, nexus, service_username: str, api_key: str) -> None:
        self._nexus = nexus
        self._request_topic = f"users/{service_username}/search/request"
        self._result_topic = f"users/{service_username}/search/result"

        nexus.subscribe(self._result_topic, self._on_result)

        # Trigger the search agent to set up topics for this service user.
        # The search agent only checks username/password are non-empty; actual
        # auth is handled at MQTT broker level via the API key.
        await nexus.publish("common/user_connected", {
            "username": service_username,
            "password": api_key,
            "private_topics": [{
                "topics": [{
                    "topic": f"users/{service_username}/agent_topics"
                }]
            }],
        })
        logger.info(f"user_connected publié pour {service_username}")
        await asyncio.sleep(SEARCH_INIT_DELAY)
        self._ready = True

    async def _on_result(self, topic: str, payload: dict) -> None:
        if self._pending and not self._pending.done():
            self._pending.set_result(payload)

    async def search(self, query: str, categories: str = "news",
                     detail_level: int = 3) -> dict:
        if not self._ready:
            return {"report": "", "sources": [], "topic": query[:60]}

        async with self._lock:
            loop = asyncio.get_event_loop()
            self._pending = loop.create_future()

            await self._nexus.publish(self._request_topic, {
                "query": query,
                "categories": categories,
                "n_results": 10,
                "detail_level": detail_level,
            })

            try:
                result = await asyncio.wait_for(
                    asyncio.shield(self._pending), timeout=SEARCH_TIMEOUT
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Timeout recherche: {query[:60]!r}")
                return {"report": "", "sources": [], "topic": query[:60]}
            finally:
                self._pending = None
                await asyncio.sleep(SEARCH_INTER_DELAY)

    async def search_many(self, queries: list[str],
                          categories: str = "news") -> list[dict]:
        results = []
        for i, q in enumerate(queries):
            logger.debug(f"  Recherche {i+1}/{len(queries)}: {q[:60]!r}")
            r = await self.search(q, categories)
            results.append(r)
        return results
