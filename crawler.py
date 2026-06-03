import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 3000
MAX_ARTICLES = 300

# Domain → (publisher name, country)
DOMAIN_MAP = {
    "apnews.com":          ("AP News", "US"),
    "washingtonpost.com":  ("Washington Post", "US"),
    "theverge.com":        ("The Verge", "US"),
    "wired.com":           ("Wired", "US"),
    "npr.org":             ("NPR", "US"),
    "theatlantic.com":     ("The Atlantic", "US"),
    "arstechnica.com":     ("Ars Technica", "US"),
    "bloomberg.com":       ("Bloomberg", "US"),
    "techcrunch.com":      ("TechCrunch", "US"),
    "politico.com":        ("Politico", "US"),
    "theguardian.com":     ("The Guardian", "UK"),
    "bbc.com":             ("BBC", "UK"),
    "bbc.co.uk":           ("BBC", "UK"),
    "theeconomist.com":    ("The Economist", "UK"),
    "independent.co.uk":   ("The Independent", "UK"),
    "spiegel.de":          ("Der Spiegel", "DE"),
    "zeit.de":             ("Die Zeit", "DE"),
    "heise.de":            ("Heise", "DE"),
    "faz.net":             ("FAZ", "DE"),
    "handelsblatt.com":    ("Handelsblatt", "DE"),
    "sueddeutsche.de":     ("Süddeutsche Zeitung", "DE"),
    "lemonde.fr":          ("Le Monde", "FR"),
    "lefigaro.fr":         ("Le Figaro", "FR"),
    "liberation.fr":       ("Libération", "FR"),
    "leparisien.fr":       ("Le Parisien", "FR"),
    "elpais.com":          ("El País", "ES"),
    "elmundo.es":          ("El Mundo", "ES"),
    "aljazeera.com":       ("Al Jazeera", "INTL"),
    "reuters.com":         ("Reuters", "INTL"),
}

REGIONS = ["us", "uk", "de", "fr", "es", "at", "ch", "nl"]


@dataclass
class RawArticle:
    url: str
    title: str
    body: str
    publisher: str
    country: str
    published_at: datetime | None


def _get_publisher_info(url: str) -> tuple[str, str]:
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
        for key, value in DOMAIN_MAP.items():
            if key in domain:
                return value
        parts = domain.split(".")
        if len(parts) >= 2:
            tld = parts[-1]
            country_map = {"de": "DE", "fr": "FR", "es": "ES", "uk": "UK",
                           "co": "UK", "at": "AT", "ch": "CH", "nl": "NL"}
            country = country_map.get(tld, "US" if tld == "com" else "?")
            return (domain, country)
    except Exception:
        pass
    return ("Unknown", "?")


def _silence_fundus_loggers() -> None:
    """Mute fundus sub-loggers after import (they register themselves on import)."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("fundus"):
            logging.getLogger(name).setLevel(logging.ERROR)


def _crawl_sync() -> list[RawArticle]:
    try:
        from fundus import Crawler, PublisherCollection
    except ImportError:
        logger.error("fundus non installé")
        return []

    _silence_fundus_loggers()

    collections = []
    for region in REGIONS:
        coll = getattr(PublisherCollection, region, None)
        if coll is not None:
            collections.append(coll)

    if not collections:
        logger.error("Aucune collection fundus disponible")
        return []

    logger.info(f"Crawl: {len(collections)} régions, max {MAX_ARTICLES} articles")
    articles = []
    n_seen = n_skip_body = n_skip_short = n_error = 0

    try:
        crawler = Crawler(*collections)
        for raw in crawler.crawl(max_articles=MAX_ARTICLES, timeout=90):
            n_seen += 1
            try:
                # URL
                try:
                    url = str(raw.html.responded_url)
                except AttributeError:
                    url = str(raw.html.requested_url)

                # Title
                title = (raw.title or "").strip()

                # Body — try .text, fall back to str()
                body = ""
                if raw.body is not None:
                    try:
                        body = (raw.body.text or "").strip()
                    except AttributeError:
                        body = str(raw.body).strip()

                if not body:
                    n_skip_body += 1
                    continue
                if len(body) < 100:
                    n_skip_short += 1
                    continue
                if not title:
                    continue

                publisher, country = _get_publisher_info(url)
                articles.append(RawArticle(
                    url=url,
                    title=title,
                    body=body[:MAX_BODY_CHARS],
                    publisher=publisher,
                    country=country,
                    published_at=raw.publishing_date,
                ))
            except Exception as e:
                n_error += 1
                if n_error <= 3:
                    logger.warning(f"Article ignoré ({type(e).__name__}): {e}")
    except Exception as e:
        logger.error(f"Erreur crawl fundus: {e}")

    logger.info(
        f"Crawl terminé: {n_seen} reçus → {len(articles)} valides "
        f"({n_skip_body} sans body, {n_skip_short} trop courts, {n_error} erreurs)"
    )
    return articles


async def crawl_articles() -> list[RawArticle]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _crawl_sync)
