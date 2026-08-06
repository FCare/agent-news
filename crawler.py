import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import feedparser

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 3000
MAX_ARTICLES_FUNDUS = 300
MAX_ARTICLES_PER_FEED = 20

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
    "korben.info":         ("Korben", "FR"),
    "lesnumeriques.com":   ("Les Numériques", "FR"),
    "nextinpact.com":      ("Next Inpact", "FR"),
    "numerama.com":        ("Numerama", "FR"),
    "01net.com":           ("01net", "FR"),
    "phoronix.com":        ("Phoronix", "US"),
    "arstechnica.com":     ("Ars Technica", "US"),
    "linuxfr.org":         ("LinuxFR", "FR"),
    "elpais.com":          ("El País", "ES"),
    "elmundo.es":          ("El Mundo", "ES"),
    "aljazeera.com":       ("Al Jazeera", "INTL"),
    "reuters.com":         ("Reuters", "INTL"),
}

# RSS/Atom feeds to crawl in addition to fundus
# (url, publisher_name, country)
RSS_FEEDS = [
    # ── Actualités générales FR ──────────────────────────────────────────
    ("https://www.lemonde.fr/rss/une.xml",                               "Le Monde",            "FR"),
    ("https://www.lemonde.fr/economie/rss_full.xml",                     "Le Monde Éco",        "FR"),
    ("https://www.lemonde.fr/sciences/rss_full.xml",                     "Le Monde Sciences",   "FR"),
    ("https://www.lemonde.fr/pixels/rss_full.xml",                       "Le Monde Tech",       "FR"),
    ("https://www.lefigaro.fr/rss/figaro_actualites.xml",                "Le Figaro",           "FR"),
    ("https://www.liberation.fr/arc/outboundfeeds/rss/",                 "Libération",          "FR"),
    ("https://www.francetvinfo.fr/titres.rss",                           "France TV Info",      "FR"),
    ("https://www.europe1.fr/rss.xml",                                   "Europe 1",            "FR"),
    ("https://www.bfmtv.com/rss/info/flux-rss/flux-toutes-les-actualites/", "BFM TV",          "FR"),
    ("https://www.nouvelobs.com/a-la-une/rss.xml",                       "Le Nouvel Obs",       "FR"),
    ("https://www.nouvelobs.com/politique/rss.xml",                      "Le Nouvel Obs Politique", "FR"),
    ("https://www.nouvelobs.com/monde/rss.xml",                          "Le Nouvel Obs Monde", "FR"),
    ("https://www.nouvelobs.com/economie/rss.xml",                       "Le Nouvel Obs Éco",   "FR"),
    ("https://www.nouvelobs.com/culture/rss.xml",                        "Le Nouvel Obs Culture","FR"),
    ("https://www.nouvelobs.com/sciences/rss.xml",                       "Le Nouvel Obs Sciences","FR"),
    # ── Actualités internationales ───────────────────────────────────────
    ("https://www.france24.com/fr/rss",                                  "France 24 FR",        "FR"),
    ("https://www.france24.com/en/rss",                                  "France 24 EN",        "FR"),
    ("https://feeds.bbci.co.uk/news/rss.xml",                            "BBC News",            "UK"),
    ("https://feeds.bbci.co.uk/news/world/rss.xml",                      "BBC World",           "UK"),
    ("https://www.aljazeera.com/xml/rss/all.xml",                        "Al Jazeera",          "INTL"),
    ("https://ecfr.eu/feed/",                                            "ECFR",                "EU"),
    ("https://politico.eu/feed/",                                        "Politico EU",         "EU"),
    # ── Tech & IT FR ─────────────────────────────────────────────────────
    ("https://korben.info/feed",                                         "Korben",              "FR"),
    ("https://next.ink/feed/",                                           "Next.ink",            "FR"),
    ("https://www.numerama.com/feed/",                                   "Numerama",            "FR"),
    # ── 01net ────────────────────────────────────────────────────────────
    ("https://www.01net.com/feed/",                                      "01net",               "FR"),
    ("https://www.01net.com/actualites/feed/",                           "01net Actu",          "FR"),
    ("https://www.01net.com/actualites/applis-logiciels/feed/",          "01net Applis",        "FR"),
    ("https://www.01net.com/actualites/technos/feed/",                   "01net Technos",       "FR"),
    ("https://www.01net.com/actualites/securite/feed/",                  "01net Sécurité",      "FR"),
    ("https://www.01net.com/actualites/buzz-societe/feed/",              "01net Buzz",          "FR"),
    ("https://www.01net.com/actualites/culture-medias/feed/",            "01net Culture",       "FR"),
    ("https://www.01net.com/actualites/politique-droits/feed/",          "01net Politique",     "FR"),
    ("https://www.01net.com/actualites/science-recherche/feed/",         "01net Science",       "FR"),
    ("https://www.01net.com/actualites/smartphones/feed/",               "01net Smartphones",   "FR"),
    ("https://www.01net.com/actualites/tablettes/feed/",                 "01net Tablettes",     "FR"),
    ("https://www.01net.com/actualites/pc-portables/feed/",              "01net PC Portables",  "FR"),
    ("https://www.01net.com/actualites/pc-peripheriques/feed/",          "01net PC Périph.",    "FR"),
    ("https://www.01net.com/actualites/voiture-connectee/feed/",         "01net Auto",          "FR"),
    ("https://www.01net.com/actualites/photo/feed/",                     "01net Photo",         "FR"),
    ("https://www.01net.com/actualites/tv-video/feed/",                  "01net TV",            "FR"),
    ("https://www.01net.com/actualites/audio/feed/",                     "01net Audio",         "FR"),
    ("https://www.01net.com/actualites/objets-connectes/feed/",          "01net IoT",           "FR"),
    ("https://www.01net.com/actualites/maison-connectee/feed/",          "01net Maison",        "FR"),
    ("https://www.01net.com/actualites/jeux-video/feed/",                "01net Jeux",          "FR"),
    ("https://www.01net.com/actualites/produits/feed/",                  "01net Produits",      "FR"),
    ("https://www.01net.com/tests/feed/",                                "01net Tests",         "FR"),
    ("https://www.01net.com/dossiers/feed/",                             "01net Dossiers",      "FR"),
    # ── Les Numériques ───────────────────────────────────────────────────
    ("https://www.lesnumeriques.com/rss.xml",                            "Les Numériques",      "FR"),
    ("https://www.lesnumeriques.com/rss-news.xml",                       "LN Actu",             "FR"),
    ("https://www.lesnumeriques.com/rss-tests.xml",                      "LN Tests",            "FR"),
    ("https://www.lesnumeriques.com/rss-gooddeals.xml",                  "LN Bons plans",       "FR"),
    ("https://www.lesnumeriques.com/informatique/rss.xml",               "LN Informatique",     "FR"),
    ("https://www.lesnumeriques.com/intelligence-artificielle/rss.xml",  "LN IA",               "FR"),
    ("https://www.lesnumeriques.com/audio/rss.xml",                      "LN Audio",            "FR"),
    ("https://www.lesnumeriques.com/voitures-co/rss.xml",               "LN Voitures",         "FR"),
    ("https://www.lesnumeriques.com/gaming/rss.xml",                     "LN Gaming",           "FR"),
    ("https://www.lesnumeriques.com/loisirs/rss.xml",                    "LN Loisirs",          "FR"),
    ("https://www.lesnumeriques.com/electromenager/rss.xml",             "LN Électroménager",   "FR"),
    ("https://www.lesnumeriques.com/photo/rss.xml",                      "LN Photo",            "FR"),
    ("https://www.lesnumeriques.com/pro/rss.xml",                        "LN Pro",              "FR"),
    ("https://www.lesnumeriques.com/sante-sport/rss.xml",               "LN Santé Sport",      "FR"),
    ("https://www.lesnumeriques.com/science-espace/rss.xml",            "LN Sciences",         "FR"),
    ("https://www.lesnumeriques.com/societe-numerique/rss.xml",         "LN Société",          "FR"),
    ("https://www.lesnumeriques.com/mobilite-c17/rss.xml",              "LN Mobilité",         "FR"),
    ("https://www.lesnumeriques.com/tv-video/rss.xml",                   "LN TV Vidéo",         "FR"),
    ("https://www.lesnumeriques.com/velos-trottinettes/rss.xml",        "LN Vélos",            "FR"),
    ("https://www.zdnet.fr/feeds/rss/actualites/",                       "ZDNet FR",            "FR"),
    ("https://www.frandroid.com/feed",                                   "Frandroid",           "FR"),
    ("https://www.silicon.fr/feed",                                      "Silicon.fr",          "FR"),
    ("https://www.journaldugeek.com/feed/",                              "Journal du Geek",     "FR"),
    ("https://www.lebigdata.fr/feed",                                    "LeBigData.fr",        "FR"),
    ("https://www.actuia.com/feed/",                                     "ActuIA",              "FR"),
    ("https://linuxfr.org/news.atom",                                    "LinuxFR",             "FR"),
    ("https://www.undernews.fr/feed",                                    "UnderNews",           "FR"),
    ("https://www.zataz.com/feed/",                                      "ZATAZ",               "FR"),
    ("https://www.developpez.com/index/rss",                             "Developpez.com",      "FR"),
    ("https://www.blogdumoderateur.com/feed/",                           "Blog du Modérateur",  "FR"),
    ("https://stackoverflow.blog/feed/",                                 "Stack Overflow Blog", "US"),
    # ── Tech & IT EN ─────────────────────────────────────────────────────
    ("https://arstechnica.com/feed/",                                    "Ars Technica",        "US"),
    ("https://www.theverge.com/rss/index.xml",                           "The Verge",           "US"),
    ("https://www.phoronix.com/rss.php",                                 "Phoronix",            "US"),
    ("https://theregister.com/headlines.atom",                           "The Register",        "UK"),
    ("https://slashdot.org/rss/slashdot.rss",                            "Slashdot",            "US"),
    ("https://hackaday.com/blog/feed/",                                  "Hackaday",            "US"),
    ("https://www.techradar.com/feeds.xml",                              "TechRadar",           "UK"),
    ("https://feeds.macrumors.com/MacRumors-Front",                      "MacRumors",           "US"),
    ("https://www.technologyreview.com/feed/",                           "MIT Tech Review",     "US"),
    ("https://feed.infoq.com/",                                          "InfoQ",               "US"),
    ("https://spectrum.ieee.org/rss/fulltext",                           "IEEE Spectrum",       "US"),
    ("https://www.9to5google.com/feed/",                                 "9to5Google",          "US"),
    ("https://news.ycombinator.com/rss",                                 "Hacker News",         "US"),
    # ── Cybersécurité ────────────────────────────────────────────────────
    ("https://krebsonsecurity.com/feed/",                                "Krebs on Security",   "US"),
    ("https://www.schneier.com/feed/atom/",                              "Schneier on Security","US"),
    # ── Science & Environnement ──────────────────────────────────────────
    ("https://www.sciencesetavenir.fr/rss.xml",                          "Sciences et Avenir",  "FR"),
    ("https://trustmyscience.com/feed/",                                 "Trust My Science",    "FR"),
    ("https://reporterre.net/spip.php?page=backend",                     "Reporterre",          "FR"),
    ("https://www.actu-environnement.com/rss/",                          "Actu-Environnement",  "FR"),
    ("https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",   "BBC Science",         "UK"),
    ("https://www.newscientist.com/feed/home/",                          "New Scientist",       "UK"),
    ("https://www.sciencedaily.com/rss/all.xml",                         "ScienceDaily",        "US"),
    ("https://phys.org/rss-feed/",                                       "Phys.org",            "INTL"),
    ("https://rss.arxiv.org/rss/cs.AI",                                  "arXiv IA",            "INTL"),
    # ── Économie ─────────────────────────────────────────────────────────
    ("https://www.bruegel.org/rss.xml",                                  "Bruegel",             "EU"),
    # ── BD & Comics ──────────────────────────────────────────────────────
    ("https://www.actuabd.com/?page=backend",                            "ActuaBD",             "FR"),
    ("https://www.cbr.com/feed/",                                        "CBR Comics",          "US"),
    # ── Cinéma & Musique (sorties) ───────────────────────────────────────
    ("https://www.allocine.fr/rss/news.xml",                             "AlloCiné",            "FR"),
]

REGIONS = ["us", "uk", "fr", "es", "nl"]


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


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _entry_body(entry) -> str:
    """Extract best available body text from a feedparser entry."""
    # content[] is the richest field (content:encoded or atom:content)
    for c in entry.get("content", []):
        v = (c.get("value") or "").strip()
        if v:
            return _strip_html(v)
    for field in ("summary", "description"):
        v = (entry.get(field) or "").strip()
        if v:
            return _strip_html(v)
    return ""


def _entry_date(entry) -> datetime | None:
    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def _fetch_rss(url: str, publisher: str, country: str) -> list[RawArticle]:
    """Fetch and parse an RSS/Atom feed using feedparser (handles malformed XML)."""
    try:
        feed = feedparser.parse(
            url,
            agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            request_headers={"Accept-Encoding": "gzip, deflate"},
        )
    except Exception as e:
        logger.warning(f"RSS fetch failed ({publisher}): {e}")
        return []

    if feed.get("bozo") and not feed.entries:
        exc = feed.get("bozo_exception", "")
        logger.warning(f"RSS parse error ({publisher}): {exc}")
        return []

    articles = []
    for entry in feed.entries[:MAX_ARTICLES_PER_FEED]:
        try:
            title = _strip_html(entry.get("title", "").strip())
            article_url = entry.get("link", "").strip()
            body = _entry_body(entry)
            published_at = _entry_date(entry)

            if not title or len(body) < 50:
                continue

            articles.append(RawArticle(
                url=article_url,
                title=title,
                body=body[:MAX_BODY_CHARS],
                publisher=publisher,
                country=country,
                published_at=published_at,
            ))
        except Exception as e:
            logger.debug(f"RSS entry skip ({publisher}): {e}")

    return articles


def _crawl_rss_feeds() -> list[RawArticle]:
    articles = []
    for feed_url, publisher, country in RSS_FEEDS:
        feed_articles = _fetch_rss(feed_url, publisher, country)
        articles.extend(feed_articles)
        logger.info(f"  RSS {publisher}: {len(feed_articles)} articles")
    return articles


def _silence_fundus_loggers() -> None:
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("fundus"):
            logging.getLogger(name).setLevel(logging.ERROR)


def _crawl_fundus() -> list[RawArticle]:
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
        return []

    articles = []
    n_seen = n_skip_body = n_skip_short = n_error = 0

    try:
        crawler = Crawler(*collections)
        for raw in crawler.crawl(max_articles=MAX_ARTICLES_FUNDUS, timeout=90):
            n_seen += 1
            try:
                try:
                    url = str(raw.html.responded_url)
                except AttributeError:
                    url = str(raw.html.requested_url)

                title = (raw.title or "").strip()
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
                    url=url, title=title, body=body[:MAX_BODY_CHARS],
                    publisher=publisher, country=country,
                    published_at=raw.publishing_date,
                ))
            except Exception as e:
                n_error += 1
                if n_error <= 3:
                    logger.warning(f"Article ignoré ({type(e).__name__}): {e}")
    except Exception as e:
        logger.error(f"Erreur crawl fundus: {e}")

    logger.info(
        f"  fundus: {len(articles)}/{n_seen} articles valides "
        f"({n_skip_body} sans body, {n_skip_short} trop courts, {n_error} erreurs)"
    )
    return articles


def _crawl_sync() -> list[RawArticle]:
    logger.info(f"Crawl RSS ({len(RSS_FEEDS)} flux)...")

    rss_articles = _crawl_rss_feeds()

    logger.info(f"Crawl fundus ({len(REGIONS)} régions, max {MAX_ARTICLES_FUNDUS})...")
    fundus_articles = _crawl_fundus()

    # Deduplicate by URL
    seen_urls: set[str] = set()
    all_articles = []
    for a in rss_articles + fundus_articles:
        if a.url and a.url not in seen_urls:
            seen_urls.add(a.url)
            all_articles.append(a)

    logger.info(f"Crawl terminé: {len(all_articles)} articles uniques "
                f"(RSS: {len(rss_articles)}, fundus: {len(fundus_articles)})")
    return all_articles


async def crawl_articles() -> list[RawArticle]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _crawl_sync)
