import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path("/data/news.db")
BULLETIN_RETENTION_DAYS = 90
ARTICLES_TTL_HOURS = 6


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                publisher TEXT,
                country TEXT,
                published_at TEXT,
                crawled_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bulletins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                generated_at TEXT NOT NULL,
                flash TEXT NOT NULL,
                headline TEXT NOT NULL,
                bulletin_json TEXT NOT NULL,
                n_articles INTEGER DEFAULT 0,
                n_topics INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_articles_crawled ON articles(crawled_at);
            CREATE INDEX IF NOT EXISTS idx_bulletins_date ON bulletins(date);
        """)
        await db.commit()
    logger.info("DB initialisée")


async def save_articles(articles: list[dict]) -> int:
    if not articles:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    saved = 0
    async with aiosqlite.connect(DB_PATH) as db:
        for a in articles:
            try:
                await db.execute(
                    "INSERT OR IGNORE INTO articles (url, title, body, publisher, country, published_at, crawled_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (a["url"], a["title"], a.get("body", ""), a.get("publisher", ""),
                     a.get("country", ""), a.get("published_at"), now),
                )
                saved += 1
            except Exception as e:
                logger.debug(f"Article skip: {e}")
        await db.commit()
    return saved


async def get_recent_articles(hours: int = ARTICLES_TTL_HOURS) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM articles WHERE crawled_at > ? ORDER BY crawled_at DESC",
            (cutoff,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def save_bulletin(date: str, flash: str, headline: str, bulletin_json: dict,
                        n_articles: int = 0, n_topics: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO bulletins (date, generated_at, flash, headline, bulletin_json, n_articles, n_topics) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (date, now, flash, headline, json.dumps(bulletin_json, ensure_ascii=False),
             n_articles, n_topics),
        )
        await db.commit()
    logger.info(f"Bulletin {date} sauvegardé ({n_topics} sujets, {n_articles} articles)")


async def get_latest_bulletin() -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM bulletins ORDER BY date DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return _parse_bulletin_row(dict(row))


async def get_bulletin_by_date(date: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM bulletins WHERE date = ?", (date,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return _parse_bulletin_row(dict(row))


async def get_history_list(limit: int = 90) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT date, generated_at, headline, n_articles, n_topics "
            "FROM bulletins ORDER BY date DESC LIMIT ?",
            (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_articles_by_publisher(publisher: str, hours: int = ARTICLES_TTL_HOURS) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT title, url, published_at FROM articles "
            "WHERE crawled_at > ? AND publisher LIKE ? "
            "ORDER BY published_at DESC",
            (cutoff, f"%{publisher}%")
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def purge_old_data() -> None:
    articles_cutoff = (datetime.now(timezone.utc) - timedelta(hours=ARTICLES_TTL_HOURS)).isoformat()
    bulletins_cutoff = (datetime.now(timezone.utc) - timedelta(days=BULLETIN_RETENTION_DAYS)).strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        res = await db.execute("DELETE FROM articles WHERE crawled_at < ?", (articles_cutoff,))
        a_deleted = res.rowcount
        res = await db.execute("DELETE FROM bulletins WHERE date < ?", (bulletins_cutoff,))
        b_deleted = res.rowcount
        await db.commit()
    if a_deleted or b_deleted:
        logger.info(f"Purge: {a_deleted} articles, {b_deleted} bulletins supprimés")


def _parse_bulletin_row(row: dict) -> dict:
    row["bulletin_json"] = json.loads(row["bulletin_json"])
    return row
