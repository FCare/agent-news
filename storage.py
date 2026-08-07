import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path("/data/news.db")
BULLETIN_RETENTION_DAYS = 90
ARTICLES_TTL_HOURS = 6
# Un "sujet" (voir subject_consolidation.py) est la vue CONSOLIDÉE d'une actualité qui
# dure plusieurs jours (ex: "Guerre en Ukraine"), distincte du bulletin brut qui reste
# inchangé et gardé 90 jours (BULLETIN_RETENTION_DAYS). Rétention bien plus longue :
# un sujet vivant (mis à jour régulièrement) doit survivre largement au-delà de 90
# jours, seule l'inactivité prolongée (aucune nouvelle édition depuis un an) le purge.
SUBJECT_RETENTION_DAYS = 365


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

            -- Sujet consolidé (voir subject_consolidation.py) : titre/résumé COURANTS,
            -- mis à jour à chaque édition rapprochée (pas de nouvelle ligne par jour,
            -- contrairement à bulletins) — l'historique jour par jour vit dans
            -- subject_editions.
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                first_seen_date TEXT NOT NULL,
                last_updated_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            -- Une édition = l'état du topic tel que généré par bulletin_gen.py CE
            -- jour-là, avant fusion — gardé tel quel même après consolidation du
            -- résumé parent, pour ne jamais perdre le détail d'un jour précis.
            CREATE TABLE IF NOT EXISTS subject_editions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
                date TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                deep_dive TEXT,
                what_to_watch TEXT,
                sources TEXT NOT NULL,
                date_range TEXT,
                category TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(subject_id, date)
            );
            CREATE INDEX IF NOT EXISTS idx_subjects_last_updated ON subjects(last_updated_date);
            CREATE INDEX IF NOT EXISTS idx_subjects_category ON subjects(category);
            CREATE INDEX IF NOT EXISTS idx_subject_editions_date ON subject_editions(date);
        """)
        # Migration pour les DB créées avant l'ajout de la colonne category (vote
        # majoritaire, voir update_subject) : ALTER TABLE échoue si la colonne existe
        # déjà, ce qui est le signal normal qu'il n'y a rien à faire.
        try:
            await db.execute("ALTER TABLE subject_editions ADD COLUMN category TEXT")
            await db.commit()
        except aiosqlite.OperationalError:
            pass
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


async def get_distinct_publishers(hours: int = ARTICLES_TTL_HOURS) -> list[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT publisher FROM articles WHERE crawled_at > ? AND publisher != '' ORDER BY publisher",
            (cutoff,)
        ) as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows]


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


async def _insert_edition(db: aiosqlite.Connection, subject_id: int, date: str, title: str,
                           summary: str, deep_dive: str | None, what_to_watch: str | None,
                           sources: list, date_range: str | None, category: str | None = None) -> None:
    # INSERT OR REPLACE sur (subject_id, date) : un même sujet ne peut avoir qu'une
    # édition par jour — rejouer la consolidation le même jour (ex: cron 8h puis 20h)
    # remplace l'édition du jour plutôt que d'en empiler une deuxième.
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT OR REPLACE INTO subject_editions "
        "(subject_id, date, title, summary, deep_dive, what_to_watch, sources, date_range, category, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (subject_id, date, title, summary, deep_dive, what_to_watch,
         json.dumps(sources, ensure_ascii=False), date_range, category, now),
    )


async def _majority_category(db: aiosqlite.Connection, subject_id: int, fallback: str) -> str:
    """Catégorie la plus fréquente parmi les éditions catégorisées de ce sujet (la
    catégorisation quotidienne de bulletin_gen.py est bruitée — voir
    subject_consolidation.py — donc figer la catégorie sur la première édition
    produit parfois une catégorie absurde qui ne se corrige jamais). Les éditions
    d'avant cette migration n'ont pas de catégorie (NULL) et sont ignorées du vote ;
    si aucune édition catégorisée n'existe encore, on garde `fallback` (la catégorie
    actuelle du sujet).
    
    Ne considère que les catégories valides (définies dans CATEGORIES) pour éviter
    les catégories invalides qui résulteraient d'un vote entre deux catégories
    proches (ex: hallucination fusionnant 'Santé' et 'Environnement').
    """
    # Catégories valides — doit correspondre à bulletin_gen.py::CATEGORIES
    VALID_CATEGORIES = {
        "Politique", "France", "International", "Économie & Finance",
        "Tech & Numérique", "Science & Espace", "Santé", "Sport",
        "Environnement", "Société", "Justice",
        "Éducation & Recherche", "Culture & Médias", "People", "Bons plans",
    }
    
    async with db.execute(
        "SELECT category, COUNT(*) AS n FROM subject_editions "
        "WHERE subject_id = ? AND category IS NOT NULL AND category != '' "
        "AND category IN ({}) "
        "GROUP BY category ORDER BY n DESC, category ASC LIMIT 1".format(
            ','.join('?' * len(VALID_CATEGORIES))
        ),
        (subject_id,) + tuple(VALID_CATEGORIES),
    ) as cur:
        row = await cur.fetchone()
    return row[0] if row else fallback


async def create_subject(title: str, category: str, summary: str, date: str,
                          deep_dive: str | None = None, what_to_watch: str | None = None,
                          sources: list | None = None, date_range: str | None = None) -> int:
    """Nouveau sujet, sans candidat de continuité trouvé (voir subject_consolidation.py) —
    first_seen_date = last_updated_date = sa première édition."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO subjects (title, category, summary, first_seen_date, last_updated_date, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, category, summary, date, date, now, now),
        )
        subject_id = cur.lastrowid
        await _insert_edition(db, subject_id, date, title, summary, deep_dive, what_to_watch,
                               sources or [], date_range, category)
        await db.commit()
    return subject_id


async def update_subject(subject_id: int, title: str, consolidated_summary: str, date: str,
                          edition_title: str, edition_summary: str, deep_dive: str | None = None,
                          what_to_watch: str | None = None, sources: list | None = None,
                          date_range: str | None = None, category: str | None = None) -> None:
    """Rapprochement confirmé (voir subject_consolidation.py) : `consolidated_summary`
    remplace le résumé courant du sujet (généré par LLM à partir de l'historique complet),
    tandis que `edition_summary` (le résumé du topic TEL QUE généré aujourd'hui, avant
    fusion) est conservé tel quel dans subject_editions — jamais perdu même si le résumé
    consolidé évolue encore par la suite. `category` (catégorie du topic du jour) alimente
    le vote majoritaire (_majority_category) qui remplace la catégorie du sujet — plutôt
    que de la figer sur sa toute première édition, potentiellement mal classée."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        # Lu AVANT _insert_edition (même connexion) : une fois l'INSERT posé, cette
        # connexion porte une transaction d'écriture non commitée — une deuxième
        # connexion (ex: get_subject, qui en ouvre une nouvelle) se bloquerait dessus
        # (SQLITE_BUSY) jusqu'au commit final.
        async with db.execute("SELECT category FROM subjects WHERE id = ?", (subject_id,)) as cur:
            row = await cur.fetchone()
        fallback = row[0] if row else category

        await _insert_edition(db, subject_id, date, edition_title, edition_summary, deep_dive,
                               what_to_watch, sources or [], date_range, category)
        new_category = await _majority_category(db, subject_id, fallback)
        await db.execute(
            "UPDATE subjects SET title = ?, summary = ?, category = ?, last_updated_date = ?, updated_at = ? "
            "WHERE id = ?",
            (title, consolidated_summary, new_category, date, now, subject_id),
        )
        await db.commit()


async def get_subject(subject_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def get_subjects_by_category(category: str, active_within_days: int | None = None) -> list[dict]:
    """Candidats de continuité pour subject_consolidation.py — restreint à la même
    catégorie (voir sa docstring : réduit fortement le risque de faux rapprochement,
    ex: ne jamais comparer un sujet "Économie" à un sujet "Sport")."""
    sql = "SELECT * FROM subjects WHERE category = ?"
    params: list = [category]
    if active_within_days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=active_within_days)).strftime("%Y-%m-%d")
        sql += " AND last_updated_date >= ?"
        params.append(cutoff)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_subject_editions(subject_id: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM subject_editions WHERE subject_id = ? ORDER BY date", (subject_id,)
        ) as cur:
            rows = await cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else []
        result.append(d)
    return result


async def get_subjects_by_date(date: str) -> list[dict]:
    """Tous les sujets ayant eu une édition CE jour précis — page wiki dates/<date>.md
    ("tous les sujets du 18 juillet"), avec le contenu complet de l'édition du jour
    (deep_dive/what_to_watch/sources) pour pouvoir afficher le bulletin du jour tel
    quel, pas seulement un lien vers le sujet consolidé."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT s.*, se.title AS edition_title, se.summary AS edition_summary, "
            "se.deep_dive AS edition_deep_dive, se.what_to_watch AS edition_what_to_watch, "
            "se.sources AS edition_sources "
            "FROM subjects s JOIN subject_editions se ON se.subject_id = s.id "
            "WHERE se.date = ? ORDER BY s.category, s.title",
            (date,),
        ) as cur:
            rows = await cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["edition_sources"] = json.loads(d["edition_sources"]) if d["edition_sources"] else []
        except (json.JSONDecodeError, TypeError):
            d["edition_sources"] = []
        result.append(d)
    return result


async def get_all_subjects() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM subjects ORDER BY category, title") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_distinct_edition_dates() -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT DISTINCT date FROM subject_editions ORDER BY date DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def purge_old_subjects() -> int:
    """Distinct de purge_old_data (90j, données brutes) : un sujet survit tant qu'il
    est réactualisé, purgé seulement après SUBJECT_RETENTION_DAYS sans nouvelle édition.
    Suppression explicite des éditions enfants avant le sujet parent plutôt que de
    compter sur ON DELETE CASCADE : les contraintes de clé étrangère SQLite ne sont pas
    actives par défaut sans PRAGMA foreign_keys=ON, jamais activé dans ce module."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SUBJECT_RETENTION_DAYS)).strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM subject_editions WHERE subject_id IN "
            "(SELECT id FROM subjects WHERE last_updated_date < ?)",
            (cutoff,),
        )
        res = await db.execute("DELETE FROM subjects WHERE last_updated_date < ?", (cutoff,))
        deleted = res.rowcount
        await db.commit()
    if deleted:
        logger.info(f"Purge: {deleted} sujets inactifs depuis plus d'un an supprimés")
    return deleted
