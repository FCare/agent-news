"""Génère un wiki statique (MkDocs) depuis les sujets consolidés (storage.py:
subjects/subject_editions, voir subject_consolidation.py) : une fiche par sujet
(résumé courant + historique des éditions), un index par date ("tous les sujets
du 18 juillet"), un index par catégorie (les 17 fixes, clustering.CATEGORIES), un
index par source citée, et un accueil avec recherche sémantique réutilisant
vector_store.search_subjects tel quel.

Étape appelée automatiquement en fin de run_bulletin_pipeline (main.py), rejouable
manuellement : `python3 wiki_build.py`.

Régénération complète à chaque exécution (comme reference/build_wiki.py côté
contes-agent) : c'est une pure mise en forme de ce qui est déjà en base, pas
d'état incrémental à gérer.
"""

import logging
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import storage
from clustering import CATEGORIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WIKI_SRC_DIR = Path(os.environ.get("WIKI_SRC_DIR", "/wiki/src"))
WIKI_SITE_DIR = Path(os.environ.get("WIKI_SITE_DIR", "/wiki/site"))
MKDOCS_CONFIG = Path(__file__).resolve().parent / "mkdocs.yml"
WIKI_ASSETS_SRC = Path(__file__).resolve().parent / "wiki_theme"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def _format_date_fr(date_str: str) -> str:
    """'2026-07-22' -> '22 juillet 2026' — affichage uniquement, les slugs/noms de
    fichiers/ancres restent en ISO (YYYY-MM-DD), triable et sans ambiguïté."""
    try:
        year, month, day = date_str.split("-")
        return f"{int(day)} {_MONTHS_FR[int(month) - 1]} {year}"
    except (ValueError, IndexError):
        return date_str


# ---------------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def _slugify(text: str, suffix: int | str | None = None) -> str:
    base = _SLUG_RE.sub("-", _normalize(text)).strip("-") or "x"
    return f"{base}-{suffix}" if suffix is not None else base


def _subject_slug(title: str, subject_id: int) -> str:
    # Suffixe subject_id : deux sujets distincts peuvent partager un titre proche
    # (ex: deux affaires judiciaires nommées d'après la même ville).
    return _slugify(title, subject_id)


_INVISIBLE_CHARS_RE = re.compile("[​‌‍﻿]")


def _clean_text(text: str) -> str:
    """Retire les caractères zero-width/BOM parfois capturés par le scraping
    (constaté sur données réelles : un blog basé sur le CMS Sanity.io y encode ses
    métadonnées d'édition visuelle en zero-width characters, invisibles à l'affichage
    normal mais qui polluent le titre une fois stocké tel quel)."""
    return _INVISIBLE_CHARS_RE.sub("", text) if text else text


_MAX_PUBLISHER_LEN = 60


def _clean_sources(sources: list) -> list[dict]:
    """Normalise + filtre les entrées "sources" : retourne [{"name", "url"}], url
    étant None pour les éditions antérieures à la migration (sources = simples noms
    de médias sans lien, voir bulletin_gen.py::_generate_deep_dive — depuis, sources
    est construit depuis les articles réellement appariés, avec leur URL exacte).
    Filtre aussi les entrées implausibles comme nom de média : constaté sur données
    réelles, un échec ponctuel du parsing JSON de la tool-call LLM fait parfois
    fuiter un fragment entier de résumé/title_fr dans le tableau sources (jusqu'à
    plusieurs centaines de caractères) — sans filtre, ça produit un slug de fichier
    trop long (OSError: File name too long). Même parsing bancal, des guillemets
    littéraux restent parfois collés au nom (ex: '"Phys.org"') — on les retire aussi."""
    result = []
    for s in sources:
        if isinstance(s, dict):
            name = (s.get("name") or "").strip().strip('"')
            url = s.get("url") or None
        elif isinstance(s, str):
            name, url = s.strip().strip('"'), None
        else:
            continue
        if name and len(name) <= _MAX_PUBLISHER_LEN:
            result.append({"name": name, "url": url})
    return result


def _format_sources(sources: list[dict]) -> str:
    return ", ".join(f"[{s['name']}]({s['url']})" if s["url"] else s["name"] for s in sources)


# ---------------------------------------------------------------------------
# Rendu Markdown
# ---------------------------------------------------------------------------

def _render_subject_page(subject: dict, editions: list[dict]) -> str:
    lines = [f"# {subject['title']}", ""]
    lines.append(f"**Catégorie** : [{subject['category']}](../categories/{_slugify(subject['category'])}.md)  ")
    lines.append(f"**Suivi depuis le** {_format_date_fr(subject['first_seen_date'])} — "
                 f"**dernière mise à jour le** {_format_date_fr(subject['last_updated_date'])} "
                 f"({len(editions)} édition(s))")
    lines.append("")
    lines += ["## Résumé", "", subject["summary"], ""]

    lines += ["## Historique des éditions", ""]
    for e in sorted(editions, key=lambda x: x["date"], reverse=True):
        # Ancre #edition-{date} en ISO (technique, liée depuis d'autres pages) ;
        # seul le texte affiché passe par _format_date_fr.
        lines.append(f"### {_format_date_fr(e['date'])} — {e['title']} {{: #edition-{e['date']} }}")
        lines.append("")
        lines.append(e["summary"])
        if e.get("deep_dive"):
            lines.append("")
            lines.append(f"??? note \"Analyse détaillée du {_format_date_fr(e['date'])}\"")
            lines.append("")
            for paragraph in e["deep_dive"].split("\n"):
                if paragraph.strip():
                    lines.append(f"    {paragraph}")
            lines.append("")
        clean_sources = _clean_sources(e.get("sources") or [])
        if clean_sources:
            lines.append(f"*Sources : {_format_sources(clean_sources)}*")
        lines.append("")
    return "\n".join(lines)


def _render_list_page(title: str, intro: str, items: list[tuple[str, str, str]]) -> str:
    """items = [(label, lien_relatif, info_annexe)]."""
    lines = [f"# {title}", ""]
    if intro:
        lines += [intro, ""]
    for label, link, extra in items:
        suffix = f" — {extra}" if extra else ""
        lines.append(f"- [{label}]({link}){suffix}")
    lines.append("")
    return "\n".join(lines)


def _render_category_page(title: str, intro: str, items: list[tuple[str, str, str, str]]) -> str:
    """items = [(label, lien_relatif, info_annexe, date)] — regroupés par date de
    dernière mise à jour du sujet (plus récent en premier), date en séparateur ## ."""
    by_date: dict[str, list[tuple[str, str, str]]] = {}
    for label, link, extra, date in items:
        by_date.setdefault(date, []).append((label, link, extra))

    lines = [f"# {title}", ""]
    if intro:
        lines += [intro, ""]
    for date in sorted(by_date, reverse=True):
        lines.append(f"## {_format_date_fr(date)}")
        lines.append("")
        for label, link, extra in sorted(by_date[date], key=lambda x: _normalize(x[0])):
            suffix = f" — {extra}" if extra else ""
            lines.append(f"- [{label}]({link}){suffix}")
        lines.append("")
    return "\n".join(lines)


def _render_date_page(date: str, intro: str, items: list[tuple[str, str, str]],
                       category_summaries: dict[str, str], headline: str, flash: str) -> str:
    """items = [(label, lien_relatif, catégorie)] — liste de liens simple, regroupée
    par catégorie (ordre CATEGORIES). Entre le titre de la catégorie et sa liste, un
    paragraphe de synthèse (bulletin_gen.py::_generate_category_summaries, stocké dans
    bulletin_json["category_summaries"]) qui résume TOUS les sujets de cette catégorie
    pour la journée en un seul texte — pas un résumé par sujet. headline/flash : le
    bulletin global du jour (bulletin_gen.py::_generate_flash), déjà généré par le
    pipeline mais jamais affiché jusqu'ici — aucun appel LLM supplémentaire ici."""
    by_category: dict[str, list[tuple[str, str]]] = {}
    for label, link, category in items:
        by_category.setdefault(category, []).append((label, link))

    lines = [f"# Sujets du {_format_date_fr(date)}", ""]
    if intro:
        lines += [intro, ""]
    if headline or flash:
        lines.append("## Bulletin du jour {: .bulletin-heading }")
        lines.append("")
        if headline:
            lines.append(f"**{headline}**")
            lines.append("")
        if flash:
            lines.append(flash)
            lines.append("")
    ordered_categories = [c for c in CATEGORIES if c in by_category]
    ordered_categories += [c for c in by_category if c not in CATEGORIES]  # robustesse
    for category in ordered_categories:
        cslug = _slugify(category)
        lines.append(f"## [{category}](../categories/{cslug}.md)")
        lines.append("")
        summary = category_summaries.get(category)
        if summary:
            lines.append(summary)
            lines.append("")
        for label, link in by_category[category]:
            lines.append(f"- [{label}]({link})")
        lines.append("")
    return "\n".join(lines)


def _render_index_page(n_subjects: int, n_dates: int, n_categories: int, n_publishers: int) -> str:
    return "\n".join([
        "# Actualités — sujets suivis",
        "",
        f"{n_subjects} sujets suivis, consolidés au fil de leurs éditions successives.",
        "",
        "## Rechercher",
        "",
        "Recherche libre sur les sujets (titre, thème, un détail précis) :",
        "",
        '<div class="wiki-search">',
        '  <input id="wiki-search-input" type="search" placeholder="ex: guerre en Ukraine, crise du café…">',
        '  <div id="wiki-search-results"></div>',
        "</div>",
        "",
        "## Explorer par...",
        "",
        f"- [Catégorie](categories/index.md) ({n_categories})",
        f"- [Date](dates/index.md) ({n_dates})",
        f"- [Source](publishers/index.md) ({n_publishers})",
        "",
    ])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def run() -> dict:
    if WIKI_SRC_DIR.exists():
        shutil.rmtree(WIKI_SRC_DIR)
    WIKI_SRC_DIR.mkdir(parents=True, exist_ok=True)
    if WIKI_ASSETS_SRC.exists():
        shutil.copytree(WIKI_ASSETS_SRC, WIKI_SRC_DIR / "assets", dirs_exist_ok=True)

    subjects = await storage.get_all_subjects()
    for s in subjects:
        s["title"] = _clean_text(s["title"])
        s["summary"] = _clean_text(s["summary"])

    # Sujets
    subject_items_by_category: dict[str, list[tuple[str, str, str]]] = {}
    publishers_index: dict[str, set[int]] = {}
    dates_seen: set[str] = set()
    for s in subjects:
        editions = await storage.get_subject_editions(s["id"])
        for e in editions:
            e["title"] = _clean_text(e["title"])
            e["summary"] = _clean_text(e["summary"])
            if e.get("deep_dive"):
                e["deep_dive"] = _clean_text(e["deep_dive"])
        slug = _subject_slug(s["title"], s["id"])
        await _write(WIKI_SRC_DIR / "sujets" / f"{slug}.md", _render_subject_page(s, editions))

        # Le 4e élément (date) sert uniquement au regroupement dans
        # _render_category_page (## {date} en séparateur) — pas répété dans "extra"
        # pour éviter la redondance avec le séparateur.
        entry = (s["title"], f"../sujets/{slug}.md", f"{len(editions)} édition(s)", s["last_updated_date"])
        subject_items_by_category.setdefault(s["category"], []).append(entry)

        for e in editions:
            dates_seen.add(e["date"])
            for pub in _clean_sources(e.get("sources") or []):
                publishers_index.setdefault(pub["name"], set()).add(s["id"])

    # Index racine des sujets (utile pour un lien direct, pas dans le menu — voir mkdocs.yml)
    # Liens relatifs à sujets/index.md lui-même, donc pas de préfixe "sujets/".
    all_items = [
        (s["title"], f"{_subject_slug(s['title'], s['id'])}.md", s["category"])
        for s in sorted(subjects, key=lambda x: _normalize(x["title"]))
    ]
    await _write(WIKI_SRC_DIR / "sujets" / "index.md", _render_list_page("Tous les sujets", "", all_items))

    # Catégories (vocabulaire fixe, voir clustering.CATEGORIES)
    category_items = []
    for cat in CATEGORIES:
        members = subject_items_by_category.get(cat, [])
        if not members:
            continue
        cslug = _slugify(cat)
        await _write(
            WIKI_SRC_DIR / "categories" / f"{cslug}.md",
            _render_category_page(cat, f"{len(members)} sujet(s).", members),
        )
        category_items.append((cat, f"{cslug}.md", f"{len(members)} sujet(s)"))
    await _write(WIKI_SRC_DIR / "categories" / "index.md", _render_list_page("Catégories", "", category_items))

    # Dates : tous les sujets ayant eu une édition ce jour-là — affiche le titre du
    # SUJET (m["title"]), pas celui de l'édition du jour (m["edition_title"]) : ce
    # dernier est délibérément préservé tel quel comme archive (voir storage.py::
    # update_subject) et n'est donc jamais retraduit, contrairement au titre du sujet.
    date_items = []
    for date in sorted(dates_seen, reverse=True):
        members = await storage.get_subjects_by_date(date)
        items = [
            (_clean_text(m["title"]), f"../sujets/{_subject_slug(m['title'], m['id'])}.md", m["category"])
            for m in members
        ]
        bulletin = await storage.get_bulletin_by_date(date)
        bulletin_json = (bulletin or {}).get("bulletin_json", {})
        category_summaries = bulletin_json.get("category_summaries", {})
        headline = _clean_text(bulletin_json.get("headline", "")) or ""
        flash = _clean_text(bulletin_json.get("flash", "")) or ""
        await _write(
            WIKI_SRC_DIR / "dates" / f"{date}.md",
            _render_date_page(date, f"{len(items)} édition(s) publiée(s) ce jour-là.", items,
                               category_summaries, headline, flash),
        )
        date_items.append((_format_date_fr(date), f"{date}.md", f"{len(items)} édition(s)"))
    await _write(WIKI_SRC_DIR / "dates" / "index.md", _render_list_page("Dates", "", date_items))

    # Sources
    publisher_items = []
    for pub, subject_ids in sorted(publishers_index.items(), key=lambda kv: _normalize(kv[0])):
        pslug = _slugify(pub)
        items = [
            (s["title"], f"../sujets/{_subject_slug(s['title'], s['id'])}.md", s["category"])
            for s in subjects if s["id"] in subject_ids
        ]
        await _write(
            WIKI_SRC_DIR / "publishers" / f"{pslug}.md",
            _render_list_page(pub, f"{len(items)} sujet(s) citant cette source.", items),
        )
        publisher_items.append((pub, f"{pslug}.md", f"{len(items)} sujet(s)"))
    await _write(WIKI_SRC_DIR / "publishers" / "index.md", _render_list_page("Sources", "", publisher_items))

    # Accueil
    await _write(
        WIKI_SRC_DIR / "index.md",
        _render_index_page(len(subjects), len(date_items), len(category_items), len(publisher_items)),
    )

    from mkdocs.commands.build import build as mkdocs_build
    from mkdocs.config import load_config

    if WIKI_SITE_DIR.exists():
        shutil.rmtree(WIKI_SITE_DIR)
    cfg = load_config(str(MKDOCS_CONFIG), docs_dir=str(WIKI_SRC_DIR), site_dir=str(WIKI_SITE_DIR))
    mkdocs_build(cfg)

    result = {
        "subjects": len(subjects),
        "dates": len(date_items),
        "categories": len(category_items),
        "publishers": len(publisher_items),
    }
    logger.info(f"wiki_build: {result}")
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
