import asyncio
import json
import logging
import math
import os
import re
from datetime import datetime, timezone

import openai

from clustering import cluster_articles
from crawler import RawArticle
from search_client import SearchClient

logger = logging.getLogger(__name__)

_LLM_SEMAPHORE = asyncio.Semaphore(1)

CATEGORIES = [
    "International",
    "Europe",
    "France",
    "Économie & Finance",
    "Géopolitique & Défense",
    "Informatique & IA",
    "Astronomie & Espace",
    "Science & Technologie",
    "Médecine & Santé",
    "Sport",
    "Automobile & Mobilité",
    "Immobilier & Logement",
    "Voyages & Tourisme",
    "Droit & Justice",
    "Éducation & Recherche",
    "Société & Environnement",
    "Culture & Médias",
]

# Context budget: 128k tokens available.
# Worst-case deep dive: 8 articles × 3 000 chars + 10 searches × 2 000 chars
# ≈ (24 000 + 20 000) / 4 ≈ 11 000 tokens — well within limit.
MAX_TOPICS = int(os.environ.get("MAX_TOPICS", 25))  # cap after TF-IDF clustering
MAX_BODY_IN_DEEP_DIVE = 3000         # full article body (matches crawler MAX_BODY_CHARS)
MAX_ARTICLES_IN_DEEP_DIVE = 8        # max articles per topic in deep dive
MAX_BODY_IN_QUERY_GEN = 600          # assez pour lever l'ambiguïté d'un titre imagé
# Par volet, pas par rapport : agent-web-search renvoie un volet 'background'
# (encyclopédie et web) et un volet 'recent' (presse). Tronquer le rapport
# concaténé faisait disparaître le second, qui arrive en fin de chaîne — le
# deep dive n'aurait vu que du contexte historique, jamais l'actualité.
MAX_SEARCH_SECTION_IN_DEEP_DIVE = 1200
SEARCH_QUERIES_PER_TOPIC = int(os.environ.get("SEARCH_QUERIES_PER_TOPIC", 5))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


_SEARCH_QUERIES_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_search_queries",
        "description": "Génère exactement 10 requêtes web pour approfondir un sujet d'actualité",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "5 requêtes couvrant: (1) faits récents, (2) contexte historique, "
                        "(3) réactions officielles, (4) implications économiques/géopolitiques, "
                        "(5) analyses d'experts et perspectives"
                    ),
                    "minItems": 5,
                    "maxItems": 5,
                }
            },
            "required": ["queries"],
        }
    }
}]

_DEEP_DIVE_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_analysis",
        "description": "Génère l'analyse complète d'un sujet pour le journal télévisé",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": CATEGORIES,
                    "description": (
                        "Catégorie la plus appropriée pour CE sujet précis, choisie à partir "
                        "de son titre et de son contenu réel — pas une catégorie générique par "
                        "défaut. Ex: un sujet sur la guerre en Ukraine est 'Géopolitique & "
                        "Défense', pas 'Informatique & IA'. Distinction FRANCE vs "
                        "INTERNATIONAL : un événement qui se déroule EN FRANCE ou qui concerne "
                        "au premier chef un acteur/une institution française (ministre, "
                        "collectivité, entreprise française...) est TOUJOURS 'France', même "
                        "traité par un média étranger ou avec une portée internationale — "
                        "'International' est réservé aux sujets qui se déroulent hors de "
                        "France ou concernent plusieurs pays sans la France comme acteur "
                        "principal."
                    ),
                },
                "title_fr": {
                    "type": "string",
                    "description": (
                        "Titre du sujet TOUJOURS EN FRANÇAIS, court et accrocheur, même si "
                        "toutes les sources sont en anglais ou dans une autre langue — traduis, "
                        "ne recopie jamais un titre source tel quel."
                    )
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Résumé factuel en 3-5 phrases, style JT 20h. "
                        "Faits précis, chiffres si disponibles, ton oral. TOUJOURS EN FRANÇAIS."
                    )
                },
                "deep_dive": {
                    "type": "string",
                    "description": (
                        "Analyse approfondie: contexte historique, enjeux stratégiques, "
                        "chiffres clés, perspectives EU et US, déclarations importantes, "
                        "conséquences potentielles. Ton oral sérieux, 6-10 phrases, sans liste. "
                        "TOUJOURS EN FRANÇAIS."
                    )
                },
                "what_to_watch": {
                    "type": "string",
                    "description": "Ce qu'il faut surveiller / prochaines étapes. 1-2 phrases. TOUJOURS EN FRANÇAIS."
                },
            },
            "required": ["category", "title_fr", "summary", "deep_dive", "what_to_watch"],
        }
    }
}]

_FLASH_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_flash",
        "description": "Génère le flash info et le titre du JT du jour",
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string",
                    "description": "Titre principal du jour, une phrase percutante"
                },
                "flash": {
                    "type": "string",
                    "description": (
                        "6 phrases résumant les sujets les plus importants du jour. "
                        "Style factuel et direct, sans formule d'introduction."
                    )
                },
            },
            "required": ["headline", "flash"],
        }
    }
}]

_CATEGORY_SUMMARY_TOOL = [{
    "type": "function",
    "function": {
        "name": "generate_category_summaries",
        "description": (
            "Génère, pour chaque catégorie fournie, un court bulletin de synthèse "
            "couvrant l'ENSEMBLE des sujets listés pour cette catégorie ce jour-là "
            "(pas un résumé par sujet — une synthèse globale de la catégorie)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summaries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "description": "Nom exact de la catégorie, tel que fourni."},
                            "summary": {
                                "type": "string",
                                "description": (
                                    "2-4 phrases, style bulletin radio, synthétisant l'ensemble des "
                                    "sujets de cette catégorie pour la journée — pas une liste, un "
                                    "texte suivi. TOUJOURS EN FRANÇAIS."
                                ),
                            },
                        },
                        "required": ["category", "summary"],
                    },
                },
            },
            "required": ["summaries"],
        }
    }
}]

_PROOFREAD_TOOL = [{
    "type": "function",
    "function": {
        "name": "proofread_text",
        "description": (
            "Relit des textes en français générés automatiquement et corrige les "
            "artefacts de génération ponctuels : caractères d'un autre alphabet insérés "
            "au milieu d'un mot (ex: 'seברète' au lieu de 's'apprête'), mots parasites "
            "incohérents insérés dans une phrase (ex: 'se suku estimant dupée' au lieu de "
            "'s'estimant dupée'), fragments de texte corrompus ou tronqués de façon "
            "incohérente. Corrige UNIQUEMENT ce type d'artefact — ne reformule pas, ne "
            "change ni le sens ni le style de ce qui est déjà correct. Si un champ n'a "
            "aucun artefact, renvoie-le à l'IDENTIQUE."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Titre corrigé (identique à l'entrée si déjà propre)."},
                "summary": {"type": "string", "description": "Résumé corrigé (identique à l'entrée si déjà propre)."},
                "deep_dive": {"type": "string", "description": "Analyse corrigée (identique à l'entrée si déjà propre)."},
                "what_to_watch": {"type": "string", "description": "Texte 'à suivre' corrigé (identique à l'entrée si déjà propre)."},
            },
            "required": ["title", "summary", "deep_dive", "what_to_watch"],
        }
    }
}]

_TRANSLATE_TOOL = [{
    "type": "function",
    "function": {
        "name": "translate_excerpts",
        "parameters": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extraits traduits en français, dans le même ordre que l'entrée",
                }
            },
            "required": ["translations"],
        }
    }
}]

_ANSWER_TOOL = [{
    "type": "function",
    "function": {
        "name": "answer_question",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Réponse complète à la question en français, style oral. "
                        "Basée sur les informations disponibles dans le bulletin."
                    )
                },
                "needs_web_search": {
                    "type": "boolean",
                    "description": "True si le bulletin ne couvre pas ce sujet"
                },
            },
            "required": ["answer", "needs_web_search"],
        }
    }
}]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _recover_partial_json(raw: str) -> dict:
    """Try to extract complete objects from truncated tool-call JSON."""
    # Find last complete object before truncation
    last = raw.rfind("},")
    if last == -1:
        last = raw.rfind("}")
    if last == -1:
        return {}
    fragment = raw[:last + 1]
    # Locate the opening of the topics array
    topics_pos = fragment.find('"topics"')
    if topics_pos == -1:
        return {}
    array_start = fragment.find("[", topics_pos)
    if array_start == -1:
        return {}
    try:
        return json.loads(fragment[: last + 1] + "]}")
    except Exception:
        return {}


def _llm_call(llm_client: openai.OpenAI, model: str, system: str,
              user: str, tool: list, max_tokens: int | None = None) -> dict:
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        tools=tool,
        tool_choice="required",
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    resp = llm_client.chat.completions.create(**kwargs, timeout=120)
    calls = resp.choices[0].message.tool_calls
    if not calls:
        return {}

    raw = calls[0].function.arguments
    finish = resp.choices[0].finish_reason
    if finish == "length":
        logger.warning(f"LLM output truncated (finish=length), attempting JSON recovery "
                       f"({len(raw)} chars)")
        logger.debug(f"  raw[:500]: {raw[:500]}")
        logger.debug(f"  raw[-200]: {raw[-200:]}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON parse error: {e} — raw length={len(raw)}")
        logger.warning(f"  first 300 chars: {raw[:300]}")
        logger.warning(f"  last  200 chars: {raw[-200:]}")
        recovered = _recover_partial_json(raw)
        if recovered:
            n = len(recovered.get("topics", []))
            logger.warning(f"  JSON recovery: {n} sujets récupérés")
        return recovered


# Un texte rendu en entier se termine par une ponctuation ; un fragment coupé
# par le parseur s'arrête n'importe où. Second signal, qui évite de prendre pour
# une corruption un texte ouvrant légitimement sur une citation.
_FINS_DE_PHRASE = (".", "!", "?", "…", "»", ":", ";")

# Caractère d'un autre alphabet collé à des lettres latines : le modèle en glisse
# parfois un au milieu d'un mot français ('faits divers 온vients' pour 'récents').
# Exiger le contact avec du latin distingue l'artefact d'une citation légitime en
# alphabet non latin, où les caractères vont par groupes.
_ALPHABET_ETRANGER = re.compile(
    r"[a-zA-ZÀ-ÿ][Ѐ-ӿ֐-׿؀-ۿ぀-ヿ一-鿿가-힯]"
    r"|[Ѐ-ӿ֐-׿؀-ۿ぀-ヿ一-鿿가-힯][a-zA-ZÀ-ÿ]"
)

# Artefacts de décodage du modèle quantifié : tokens de contrôle qui "fuient"
# dans le texte généré. Observés: '<|"|>', '<|', '|>', etc. Ces séquences
# n'ont aucun sens en français et trahissent une corruption de la génération.
_ARTEFACT_TOKENS = re.compile(
    r"<\|\"?\|>"      # <|"<|> ou <|>
    r"|<\|"           # <| seul
    r"\"\|>"          # "|> seul
    r"|recherche scientifique annonce"  # Fragment spécifique observé
    r"|^[^a-zA-ZÀ-ÿ]*<\|"  # Commence par des caractères non-alphabétiques suivis de <|
)


def _looks_truncated(result: dict) -> str | None:
    """Détecte une valeur amputée par le parseur de tool call du backend.

    Le modèle produit parfois un guillemet droit non échappé en ouverture de
    valeur ; le parseur coupe alors sur le guillemet ou la virgule suivante et
    rend un fragment. Le JSON reste valide, aucune exception n'est levée, et le
    fragment part dans le bulletin — un flash réduit à '"Un séisme de magnitude 7'
    a fait échouer toute une génération. Le guillemet ouvrant est le marqueur
    commun à tous les cas observés (flash, analyse, résumé, à-suivre).

    Le guillemet doit être resté ouvert : une valeur qui ouvre ET referme est
    une citation légitime, ou une chaîne vide rendue littéralement ('""'), que
    la consolidation produit à chaque fois qu'elle ne veut rien changer. Les
    compter comme corrompues coûtait un appel LLM pour rien.

    Retourne (chemin du champ fautif, sa valeur), ou None si tout va bien. La
    valeur est rendue avec le chemin parce que celui-ci peut désigner une entrée
    de tableau ('textes[texte]') et n'est donc pas indexable sur le résultat.
    """
    for champ, valeur in result.items():
        if isinstance(valeur, list):
            # Réponses en tableau (synthèses par catégorie, sujets) : inspecter
            # les valeurs de chaque entrée.
            for item in valeur:
                if isinstance(item, dict) and (fautif := _looks_truncated(item)):
                    return f"{champ}[{fautif[0]}]", fautif[1]
            continue
        if not isinstance(valeur, str):
            continue
        v = valeur.strip()
        if v.startswith('"') and not v.endswith('"') and not v.endswith(_FINS_DE_PHRASE):
            return champ, valeur
        if _ALPHABET_ETRANGER.search(v):
            return champ, valeur
        if _ARTEFACT_TOKENS.search(v):
            return champ, valeur
    return None


async def _llm(llm_client: openai.OpenAI, model: str, system: str,
               user: str, tool: list, max_tokens: int | None = None,
               tentatives: int = 2) -> dict:
    """Appelle le LLM, en rejouant une fois si la réponse revient amputée.

    La corruption est intermittente (~10% des appels sur ce backend), donc un
    second essai suffit presque toujours.
    """
    loop = asyncio.get_event_loop()
    for essai in range(1, tentatives + 1):
        async with _LLM_SEMAPHORE:
            try:
                result = await loop.run_in_executor(
                    None, _llm_call, llm_client, model, system, user, tool, max_tokens
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return {}

        if not result:
            return {}
        fautif = _looks_truncated(result)
        if fautif is None:
            return result
        # La valeur vient de _looks_truncated, jamais d'une indexation de result :
        # pour une corruption nichée dans un tableau, le nom rapporté est un
        # chemin ('textes[texte]') et non une clé — l'indexer levait un KeyError
        # qui a fait échouer toute une génération.
        champ, valeur = fautif
        if essai < tentatives:
            logger.warning(
                f"  réponse LLM amputée sur '{champ}' ({len(valeur)} car.) — "
                f"nouvel essai {essai + 1}/{tentatives}"
            )
        else:
            logger.error(f"  réponse LLM toujours amputée sur '{champ}' après {tentatives} essais")
    return result


# ---------------------------------------------------------------------------
# Article matching
# ---------------------------------------------------------------------------

def _find_articles_for_topic(topic: dict, articles: list[RawArticle],
                              n: int = MAX_ARTICLES_IN_DEEP_DIVE) -> list[RawArticle]:
    """
    Use cluster indices from TF-IDF clustering when available,
    fall back to keyword matching otherwise.
    """
    indices = topic.get("_article_indices")
    if indices:
        return [articles[i] for i in indices[:n] if i < len(articles)]

    # Fallback: keyword matching
    keywords = [w.lower() for w in topic.get("keywords", [])]
    title_words = [w.lower() for w in topic["title"].split() if len(w) > 3]
    all_terms = set(keywords + title_words)

    def score(a: RawArticle) -> int:
        text = (a.title + " " + a.body[:200]).lower()
        return sum(1 for term in all_terms if term in text)

    ranked = sorted(articles, key=score, reverse=True)
    return [a for a in ranked[:n] if score(a) > 0] or ranked[:3]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _do_cluster_topics(articles: list[RawArticle]) -> list[dict]:
    """TF-IDF + DBSCAN clustering — no LLM, instant.
    
    Garantit au moins 1 sujet par catégorie (diversité éditoriale), puis remplit
    avec les sujets les plus importants jusqu'à MAX_TOPICS.
    """
    all_topics = cluster_articles(articles)
    
    # 1. Garantir au moins 1 sujet par catégorie (le plus important de chaque)
    selected = []
    seen_ids = set()
    by_category: dict[str, list[dict]] = {}
    for t in all_topics:
        cat = t.get("category", "International")
        by_category.setdefault(cat, []).append(t)
    
    for cat, topics_in_cat in by_category.items():
        if topics_in_cat:
            best = topics_in_cat[0]  # Déjà triés par importance dans cluster_articles
            selected.append(best)
            seen_ids.add(id(best))
    
    # 2. Remplir avec les sujets restants les plus importants jusqu'à MAX_TOPICS
    for t in all_topics:
        if len(selected) >= MAX_TOPICS:
            break
        if id(t) not in seen_ids:
            selected.append(t)
            seen_ids.add(id(t))
    
    # Trier par importance décroissante pour l'affichage
    selected.sort(key=lambda t: t.get("importance", 0), reverse=True)
    topics = selected[:MAX_TOPICS]
    
    for t in topics:
        logger.info(
            f"  [{t['article_count']:3d} art.] [{t['category']}] {t['title']}"
        )
    return topics


async def _generate_search_queries(topic: dict, articles: list[RawArticle],
                                    llm_client: openai.OpenAI, model: str) -> list[str]:
    """Génère les requêtes d'enrichissement à partir du texte, pas du seul titre.

    Un titre de presse est souvent métaphorique, et le LLM le prend au pied de
    la lettre. Constaté sur « la Gironde submergée par un élan de solidarité » :
    il a compris "inondations", généré des requêtes sur les crues, et le web a
    confirmé l'erreur — il existe de vraies crues en Gironde dans les archives.
    Le bulletin a décrit des inondations avec 200 000 sinistrés et 3 milliards
    de dégâts, alors que le département brûlait. Les extraits d'articles lèvent
    l'ambiguïté que le titre seul entretient.
    """
    logger.info(f"    → appel LLM requêtes...")
    matched = _find_articles_for_topic(topic, articles, n=3)
    extraits = "\n\n".join(
        f"[{a.publisher}] {a.title}\n{(a.body or '')[:MAX_BODY_IN_QUERY_GEN]}"
        for a in matched
    )
    result = await _llm(
        llm_client, model,
        system=(
            f"Tu es journaliste d'investigation. Génère exactement {SEARCH_QUERIES_PER_TOPIC} requêtes "
            "de recherche web pour enrichir un sujet d'actualité. "
            "Couvre: (1) faits récents, (2) contexte/historique, (3) réactions officielles, "
            "(4) implications économiques ou géopolitiques, (5) perspectives d'experts. "
            "Requêtes courtes et efficaces pour un moteur d'actualités. "
            "IMPORTANT: le titre peut être imagé ou accrocheur — identifie de quoi il "
            "s'agit RÉELLEMENT d'après les extraits d'articles, jamais d'après le seul "
            "titre, et n'introduis dans tes requêtes aucun fait qui n'y figure pas. "
            "Appelle generate_search_queries."
        ),
        user=(
            f"Sujet: {topic['title']}\n"
            f"Catégorie: {topic['category']}\n"
            + (f"\n=== EXTRAITS D'ARTICLES ===\n{extraits}" if extraits else "")
        ),
        tool=_SEARCH_QUERIES_TOOL,
    )
    queries = result.get("queries", [])
    if not queries:
        queries = [topic["title"]]
    queries = queries[:SEARCH_QUERIES_PER_TOPIC]
    for i, q in enumerate(queries, 1):
        logger.info(f"    requête {i:2d}: {q}")
    return queries


def _fmt_date(dt) -> str:
    if not dt:
        return ""
    try:
        from zoneinfo import ZoneInfo
        if dt.tzinfo is None:
            from datetime import timezone as _tz
            dt = dt.replace(tzinfo=_tz.utc)
        dt = dt.astimezone(ZoneInfo("Europe/Paris"))
        return dt.strftime("%-d %b %Y %Hh%M")
    except Exception:
        return str(dt)[:16]


async def _generate_deep_dive(topic: dict, articles: list[RawArticle],
                               search_results: list[dict],
                               llm_client: openai.OpenAI, model: str) -> dict:
    # Find relevant articles — include publication date in excerpts
    matched = _find_articles_for_topic(topic, articles)
    article_excerpts = []
    article_dates = []
    for a in matched:
        date_tag = _fmt_date(a.published_at)
        article_dates.append(a.published_at)
        header = f"[{a.publisher}{' — ' + date_tag if date_tag else ''}] {a.title}"
        article_excerpts.append(f"{header}\n{a.body[:MAX_BODY_IN_DEEP_DIVE]}")

    # Date range of sources
    valid_dates = [d for d in article_dates if d]
    if valid_dates:
        from datetime import timezone as _tz
        def _to_utc(d):
            return d.replace(tzinfo=_tz.utc) if d.tzinfo is None else d
        oldest = min(valid_dates, key=_to_utc)
        newest = max(valid_dates, key=_to_utc)
        date_range = f"{_fmt_date(oldest)} → {_fmt_date(newest)}"
    else:
        date_range = ""

    # Gather search reports, en tronquant chaque volet séparément pour que
    # l'actualité survive au budget de contexte (voir la constante).
    search_excerpts = []
    for r in search_results:
        sections = []
        for label, key in (("Fond", "background"), ("Actualité", "recent")):
            section = (r.get(key) or "").strip()
            if section:
                sections.append(f"[{label}]\n{section[:MAX_SEARCH_SECTION_IN_DEEP_DIVE]}")
        if sections:
            search_excerpts.append("\n\n".join(sections))
        elif (report := (r.get("report") or "").strip()):
            # Réponse d'un agent qui ne renvoie pas encore les deux volets
            search_excerpts.append(report[:MAX_SEARCH_SECTION_IN_DEEP_DIVE * 2])

    articles_block = "\n\n---\n\n".join(article_excerpts) or "(aucun extrait)"
    searches_block = "\n\n---\n\n".join(search_excerpts) or "(aucun résultat)"

    user_content = (
        f"SUJET: {topic['title']}\n"
        # Pas de "CATÉGORIE: ..." ici : topic['category'] vient d'un classifieur par
        # mots-clés (clustering.py::_assign_category) peu fiable — l'afficher comme un
        # fait biaisait le LLM à le recopier au lieu de déterminer la vraie catégorie
        # (voir le champ "category" de generate_analysis, calculé à partir du contenu
        # réel ci-dessous).
        + (f"PÉRIODE DES SOURCES: {date_range}\n" if date_range else "")
        + f"\n=== EXTRAITS D'ARTICLES ({len(article_excerpts)}) ===\n{articles_block}\n\n"
        f"=== RÉSULTATS DE RECHERCHE ({len(search_excerpts)}) ===\n{searches_block}"
    )

    logger.info(f"    → appel LLM deep dive ({len(article_excerpts)} articles, {len(search_excerpts)} recherches)...")
    result = await _llm(
        llm_client, model,
        system=(
            "Tu es un expert journaliste du Monde diplomatique. Tu reçois des extraits d'articles "
            "et des résultats de recherches approfondies sur un sujet d'actualité. "
            "Détermine d'abord la catégorie la plus appropriée pour CE sujet parmi la liste donnée. "
            "Rédige ensuite une analyse complète, INTÉGRALEMENT EN FRANÇAIS même si les extraits "
            "d'articles et résultats de recherche fournis sont en anglais ou dans une autre langue "
            "(traduis, ne recopie jamais un texte source tel quel) : faits précis avec chiffres, "
            "contexte historique, enjeux, perspectives EU et US, ce qu'il faut retenir. "
            "Ton oral, sérieux, factuel, sans liste ni tirets. "
            "Appelle generate_analysis."
        ),
        user=user_content,
        tool=_DEEP_DIVE_TOOL,
    )

    title_fr  = result.get("title_fr",      topic["title"])
    category  = result.get("category",      topic["category"])
    if category not in CATEGORIES:
        # Constaté sur données réelles : le LLM peut renvoyer une catégorie hors de
        # l'enum du schéma malgré la contrainte (ex: "Santé & Environnement", fusion
        # halluciné de deux catégories valides distinctes) — l'enum JSON schema n'est
        # apparemment pas strictement appliqué par ce backend.
        logger.warning(f"  catégorie hors liste reçue du LLM: {category!r} -> repli sur {topic['category']!r}")
        category = topic["category"] if topic["category"] in CATEGORIES else "International"
    summary   = result.get("summary",       topic["title"])
    deep_dive = result.get("deep_dive",     "")
    watch     = result.get("what_to_watch", "")
    # Construites depuis les vrais articles appariés (`matched`), PAS devinées par le
    # LLM (l'ancien champ "sources" du tool ne donnait que des noms de médias, jamais
    # d'URL, et le LLM ne voit de toute façon jamais les URLs sources dans son prompt
    # — voir article_excerpts ci-dessus, qui n'inclut que publisher+titre+date).
    seen_urls: set[str] = set()
    sources = []
    for a in matched:
        if a.url in seen_urls:
            continue
        seen_urls.add(a.url)
        sources.append({"name": a.publisher, "url": a.url})
    logger.info(f"  titre_fr : {title_fr}")
    if category != topic["category"]:
        logger.info(f"  catégorie: {topic['category']} -> {category} (corrigée par le LLM)")
    logger.info(f"  résumé   : {summary}")
    logger.info(f"  analyse  : {deep_dive[:200]}{'…' if len(deep_dive) > 200 else ''}")
    if watch:
        logger.info(f"  à suivre : {watch}")
    if sources:
        logger.info(f"  sources  : {', '.join(s['name'] for s in sources)}")

    title_fr, summary, deep_dive, watch = await _proofread(
        title_fr, summary, deep_dive, watch, llm_client, model
    )

    return {
        "title":        title_fr,
        "category":     category,
        "importance":   topic.get("importance", 5),
        "date_range":   date_range,
        "summary":      summary,
        "deep_dive":    deep_dive,
        "what_to_watch": watch,
        "sources":      sources,
    }


def _keep_if_intact(champ: str, original: str, relu: str | None,
                    seuil: float = 0.7) -> str:
    """Garde la version relue seulement si elle n'a pas perdu de contenu.

    La relecture ne corrige que des artefacts ponctuels : sa sortie doit faire
    la même longueur que l'entrée. Or elle rend parfois un fragment — constaté
    sur 6 analyses d'une édition, réduites à quelques dizaines de caractères
    ouvrant sur un guillemet ('"Le conflit à Gaza') alors que l'original en
    faisait 1400. Le garde-fou ne couvrait que la réponse vide, si bien qu'un
    fragment écrasait l'analyse complète et arrivait tel quel dans le bulletin.
    """
    if not relu:
        return original
    if original and len(relu) < len(original) * seuil:
        logger.warning(
            f"  relecture ignorée ({champ}): {len(relu)} car. contre {len(original)} "
            f"à l'origine — troncature, pas une correction"
        )
        return original
    return relu


async def _proofread(title: str, summary: str, deep_dive: str, watch: str,
                      llm_client: openai.OpenAI, model: str) -> tuple[str, str, str, str]:
    """Relecture systématique du texte généré (title/summary/deep_dive/watch) pour
    rattraper les artefacts ponctuels de génération constatés sur données réelles avec
    ce backend quantifié (w4a16) : caractères d'un autre alphabet insérés au milieu
    d'un mot français, mots parasites incohérents. Champs nommés (pas un tableau
    positionné par un préfixe "1)/2)/...") : un format numéroté a déjà fait fuiter la
    numérotation dans le texte traduit (voir backfill_translate.py) — les champs JSON
    nommés éliminent ce risque. En cas d'échec de l'appel, renvoie le texte original
    inchangé plutôt que de risquer de perdre du contenu."""
    result = await _llm(
        llm_client, model,
        system=(
            "Tu relis des textes en français générés automatiquement par un autre "
            "modèle et corriges UNIQUEMENT les artefacts de génération : caractères "
            "d'un autre alphabet insérés au milieu d'un mot, mots parasites "
            "incohérents, fragments corrompus. Ne reformule rien d'autre — un champ "
            "déjà propre doit être renvoyé strictement identique. Appelle proofread_text."
        ),
        user=f"TITRE: {title}\nRÉSUMÉ: {summary}\nANALYSE: {deep_dive}\nÀ_SUIVRE: {watch}",
        tool=_PROOFREAD_TOOL,
    )
    if not result:
        return title, summary, deep_dive, watch
    new_title      = _keep_if_intact("titre",     title,      result.get("title"))
    new_summary    = _keep_if_intact("résumé",    summary,    result.get("summary"))
    new_deep_dive  = _keep_if_intact("analyse",   deep_dive,  result.get("deep_dive"))
    new_watch      = _keep_if_intact("à suivre",  watch,      result.get("what_to_watch"))
    if (new_title, new_summary, new_deep_dive, new_watch) != (title, summary, deep_dive, watch):
        logger.info("  relecture: artefact(s) corrigé(s)")
    return new_title, new_summary, new_deep_dive, new_watch


_PROOFREAD_MAPPING_TOOL = [{
    "type": "function",
    "function": {
        "name": "proofread_texts",
        "description": "Renvoie chaque texte corrigé de ses artefacts de génération.",
        "parameters": {
            "type": "object",
            "properties": {
                "textes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cle": {"type": "string", "description": "Clé du texte, recopiée à l'identique."},
                            "texte": {"type": "string", "description": "Texte corrigé (identique à l'entrée si déjà propre)."},
                        },
                        "required": ["cle", "texte"],
                    },
                }
            },
            "required": ["textes"],
        },
    },
}]


async def _proofread_mapping(textes: dict[str, str], llm_client: openai.OpenAI,
                             model: str, libelle: str = "texte") -> dict[str, str]:
    """Relit un ensemble de textes nommés, en un seul appel.

    Même objet que _proofread, mais pour les textes du bulletin qui ne sont pas
    rattachés à un sujet — flash, titre du jour, synthèses par catégorie. Ils
    n'étaient relus par personne : une synthèse a été publiée avec un caractère
    coréen inséré dans un mot ('faits divers 온vients'), exactement l'artefact
    que la relecture des sujets rattrape depuis toujours.

    Chaque texte est repris individuellement par _keep_if_intact : une clé
    absente de la réponse, ou raccourcie, garde sa version d'origine.
    """
    textes = {k: v for k, v in textes.items() if v}
    if not textes:
        return {}

    result = await _llm(
        llm_client, model,
        system=(
            "Tu relis des textes en français générés automatiquement par un autre "
            "modèle et corriges UNIQUEMENT les artefacts de génération : caractères "
            "d'un autre alphabet insérés au milieu d'un mot, mots parasites "
            "incohérents, fragments corrompus. Ne reformule rien d'autre — un texte "
            "déjà propre doit être renvoyé strictement identique, et sa clé recopiée "
            "telle quelle. Appelle proofread_texts."
        ),
        user="\n\n".join(f"[{cle}]\n{texte}" for cle, texte in textes.items()),
        tool=_PROOFREAD_MAPPING_TOOL,
    )

    relus = {
        t["cle"]: t["texte"]
        for t in result.get("textes", [])
        if isinstance(t, dict) and t.get("cle") and t.get("texte")
    }
    sortie = {
        cle: _keep_if_intact(f"{libelle} {cle}", original, relus.get(cle))
        for cle, original in textes.items()
    }
    if sortie != textes:
        logger.info(f"  relecture {libelle}: artefact(s) corrigé(s)")
    return sortie


async def _generate_flash(topics: list[dict],
                           llm_client: openai.OpenAI, model: str) -> tuple[str, str]:
    top = topics[:15]  # 6 phrases, répartition libre entre sujets — vivier élargi (était 10 pour 3 phrases/3 sujets fixes)
    topics_text = "\n".join(
        f"- [{t['category']}] {t['title']}: {t.get('summary', '')[:200]}"
        for t in top
    )
    logger.info(f"  → appel LLM flash ({len(top)} sujets)...")
    result = await _llm(
        llm_client, model,
        system=(
            "Rédige un titre accrocheur et un flash info du jour à partir des sujets principaux. "
            "Style factuel et direct. "
            "INTERDIT : toute formule d'introduction ('Ce soir', 'Bonsoir', 'Au programme'), "
            "toute formule de transition ('à suivre', 'restez avec nous', 'on en parle'). "
            "Commence directement par les faits. Le flash se termine sur un fait, pas une promesse. "
            "Appelle generate_flash."
        ),
        user=f"Sujets du jour:\n\n{topics_text}",
        tool=_FLASH_TOOL,
    )
    headline = result.get("headline", "L'actualité du jour")
    flash    = result.get("flash",    "")
    logger.info(f"[TITRE]  {headline}")
    logger.info(f"[FLASH]  {flash}")
    return headline, flash


def _assemble_bulletin(flash: str, headline: str, topics: list[dict],
                        category_summaries: dict[str, str] | None = None) -> dict:
    categories: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for t in topics:
        cat = t.get("category", "International")
        if cat not in categories:
            cat = "International"
        categories[cat].append(t)
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}
    return {
        "flash": flash,
        "headline": headline,
        "categories": categories,
        "category_summaries": category_summaries or {},
    }


async def _generate_category_summaries(categories: dict[str, list[dict]],
                                        llm_client: openai.OpenAI, model: str) -> dict[str, str]:
    """Un bulletin de synthèse par catégorie (pas par sujet) — ex: pour 'International'
    avec 5 sujets aujourd'hui, un paragraphe qui les résume ENSEMBLE, affiché sur la
    page wiki dates/<date>.md entre le titre de la catégorie et la liste des sujets.

    Génère une synthèse par catégorie individuellement (appel LLM séparé) pour éviter
    les problèmes de contexte trop long qui corrompaient les réponses (tokens de contrôle
    qui fuient dans le texte avec le modèle quantifié).
    """
    if not categories:
        return {}

    summaries = {}
    for cat, stories in categories.items():
        stories_text = "\n".join(f"- {s['title']}: {s.get('summary', '')[:200]}" for s in stories)
        user_content = f"### {cat} ({len(stories)} sujet(s))\n{stories_text}"

        logger.info(f"  → appel LLM synthèse catégorie '{cat}' ({len(stories)} sujets)...")
        result = await _llm(
            llm_client, model,
            system=(
                "Tu rédiges un bulletin de synthèse pour une catégorie d'actualité, à "
                "partir de la liste de ses sujets du jour. Un seul paragraphe qui couvre "
                "l'ensemble des sujets listés — pas un résumé sujet par sujet, une synthèse "
                "d'ensemble façon flash radio. Style factuel et direct, sans formule d'introduction. "
                "Appelle generate_category_summaries."
            ),
            user=user_content,
            tool=_CATEGORY_SUMMARY_TOOL,
        )

        for s in result.get("summaries", []):
            if s.get("category") and s.get("summary"):
                summaries[s["category"]] = s["summary"]
                break

    return summaries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def generate_bulletin(articles: list[RawArticle], search_client: SearchClient,
                             llm_client: openai.OpenAI, model: str) -> dict:
    logger.info(f"Génération bulletin: {len(articles)} articles")

    # Step 1: Cluster into topics (TF-IDF + DBSCAN — no LLM, instant)
    loop = asyncio.get_event_loop()
    topics = await loop.run_in_executor(None, _do_cluster_topics, articles)
    if not topics:
        logger.error("Aucun sujet identifié")
        return {}

    # Step 2-4: For each topic, generate queries, search, build deep dive — all in parallel.
    # LLM calls are serialized by _LLM_SEMAPHORE; searches run fully in parallel.
    total_searches = len(topics) * SEARCH_QUERIES_PER_TOPIC
    logger.info(f"[2/4] Enrichissement: {len(topics)} sujets × {SEARCH_QUERIES_PER_TOPIC} requêtes = {total_searches} recherches (parallèles)")

    async def _process_topic(topic: dict, i: int) -> dict:
        logger.info(f"  sujet [{i+1}/{len(topics)}] {topic['title']!r} ({topic['category']})")
        queries = await _generate_search_queries(topic, articles, llm_client, model)
        search_results = await search_client.search_many(queries)
        n_results = sum(1 for r in search_results if r.get("report"))
        logger.info(f"  sujet [{i+1}/{len(topics)}] {n_results}/{len(queries)} résultats → deep dive en cours")
        deep = await _generate_deep_dive(topic, articles, search_results, llm_client, model)
        logger.info(f"  sujet [{i+1}/{len(topics)}] OK")
        return deep

    enriched_topics = list(await asyncio.gather(
        *[_process_topic(t, i) for i, t in enumerate(topics)]
    ))

    logger.info(f"[2/4] Enrichissement terminé")
    logger.info(f"[3/4] Génération du flash...")
    # Step 5: Flash
    headline, flash = await _generate_flash(enriched_topics, llm_client, model)

    # Step 6: Assemble (catégories d'abord, la synthèse par catégorie en a besoin)
    bulletin = _assemble_bulletin(flash, headline, enriched_topics)
    logger.info(f"[3/4] Génération des synthèses par catégorie...")
    bulletin["category_summaries"] = await _generate_category_summaries(
        bulletin["categories"], llm_client, model
    )

    # Relecture des textes hors sujets : ils échappaient à _proofread, qui ne
    # couvre que les champs d'un sujet. Le flash et le titre passent en un appel,
    # les synthèses par catégorie en un second.
    #
    # Sous try/except : cette relecture n'est qu'une amélioration cosmétique du
    # bulletin, elle ne doit jamais pouvoir le faire perdre. Un KeyError dans son
    # propre message de log a déjà fait échouer une génération entière alors que
    # les 100 sujets étaient prêts.
    logger.info(f"[3/4] Relecture du flash et des synthèses...")
    try:
        entete = await _proofread_mapping(
            {"headline": bulletin["headline"], "flash": bulletin["flash"]},
            llm_client, model, libelle="entête",
        )
        bulletin["headline"] = entete.get("headline", bulletin["headline"])
        bulletin["flash"] = entete.get("flash", bulletin["flash"])
        if bulletin["category_summaries"]:
            bulletin["category_summaries"] = await _proofread_mapping(
                bulletin["category_summaries"], llm_client, model, libelle="catégorie",
            )
    except Exception as e:
        logger.exception(f"[3/4] Relecture du flash et des synthèses échouée, on garde le texte tel quel: {e}")

    logger.info(
        f"Bulletin généré: {sum(len(v) for v in bulletin['categories'].values())} sujets "
        f"en {len(bulletin['categories'])} catégories"
    )
    return bulletin


# ---------------------------------------------------------------------------
# Question answering (ChromaDB semantic search)
# ---------------------------------------------------------------------------

_SCORE_MIN = 0.40    # absolute floor — below this, always irrelevant
_SCORE_DELTA = 0.15  # keep only hits within this range of the best score


def _format_topic_hits(query: str, hits: list[dict]) -> str:
    """Format bulletin topic hits with title + summary."""
    most_recent_date = max((h["metadata"].get("date", "") for h in hits), default="")
    parts = [f"[{most_recent_date}] {query.upper()}", ""]
    for h in hits:
        meta = h["metadata"]
        title = meta.get("title", "")
        date = meta.get("date", "")
        # content = "title\nsummary\ndeep_dive" — extract summary (second paragraph)
        lines = h["content"].split("\n", 2)
        summary = lines[1].strip() if len(lines) > 1 else ""
        parts.append(f"• [{date}] [{meta.get('category', '')}] {title}")
        if summary:
            parts.append(f"  {summary}")
        parts.append("")
    return "\n".join(parts)


def _format_article_hits(query: str, hits: list[dict]) -> str:
    """Format raw article hits with title + excerpt."""
    parts = [f"ARTICLES RÉCENTS — {query.upper()}", ""]
    for h in hits:
        meta = h["metadata"]
        title = meta.get("title") or h["content"].split("\n")[0]
        publisher = meta.get("publisher", "")
        lines = h["content"].split("\n", 1)
        excerpt = lines[1].strip() if len(lines) > 1 else ""
        parts.append(f"• [{publisher}] {title}")
        if excerpt:
            parts.append(f"  {excerpt}")
        parts.append("")
    return "\n".join(parts)


_RECENCY_HALFLIFE_DAYS = 3.0  # score halves every 3 days


_WINDOW_DAYS = 7          # ChromaDB pre-filter: ignore content older than this
_RECENCY_HALFLIFE_DAYS = 3.0  # score halves every 3 days


def _item_ts(meta: dict, ref_ts: float) -> float:
    """Extract a Unix timestamp from a hit's metadata, with fallbacks."""
    # Articles: crawled_ts (int) or crawled_at (ISO string)
    ts = meta.get("crawled_ts")
    if ts is not None:
        return float(ts)
    crawled_at = meta.get("crawled_at", "")
    if crawled_at:
        try:
            return datetime.fromisoformat(crawled_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    # Topics: date (YYYY-MM-DD)
    date_str = meta.get("date", "")
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return ref_ts  # unknown age → treat as on-time


def _apply_recency(hits: list[dict], ref_ts: float,
                   halflife_days: float = _RECENCY_HALFLIFE_DAYS) -> list[dict]:
    """Re-score hits: semantic * 0.7 + recency_decay * 0.3, relative to ref_ts."""
    decay = 0.693 / (halflife_days * 86400)
    result = []
    for h in hits:
        ts = _item_ts(h["metadata"], ref_ts)
        age_s = abs(ref_ts - ts)
        recency = math.exp(-decay * age_s)
        combined = h["score"] * 0.7 + recency * 0.3
        result.append({**h, "score": round(combined, 3), "score_sem": h["score"], "recency": round(recency, 3)})
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


async def answer_question(query: str, bulletin: dict, search_client: SearchClient,
                           llm_client: openai.OpenAI, model: str,
                           ref_date: str | None = None) -> str:
    import vector_store

    loop = asyncio.get_event_loop()

    from datetime import timedelta
    if ref_date:
        try:
            ref_dt = datetime.strptime(ref_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            ref_dt = datetime.now(timezone.utc)
    else:
        ref_dt = datetime.now(timezone.utc)
    ref_ts = ref_dt.timestamp()

    # Pre-filter: only content within _WINDOW_DAYS before the reference date
    window_start = (ref_dt - timedelta(days=_WINDOW_DAYS)).strftime("%Y-%m-%d")
    window_start_ts = int((ref_dt - timedelta(days=_WINDOW_DAYS)).timestamp())

    # 1. Semantic search in bulletin topics (deep dives) — date-filtered
    topic_hits = await loop.run_in_executor(
        None, vector_store.search_topics, query, 10, window_start
    )
    topic_hits = _apply_recency(topic_hits, ref_ts)
    top_topic_score = topic_hits[0]["score"] if topic_hits else 0
    topic_threshold = max(_SCORE_MIN, top_topic_score - _SCORE_DELTA)
    relevant_topics = [h for h in topic_hits if h["score"] >= topic_threshold]

    # 2. Semantic search in raw articles — crawled_ts-filtered
    article_hits = await loop.run_in_executor(
        None, vector_store.search_articles, query, 10, window_start_ts
    )
    article_hits = _apply_recency(article_hits, ref_ts)
    top_article_score = article_hits[0]["score"] if article_hits else 0
    article_threshold = max(_SCORE_MIN, top_article_score - _SCORE_DELTA)
    relevant_articles = [h for h in article_hits if h["score"] >= article_threshold]

    for h in topic_hits:
        date = h["metadata"].get("date", "?")
        logger.info(f"  topic  score={h['score']:.3f} (sem={h.get('score_sem','?'):.3f} rec={h.get('recency','?'):.3f}) [{date}] : {h['metadata'].get('title','')[:60]}")
    for h in article_hits:
        crawled = h["metadata"].get("crawled_at", h["metadata"].get("crawled_ts", "?"))
        if isinstance(crawled, (int, float)):
            crawled = datetime.fromtimestamp(crawled, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(crawled, str) and "T" in crawled:
            crawled = crawled[:10]
        logger.info(f"  article score={h['score']:.3f} (sem={h.get('score_sem','?'):.3f} rec={h.get('recency','?'):.3f}) [{crawled}] : {h['metadata'].get('title','')[:60]}")
    logger.info(
        f"ChromaDB '{query[:50]}': {len(relevant_topics)}/{len(topic_hits)} topics "
        f"(seuil={topic_threshold:.2f}), "
        f"{len(relevant_articles)}/{len(article_hits)} articles (seuil={article_threshold:.2f})"
    )

    # Return directly if we have relevant results — no LLM needed
    # Articles first (most recent), then topics (deeper analysis from bulletin)
    if relevant_articles or relevant_topics:
        parts = []
        if relevant_articles:
            parts.append(_format_article_hits(query, relevant_articles))
        if relevant_topics:
            parts.append(_format_topic_hits(query, relevant_topics))
        return "\n".join(parts)

    # 3. Web search fallback (no relevant ChromaDB results)
    logger.info(f"Question '{query[:50]}' → pas de résultat ChromaDB, recherche web")
    search_results = await search_client.search_many(
        [query, f"{query} actualité récente"], "news"
    )
    reports = [r.get("report", "") for r in search_results if r.get("report")]
    if not reports:
        return "Je n'ai pas trouvé d'information sur ce sujet dans l'actualité récente."

    search_block = "\n\n".join(r[:800] for r in reports[:4])
    now = datetime.now(timezone.utc)
    result = await _llm(
        llm_client, model,
        system=(
            f"Nous sommes le {now.strftime('%A %d %B %Y à %Hh%M UTC')}. "
            "Tu es expert en actualités. Réponds à la question en français, "
            "ton oral, factuel, à partir des résultats de recherche fournis. "
            "Appelle answer_question avec needs_web_search=false."
        ),
        user=f"Question: {query}\n\nRecherches:\n{search_block}",
        tool=_ANSWER_TOOL,
    )
    return result.get("answer") or "Je n'ai pas trouvé d'information sur ce sujet dans l'actualité récente."
