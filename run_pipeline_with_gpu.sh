#!/usr/bin/env bash
# Lance un cycle du pipeline agent-news en libérant temporairement la VRAM du GPU
# partagé le temps STRICTEMENT nécessaire : voxcpm2-api en est un gros consommateur
# (~10,5 Go), suffisant pour empêcher le modèle d'embedding bge-m3 (vector_store.py)
# de charger sur GPU et le faire basculer sur CPU. Mais bge-m3, une fois chargé,
# reste résident en VRAM pour tout le cycle (pas de rechargement entre les étapes)
# — seule la toute première indexation (upsert_articles/upsert_publishers, juste
# après le crawl) a besoin de VRAM libre pour charger le modèle. Les appels
# d'embedding plus tardifs (consolidation, ~100 requêtes courtes) réutilisent le
# modèle déjà chargé ; s'ils retombaient malgré tout sur CPU une fois voxcpm2
# redémarré, le coût resterait négligeable (courtes requêtes, pas 1800 articles).
# On redémarre donc voxcpm2-api dès que l'indexation initiale est finie (log
# "[2/4] Clustering"), PAS en attendant la fin complète du pipeline (~35-40 min
# d'enrichissement LLM ensuite, sans rapport avec le GPU).
#
# Le conteneur agent-news est redémarré avant de déclencher le pipeline : un cycle
# déjà en cours tourne comme tâche asyncio dans le process principal, pas moyen de
# le tuer individuellement autrement qu'en relançant le conteneur (contrairement au
# pipeline contes-agent, un script CLI qu'on peut tuer par PID — voir
# contes-agent/reindex_with_gpu.sh, même principe sinon).
#
# voxcpm2-api est TOUJOURS redémarré en sortie (succès, échec, Ctrl-C) via le trap
# ci-dessous (docker start sur un conteneur déjà démarré ne fait rien — sans danger
# si le redémarrage anticipé a déjà eu lieu).
set -uo pipefail

CONTAINER=agent-news
VOXCPM_CONTAINER=voxcpm2-api
AGENT_NEWS_URL=http://localhost:8085

restart_voxcpm() {
    echo "[gpu] Redémarrage de ${VOXCPM_CONTAINER}..."
    docker start "${VOXCPM_CONTAINER}"
}
trap restart_voxcpm EXIT

echo "[gpu] Arrêt de ${VOXCPM_CONTAINER} pour libérer de la VRAM..."
docker stop "${VOXCPM_CONTAINER}"

echo "[gpu] Redémarrage de ${CONTAINER} pour repartir sur un chargement GPU propre..."
docker restart "${CONTAINER}"

echo "[gpu] Attente du démarrage du service..."
until curl -sf -o /dev/null "${AGENT_NEWS_URL}/bulletin"; do
    sleep 2
done

echo "[gpu] Déclenchement du pipeline..."
curl -sf -X POST "${AGENT_NEWS_URL}/pipeline/run"
echo

echo "[gpu] Attente de la fin de l'indexation initiale (seule étape sensible à la VRAM)..."
docker logs -f "${CONTAINER}" 2>&1 | grep --line-buffered -m1 "\[2/4\] Clustering"

echo "[gpu] Indexation terminée — redémarrage anticipé de ${VOXCPM_CONTAINER}..."
docker start "${VOXCPM_CONTAINER}"

echo "[gpu] Pipeline toujours en cours en arrière-plan (enrichissement LLM + consolidation, sans impact GPU) — suivi jusqu'à la régénération du wiki..."
docker logs -f "${CONTAINER}" 2>&1 | grep --line-buffered -m1 -E "Wiki régénéré|Régénération du wiki échouée"

echo "[gpu] Terminé."
