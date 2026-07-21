// Recherche sémantique de l'accueil : appelle /api/wiki/search (voir wiki_api.py,
// réutilise directement vector_store.search_subjects) plutôt que de dupliquer un
// index de recherche côté site statique.
(function () {
  function debounce(fn, delay) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  function renderResults(container, data) {
    const subjects = data.subjects || [];
    if (!subjects.length) {
      container.innerHTML = "<p><em>Aucun résultat.</em></p>";
      return;
    }
    let html = "<ul>";
    for (const s of subjects) {
      html += `<li><a href="${escapeHtml(s.url)}">${escapeHtml(s.title)}</a>`;
      html += ` <em>(${escapeHtml(s.category)})</em>`;
      if (s.summary) html += ` — ${escapeHtml(s.summary)}`;
      html += "</li>";
    }
    html += "</ul>";
    container.innerHTML = html;
  }

  function runSearch(input, results) {
    const q = input.value.trim();
    if (!q) {
      results.innerHTML = "";
      return;
    }
    results.innerHTML = "<p><em>Recherche…</em></p>";
    fetch("/api/wiki/search?q=" + encodeURIComponent(q))
      .then((r) => r.json())
      .then((data) => renderResults(results, data))
      .catch(() => {
        results.innerHTML = "<p><em>Recherche indisponible pour le moment.</em></p>";
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    const input = document.getElementById("wiki-search-input");
    const results = document.getElementById("wiki-search-results");
    if (!input || !results) return;
    input.addEventListener("input", debounce(() => runSearch(input, results), 300));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        runSearch(input, results);
      }
    });
  });
})();
