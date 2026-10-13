(function () {
  const LANES = ["jev", "laya", "vega", "openai"];
  const NAME = { jev: "Jev", laya: "Laya", vega: "Vega", openai: "OpenAI" };

  const datasetSelect = document.getElementById("dataset-select");
  const countSelect = document.getElementById("count-select");
  const typeFilterSelect = document.getElementById("type-filter");
  const runBtn = document.getElementById("run-btn");
  const errorBanner = document.getElementById("error-banner");

  const benchStatus = document.getElementById("bench-status");
  const benchProgressFill = document.getElementById("bench-progress-fill");
  const ticketLabel = document.getElementById("ticket-label");
  const ticketFeed = document.getElementById("ticket-feed");
  const qtableBody = document.getElementById("qtable-body");
  const resultsSection = document.getElementById("results");
  const resultsBody = document.getElementById("results-body");
  const resultsFile = document.getElementById("results-file");
  const exportXlsx = document.getElementById("export-xlsx");
  document.getElementById("export-pdf").addEventListener("click", () => window.print());

  const scoreEls = {};
  LANES.forEach((lane) => {
    scoreEls[lane] = {
      pct: document.getElementById(`score-pct-${lane}`),
      frac: document.getElementById(`score-frac-${lane}`),
      bar: document.getElementById(`score-bar-${lane}`),
      conf: document.getElementById(`score-conf-${lane}`),
      tokens: document.getElementById(`score-tokens-${lane}`),
      latency: document.getElementById(`score-latency-${lane}`),
    };
  });

  const firstDataset = datasetSelect.querySelector(".opt");
  let selectedDataset = firstDataset ? firstDataset.dataset.value : null;
  let selectedCount = null; // null = every item
  let running = false;
  let eventSource = null;
  let totalItems = 0;
  let feedCount = 0;

  let questionIds = [];
  let questionTypes = {};
  let scoreLevels = {};
  let typeFilter = "all";
  const qTotals = {};
  LANES.forEach((lane) => (qTotals[lane] = {}));
  const confSum = {}, confN = {}, latencySum = {}, latencyN = {};
  LANES.forEach((lane) => {
    confSum[lane] = 0; confN[lane] = 0; latencySum[lane] = 0; latencyN[lane] = 0;
  });

  let currentItem = null;

  function setActive(group, value) {
    group.querySelectorAll(".opt").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === String(value));
    });
  }

  datasetSelect.addEventListener("click", (e) => {
    const btn = e.target.closest(".opt");
    if (!btn || running) return;
    selectedDataset = btn.dataset.value;
    setActive(datasetSelect, selectedDataset);
    benchStatus.textContent = btn.dataset.description;
  });

  countSelect.addEventListener("click", (e) => {
    const btn = e.target.closest(".opt");
    if (!btn || running) return;
    selectedCount = btn.dataset.value === "all" ? null : parseInt(btn.dataset.value, 10);
    setActive(countSelect, btn.dataset.value);
  });

  typeFilterSelect.addEventListener("click", (e) => {
    const btn = e.target.closest(".opt");
    if (!btn) return;
    typeFilter = btn.dataset.value;
    setActive(typeFilterSelect, typeFilter);
    applyTypeFilter();
  });

  function applyTypeFilter() {
    ticketFeed.querySelectorAll(".ticket-row[data-qtype]").forEach((row) => {
      row.style.display = typeFilter === "all" || row.dataset.qtype === typeFilter ? "" : "none";
    });
  }

  function clearError() {
    errorBanner.classList.add("hidden");
    errorBanner.textContent = "";
  }

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
  }

  function setRunning(isRunning) {
    running = isRunning;
    datasetSelect.querySelectorAll(".opt").forEach((b) => (b.disabled = isRunning));
    countSelect.querySelectorAll(".opt").forEach((b) => (b.disabled = isRunning));
    runBtn.disabled = isRunning;
    runBtn.textContent = isRunning ? "Running..." : "Run benchmark";
  }

  function resetUI() {
    clearError();
    benchStatus.textContent = "Starting...";
    benchProgressFill.style.width = "0%";
    ticketLabel.textContent = "";
    ticketFeed.innerHTML = "";
    qtableBody.innerHTML = "";
    resultsSection.classList.add("hidden");
    resultsBody.innerHTML = "";
    resultsFile.textContent = "";
    feedCount = 0;
    currentItem = null;
    questionIds = [];

    LANES.forEach((lane) => {
      qTotals[lane] = {};
      confSum[lane] = 0; confN[lane] = 0; latencySum[lane] = 0; latencyN[lane] = 0;
      scoreEls[lane].pct.textContent = "—";
      scoreEls[lane].frac.textContent = "0 / 0 correct decisions";
      scoreEls[lane].bar.style.width = "0%";
      scoreEls[lane].conf.textContent = "—";
      scoreEls[lane].tokens.textContent = "—";
      scoreEls[lane].latency.textContent = "—";
    });
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function buildQuestionTable() {
    qtableBody.innerHTML = "";
    questionIds.forEach((qid) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, qid));
      tr.appendChild(el("td", "qtype", questionTypes[qid]));
      LANES.forEach((lane) => {
        const td = el("td", `acc lane-${lane}`, "—");
        td.id = `q-${qid}-${lane}-acc`;
        tr.appendChild(td);
      });
      const dis = el("td", null, "0");
      dis.id = `q-${qid}-disagree`;
      tr.appendChild(dis);
      const h2h = el("td", "h2h-cell", "—");
      h2h.id = `q-${qid}-h2h`;
      tr.appendChild(h2h);
      qtableBody.appendChild(tr);
    });
  }

  async function run() {
    if (!selectedDataset) return;
    resetUI();
    setRunning(true);

    let resp;
    try {
      resp = await fetch("/api/benchmark/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: selectedDataset, limit: selectedCount }),
      });
    } catch (err) {
      showError("Could not reach the server.");
      setRunning(false);
      return;
    }

    const data = await resp.json();
    if (!resp.ok) {
      showError(data.error || "Failed to start benchmark.");
      setRunning(false);
      return;
    }

    connectStream();
  }

  function connectStream() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource("/api/benchmark/stream");
    eventSource.onmessage = (e) => handleEvent(JSON.parse(e.data));
    eventSource.onerror = () => eventSource.close();
  }

  function updateOverallScoreboard() {
    LANES.forEach((lane) => {
      let hits = 0, n = 0;
      questionIds.forEach((qid) => {
        const t = qTotals[lane][qid];
        if (t) { hits += t.hits; n += t.n; }
      });
      const pct = n > 0 ? (hits / n) * 100 : 0;
      scoreEls[lane].pct.textContent = n > 0 ? `${pct.toFixed(1)}%` : "—";
      scoreEls[lane].frac.textContent = `${hits} / ${n} correct decisions`;
      scoreEls[lane].bar.style.width = `${pct}%`;
      scoreEls[lane].conf.textContent = confN[lane] > 0 ? `${((confSum[lane] / confN[lane]) * 100).toFixed(0)}%` : "—";
    });
  }

  function updateQuestionRow(qid) {
    LANES.forEach((lane) => {
      const t = qTotals[lane][qid];
      const cell = document.getElementById(`q-${qid}-${lane}-acc`);
      if (!cell || !t) return;
      cell.textContent = t.n > 0 ? `${((t.hits / t.n) * 100).toFixed(0)}% (${t.hits}/${t.n})` : "—";
    });
  }

  function formatMs(ms) {
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
  }

  function updateLatency() {
    LANES.forEach((lane) => {
      scoreEls[lane].latency.textContent = latencyN[lane] > 0 ? formatMs(latencySum[lane] / latencyN[lane]) : "—";
    });
  }

  function updateTokens(tokenTotals, itemsDone) {
    LANES.forEach((lane) => {
      const t = tokenTotals[lane];
      if (!t) return;
      const total = t.input_tokens + t.output_tokens;
      const perItem = itemsDone > 0 ? Math.round(total / itemsDone) : 0;
      scoreEls[lane].tokens.textContent = `${total.toLocaleString()} (${perItem.toLocaleString()} / item)`;
      scoreEls[lane].tokens.title = `input ${t.input_tokens.toLocaleString()} · output ${t.output_tokens.toLocaleString()}`;
    });
  }

  function formatPredicted(qid, value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "yes" : "no";
    const levels = scoreLevels[qid];
    if (levels && typeof value === "number") {
      // Rubric levels read "cosmetic: ..." in some datasets; the part before the colon is enough here.
      const label = String(levels[value] || "").split(":")[0];
      return `${value} (${label.length > 18 ? label.slice(0, 17) + "…" : label})`;
    }
    return String(value);
  }

  function renderItemCard(item) {
    const card = el("div", "ticket-card");
    const head = el("div", "ticket-head");
    head.appendChild(el("span", "ticket-id", `#${item.index + 1} · ${item.id}`));
    item.tags.forEach((tag) => head.appendChild(el("span", "ticket-tag", tag)));
    card.appendChild(head);
    card.appendChild(el("div", "ticket-message", item.text));

    const rows = el("div", "ticket-rows");
    questionIds.forEach((qid) => {
      const answer = item.answers[qid];
      if (!answer) return;
      const row = el("div", "ticket-row");
      row.dataset.qtype = answer.qtype || questionTypes[qid] || "";
      if (typeFilter !== "all" && row.dataset.qtype !== typeFilter) row.style.display = "none";
      const qname = el("span", "qname", qid);
      qname.appendChild(el("span", "gold", ` = ${formatPredicted(qid, answer.correct)}`));
      row.appendChild(qname);

      LANES.forEach((lane) => {
        const p = answer.picks[lane];
        const span = el("span");
        if (p) {
          span.className = p.hit ? "pick-hit" : "pick-miss";
          const shown = p.refused ? "refused" : p.errored ? "error" : formatPredicted(qid, p.predicted);
          span.textContent = `${NAME[lane]}: ${shown} ${p.hit ? "✓" : "✗"} `;
          if (!p.refused && !p.errored) span.appendChild(el("span", "pick-conf", `(${(p.confidence * 100).toFixed(0)}%)`));
        }
        row.appendChild(span);
      });
      rows.appendChild(row);
    });
    card.appendChild(rows);

    if (item.tokens || item.latency) {
      const meta = el("div", "ticket-row ticket-tokens");
      meta.appendChild(el("span"));
      LANES.forEach((lane) => {
        const parts = [];
        if (item.latency && item.latency[lane] != null) parts.push(formatMs(item.latency[lane]));
        const t = item.tokens && item.tokens[lane];
        if (t) parts.push(`${t.input_tokens} in / ${t.output_tokens} out tok`);
        meta.appendChild(el("span", null, parts.length ? `${NAME[lane]}: ${parts.join(" · ")}` : ""));
      });
      card.appendChild(meta);
    }

    ticketFeed.insertBefore(card, ticketFeed.firstChild);
    feedCount += 1;
    while (ticketFeed.children.length > 40) ticketFeed.removeChild(ticketFeed.lastChild);
  }

  // ---- Final results -------------------------------------------------------

  const HIGHER_BETTER = new Set(["accuracy", "macro_f1", "precision", "recall", "f1", "auc", "within_one", "qwk"]);
  const LOWER_BETTER = new Set(["ece", "brier", "mae", "p50", "p95", "per_item", "refusals", "errors"]);
  const PERCENT = new Set(["accuracy", "avg_confidence", "within_one", "precision", "recall"]);
  const METRIC_LABEL = {
    accuracy: "accuracy", avg_confidence: "avg confidence", ece: "calibration error (ECE)",
    refusals: "refusals", errors: "failed calls", p50: "latency p50", p95: "latency p95",
    per_item: "tokens / item", macro_f1: "macro F1", precision: "precision", recall: "recall",
    f1: "F1", brier: "Brier score", auc: "AUC", mae: "mean abs error", within_one: "within one level",
    qwk: "weighted kappa (QWK)",
  };
  const QUESTION_METRICS = {
    choice: ["accuracy", "macro_f1", "ece"],
    noul: ["accuracy", "precision", "recall", "f1", "auc", "brier", "ece"],
    score: ["accuracy", "within_one", "mae", "qwk", "ece"],
  };

  function formatMetric(key, value) {
    if (value === null || value === undefined) return "—";
    if (PERCENT.has(key)) return `${(value * 100).toFixed(1)}%`;
    if (key === "p50" || key === "p95") return formatMs(value);
    if (key === "per_item") return Math.round(value).toLocaleString();
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(3);
  }

  function metricTable(title, rows, lanes) {
    // rows: [{key, label, values: {lane: number|null}}]
    const wrap = el("div", "metric-block");
    wrap.appendChild(el("h3", null, title));
    const scroll = el("div", "table-scroll");
    const table = el("table", "qtable metric-table");
    const head = el("tr");
    head.appendChild(el("th", null, "metric"));
    lanes.forEach((lane) => head.appendChild(el("th", null, NAME[lane])));
    table.appendChild(el("thead")).appendChild(head);
    const body = el("tbody");
    rows.forEach((row) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, row.label));
      const nums = lanes.map((lane) => row.values[lane]).filter((v) => v !== null && v !== undefined);
      const better = HIGHER_BETTER.has(row.key) ? Math.max : LOWER_BETTER.has(row.key) ? Math.min : null;
      const best = better && nums.length > 1 ? better(...nums) : null;
      lanes.forEach((lane) => {
        const v = row.values[lane];
        const td = el("td", `lane-${lane}`, formatMetric(row.key, v));
        if (best !== null && v === best && new Set(nums).size > 1) td.classList.add("best");
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    scroll.appendChild(table);
    wrap.appendChild(scroll);
    return wrap;
  }

  function renderResults(event) {
    const m = event.metrics;
    const lanes = LANES.filter((lane) => m.overall[lane]);
    resultsBody.innerHTML = "";
    resultsFile.textContent = `saved to ${event.results_file}`;
    const runName = event.results_file.split("/").pop().replace(/\.jsonl$/, "");
    exportXlsx.href = `/api/results/${encodeURIComponent(runName)}.xlsx`;

    const overallRows = ["accuracy", "avg_confidence", "ece", "refusals", "errors"].map((key) => ({
      key, label: METRIC_LABEL[key], values: Object.fromEntries(lanes.map((l) => [l, m.overall[l][key]])),
    }));
    ["p50", "p95"].forEach((key) => overallRows.push({
      key, label: METRIC_LABEL[key], values: Object.fromEntries(lanes.map((l) => [l, m.overall[l].latency_ms[key]])),
    }));
    overallRows.push({
      key: "per_item", label: METRIC_LABEL.per_item,
      values: Object.fromEntries(lanes.map((l) => [l, m.overall[l].tokens.per_item])),
    });

    const grid = el("div", "metric-grid");
    grid.appendChild(metricTable(`Overall (${m.items} items)`, overallRows, lanes));

    Object.entries(m.questions).forEach(([qid, q]) => {
      const rows = QUESTION_METRICS[q.type].map((key) => ({
        key, label: METRIC_LABEL[key], values: Object.fromEntries(lanes.map((l) => [l, q.models[l][key]])),
      }));
      if (q.type === "choice") {
        const classes = Object.keys(q.models[lanes[0]].per_class);
        classes.forEach((cls) => rows.push({
          key: "f1", label: `F1 · ${cls} (${q.models[lanes[0]].per_class[cls].support})`,
          values: Object.fromEntries(lanes.map((l) => [l, q.models[l].per_class[cls].f1])),
        }));
      }
      grid.appendChild(metricTable(`${qid} (${q.type})`, rows, lanes));
    });

    const tagRows = Object.entries(m.by_tag).map(([tag, t]) => ({
      key: "accuracy", label: `${tag} (${t.items})`,
      values: Object.fromEntries(lanes.map((l) => [l, t.models[l].accuracy])),
    }));
    if (tagRows.length) grid.appendChild(metricTable("Accuracy by tag", tagRows, lanes));

    resultsBody.appendChild(grid);
    resultsSection.classList.remove("hidden");
  }

  function handleEvent(event) {
    switch (event.type) {
      case "status":
        benchStatus.textContent = event.message;
        break;

      case "model_unavailable":
        showError(`${NAME[event.model] || event.model} unavailable: ${event.message}`);
        break;

      case "model_error":
        showError(`${NAME[event.model] || event.model} failed on item ${event.item + 1}: ${event.message}`);
        break;

      case "benchmark_started":
        totalItems = event.items;
        questionIds = Object.keys(event.questions);
        questionTypes = event.questions;
        scoreLevels = event.score_levels || {};
        questionIds.forEach((qid) => LANES.forEach((lane) => (qTotals[lane][qid] = { hits: 0, n: 0 })));
        buildQuestionTable();
        benchStatus.textContent = `Running ${event.title}: ${event.items} items...`;
        break;

      case "item_start":
        ticketLabel.textContent = `— item ${event.item + 1} / ${event.total}`;
        benchProgressFill.style.width = `${(event.item / event.total) * 100}%`;
        currentItem = { index: event.item, id: event.id, text: event.text, tags: event.tags, answers: {} };
        break;

      case "item_timing":
        LANES.forEach((lane) => {
          const ms = event.latency_ms[lane];
          if (ms == null) return;
          latencySum[lane] += ms;
          latencyN[lane] += 1;
        });
        if (currentItem && currentItem.index === event.item) currentItem.latency = event.latency_ms;
        updateLatency();
        break;

      case "answer": {
        const qid = event.question;
        LANES.forEach((lane) => {
          const p = event.picks[lane];
          if (!p) return;
          qTotals[lane][qid] = event.totals[lane];
          if (p.refused || p.errored) return;
          confSum[lane] += p.confidence;
          confN[lane] += 1;
        });
        const disCell = document.getElementById(`q-${qid}-disagree`);
        if (disCell) disCell.textContent = String(event.disagreements);
        const h2hCell = document.getElementById(`q-${qid}-h2h`);
        if (h2hCell) {
          const w = event.disagreement_wins;
          h2hCell.textContent = event.disagreements > 0
            ? LANES.filter((lane) => w[lane] != null).map((lane) => `${NAME[lane]} ${w[lane]}`).join(" / ")
            : "—";
        }
        updateQuestionRow(qid);
        updateOverallScoreboard();
        if (currentItem && currentItem.index === event.item) currentItem.answers[qid] = event;
        break;
      }

      case "item_done":
        if (currentItem && currentItem.index === event.item) {
          currentItem.tokens = event.tokens;
          renderItemCard(currentItem);
        }
        updateTokens(event.token_totals, event.item + 1);
        break;

      case "done":
        benchProgressFill.style.width = "100%";
        benchStatus.textContent = `Done: ${totalItems} items in ${event.elapsed.toFixed(1)}s`;
        renderResults(event);
        setRunning(false);
        break;

      case "error":
        showError(event.message);
        benchStatus.textContent = "Error";
        setRunning(false);
        break;

      case "stream_end":
        if (eventSource) eventSource.close();
        break;

      default:
        break;
    }
  }

  runBtn.addEventListener("click", run);
})();
