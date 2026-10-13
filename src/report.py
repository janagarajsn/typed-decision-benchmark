"""One-page performance dashboard (JPG) across saved runs, drawn with matplotlib.

By default it uses the latest full run (no item limit) of each dataset in results/, falling
back to the latest run of any size when a dataset has no full run:

    python src/report.py                         # writes results/performance_dashboard.jpg
    python src/report.py results/a.jsonl ...     # only these runs
    python src/report.py --out /tmp/board.jpg

The page is laid out on a 1600-unit-wide canvas (y grows downward) and saved at 2x, so one
unit is one CSS pixel of the equivalent web page. Every number is recomputed from the
records with metrics.compute; the "Key findings" sentences are generated from those numbers.
"""
import argparse
import io
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from benchmark_engine import LANES, RESULTS_DIR  # noqa: E402
from export import NAMES  # noqa: E402
from metrics import _calibration, _percentile, compute, load_run, pick_and_confidence  # noqa: E402

DEFAULT_OUT = RESULTS_DIR / "performance_dashboard.jpg"
WIDTH, MARGIN, GAP, SIDE = 1600, 48, 16, 486
# Categorical slots validated together on the light surface (CVD and normal-vision separation);
# every bar also carries a printed value, since the green sits below 3:1 contrast.
COLOR = {"jev": "#2a78d6", "laya": "#1baf7a", "vega": "#6250d6", "openai": "#eb6834"}
KIND = {"jev": "cloud API", "laya": "local", "vega": "local", "openai": "cloud API"}
PAGE, PANEL, BORDER, RULE = "#f4f3f0", "#ffffff", "#e6e5e1", "#f0efec"
TEXT, TEXT2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#c3c2b7"
# Sequential blue ramp for the accuracy heatmap: (lower bound, fill, ink).
HEAT = [(0.0, "#f3f2ef", TEXT), (0.3, "#cde2fb", TEXT), (0.5, "#9ec5f4", TEXT), (0.7, "#6da7ec", TEXT),
        (0.85, "#3987e5", "#ffffff"), (0.95, "#256abf", "#ffffff"), (0.99, "#184f95", "#ffffff")]


def _first_font(*families):
    # matplotlib warns about every missing name in a family list, so pick one installed font up front.
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in families if f in installed), families[-1])


SANS = _first_font("Arial", "Liberation Sans", "DejaVu Sans")
MONO = _first_font("Menlo", "DejaVu Sans Mono")


def pick_runs(paths=None):
    """{dataset: (path, run, records)} in a stable order."""
    if paths:
        chosen = {}
        for p in paths:
            run, records = load_run(p)
            chosen[run["dataset"]] = (Path(p), run, records)
        return chosen
    best = {}
    for p in sorted(RESULTS_DIR.glob("*.jsonl")):  # names start with a timestamp, so later wins
        run, records = load_run(p)
        if not records:
            continue
        full = run.get("limit") is None
        current = best.get(run["dataset"])
        if current is None or full or not current[0]:
            best[run["dataset"]] = (full, p, run, records)
    return {d: v[1:] for d, v in sorted(best.items())}


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _pct(v, digits=1):
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _round_pct(v):
    return int(v * 100 + 0.5)  # half-up, so 40.5% shows as 41 like a person would round it


def summarize(runs):
    datasets = list(runs)
    models = [m for m in LANES if any(m in r[1]["models"] for r in runs.values())]
    per_ds, questions = {}, []
    pooled = {m: {"hits": 0, "n": 0, "conf": [], "lat": [], "tok": 0, "items": 0, "datasets": 0,
                  "easy": [0, 0], "hard": [0, 0], "refusals": 0, "errors": 0} for m in models}
    for d in datasets:
        _, run, records = runs[d]
        met = compute(run["questions"], records, run["models"])
        per_ds[d] = met
        for qid, qm in met["questions"].items():
            questions.append((d, qid, qm["type"], {m: qm["models"][m]["accuracy"] for m in run["models"]}))
        for m in run["models"]:
            o, p = met["overall"][m], pooled[m]
            p["datasets"] += 1
            p["hits"] += o["hits"]
            p["n"] += o["n"]
            p["refusals"] += o["refusals"]
            p["errors"] += o["errors"]
            p["tok"] += o["tokens"]["input_tokens"] + o["tokens"]["output_tokens"]
            p["items"] += len(records)
            for rec in records:
                if m in rec["latency_ms"]:
                    p["lat"].append(rec["latency_ms"][m])
                for qid, q in run["questions"].items():
                    ans = rec["answers"].get(m, {}).get(qid)
                    if ans and ans["type"] not in ("refusal", "error"):
                        pred, conf = pick_and_confidence(ans, q["type"])
                        p["conf"].append((conf, pred == rec["labels"][qid]))
            for tag in ("easy", "hard"):
                if tag in met["by_tag"]:
                    p[tag][0] += met["by_tag"][tag]["models"][m]["hits"]
                    p[tag][1] += met["by_tag"][tag]["models"][m]["n"]

    summary = {}
    for m in models:
        p = pooled[m]
        _, ece = _calibration(p["conf"])
        summary[m] = {
            "acc": p["hits"] / p["n"] if p["n"] else None, "hits": p["hits"], "n": p["n"],
            "conf": sum(c for c, _ in p["conf"]) / len(p["conf"]) if p["conf"] else None, "ece": ece,
            "p50": _percentile(p["lat"], 50), "p95": _percentile(p["lat"], 95),
            "tok": p["tok"] / p["items"] if p["items"] else None,
            "easy": p["easy"][0] / p["easy"][1] if p["easy"][1] else None,
            "hard": p["hard"][0] / p["hard"][1] if p["hard"][1] else None,
            "datasets": p["datasets"], "refusals": p["refusals"], "errors": p["errors"],
        }
    return {"datasets": datasets, "models": models, "per_ds": per_ds, "questions": questions,
            "summary": summary, "runs": runs}


def findings(s):
    S, models, per_ds = s["summary"], s["models"], s["per_ds"]
    ranked = sorted((m for m in models if S[m]["acc"] is not None), key=lambda m: -S[m]["acc"])
    out = []
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        wins = {m: 0 for m in models}
        for d in s["datasets"]:
            accs = {m: per_ds[d]["overall"][m]["accuracy"] for m in per_ds[d]["overall"]}
            top = max(accs.values())
            for m, v in accs.items():
                wins[m] += v == top
        if _pct(S[a]["acc"]) == _pct(S[b]["acc"]):
            head = f"{NAMES[a]} and {NAMES[b]} are effectively tied"
            body = f"Both at {_pct(S[a]['acc'])} ({S[a]['hits']:,} vs {S[b]['hits']:,} decisions correct)."
        else:
            head = f"{NAMES[a]} leads overall"
            body = f"{_pct(S[a]['acc'])}, ahead of {NAMES[b]} at {_pct(S[b]['acc'])}."
        if len(s["datasets"]) > 1:
            body += f" Best or tied per dataset (of {len(s['datasets'])}): " + ", ".join(
                f"{NAMES[m]} {wins[m]}" for m in sorted(models, key=lambda m: -wins[m]) if wins[m]
            ) + "."
        out.append((head, body))

        fast = min((m for m in models if S[m]["p50"] is not None), key=lambda m: S[m]["p50"])
        if fast != a:
            out.append(("Speed costs accuracy",
                        f"{NAMES[fast]} is fastest at {S[fast]['p50']:.0f} ms median vs {S[a]['p50']:.0f} ms for "
                        f"{NAMES[a]}, at {_pct(S[fast]['acc'])} vs {_pct(S[a]['acc'])} accuracy."))

    full = [(d, q, accs) for d, q, _, accs in s["questions"] if len(accs) >= 2]
    if full:
        d, q, accs = max(full, key=lambda r: max(r[2].values()) - min(r[2].values()))
        hi, lo = max(accs, key=accs.get), min(accs, key=accs.get)
        out.append(("Widest gap between models",
                    f"{q} ({s['runs'][d][1]['title']}): {NAMES[hi]} {_pct(accs[hi], 0)} vs {NAMES[lo]} {_pct(accs[lo], 0)}."))
        d, q, accs = min(full, key=lambda r: max(r[2].values()))
        best = max(accs, key=accs.get)
        out.append(("Hardest question",
                    f"{q} ({s['runs'][d][1]['title']}): the best score is {_pct(accs[best], 0)} ({NAMES[best]})."))

    drops = {m: S[m]["easy"] - S[m]["hard"] for m in models if S[m]["easy"] is not None and S[m]["hard"] is not None}
    if drops:
        out.append(("Hard items widen the gap", "Going from easy to hard items costs " + ", ".join(
            f"{NAMES[m]} {d * 100:.0f} pt" for m, d in sorted(drops.items(), key=lambda kv: kv[1])) + "."))

    cal = [m for m in models if S[m]["ece"] is not None]
    if cal:
        best = min(cal, key=lambda m: S[m]["ece"])
        over = max(cal, key=lambda m: S[m]["conf"] - S[m]["acc"])
        body = f"{NAMES[best]} is best calibrated (ECE {S[best]['ece']:.3f})."
        if S[over]["conf"] - S[over]["acc"] > 0.05:
            body += (f" {NAMES[over]} is the most overconfident: {_pct(S[over]['conf'], 0)} average confidence "
                     f"vs {_pct(S[over]['acc'], 0)} accuracy.")
        out.append(("Calibration", body))
    return out


class Canvas:
    """Draws in page units (1600 wide, y down) on a single full-figure axes."""

    def __init__(self, height):
        self.height = height
        self.fig = plt.figure(figsize=(WIDTH / 100, height / 100), dpi=100)
        self.fig.patch.set_facecolor(PAGE)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size=13, color=TEXT, weight="normal", ha="left", va="baseline", mono=False, style="normal"):
        # Font sizes are CSS pixels; at 100 units per inch one pixel is 0.72 pt.
        self.ax.text(x, y, s, fontsize=size * 0.72, color=color, fontweight=weight, ha=ha, va=va,
                     family=MONO if mono else SANS, fontstyle=style)

    def box(self, x, y, w, h, fill, radius=0, edge=None, lw=1.0, z=1):
        if radius:
            patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor=fill, edgecolor=edge or "none", linewidth=lw, zorder=z)
        else:
            patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge or "none", linewidth=lw, zorder=z)
        self.ax.add_patch(patch)

    def bar_up(self, x, base, w, h, fill):
        """Vertical bar with a 4-unit rounded top and a square baseline."""
        if h <= 0:
            return
        r = min(4, h)
        self.box(x, base - h, w, h, fill, radius=r, z=2)
        self.box(x, base - h / 2, w, h / 2, fill, z=2)

    def bar_right(self, x, y, w, h, fill):
        """Horizontal bar with a rounded end and a square start."""
        if w <= 0:
            return
        r = min(4, w)
        self.box(x, y, w, h, fill, radius=r, z=2)
        self.box(x, y, w / 2, h, fill, z=2)

    def line(self, x1, y1, x2, y2, color=GRID, lw=1.0, z=1):
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw * 0.72, zorder=z, solid_capstyle="butt")

    def dot(self, x, y, r, fill, edge=None, lw=2.0):
        self.ax.scatter([x], [y], s=(r * 0.72 * 2) ** 2, facecolor=fill, edgecolor=edge or fill,
                        linewidths=lw * 0.72, zorder=3)

    def panel(self, x, y, w, h, title, desc=None):
        self.box(x, y, w, h, PANEL, radius=12, edge=BORDER)
        self.text(x + 20, y + 30, title, size=15, weight="bold")
        if desc:
            self.text(x + 20, y + 49, desc, size=12.5, color=MUTED)

    def save(self, dpi_scale=2):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="jpg", dpi=100 * dpi_scale, facecolor=PAGE, pil_kwargs={"quality": 92})
        plt.close(self.fig)
        return buf.getvalue()


def _wrap(text, width):
    return textwrap.wrap(text, width) or [""]


def render(s):
    datasets, models, S, per_ds, runs = s["datasets"], s["models"], s["summary"], s["per_ds"], s["runs"]
    content = WIDTH - 2 * MARGIN
    left_w = content - SIDE - GAP
    side_x = MARGIN + left_w + GAP
    found = findings(s)
    has_tags = any(S[m]["easy"] is not None for m in models)

    # Section heights, computed up front so the canvas fits exactly.
    cards_h = 214
    find_lines = sum(1 + len(_wrap(b, 64)) for _, b in found)
    row2_h = max(440, 100 + find_lines * 20 + len(found) * 10)
    heat_h = 100 + 33 * len(s["questions"])
    ece_h = 100 + 33 * len(datasets)
    hbar_h = 72 + 34 * len(models)
    dumb_h = 110 + 42 * len(models) if has_tags else 0
    side_h = (dumb_h + GAP if has_tags else 0) + 2 * hbar_h + GAP
    row3_h = max(heat_h + GAP + ece_h, side_h)
    y_cards, y_row2 = 124, 124 + cards_h + GAP
    y_row3 = y_row2 + row2_h + GAP
    height = int(y_row3 + row3_h + 60)
    c = Canvas(height)

    # Header
    total_items = sum(len(runs[d][2]) for d in datasets)
    decisions = max((S[m]["n"] for m in models), default=0)
    dates = sorted(runs[d][1]["started_at"][:10] for d in datasets)
    c.text(MARGIN, 66, f"Typed-decision benchmark: {' vs '.join(NAMES[m] for m in models)}", size=30, weight="bold")
    partial = [m for m in models if S[m]["datasets"] < len(datasets)]
    coverage = ("every model answered the same items" if not partial else
                "not run on every dataset: " + ", ".join(NAMES[m] for m in partial))
    c.text(MARGIN, 94, f"{_plural(len(datasets), 'golden dataset')} · {total_items} items · "
                       f"up to {decisions:,} scored decisions per model · {coverage}" if partial else
                       f"{_plural(len(datasets), 'golden dataset')} · {total_items} items · "
                       f"{decisions:,} scored decisions per model · {coverage}", size=15, color=TEXT2)
    date_text = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"
    c.text(WIDTH - MARGIN, 72, f"Runs of {date_text}", size=13.5, color=TEXT2, ha="right")
    c.text(WIDTH - MARGIN, 94, "Accuracy = share of decisions matching the gold answer", size=13.5, color=TEXT2, ha="right")

    # Model cards, best first; models with equal displayed accuracy share a rank.
    ranked = sorted(models, key=lambda m: -(S[m]["acc"] or 0))
    cw = (content - GAP * (len(models) - 1)) / max(len(models), 1)
    for i, m in enumerate(ranked):
        x, st = MARGIN + i * (cw + GAP), S[m]
        rank = 1 + sum(1 for o in models if float(_pct(S[o]["acc"])[:-1] or 0) > float(_pct(st["acc"])[:-1] or 0))
        c.box(x, y_cards, cw, cards_h, PANEL, radius=12, edge=BORDER)
        c.box(x + 20, y_cards + 22, 12, 12, COLOR.get(m, MUTED), radius=3)
        c.text(x + 40, y_cards + 33, NAMES.get(m, m), size=16, weight="bold")
        c.text(x + 40 + 9.5 * len(NAMES.get(m, m)) + 12, y_cards + 33, KIND.get(m, ""), size=12, color=MUTED)
        c.box(x + cw - 50, y_cards + 18, 30, 20, RULE, radius=10)
        c.text(x + cw - 35, y_cards + 33, f"#{rank}", size=12, weight="bold", color=TEXT2, ha="center")
        c.text(x + 20, y_cards + 84, _pct(st["acc"]), size=40, weight="bold")
        sub = f"{st['hits']:,} / {st['n']:,} decisions correct"
        if st["datasets"] < len(datasets):
            sub += f" · ran on {st['datasets']} of {len(datasets)} datasets"
        c.text(x + 20, y_cards + 106, sub, size=13, color=TEXT2)
        c.box(x + 20, y_cards + 118, cw - 40, 8, "#efeeea", radius=4)
        c.box(x + 20, y_cards + 118, (cw - 40) * (st["acc"] or 0), 8, COLOR.get(m, MUTED), radius=4, z=2)
        stats = [("median latency", f"{st['p50']:.0f} ms" if st["p50"] is not None else "—"),
                 ("avg confidence", _pct(st["conf"], 0)),
                 ("calibration error", f"{st['ece']:.3f}" if st["ece"] is not None else "—"),
                 ("tokens / item", f"{st['tok']:.0f}" if st["tok"] is not None else "—")]
        half = (cw - 40 - 16) / 2
        for j, (label, value) in enumerate(stats):
            sx = x + 20 + (j % 2) * (half + 16)
            sy = y_cards + 146 + (j // 2) * 32
            c.line(sx, sy, sx + half, sy, RULE)
            c.text(sx, sy + 20, label, size=13, color=TEXT2)
            c.text(sx + half, sy + 20, value, size=13, weight="bold", ha="right")

    # Accuracy by dataset: grouped bars with a value on every bar.
    c.panel(MARGIN, y_row2, left_w, row2_h, "Accuracy by dataset", "All questions in each dataset; value = accuracy %")
    lx = MARGIN + 20
    for m in models:
        c.box(lx, y_row2 + 66, 12, 12, COLOR.get(m, MUTED), radius=3)
        c.text(lx + 18, y_row2 + 77, NAMES[m], size=13, color=TEXT2)
        lx += 32 + 8.5 * len(NAMES[m])
    px0, px1 = MARGIN + 64, MARGIN + left_w - 28
    top, base = y_row2 + 110, y_row2 + row2_h - 66
    ph = base - top
    for t in (0, 25, 50, 75, 100):
        y = base - ph * t / 100
        c.line(px0, y, px1, y, GRID)
        c.text(px0 - 8, y + 4, f"{t}%", size=12, color=MUTED, ha="right")
    c.line(px0, base, px1, base, AXIS)
    gw = (px1 - px0) / max(len(datasets), 1)
    bw = min(34, (gw - 30) / max(len(models), 1) - 2)
    for gi, d in enumerate(datasets):
        overall = per_ds[d]["overall"]
        gx = px0 + gi * gw + (gw - len(models) * (bw + 2) + 2) / 2
        for mi, m in enumerate(models):
            x = gx + mi * (bw + 2)
            if m not in overall:
                c.text(x + bw / 2, base - 6, "n/a", size=10, color=MUTED, ha="center")
                continue
            v = overall[m]["accuracy"] or 0
            c.bar_up(x, base, bw, ph * v, COLOR.get(m, MUTED))
            c.text(x + bw / 2, base - ph * v - 6, str(_round_pct(v)), size=11.5, weight="bold", color=TEXT2, ha="center")
        cx = px0 + gi * gw + gw / 2
        run = runs[d][1]
        qtypes = " / ".join(sorted({q["type"] for q in run["questions"].values()}))
        nq = len(run["questions"])
        c.text(cx, base + 22, run["title"], size=13.5, weight="bold", ha="center")
        c.text(cx, base + 40, f"{nq} question{'s' if nq != 1 else ''} · {qtypes}", size=11.5, color=MUTED, ha="center")

    # Key findings
    c.panel(side_x, y_row2, SIDE, row2_h, "Key findings", "Computed from the saved runs")
    y = y_row2 + 82
    for head, body in found:
        c.box(side_x + 22, y - 10, 4, 4 + 20 * len(_wrap(body, 64)), BORDER)
        c.text(side_x + 36, y, head.rstrip(":"), size=13.5, weight="bold")
        y += 20
        for ln in _wrap(body, 64):
            c.text(side_x + 36, y, ln, size=13.5, color=TEXT2)
            y += 20
        y += 10

    # Accuracy by question heatmap
    c.panel(MARGIN, y_row3, left_w, heat_h, "Accuracy by question", "Darker = more accurate; outlined cell = best on that question")
    cell_w, cell_x0 = 84, MARGIN + left_w - 20 - len(models) * 84
    hy = y_row3 + 82
    qx = MARGIN + 240  # room for the longest dataset title beside its first question
    c.text(MARGIN + 28, hy, "DATASET", size=11.5, color=MUTED, weight="bold")
    c.text(qx, hy, "QUESTION", size=11.5, color=MUTED, weight="bold")
    c.text(cell_x0 - 56, hy, "TYPE", size=11.5, color=MUTED, weight="bold")
    for j, m in enumerate(models):
        c.text(cell_x0 + j * cell_w + cell_w / 2, hy, NAMES[m].upper(), size=11.5, color=MUTED, weight="bold", ha="center")
    y, prev = y_row3 + 92, None
    for d, qid, qtype, accs in s["questions"]:
        if d != prev:
            c.line(MARGIN + 20, y, MARGIN + left_w - 20, y, BORDER)
            c.text(MARGIN + 28, y + 22, runs[d][1]["title"], size=13, weight="bold")
            prev = d
        c.text(qx, y + 22, qid, size=12.5, mono=True)
        c.text(cell_x0 - 56, y + 22, qtype, size=12, color=MUTED)
        top_v = max(accs.values())
        for j, m in enumerate(models):
            cx = cell_x0 + j * cell_w
            if m not in accs or accs[m] is None:
                c.text(cx + cell_w / 2, y + 22, "n/a", size=12, color=MUTED, ha="center")
                continue
            fill, ink = next((f, k) for t, f, k in reversed(HEAT) if accs[m] >= t)
            best = accs[m] == top_v
            c.box(cx + 2, y + 3, cell_w - 4, 27, fill, radius=5, edge="#0b0b0b88" if best else None, lw=1.6)
            c.text(cx + cell_w / 2, y + 22, f"{_round_pct(accs[m])}%", size=13, weight="bold", color=ink, ha="center")
        y += 33

    # Calibration error table
    ey = y_row3 + heat_h + GAP
    c.panel(MARGIN, ey, left_w, ece_h,
            "Calibration error by dataset", "Gap between stated confidence and actual accuracy (ECE, lower is better)")
    col_w = (left_w - 240) / max(len(models), 1)
    c.text(MARGIN + 28, ey + 82, "DATASET", size=11.5, color=MUTED, weight="bold")
    for j, m in enumerate(models):
        c.text(MARGIN + 220 + j * col_w + col_w / 2, ey + 82, NAMES[m].upper(), size=11.5, color=MUTED, weight="bold", ha="center")
    y = ey + 92
    for d in datasets:
        c.line(MARGIN + 20, y, MARGIN + left_w - 20, y, RULE)
        c.text(MARGIN + 28, y + 22, runs[d][1]["title"], size=13, weight="bold")
        for j, m in enumerate(models):
            v = per_ds[d]["overall"].get(m, {}).get("ece")
            c.text(MARGIN + 220 + j * col_w + col_w / 2, y + 22, "—" if v is None else f"{v:.3f}", size=13, color=TEXT2, ha="center")
        y += 33

    # Right column: easy vs hard, latency, tokens
    y = y_row3
    if has_tags:
        c.panel(side_x, y, SIDE, dumb_h, "Easy vs hard items", "Items tagged easy or hard in the datasets above")
        c.dot(side_x + 96, y + 74, 5.5, PANEL, TEXT2)
        c.text(side_x + 108, y + 78, "easy items", size=12, color=TEXT2)
        c.dot(side_x + 194, y + 74, 5.5, TEXT2)
        c.text(side_x + 206, y + 78, "hard items", size=12, color=TEXT2)
        x0, x1, lo = side_x + 90, side_x + SIDE - 70, 0.0
        tags = [m for m in models if S[m]["easy"] is not None and S[m]["hard"] is not None]
        lo = max(0.0, min(min(S[m]["easy"], S[m]["hard"]) for m in tags) // 0.2 * 0.2) if tags else 0.0
        sx = lambda v: x0 + (x1 - x0) * (v - lo) / (1 - lo)  # noqa: E731
        ry0, ry1 = y + 100, y + dumb_h - 40
        steps = [lo + k * 0.2 for k in range(int(round((1 - lo) / 0.2)) + 1)]
        for t in steps:
            c.line(sx(t), ry0 - 6, sx(t), ry1 + 6, GRID)
            c.text(sx(t), ry1 + 26, f"{t * 100:.0f}%", size=11.5, color=MUTED, ha="center")
        for i, m in enumerate(models):
            ry = ry0 + (ry1 - ry0) * (i + 0.5) / len(models)
            c.text(side_x + 76, ry + 5, NAMES[m], size=13, weight="bold", ha="right")
            if m not in tags:
                c.text(side_x + 90, ry + 5, "no tagged items", size=12, color=MUTED)
                continue
            e, h = S[m]["easy"], S[m]["hard"]
            c.line(sx(e), ry, sx(h), ry, COLOR.get(m, MUTED), lw=2.5, z=2)
            c.dot(sx(e), ry, 6, PANEL, COLOR.get(m, MUTED), lw=2.5)
            c.dot(sx(h), ry, 6, COLOR.get(m, MUTED), PANEL, lw=2)
            drop = (e - h) * 100
            c.text(max(sx(e), sx(h)) + 12, ry + 4.5, f"{'−' if drop > 0 else '+'}{abs(drop):.0f} pt", size=12, color=TEXT2)
        y += dumb_h + GAP

    for title, desc, key, fmt in (
        ("Median latency per call", f"All {total_items} items, one call per item; label shows median / p95", "p50",
         lambda m: f"{S[m]['p50']:.0f} / {S[m]['p95']:.0f} ms"),
        ("Tokens per item", "As each model reports them; not directly comparable across models", "tok",
         lambda m: f"{S[m]['tok']:.0f}"),
    ):
        c.panel(side_x, y, SIDE, hbar_h, title, desc)
        vals = {m: S[m][key] for m in models if S[m][key] is not None}
        vmax = max(vals.values(), default=1) * 1.05 or 1
        bx0, bx1 = side_x + 90, side_x + SIDE - 110
        c.line(bx0, y + 66, bx0, y + hbar_h - 10, AXIS)
        for i, m in enumerate(models):
            by = y + 70 + i * 34
            c.text(side_x + 76, by + 15, NAMES[m], size=13, weight="bold", ha="right")
            if m not in vals:
                c.text(bx0 + 8, by + 15, "—", size=12.5, color=MUTED)
                continue
            w = max((bx1 - bx0) * vals[m] / vmax, 2)
            c.bar_right(bx0, by + 2, w, 20, COLOR.get(m, MUTED))
            c.text(bx0 + w + 8, by + 16, fmt(m), size=12.5, color=TEXT2)
        y += hbar_h + GAP

    fy = height - 26
    c.text(MARGIN, fy, "Refusals: " + ", ".join(f"{NAMES[m]} {S[m]['refusals']}" for m in models)
           + " · Failed calls: " + ", ".join(f"{NAMES[m]} {S[m]['errors']}" for m in models), size=12, color=MUTED)
    c.text(WIDTH - MARGIN, fy, f"Source: {_plural(len(datasets), 'run')} in results/ · generated {datetime.now():%Y-%m-%d %H:%M}",
           size=12, color=MUTED, ha="right")
    return c.save()


def build_dashboard(paths=None, out=DEFAULT_OUT):
    """Render the dashboard from the chosen runs, write it to `out` and return the JPG bytes."""
    runs = pick_runs(paths)
    if not runs:
        raise ValueError("No saved runs in results/ yet. Run a benchmark first.")
    data = render(summarize(runs))
    Path(out).write_bytes(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*", help="results/*.jsonl files (default: latest full run of each dataset)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"output JPG (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    build_dashboard(args.runs or None, args.out)
    print(f"wrote {args.out}")
