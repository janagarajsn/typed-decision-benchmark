"""Scoring for a finished benchmark run, computed from the saved per-item records.

Everything here is a pure function of (questions, records), so a saved run in results/
can be rescored without calling any model again:

    python src/metrics.py results/<run>.jsonl

A record is one item: {"id", "tags", "labels": {qid: gold}, "answers": {model: {qid: normalized}},
"latency_ms": {model: float}, "usage": {model: {"input_tokens", "output_tokens"}}}, where
normalized answers use the shape documented in decision_models.py, plus {"type": "error"}
when the call itself failed.
"""
import json
import sys

N_BINS = 5


def pick_and_confidence(ans, qtype):
    """Value the model actually predicted, and the probability mass it put on that value.

    Deliberately not the vendor's own "confidence" field -- that metric reads
    oddly near 50/50 calls. This is always "probability assigned to whatever
    ended up being predicted," consistent across choice/noul/score so
    calibration bucketing means the same thing for all three.

    A refusal (OpenAI may decline a question) or a failed call predicts nothing: (None, 0.0).
    """
    if ans["type"] in ("refusal", "error"):
        return None, 0.0
    if qtype == "choice":
        predicted = ans["choice"]
        return predicted, ans["probabilities"].get(predicted, 0.0)
    if qtype == "noul":
        predicted = ans["noul"] >= 0.5
        return predicted, (ans["noul"] if predicted else 1.0 - ans["noul"])
    # score: the most likely rubric level. Rounding the expected value instead would
    # drag a spread-out distribution toward the middle level, even when the model's
    # own top pick is an extreme (e.g. {0: .55, 1: .32, 2: .13} -> E=0.58 -> 1).
    level, prob = max(ans["probabilities"].items(), key=lambda kv: kv[1])
    return int(level), prob


def _ratio(a, b):
    return round(a / b, 4) if b else None


def _percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def _calibration(pairs):
    """pairs: [(confidence, hit)]. Returns (bins, expected calibration error)."""
    bins = [{"n": 0, "hits": 0, "conf_sum": 0.0} for _ in range(N_BINS)]
    for conf, hit in pairs:
        b = bins[min(int(conf * N_BINS), N_BINS - 1)]
        b["n"] += 1
        b["hits"] += hit
        b["conf_sum"] += conf
    total = sum(b["n"] for b in bins)
    ece = sum(abs(b["hits"] / b["n"] - b["conf_sum"] / b["n"]) * b["n"] for b in bins if b["n"]) / total if total else None
    out = [{
        "lo": i / N_BINS, "hi": (i + 1) / N_BINS, "n": b["n"], "hits": b["hits"],
        "accuracy": _ratio(b["hits"], b["n"]), "avg_confidence": _ratio(b["conf_sum"], b["n"]),
    } for i, b in enumerate(bins)]
    return out, (round(ece, 4) if ece is not None else None)


def _auc(scored):
    """Probability a random positive outranks a random negative (ties count half)."""
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def _qwk(pairs, k):
    """Quadratic weighted kappa between gold and predicted levels 0..k-1."""
    if not pairs or k < 2:
        return None
    n = len(pairs)
    observed = [[0] * k for _ in range(k)]
    for gold, pred in pairs:
        observed[gold][pred] += 1
    gold_hist = [sum(row) for row in observed]
    pred_hist = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[i][j]
            den += w * gold_hist[i] * pred_hist[j] / n
    return round(1 - num / den, 4) if den else None


def _prf(tp, fp, fn):
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
    return precision, recall, f1


def _question_metrics(qid, q, records, model):
    qtype = q["type"]
    hits = n = refusals = errors = 0
    conf_pairs = []
    preds = []  # (gold, predicted, answer) for answered items
    for rec in records:
        ans = rec["answers"].get(model, {}).get(qid)
        if ans is None:
            continue
        gold = rec["labels"][qid]
        predicted, conf = pick_and_confidence(ans, qtype)
        n += 1
        if ans["type"] == "refusal":
            refusals += 1
            continue
        if ans["type"] == "error":
            errors += 1
            continue
        hit = predicted == gold
        hits += hit
        conf_pairs.append((conf, hit))
        preds.append((gold, predicted, ans))

    calibration, ece = _calibration(conf_pairs)
    out = {"accuracy": _ratio(hits, n), "hits": hits, "n": n, "refusals": refusals, "errors": errors,
           "ece": ece, "calibration": calibration}

    if qtype == "choice":
        options = list(q["criteria"].keys()) if isinstance(q["criteria"], dict) else list(q["criteria"])
        confusion = {g: {p: 0 for p in options} for g in options}
        for gold, pred, _ in preds:
            confusion[gold][pred] = confusion[gold].get(pred, 0) + 1
        per_class = {}
        for opt in options:
            tp = confusion[opt][opt]
            fp = sum(confusion[g].get(opt, 0) for g in options if g != opt)
            fn = sum(c for p, c in confusion[opt].items() if p != opt)
            precision, recall, f1 = _prf(tp, fp, fn)
            per_class[opt] = {"precision": precision, "recall": recall, "f1": f1,
                              "support": sum(1 for r in records if r["labels"][qid] == opt)}
        supported = [c["f1"] for c in per_class.values() if c["support"]]
        out.update(macro_f1=round(sum(supported) / len(supported), 4) if supported else None,
                   per_class=per_class, confusion=confusion)
    elif qtype == "noul":
        tp = sum(1 for g, p, _ in preds if g and p)
        fp = sum(1 for g, p, _ in preds if not g and p)
        fn = sum(1 for g, p, _ in preds if g and not p)
        precision, recall, f1 = _prf(tp, fp, fn)
        brier = _ratio(sum((a["noul"] - g) ** 2 for g, _, a in preds), len(preds))
        out.update(precision=precision, recall=recall, f1=f1, brier=brier,
                   auc=_auc([(a["noul"], g) for g, _, a in preds]))
    else:
        k = len(q["criteria"])
        out.update(
            mae=_ratio(sum(abs(a["score"] - g) for g, _, a in preds), len(preds)),
            within_one=_ratio(sum(1 for g, p, _ in preds if abs(p - g) <= 1), n),
            qwk=_qwk([(g, p) for g, p, _ in preds], k),
        )
    return out


def compute(questions, records, models):
    """Every metric the dashboard and the CLI report, as one JSON-serializable dict."""
    result = {"items": len(records), "questions": {}, "overall": {}, "by_tag": {}}

    for qid, q in questions.items():
        disagreements, wins = 0, {m: 0 for m in models}
        for rec in records:
            preds = {m: pick_and_confidence(rec["answers"][m][qid], q["type"])[0]
                     for m in models if qid in rec["answers"].get(m, {})}
            if len({str(p) for p in preds.values()}) > 1:
                disagreements += 1
                for m, p in preds.items():
                    wins[m] += p is not None and p == rec["labels"][qid]
        result["questions"][qid] = {
            "type": q["type"],
            "models": {m: _question_metrics(qid, q, records, m) for m in models},
            "disagreements": disagreements, "disagreement_wins": wins,
        }

    for m in models:
        per_q = [result["questions"][qid]["models"][m] for qid in questions]
        hits, n = sum(x["hits"] for x in per_q), sum(x["n"] for x in per_q)
        conf_pairs = []
        for rec in records:
            for qid, q in questions.items():
                ans = rec["answers"].get(m, {}).get(qid)
                if ans and ans["type"] not in ("refusal", "error"):
                    pred, conf = pick_and_confidence(ans, q["type"])
                    conf_pairs.append((conf, pred == rec["labels"][qid]))
        calibration, ece = _calibration(conf_pairs)
        latencies = [rec["latency_ms"][m] for rec in records if m in rec.get("latency_ms", {})]
        tokens_in = sum(rec.get("usage", {}).get(m, {}).get("input_tokens", 0) for rec in records)
        tokens_out = sum(rec.get("usage", {}).get(m, {}).get("output_tokens", 0) for rec in records)
        result["overall"][m] = {
            "accuracy": _ratio(hits, n), "hits": hits, "n": n,
            "refusals": sum(x["refusals"] for x in per_q),
            "errors": sum(1 for rec in records if m in rec.get("errors", {})),  # failed calls, not answers
            "avg_confidence": _ratio(sum(c for c, _ in conf_pairs), len(conf_pairs)),
            "ece": ece, "calibration": calibration,
            "latency_ms": {"mean": _ratio(sum(latencies), len(latencies)),
                           "p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
            "tokens": {"input_tokens": tokens_in, "output_tokens": tokens_out,
                       "per_item": _ratio(tokens_in + tokens_out, len(records))},
        }

    tags = sorted({t for rec in records for t in rec["tags"]})
    for tag in tags:
        subset = [rec for rec in records if tag in rec["tags"]]
        per_model = {}
        for m in models:
            hits = n = 0
            for rec in subset:
                for qid, q in questions.items():
                    ans = rec["answers"].get(m, {}).get(qid)
                    if ans is None:
                        continue
                    pred, _ = pick_and_confidence(ans, q["type"])
                    n += 1
                    hits += pred is not None and pred == rec["labels"][qid]
            per_model[m] = {"accuracy": _ratio(hits, n), "hits": hits, "n": n}
        result["by_tag"][tag] = {"items": len(subset), "models": per_model}

    return result


def load_run(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[0]["run"], rows[1:]


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}" if isinstance(v, float) else str(v)


def print_report(run, metrics):
    models = run["models"]
    row = lambda label, vals: print(f"  {label:<18}" + "".join(f"{v:>12}" for v in vals))
    print(f"{run['title']}: {metrics['items']} items, models {', '.join(models)}")
    row("", models)
    for key, pct in (("accuracy", True), ("avg_confidence", True), ("ece", False), ("refusals", False), ("errors", False)):
        row(key, [_fmt(metrics["overall"][m][key], pct) for m in models])
    row("latency p50 ms", [_fmt(metrics["overall"][m]["latency_ms"]["p50"]) for m in models])
    row("latency p95 ms", [_fmt(metrics["overall"][m]["latency_ms"]["p95"]) for m in models])
    for qid, qm in metrics["questions"].items():
        print(f"\n{qid} ({qm['type']}), disagreements {qm['disagreements']}")
        keys = {"choice": ["accuracy", "macro_f1", "ece"], "noul": ["accuracy", "precision", "recall", "f1", "brier", "auc", "ece"],
                "score": ["accuracy", "within_one", "mae", "qwk", "ece"]}[qm["type"]]
        for key in keys:
            row(key, [_fmt(qm["models"][m][key], key in ("accuracy", "within_one")) for m in models])
    print("\nby tag (accuracy over all questions)")
    for tag, tm in metrics["by_tag"].items():
        row(f"{tag} ({tm['items']})", [_fmt(tm["models"][m]["accuracy"], True) for m in models])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/metrics.py results/<run>.jsonl")
    run, records = load_run(sys.argv[1])
    print_report(run, compute(run["questions"], records, run["models"]))
