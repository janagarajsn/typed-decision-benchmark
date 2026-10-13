"""Streaming benchmark: one golden dataset run through Jev, Laya, Vega and the OpenAI
Decisions API, one call per item per model, scored against the dataset's known answers.

Each item is an independent decision, so every model answers the exact same state --
no path-dependency to work around. All of a dataset's questions (any mix of
choice/noul/score) are asked in a single call, so this exercises the full shape of
the typed-decision API.

Running totals are streamed while the run is in progress; the full metric set
(metrics.compute) is computed once at the end from the per-item records, which are
also written to results/ so a run can be rescored later without calling the models.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from decision_models import get_model
from golden import load_dataset, sample_items
from metrics import compute, pick_and_confidence

LANES = ("jev", "laya", "vega", "openai")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _answer_safely(model, state, questions):
    """One model call; a failed call becomes an "error" answer instead of ending the run."""
    t0 = time.perf_counter()
    try:
        answers, error = model.answer(state, questions), None
    except Exception as exc:
        answers, error = {qid: {"type": "error"} for qid in questions}, str(exc)
    latency = (time.perf_counter() - t0) * 1000
    usage = (model.last_usage or {}) if error is None else {"input_tokens": 0, "output_tokens": 0}
    return answers, latency, dict(usage), error


def run_benchmark_stream(dataset_name, emit, limit=None, seed=0):
    dataset = load_dataset(dataset_name)
    meta = dataset["meta"]
    questions = meta["questions"]
    context = meta.get("context") or {}
    items = sample_items(dataset, limit, seed)

    models = {}
    for name in LANES:
        emit({"type": "status", "message": f"Loading {name}..."})
        try:
            models[name] = get_model(name)
        except Exception as exc:
            emit({"type": "model_unavailable", "model": name, "message": str(exc)})

    if not models:
        emit({"type": "error", "message": "No usable decision models available."})
        return

    totals = {name: {qid: {"hits": 0, "n": 0} for qid in questions} for name in models}
    disagreements = {qid: 0 for qid in questions}
    disagreement_wins = {qid: {name: 0 for name in models} for qid in questions}
    token_totals = {name: {"input_tokens": 0, "output_tokens": 0} for name in models}
    records = []

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{meta['name']}.jsonl"
    results_file = results_path.open("w", encoding="utf-8")
    results_file.write(json.dumps({"run": {
        "dataset": meta["name"], "title": meta["title"], "started_at": datetime.now().isoformat(timespec="seconds"),
        "models": list(models), "limit": limit, "seed": seed, "questions": questions,
    }}, ensure_ascii=False) + "\n")

    start = time.monotonic()
    emit({
        "type": "benchmark_started", "dataset": meta["name"], "title": meta["title"],
        "items": len(items), "models": list(models),
        "questions": {qid: q["type"] for qid, q in questions.items()},
        "score_levels": {qid: q["criteria"] for qid, q in questions.items() if q["type"] == "score"},
    })

    try:
        for i, item in enumerate(items):
            emit({
                "type": "item_start", "item": i, "total": len(items), "id": item["id"],
                "text": "\n\n".join(str(v) for v in item["input"].values()),
                "tags": item["tags"], "labels": item["labels"],
            })

            state = {**context, **item["input"]}
            record = {"id": item["id"], "tags": item["tags"], "labels": item["labels"],
                      "answers": {}, "latency_ms": {}, "usage": {}}
            for name, model in models.items():
                answers, latency, usage, error = _answer_safely(model, state, questions)
                record["answers"][name] = answers
                record["latency_ms"][name] = round(latency, 1)
                record["usage"][name] = usage
                for k in token_totals[name]:
                    token_totals[name][k] += usage.get(k, 0)
                if error:
                    record.setdefault("errors", {})[name] = error
                    emit({"type": "model_error", "model": name, "item": i, "message": error})

            emit({"type": "item_timing", "item": i, "latency_ms": record["latency_ms"]})

            for qid, qdef in questions.items():
                gold = item["labels"][qid]
                picks = {}
                for name in models:
                    ans = record["answers"][name][qid]
                    predicted, conf = pick_and_confidence(ans, qdef["type"])
                    hit = predicted is not None and predicted == gold
                    totals[name][qid]["n"] += 1
                    totals[name][qid]["hits"] += hit
                    picks[name] = {"predicted": predicted, "hit": hit, "confidence": round(conf, 4)}
                    if ans["type"] == "refusal":
                        picks[name]["refused"] = True
                    elif ans["type"] == "error":
                        picks[name]["errored"] = True

                disagree = len({str(p["predicted"]) for p in picks.values()}) > 1
                if disagree:
                    disagreements[qid] += 1
                    for name, p in picks.items():
                        disagreement_wins[qid][name] += p["hit"]

                emit({
                    "type": "answer", "item": i, "question": qid, "qtype": qdef["type"],
                    "correct": gold, "picks": picks, "disagree": disagree,
                    "totals": {name: dict(totals[name][qid]) for name in models},
                    "disagreements": disagreements[qid],
                    "disagreement_wins": dict(disagreement_wins[qid]),
                })

            records.append(record)
            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()
            emit({
                "type": "item_done", "item": i, "tokens": record["usage"],
                "token_totals": {name: dict(token_totals[name]) for name in models},
            })
    finally:
        results_file.close()

    emit({
        "type": "done",
        "elapsed": round(time.monotonic() - start, 2),
        "results_file": str(results_path.relative_to(RESULTS_DIR.parent)),
        "metrics": compute(questions, records, list(models)),
    })
