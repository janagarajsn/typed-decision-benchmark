"""Golden datasets: fixed, hand-reviewed items with known-correct answers.

Each dataset is one JSONL file in data/golden/. The first line is a header, every
following line is one item:

    {"meta": {"name": "spam_detection", "title": "Spam detection", "description": "...",
              "context": {...},            # optional: merged into every item's input
              "questions": {qid: {"type": "choice"|"noul"|"score", "instructions": str,
                                  "criteria": {value: description} | [value, ...] | [level label, ...]}},
              "tags": {tag: description}}}
    {"id": "spam-001", "input": {"email": "..."}, "labels": {qid: value},
     "tags": ["phishing", "hard"], "rationale": "...", "reviewed": false}

Label values: choice -> the option key (str), noul -> bool, score -> level index (int).
`reviewed` flips to true once a person has checked the item's answer.

Run `python src/golden.py` from the project root to validate every dataset.
"""
import json
import random
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "golden"
QTYPES = ("choice", "noul", "score")
ITEM_KEYS = {"id", "input", "labels", "tags", "rationale", "reviewed"}


class DatasetError(ValueError):
    pass


def _choice_options(q):
    criteria = q.get("criteria") or []
    return list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)


def _check_label(qid, q, value, where):
    if q["type"] == "noul":
        ok = isinstance(value, bool)
    elif q["type"] == "score":
        ok = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(q["criteria"])
    else:
        ok = value in _choice_options(q)
    if not ok:
        raise DatasetError(f"{where}: invalid label {value!r} for {q['type']} question {qid!r}")


def _validate(meta, items, path):
    for key in ("name", "title", "description", "questions", "tags"):
        if key not in meta:
            raise DatasetError(f"{path}: header is missing {key!r}")
    if meta["name"] != path.stem:
        raise DatasetError(f"{path}: header name {meta['name']!r} doesn't match the file name")
    questions = meta["questions"]
    for qid, q in questions.items():
        if q.get("type") not in QTYPES:
            raise DatasetError(f"{path}: question {qid!r} has unknown type {q.get('type')!r}")
        if not q.get("instructions"):
            raise DatasetError(f"{path}: question {qid!r} has no instructions")
        if q["type"] != "noul" and len(q.get("criteria") or []) < 2:
            raise DatasetError(f"{path}: question {qid!r} needs at least two criteria")

    seen = set()
    for item in items:
        where = f"{path.name}:{item.get('id', '?')}"
        missing = ITEM_KEYS - item.keys()
        if missing:
            raise DatasetError(f"{where}: missing {sorted(missing)}")
        if item["id"] in seen:
            raise DatasetError(f"{where}: duplicate id")
        seen.add(item["id"])
        if not isinstance(item["input"], dict) or not item["input"]:
            raise DatasetError(f"{where}: input must be a non-empty object")
        if set(item["labels"]) != set(questions):
            raise DatasetError(f"{where}: labels must cover exactly {sorted(questions)}")
        for qid, value in item["labels"].items():
            _check_label(qid, questions[qid], value, where)
        unknown = set(item["tags"]) - set(meta["tags"])
        if unknown:
            raise DatasetError(f"{where}: tags {sorted(unknown)} aren't declared in the header")
        if not isinstance(item["reviewed"], bool):
            raise DatasetError(f"{where}: reviewed must be true or false")
    if not items:
        raise DatasetError(f"{path}: no items")


def load_dataset(name):
    path = GOLDEN_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise DatasetError(f"Unknown dataset {name!r}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path.name}: invalid JSON ({exc})") from exc
    if not rows or "meta" not in rows[0]:
        raise DatasetError(f"{path.name}: the first line must be the {{\"meta\": ...}} header")
    meta, items = rows[0]["meta"], rows[1:]
    _validate(meta, items, path)
    return {"meta": meta, "items": items}


def list_datasets():
    """Summaries of every valid dataset, for the dashboard's picker."""
    out = []
    for path in sorted(GOLDEN_DIR.glob("*.jsonl")):
        try:
            ds = load_dataset(path.stem)
        except DatasetError as exc:
            print(f"[golden] skipping {path.name}: {exc}")
            continue
        meta = ds["meta"]
        out.append({
            "name": meta["name"], "title": meta["title"], "description": meta["description"],
            "items": len(ds["items"]),
            "reviewed": sum(1 for it in ds["items"] if it["reviewed"]),
            "questions": {qid: q["type"] for qid, q in meta["questions"].items()},
        })
    return out


def sample_items(dataset, limit=None, seed=0):
    """A fixed-seed shuffle, so "first 20" is the same 20 items on every run."""
    items = list(dataset["items"])
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


if __name__ == "__main__":
    for summary in list_datasets():
        print(f"{summary['name']}: {summary['items']} items ({summary['reviewed']} reviewed), "
              f"questions {summary['questions']}")
