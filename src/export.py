"""Excel export of a saved run (results/<run>.jsonl), with the full metric set and every answer.

Sheets: Summary (overall metrics per model), Questions (type-specific metrics per question),
By tag, Items (one row per item: gold answer, each model's prediction, hit and confidence),
and Confusion (one matrix per choice question and model).

    python src/export.py results/<run>.jsonl    # writes results/<run>.xlsx
"""
import io
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from golden import DatasetError, load_dataset
from metrics import compute, load_run, pick_and_confidence

NAMES = {"jev": "Jev", "laya": "Laya", "vega": "Vega", "openai": "OpenAI"}
HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="2A2F38")
TITLE_FONT = Font(bold=True, size=13)
HIT_FILL = PatternFill("solid", fgColor="D9F2DD")
MISS_FILL = PatternFill("solid", fgColor="F9D9D9")
PERCENT = {"accuracy", "avg_confidence", "within_one", "precision", "recall"}
QUESTION_METRICS = {
    "choice": ["accuracy", "macro_f1", "ece", "refusals", "errors"],
    "noul": ["accuracy", "precision", "recall", "f1", "auc", "brier", "ece", "refusals", "errors"],
    "score": ["accuracy", "within_one", "mae", "qwk", "ece", "refusals", "errors"],
}


def _header(ws, row, values):
    for col, value in enumerate(values, start=1):
        cell = ws.cell(row=row, column=col, value=value)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL


def _metric_row(ws, row, label, key, values):
    ws.cell(row=row, column=1, value=label)
    for col, value in enumerate(values, start=2):
        cell = ws.cell(row=row, column=col, value=value)
        if value is not None and key in PERCENT:
            cell.number_format = "0.0%"
        elif isinstance(value, float):
            cell.number_format = "0.000"


def _autosize(ws, max_width=60):
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(width + 2, 8), max_width)


def _display(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def build_workbook(run, records):
    """The workbook as .xlsx bytes."""
    models = run["models"]
    questions = run["questions"]
    metrics = compute(questions, records, models)
    lanes = [NAMES.get(m, m) for m in models]
    wb = Workbook()

    ws = wb.active
    ws.title = "Summary"
    ws.cell(row=1, column=1, value=f"{run['title']} benchmark").font = TITLE_FONT
    info = [("Dataset", run["dataset"]), ("Started", run["started_at"]), ("Items", metrics["items"]),
            ("Models", ", ".join(lanes))]
    for i, (k, v) in enumerate(info, start=2):
        ws.cell(row=i, column=1, value=k).font = Font(bold=True)
        ws.cell(row=i, column=2, value=v)
    row = len(info) + 3
    _header(ws, row, ["metric", *lanes])
    overall = metrics["overall"]
    rows = [("accuracy", "accuracy", [overall[m]["accuracy"] for m in models]),
            ("avg confidence", "avg_confidence", [overall[m]["avg_confidence"] for m in models]),
            ("calibration error (ECE)", "ece", [overall[m]["ece"] for m in models]),
            ("refusals", "refusals", [overall[m]["refusals"] for m in models]),
            ("failed calls", "errors", [overall[m]["errors"] for m in models]),
            ("latency mean (ms)", "ms", [overall[m]["latency_ms"]["mean"] for m in models]),
            ("latency p50 (ms)", "ms", [overall[m]["latency_ms"]["p50"] for m in models]),
            ("latency p95 (ms)", "ms", [overall[m]["latency_ms"]["p95"] for m in models]),
            ("input tokens", "tokens", [overall[m]["tokens"]["input_tokens"] for m in models]),
            ("output tokens", "tokens", [overall[m]["tokens"]["output_tokens"] for m in models]),
            ("tokens / item", "tokens", [overall[m]["tokens"]["per_item"] for m in models])]
    for label, key, values in rows:
        row += 1
        _metric_row(ws, row, label, key, values)
    _autosize(ws)

    ws = wb.create_sheet("Questions")
    row = 1
    for qid, qm in metrics["questions"].items():
        ws.cell(row=row, column=1, value=f"{qid} ({qm['type']})").font = TITLE_FONT
        row += 1
        _header(ws, row, ["metric", *lanes])
        for key in QUESTION_METRICS[qm["type"]]:
            row += 1
            _metric_row(ws, row, key, key, [qm["models"][m][key] for m in models])
        if qm["type"] == "choice":
            for cls, stats in qm["models"][models[0]]["per_class"].items():
                row += 1
                _metric_row(ws, row, f"F1 · {cls} ({stats['support']})", "f1",
                            [qm["models"][m]["per_class"][cls]["f1"] for m in models])
        row += 1
        _metric_row(ws, row, "disagreements", "n", [qm["disagreements"]] + [None] * (len(models) - 1))
        row += 1
        _metric_row(ws, row, "right when they disagreed", "n", [qm["disagreement_wins"][m] for m in models])
        row += 2
    _autosize(ws)

    ws = wb.create_sheet("By tag")
    _header(ws, 1, ["tag", "items", *[f"{n} accuracy" for n in lanes]])
    for row, (tag, tm) in enumerate(metrics["by_tag"].items(), start=2):
        ws.cell(row=row, column=1, value=tag)
        ws.cell(row=row, column=2, value=tm["items"])
        for col, m in enumerate(models, start=3):
            cell = ws.cell(row=row, column=col, value=tm["models"][m]["accuracy"])
            cell.number_format = "0.0%"
    _autosize(ws)

    try:
        texts = {it["id"]: "\n\n".join(str(v) for v in it["input"].values())
                 for it in load_dataset(run["dataset"])["items"]}
    except DatasetError:
        texts = {}
    ws = wb.create_sheet("Items")
    head = ["id", "tags", "input", "question", "correct"]
    for name in lanes:
        head += [f"{name} answer", f"{name} confidence"]
    head += [f"{name} latency (ms)" for name in lanes]
    _header(ws, 1, head)
    row = 1
    for rec in records:
        for qid, q in questions.items():
            row += 1
            gold = rec["labels"][qid]
            values = [rec["id"], ", ".join(rec["tags"]), texts.get(rec["id"]), qid, _display(gold)]
            fills = []
            for m in models:
                ans = rec["answers"].get(m, {}).get(qid)
                if ans is None:
                    values += [None, None]
                    fills.append(None)
                    continue
                pred, conf = pick_and_confidence(ans, q["type"])
                shown = {"refusal": "refused", "error": "error"}.get(ans["type"], _display(pred))
                values += [shown, round(conf, 4) if pred is not None else None]
                fills.append(HIT_FILL if pred is not None and pred == gold else MISS_FILL)
            values += [rec["latency_ms"].get(m) for m in models]
            for col, value in enumerate(values, start=1):
                ws.cell(row=row, column=col, value=value)
            for i, fill in enumerate(fills):
                if fill:
                    ws.cell(row=row, column=6 + 2 * i).fill = fill
                    ws.cell(row=row, column=7 + 2 * i).number_format = "0%"
    _autosize(ws)
    ws.column_dimensions["C"].width = 70
    for cells in ws.iter_rows(min_row=2, min_col=3, max_col=3):
        cells[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions

    choice_qs = [qid for qid, q in questions.items() if q["type"] == "choice"]
    if choice_qs:
        ws = wb.create_sheet("Confusion")
        row = 1
        for qid in choice_qs:
            for m, name in zip(models, lanes):
                confusion = metrics["questions"][qid]["models"][m]["confusion"]
                options = list(confusion)
                ws.cell(row=row, column=1, value=f"{qid} · {name} (rows = correct, columns = predicted)").font = TITLE_FONT
                row += 1
                _header(ws, row, ["", *options])
                for gold in options:
                    row += 1
                    ws.cell(row=row, column=1, value=gold).font = Font(bold=True)
                    for col, pred in enumerate(options, start=2):
                        cell = ws.cell(row=row, column=col, value=confusion[gold].get(pred, 0))
                        if gold == pred:
                            cell.fill = HIT_FILL
                row += 2
        _autosize(ws)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/export.py results/<run>.jsonl")
    src = Path(sys.argv[1])
    run, records = load_run(src)
    out = src.with_suffix(".xlsx")
    out.write_bytes(build_workbook(run, records))
    print(f"wrote {out}")
