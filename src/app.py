import json
import os
import queue
import threading

import dotenv
from flask import Flask, Response, jsonify, render_template, request

from decision_models import get_model
from benchmark_engine import LANES, RESULTS_DIR, run_benchmark_stream
from export import build_workbook
from golden import list_datasets
from metrics import load_run

dotenv.load_dotenv()

app = Flask(__name__)

_run_lock = threading.Lock()
_run_state = {"running": False, "queue": None}
# Display name and where each model runs, for the dashboard.
MODEL_INFO = {
    "jev": ("Jev", "cloud"),
    "laya": ("Laya", "local"),
    "vega": ("Vega", "local"),
    "openai": ("OpenAI", "cloud"),
}


@app.route("/")
def index():
    return render_template("dashboard.html", lanes=LANES, model_info=MODEL_INFO, datasets=list_datasets())


@app.route("/api/datasets")
def api_datasets():
    return jsonify(datasets=list_datasets())


def _run_benchmark(dataset, limit):
    q = _run_state["queue"]

    def emit(event):
        q.put(event)

    try:
        run_benchmark_stream(dataset, emit, limit=limit)
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        emit({"type": "stream_end"})
        _run_state["running"] = False


@app.route("/api/benchmark/start", methods=["POST"])
def api_benchmark_start():
    data = request.get_json(force=True) or {}
    dataset = data.get("dataset")
    limit = data.get("limit")

    if dataset not in {d["name"] for d in list_datasets()}:
        return jsonify(error=f"Unknown dataset: {dataset!r}."), 400
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        return jsonify(error="limit must be a positive integer, or null for every item."), 400

    with _run_lock:
        if _run_state["running"]:
            return jsonify(error="A benchmark run is already in progress. Wait for it to finish."), 409
        _run_state["running"] = True
        _run_state["queue"] = queue.Queue()
        thread = threading.Thread(target=_run_benchmark, args=(dataset, limit), daemon=True)
        thread.start()

    return jsonify(ok=True)


@app.route("/api/benchmark/stream")
def api_benchmark_stream():
    q = _run_state["queue"]
    if q is None:
        return Response("", mimetype="text/event-stream")

    def gen():
        while True:
            event = q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "stream_end":
                break

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/results/<name>.xlsx")
def api_results_xlsx(name):
    # <name> can't contain "/", so this only ever reads a file directly inside results/.
    path = RESULTS_DIR / f"{name}.jsonl"
    if not path.is_file():
        return jsonify(error=f"No saved run named {name!r}."), 404
    run, records = load_run(path)
    if not records:
        return jsonify(error="That run has no finished items yet."), 400
    return Response(
        build_workbook(run, records),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{name}.xlsx"'},
    )


def _prewarm_local_models():
    """Load the local models (Laya, Vega) at server startup instead of on the user's first click.

    Building the ModernBERT-large encoder from config (before the downloaded checkpoint
    weights overwrite it) takes ~20s regardless of the Hugging Face cache; doing this in
    the background while the server starts hides that cost from the first benchmark run.
    Vega's first load also downloads its checkpoint from Hugging Face.
    """
    for name, label in (("laya", "Laya"), ("vega", "Vega")):
        try:
            get_model(name)
            print(f"[prewarm] {label} decision model loaded and ready.")
        except Exception as exc:
            print(f"[prewarm] Could not preload {label}: {exc}")


if __name__ == "__main__":
    DEBUG = True
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_prewarm_local_models, daemon=True).start()
    app.run(debug=DEBUG, threaded=True, port=5050)
