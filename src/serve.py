"""
FastAPI service exposing the classifier over HTTP, for wiring into a
moderation pipeline (e.g. a bot reading Instagram/Google chat webhooks).

Run:
  MODEL_DIR=results/peft__qlora__telugu/model uvicorn src.serve:app --host 0.0.0.0 --port 8000

POST /predict
  {"texts": ["...", "..."]}
->
  {"results": [{"text": ..., "label": ..., "probability": ..., "decision": ...}, ...]}
"""
import logging
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

sys.path.insert(0, os.path.dirname(__file__))
from infer import HateSpeechClassifier

MODEL_DIR = os.environ.get("MODEL_DIR", "results/baseline__muril__full__telugu/model")
BASE_MODEL = os.environ.get("BASE_MODEL_FOR_ADAPTER")  # only needed if MODEL_DIR is an adapter
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "2000"))

# Rule 9: never log raw moderation text at INFO. Logger emits counts/ids only.
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("serve")

app = FastAPI(title="Telugu/Code-mixed Abusive Content Classifier")
_clf = None
_load_error = None


def get_classifier():
    global _clf, _load_error
    if _clf is None and _load_error is None:
        try:
            _clf = HateSpeechClassifier(MODEL_DIR, base_model_for_adapter=BASE_MODEL)
        except Exception as e:  # noqa: BLE001 -- surfaced via /readyz, not a stack trace to callers
            _load_error = str(e)
            log.error("model load failed: %s", _load_error)
    return _clf


class PredictRequest(BaseModel):
    texts: conlist(str, min_length=1, max_length=MAX_BATCH)


@app.on_event("startup")
def _load():
    get_classifier()


@app.get("/health")
def health():
    """Liveness only: process is up. Does not imply the model can run."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness: model loaded AND a real inference actually executes."""
    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")
    try:
        clf.predict(["health check"], batch_size=1)
    except Exception:
        raise HTTPException(status_code=503, detail="inference_failed")
    return {"status": "ready", "model_dir": MODEL_DIR}


@app.get("/version")
def version():
    return {"model_dir": MODEL_DIR, "base_model": BASE_MODEL}


@app.post("/predict")
def predict(req: PredictRequest):
    clf = get_classifier()
    if clf is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    oversized = [i for i, t in enumerate(req.texts) if len(t) > MAX_TEXT_CHARS]
    if oversized:
        raise HTTPException(status_code=422,
                             detail=f"{len(oversized)} text(s) exceed MAX_TEXT_CHARS={MAX_TEXT_CHARS}")

    start = time.time()
    try:
        results = clf.predict(req.texts)
    except Exception:
        log.exception("inference error (n_texts=%d)", len(req.texts))
        raise HTTPException(status_code=500, detail="inference_error")
    log.info("predict n=%d latency_ms=%.1f", len(req.texts), (time.time() - start) * 1000)
    return {"results": results}
