"""Local reranker SUBPROCESS (private path). Self-exiting: spawned on first rerank,
exits after RERANKER_IDLE_EXIT seconds of no request — so the ~2GB cross-encoder holds
NO resident RAM between bursts. The main server NEVER imports this module's model; it
only spawns this as `python -m scribbleslm.reranker_worker` and talks newline-JSON over
stdin/stdout.

Protocol:
  request : {"query": str, "documents": [str, ...]}\n
  response: {"scores": [float, ...]}\n   (one score per document, higher = better)

RERANKER_MODEL=stub uses a dependency-free lexical scorer (for lifecycle tests without
downloading the cross-encoder).
"""
from __future__ import annotations

import json
import os
import re
import select
import sys

_model = None


def _load(model_name: str):
    global _model
    if _model is not None:
        return _model
    if model_name == "stub":
        _model = "stub"
        return _model
    from sentence_transformers import CrossEncoder  # heavy; only in the subprocess
    _model = CrossEncoder(model_name)
    return _model


def _score(model, model_name: str, query: str, docs: list[str]) -> list[float]:
    if model_name == "stub":
        q = set(re.findall(r"\w+", query.lower()))
        return [len(q & set(re.findall(r"\w+", d.lower()))) / (len(q) or 1) for d in docs]
    return [float(s) for s in model.predict([(query, d) for d in docs])]


def main() -> None:
    model_name = os.environ.get("RERANKER_MODEL", "stub")
    idle = float(os.environ.get("RERANKER_IDLE_EXIT", "120"))
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], idle)
        if not ready:
            sys.stderr.write("reranker_worker: idle timeout, exiting\n")
            return
        line = sys.stdin.readline()
        if not line:  # EOF: parent closed the pipe / died
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            model = _load(model_name)
            scores = _score(model, model_name, req["query"], req.get("documents", []))
            sys.stdout.write(json.dumps({"scores": scores}) + "\n")
        except Exception as e:  # never die on a bad request
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
