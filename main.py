import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from rageval import core

ROOT = Path(__file__).parent
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
]
MAX_CONFIGS, MAX_JUDGED_RUNS = 12, 60
app = FastAPI(title="RAGBench")
core.init_db()


def records(df):
    return json.loads(df.to_json(orient="records"))


def numbers(text, name):
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, f"Invalid {name}.")


@app.get("/api/options")
def options():
    return {"models": MODELS, "llm": bool(os.getenv("GROQ_API_KEY"))}


@app.get("/api/history")
def history():
    h = core.history()
    return records(h) if not h.empty else []


@app.post("/api/run")
def run(models: str = Form(...), sizes: str = Form(...), ks: str = Form(...), overlap: int = Form(10),
        use_llm: bool = Form(False), docs: Optional[List[UploadFile]] = File(None),
        golden: Optional[UploadFile] = File(None)):
    m = [x for x in models.split(",") if x in MODELS]
    s, k = numbers(sizes, "chunk sizes"), numbers(ks, "top-K values")
    if not (m and s and k) or min(s) < 1 or min(k) < 1:
        raise HTTPException(400, "Pick at least one embedding model, chunk size and top-K.")
    if len(m) * len(s) * len(k) > MAX_CONFIGS:
        raise HTTPException(400, f"Choose at most {MAX_CONFIGS} configurations per run.")
    if overlap < 0 or overlap >= min(s):
        raise HTTPException(400, "Overlap must be smaller than the smallest chunk size.")
    if use_llm and not os.getenv("GROQ_API_KEY"):
        raise HTTPException(400, "Set GROQ_API_KEY in your .env file to judge answers.")
    if bool(docs) != bool(golden):
        raise HTTPException(400, "Add documents and a golden CSV together, or leave both empty to use the sample.")

    texts = {}
    for f in docs or []:
        try:
            texts[f.filename] = core.read_file(f.filename, f.file.read())
        except Exception:
            raise HTTPException(400, f"Could not read {f.filename}.")
    if not texts:
        texts = {p.name: p.read_text(encoding="utf-8") for p in sorted((ROOT / "data/sample").glob("*.txt"))}
    if not any(t.strip() for t in texts.values()):
        raise HTTPException(400, "No text could be extracted from the documents.")

    try:
        gold = pd.read_csv(golden.file if golden else ROOT / "data/golden.csv")
    except Exception:
        raise HTTPException(400, "Could not read the golden CSV.")
    missing = {"question", "reference_answer", "source", "evidence"} - set(gold.columns)
    if missing:
        raise HTTPException(400, f"The golden set is missing columns: {', '.join(sorted(missing))}.")
    gold = gold.dropna(subset=["question", "source"]).reset_index(drop=True)
    if gold.empty:
        raise HTTPException(400, "The golden set has no usable rows.")
    unknown = set(gold["source"]) - set(texts)
    if unknown:
        raise HTTPException(400, f"Golden questions refer to documents you did not add: {', '.join(sorted(unknown))}.")
    runs = len(m) * len(s) * len(k) * len(gold)
    if use_llm and runs > MAX_JUDGED_RUNS:
        raise HTTPException(400, f"Judging makes two API calls per question run. The limit is {MAX_JUDGED_RUNS} question runs, you asked for {runs}.")

    try:
        summary, detail = core.run_benchmark(texts, gold, m, s, k, overlap, use_llm)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("benchmark failed")
        raise HTTPException(500, f"Benchmark failed: {e}")
    return {"summary": records(summary), "detail": records(detail)}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
