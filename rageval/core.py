"""RAGBench core: chunking, Qdrant retrieval, retrieval metrics, LLM judge, SQLite run storage."""
import io
import itertools
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = os.getenv("DB_PATH", "rageval.db")
LLM_METRICS = ("faithfulness", "relevancy", "correctness")


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created TEXT, config TEXT, metrics TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS results (run_id TEXT, question TEXT, category TEXT, answer TEXT, detail TEXT)")


# ---------- documents ----------
def read_file(name, data):
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def chunk(txt, size, overlap=0):
    words, step = txt.split(), max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


# ---------- vector search (Qdrant) ----------
@lru_cache(maxsize=4)
def get_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def build_index(chunks, model_name):
    if not chunks:
        raise ValueError("No text could be extracted from the documents.")
    from qdrant_client import QdrantClient, models
    model = get_model(model_name)
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=url) if url else QdrantClient(":memory:")
    name = "bench_" + uuid.uuid4().hex[:8]
    client.create_collection(name, vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE))
    client.upsert(name, points=[models.PointStruct(id=i, vector=v.tolist(), payload=chunks[i]) for i, v in enumerate(vecs)])
    return client, name, model


def retrieve(index, question, k):
    client, name, model = index
    qv = model.encode(question, normalize_embeddings=True).tolist()
    return client.query_points(name, query=qv, limit=k).points


# ---------- metrics ----------
def is_rel(chunk_, row):
    ev = "" if pd.isna(row["evidence"]) else str(row["evidence"]).lower()
    return chunk_["source"] == row["source"] and (not ev or ev in chunk_["text"].lower())


def retrieval_metrics(rel, total_rel, k):
    got = sum(rel)
    first = next((i for i, x in enumerate(rel, 1) if x), None)
    return {
        "precision@k": got / k,
        "recall@k": got / total_rel if total_rel else 0.0,
        "hit_rate": float(got > 0),
        "mrr": 1 / first if first else 0.0,
    }


def category(m, scores=None):
    if not m["hit_rate"]:
        return "Retrieval miss"
    if scores and scores["faithfulness"] < 0.5:
        return "Hallucination"
    if scores and scores["correctness"] < 0.5:
        return "Incorrect answer"
    return "OK"


# ---------- generation + LLM judge (Groq) ----------
def llm(prompt, as_json=False):
    from groq import Groq
    kw = {"response_format": {"type": "json_object"}} if as_json else {}
    for attempt in range(5):
        try:
            r = Groq().chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=0, messages=[{"role": "user", "content": prompt}], **kw)
            return r.choices[0].message.content
        except Exception:
            if attempt == 4:
                raise
            time.sleep(4 * 2 ** attempt)


def generate(question, ctx):
    return llm(f"Answer using ONLY the context. If the answer is not there, say \"I don't know\".\n\nContext:\n{ctx}\n\nQuestion: {question}")


def judge(question, ctx, answer, reference):
    try:
        raw = json.loads(llm(
            'Score from 0 to 1 and reply as JSON {"faithfulness": x, "relevancy": x, "correctness": x}. '
            "faithfulness: every claim in the answer is supported by the context. "
            "relevancy: the answer addresses the question. correctness: the answer matches the reference.\n"
            f"Question: {question}\nContext: {ctx}\nAnswer: {answer}\nReference: {reference}", True))
        return {k: min(max(float(raw[k]), 0.0), 1.0) for k in LLM_METRICS}
    except (ValueError, KeyError, TypeError, AttributeError):
        return {k: float("nan") for k in LLM_METRICS}


# ---------- benchmark ----------
def run_benchmark(docs, golden, models, sizes, ks, overlap=10, use_llm=False, progress=lambda *a: None):
    combos = list(itertools.product(models, sizes, ks))
    cache, rows, details = {}, [], []
    for n, (mod, size, k) in enumerate(combos, 1):
        label = f"{mod.split('/')[-1]} ({size}w, k={k})"
        progress(n - 1, len(combos), label)
        if (mod, size) not in cache:
            chunks = [{"source": s, "text": c} for s, t in docs.items() for c in chunk(t, size, overlap)]
            cache[(mod, size)] = (chunks, build_index(chunks, mod))
        chunks, index = cache[(mod, size)]
        per_q, det = [], []
        for qi, (_, g) in enumerate(golden.iterrows()):
            t0 = time.perf_counter()
            hits = retrieve(index, g["question"], k)
            m = retrieval_metrics([is_rel(h.payload, g) for h in hits], sum(is_rel(c, g) for c in chunks), k)
            ans, sc, ctx = "", None, "\n\n".join(h.payload["text"] for h in hits)
            if use_llm:
                ans = generate(g["question"], ctx)
            m["latency_ms"] = (time.perf_counter() - t0) * 1000
            if use_llm:
                sc = judge(g["question"], ctx, ans, g["reference_answer"])
                m.update(sc)
            per_q.append(m)
            det.append({"config": label, "qid": qi, "question": g["question"], "category": category(m, sc), "mrr": m["mrr"],
                        "top_chunk": hits[0].payload["text"][:180] if hits else "", "answer": ans,
                        "reference": g["reference_answer"]})
        row = {"config": label, "model": mod, "chunk_words": size, "top_k": k, **pd.DataFrame(per_q).mean().to_dict()}
        rows.append(row)
        details += det
        _save(row, det, overlap, use_llm)
    progress(len(combos), len(combos), "Done")
    return pd.DataFrame(rows), pd.DataFrame(details)


def _save(row, det, overlap, use_llm):
    rid = uuid.uuid4().hex[:12]
    cfg = {k: row[k] for k in ("model", "chunk_words", "top_k")} | {"overlap": overlap, "llm": use_llm}
    met = {k: v for k, v in row.items() if k not in ("config", "model", "chunk_words", "top_k")}
    with db() as c:
        c.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                  (rid, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   json.dumps(cfg, default=str), json.dumps(met, default=float)))
        c.executemany("INSERT INTO results VALUES (?, ?, ?, ?, ?)",
                      [(rid, d["question"], d["category"], d["answer"], json.dumps(d)) for d in det])


def history():
    with db() as c:
        df = pd.read_sql_query("SELECT created, config, metrics FROM runs ORDER BY created DESC", c)
    if df.empty:
        return df
    cfg = pd.json_normalize(df["config"].map(json.loads).tolist())
    met = pd.json_normalize(df["metrics"].map(json.loads).tolist())
    return pd.concat([df[["created"]], cfg, met], axis=1)
