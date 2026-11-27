
"""RAGBench core: chunking, Qdrant retrieval, retrieval metrics, LLM judge, SQLite run storage."""

import gc
import io
import itertools
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_PATH = os.getenv("DB_PATH", "rageval.db")
LLM_METRICS = ("faithfulness", "relevancy", "correctness")
ENCODE_BATCH_SIZE = 8


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
        c.execute(
            "CREATE TABLE IF NOT EXISTS runs "
            "(id TEXT PRIMARY KEY, created TEXT, config TEXT, metrics TEXT)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS results "
            "(run_id TEXT, question TEXT, category TEXT, answer TEXT, detail TEXT)"
        )


# ---------- documents ----------

def read_file(name, data):
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(
            p.extract_text() or ""
            for p in PdfReader(io.BytesIO(data)).pages
        )
    return data.decode("utf-8", errors="ignore")


def chunk(txt, size, overlap=0):
    words = txt.split()
    step = max(size - overlap, 1)
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), step)
    ]


# ---------- vector search (Qdrant + FastEmbed) ----------

def get_model(name):
    """Load an ONNX embedding model using FastEmbed."""
    from fastembed import TextEmbedding

    return TextEmbedding(
        model_name=name,
        threads=1
    )


def _encode(model, texts, batch_size=None):
    """Encode text using FastEmbed or compatible test doubles."""

    if hasattr(model, "embed"):
        import numpy as np

        single = isinstance(texts, str)
        items = [texts] if single else texts

        vectors = np.asarray(
            list(
                model.embed(
                    items,
                    batch_size=batch_size or ENCODE_BATCH_SIZE
                )
            ),
            dtype="float32"
        )

        return vectors[0] if single else vectors

    # Compatibility with the existing unit tests.
    kwargs = {"normalize_embeddings": True}

    if batch_size is not None:
        kwargs["batch_size"] = batch_size
        kwargs["show_progress_bar"] = False
        kwargs["convert_to_numpy"] = True

    try:
        return model.encode(texts, **kwargs)
    except TypeError:
        return model.encode(
            texts,
            normalize_embeddings=True
        )


def build_index(chunks, model_name):
    if not chunks:
        raise ValueError("No text could be extracted from the documents.")

    from qdrant_client import QdrantClient, models

    model = get_model(model_name)
    client = None
    collection_name = "bench_" + uuid.uuid4().hex[:8]

    try:
        url = os.getenv("QDRANT_URL")

        client = (
            QdrantClient(url=url)
            if url
            else QdrantClient(":memory:")
        )

        # Encode only the first small batch to discover vector dimensions.
        first_batch = chunks[:ENCODE_BATCH_SIZE]

        first_vectors = _encode(
            model,
            [item["text"] for item in first_batch],
            ENCODE_BATCH_SIZE
        )

        dimension = first_vectors.shape[1]

        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE
            )
        )

        def upload_batch(start, batch, vectors):
            points = [
                models.PointStruct(
                    id=start + i,
                    vector=vector.tolist(),
                    payload=batch[i]
                )
                for i, vector in enumerate(vectors)
            ]

            client.upsert(
                collection_name=collection_name,
                points=points
            )

            del points

        upload_batch(0, first_batch, first_vectors)

        del first_vectors, first_batch
        gc.collect()

        for start in range(
            ENCODE_BATCH_SIZE,
            len(chunks),
            ENCODE_BATCH_SIZE
        ):
            batch = chunks[
                start:start + ENCODE_BATCH_SIZE
            ]

            vectors = _encode(
                model,
                [item["text"] for item in batch],
                ENCODE_BATCH_SIZE
            )

            upload_batch(start, batch, vectors)

            del vectors, batch
            gc.collect()

        return client, collection_name, model

    except Exception:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

        del model
        gc.collect()
        raise


def retrieve(index, question, k):
    client, name, model = index

    query_vector = _encode(
        model,
        question
    ).tolist()

    return client.query_points(
        collection_name=name,
        query=query_vector,
        limit=k
    ).points


def close_index(index):
    # Offline tests replace build_index with a list of chunks.
    if not isinstance(index, tuple) or len(index) != 3:
        return

    client, name, model = index

    try:
        client.delete_collection(name)
    except Exception:
        pass

    try:
        client.close()
    except Exception:
        pass

    del client, model
    gc.collect()


# ---------- metrics ----------

def is_rel(chunk_, row):
    ev = (
        ""
        if pd.isna(row["evidence"])
        else str(row["evidence"]).lower()
    )

    return (
        chunk_["source"] == row["source"]
        and (not ev or ev in chunk_["text"].lower())
    )


def retrieval_metrics(rel, total_rel, k):
    got = sum(rel)

    first = next(
        (i for i, x in enumerate(rel, 1) if x),
        None
    )

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

    kw = (
        {"response_format": {"type": "json_object"}}
        if as_json else {}
    )

    for attempt in range(5):
        try:
            response = Groq().chat.completions.create(
                model=os.getenv(
                    "GROQ_MODEL",
                    "openai/gpt-oss-120b"
                ),
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kw
            )

            return response.choices[0].message.content

        except Exception:
            if attempt == 4:
                raise

            time.sleep(4 * 2 ** attempt)


def generate(question, ctx):
    return llm(
        'Answer using ONLY the context. '
        'If the answer is not there, say "I don\'t know".\n\n'
        f"Context:\n{ctx}\n\nQuestion: {question}"
    )


def judge(question, ctx, answer, reference):
    try:
        raw = json.loads(
            llm(
                'Score from 0 to 1 and reply as JSON '
                '{"faithfulness": x, "relevancy": x, "correctness": x}. '
                "faithfulness: every claim in the answer is supported "
                "by the context. "
                "relevancy: the answer addresses the question. "
                "correctness: the answer matches the reference.\n"
                f"Question: {question}\n"
                f"Context: {ctx}\n"
                f"Answer: {answer}\n"
                f"Reference: {reference}",
                True
            )
        )

        return {
            key: min(max(float(raw[key]), 0.0), 1.0)
            for key in LLM_METRICS
        }

    except (ValueError, KeyError, TypeError, AttributeError):
        return {
            key: float("nan")
            for key in LLM_METRICS
        }


# ---------- benchmark ----------

def run_benchmark(
    docs,
    golden,
    models,
    sizes,
    ks,
    overlap=10,
    use_llm=False,
    progress=lambda *a: None
):
    rows = []
    details = []

    # Only one model/chunk-size index is kept at a time.
    configurations = list(
        itertools.product(models, sizes)
    )

    total_runs = len(configurations) * len(ks)
    completed = 0

    for mod, size in configurations:
        label_base = mod.split("/")[-1]
        label = f"{label_base} ({size}w)"

        progress(completed, total_runs, label)

        chunks = [
            {"source": source, "text": text_chunk}
            for source, document in docs.items()
            for text_chunk in chunk(document, size, overlap)
        ]

        index = None

        try:
            index = build_index(chunks, mod)

            for k in ks:
                config_label = (
                    f"{label_base} ({size}w, k={k})"
                )

                per_q = []
                det = []

                for qi, (_, g) in enumerate(golden.iterrows()):
                    t0 = time.perf_counter()

                    hits = retrieve(
                        index,
                        g["question"],
                        k
                    )

                    relevant = [
                        is_rel(hit.payload, g)
                        for hit in hits
                    ]

                    total_relevant = sum(
                        is_rel(c, g)
                        for c in chunks
                    )

                    metrics = retrieval_metrics(
                        relevant,
                        total_relevant,
                        k
                    )

                    answer = ""
                    scores = None

                    context = "\n\n".join(
                        hit.payload["text"]
                        for hit in hits
                    )

                    if use_llm:
                        answer = generate(
                            g["question"],
                            context
                        )

                    metrics["latency_ms"] = (
                        time.perf_counter() - t0
                    ) * 1000

                    if use_llm:
                        scores = judge(
                            g["question"],
                            context,
                            answer,
                            g["reference_answer"]
                        )

                        metrics.update(scores)

                    per_q.append(metrics)

                    det.append({
                        "config": config_label,
                        "qid": qi,
                        "question": g["question"],
                        "category": category(metrics, scores),
                        "mrr": metrics["mrr"],
                        "top_chunk": (
                            hits[0].payload["text"][:180]
                            if hits else ""
                        ),
                        "answer": answer,
                        "reference": g["reference_answer"]
                    })

                row = {
                    "config": config_label,
                    "model": mod,
                    "chunk_words": size,
                    "top_k": k,
                    **pd.DataFrame(per_q).mean().to_dict()
                }

                rows.append(row)
                details.extend(det)

                _save(row, det, overlap, use_llm)

                completed += 1

                progress(
                    completed,
                    total_runs,
                    config_label
                )

                del per_q, det, row
                gc.collect()

        finally:
            close_index(index)
            del chunks
            gc.collect()

    progress(total_runs, total_runs, "Done")

    return (
        pd.DataFrame(rows),
        pd.DataFrame(details)
    )


def _save(row, det, overlap, use_llm):
    run_id = uuid.uuid4().hex[:12]

    config = {
        key: row[key]
        for key in ("model", "chunk_words", "top_k")
    } | {
        "overlap": overlap,
        "llm": use_llm
    }

    metrics = {
        key: value
        for key, value in row.items()
        if key not in (
            "config",
            "model",
            "chunk_words",
            "top_k"
        )
    }

    with db() as c:
        c.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                json.dumps(config, default=str),
                json.dumps(metrics, default=float)
            )
        )

        c.executemany(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    detail["question"],
                    detail["category"],
                    detail["answer"],
                    json.dumps(detail)
                )
                for detail in det
            ]
        )


def history():
    with db() as c:
        df = pd.read_sql_query(
            "SELECT created, config, metrics "
            "FROM runs ORDER BY created DESC",
            c
        )

    if df.empty:
        return df

    config = pd.json_normalize(
        df["config"].map(json.loads).tolist()
    )

    metrics = pd.json_normalize(
        df["metrics"].map(json.loads).tolist()
    )

    return pd.concat(
        [df[["created"]], config, metrics],
        axis=1
    )
