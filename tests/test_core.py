import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rageval import core


class Hit:
    def __init__(self, payload):
        self.payload = payload


def fake_retrieve(idx, q, k):
    overlap = lambda c: -len(set(q.lower().split()) & set(c["text"].lower().split()))
    return [Hit(c) for c in sorted(idx, key=overlap)[:k]]


def offline(monkeypatch, save=False):
    monkeypatch.setattr(core, "build_index", lambda chunks, m: chunks)
    monkeypatch.setattr(core, "retrieve", fake_retrieve)
    if not save:
        monkeypatch.setattr(core, "_save", lambda *a: None)


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE = {"models": MODEL, "sizes": "40", "ks": "3"}


def test_chunk_overlap():
    assert core.chunk("a b c d e f", 4, 1) == ["a b c d", "d e f"]


def test_retrieval_metrics():
    m = core.retrieval_metrics([False, True, False], total_rel=2, k=3)
    assert m["hit_rate"] == 1 and m["mrr"] == 0.5
    assert round(m["precision@k"], 2) == 0.33 and m["recall@k"] == 0.5


def test_category():
    assert core.category({"hit_rate": 0.0}) == "Retrieval miss"
    assert core.category({"hit_rate": 1.0}, {"faithfulness": 0.2, "correctness": 1}) == "Hallucination"


def test_run_benchmark_offline(monkeypatch):
    offline(monkeypatch)
    docs = {"a.txt": "breach must be reported within 72 hours " * 3, "b.txt": "euro coins circulated in 2002 " * 3}
    gold = pd.DataFrame([{"question": "how soon is a breach reported", "reference_answer": "72 hours",
                          "source": "a.txt", "evidence": "72 hours"}])
    summary, detail = core.run_benchmark(docs, gold, ["m"], [8], [1], 0)
    assert summary.loc[0, "hit_rate"] == 1 and detail.loc[0, "category"] == "OK"


def test_api_run_on_sample(monkeypatch):
    import main
    offline(monkeypatch)
    r = TestClient(main.app).post("/api/run", data=BASE)
    assert r.status_code == 200 and len(r.json()["summary"]) == 1 and len(r.json()["detail"]) == 9


def test_qdrant_roundtrip(monkeypatch):
    pytest.importorskip("qdrant_client")
    import numpy as np

    class Fake:
        def encode(self, x, normalize_embeddings=True):
            def vec(t):
                v = np.zeros(64)
                for w in t.lower().split():
                    v[sum(map(ord, w)) % 64] += 1
                return v / (np.linalg.norm(v) or 1)
            return np.array([vec(t) for t in x]) if isinstance(x, list) else vec(x)

    monkeypatch.setattr(core, "get_model", lambda n: Fake())
    monkeypatch.setattr(core, "_save", lambda *a: None)
    root = Path(__file__).parent.parent
    docs = {p.name: p.read_text() for p in (root / "data/sample").glob("*.txt")}
    s, d = core.run_benchmark(docs, pd.read_csv(root / "data/golden.csv"), ["m"], [40], [3], 10)
    assert 0 < s.loc[0, "hit_rate"] <= 1 and len(d) == 9  # crude fake embedder, so only the Qdrant plumbing is checked


def test_api_validation(monkeypatch):
    import main
    offline(monkeypatch)
    c = TestClient(main.app)
    docs = [("docs", ("a.txt", b"hello world", "text/plain"))]
    assert c.post("/api/run", data=BASE, files=docs).status_code == 400            # docs without golden
    gold = [("golden", ("g.csv", b"question,reference_answer,source,evidence\nq,a,missing.txt,x\n", "text/csv"))]
    assert c.post("/api/run", data=BASE, files=docs + gold).status_code == 400     # unknown source
    assert c.post("/api/run", data={**BASE, "overlap": 40}).status_code == 400     # overlap >= chunk
    assert c.post("/api/run", data={**BASE, "models": "nope"}).status_code == 400  # unknown model
    assert c.post("/api/run", data={**BASE, "use_llm": "true"}).status_code == 400 # no key


def test_api_uploads_judge_and_history(monkeypatch, tmp_path):
    import main
    offline(monkeypatch, save=True)
    monkeypatch.setattr(core, "DB_PATH", str(tmp_path / "t.db"))
    core.init_db()
    monkeypatch.setenv("GROQ_API_KEY", "x")
    calls = iter(range(100))
    def fake_llm(prompt, as_json=False):
        if not as_json:
            return "an answer"
        return '{"faithfulness": 0.9, "relevancy": 1, "correctness": 0.8}' if next(calls) % 2 == 0 else "not json"
    monkeypatch.setattr(core, "llm", fake_llm)
    c = TestClient(main.app)
    files = [("docs", ("a.txt", b"breach reported within 72 hours " * 4, "text/plain")),
             ("golden", ("g.csv", b"question,reference_answer,source,evidence\nhow soon is a breach reported,72 hours,a.txt,72 hours\nsecond one,x,a.txt,\n", "text/csv"))]
    r = c.post("/api/run", data={**BASE, "use_llm": "true"}, files=files)
    assert r.status_code == 200, r.text
    row = r.json()["summary"][0]
    assert row["faithfulness"] is not None and row["mrr"] > 0       # judged question averaged, failed judge skipped
    h = c.get("/api/history").json()
    assert len(h) == 1 and h[0]["top_k"] == 3
