# RAGBench

Benchmark RAG pipelines on your own documents: compare embedding models, chunk sizes and top-K, and see which questions fail and why.

**Stack:** Python, FastAPI, Qdrant, Sentence Transformers, Groq (LLaMA 3.3 70B), SQLite, custom HTML/JS front end with three.js, pytest, Docker, GitHub Actions.

## Run
```
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY for answer judging
uvicorn main:app --reload
```
Open http://localhost:8000. The first run downloads the embedding models, so allow a minute.

Retrieval metrics (Precision@K, Recall@K, hit rate, MRR, latency) work without a key. Faithfulness, relevancy and correctness use an LLM judge and need `GROQ_API_KEY`.

Your own data: add `.txt`/`.pdf` files and a golden CSV with columns `question, reference_answer, source, evidence`. A chunk counts as relevant when it comes from `source` and contains the `evidence` phrase.

`docker compose up` starts the app with Qdrant. Tests: `python -m pytest`.
