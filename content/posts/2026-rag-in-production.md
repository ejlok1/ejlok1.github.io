---
title: "RAG in Production: What the Tutorials Don't Tell You"
date: 2026-04-14
description: "Everyone has a RAG hello-world. Few talk about what breaks in production. Here's what I've learned building RAG systems for real enterprises."
tags: ["RAG", "LLM", "embeddings", "production", "agentic-ai", "enterprise"]
categories: ["New Series"]
showToc: true
---

## The RAG Hello World Is a Lie

Every RAG tutorial goes like this:

1. Load a PDF
2. Chunk it into 512-token segments
3. Embed with OpenAI
4. Store in a vector database
5. At query time: embed the question, find the top-3 chunks, stuff into a prompt
6. 🎉 "Look, it answered a question about my document!"

This works great for demos. It falls apart in production.

I've built RAG systems for a semiconductor company's internal SOP retrieval, a financial services compliance auditing pipeline, and a hospital's clinical documentation assistant. Here's what the tutorials don't cover.

## Problem 1: Chunking Is Not Trivial

The default advice is "chunk at 512 tokens with 50 token overlap." This is fine for clean, well-structured text. Real enterprise documents are:

- **PDFs with tables**: chunking by token count splits a table in the middle, destroying the row/column relationships
- **SOPs with numbered steps**: step 3 and step 4 might be in different chunks, breaking the logical flow
- **Multi-language documents**: TSMC SOPs mix English, Traditional Chinese, and technical acronyms
- **Scanned PDFs**: poor OCR introduces noise that confuses embedding models

Better approaches:
- **Semantic chunking**: split on meaning shifts, not token counts
- **Structure-aware chunking**: respect headings, tables, lists as natural boundaries
- **Hierarchical indexing**: store both a summary and the full chunk; retrieve at summary level, return the full chunk

## Problem 2: Embedding Model Choice Actually Matters

Not all embedding models are equal, and the differences aren't marginal.

I evaluated three models on a technical document retrieval task:

| Model | Dims | NDCG@5 | Latency (ms) | Notes |
|---|---|---|---|---|
| text-embedding-3-small | 1536 | 0.71 | 12 | Good baseline |
| text-embedding-3-large | 3072 | 0.79 | 18 | Best OpenAI |
| **BGE-M3** | 1024 | **0.84** | 8 | Best overall |
| E5-large-v2 | 1024 | 0.76 | 9 | Solid open-source |

BGE-M3 won on our technical corpus. Why? It was trained with **Matryoshka Representation Learning (MRL)** — the embedding dimensions are nested, so you can truncate to 512 or 256 dimensions for storage savings without catastrophic quality loss. It's also multilingual by default, which matters for APAC enterprise deployments.

Key lesson: **always evaluate on your own data**. Benchmark numbers on public datasets don't predict performance on your specific domain.

## Problem 3: Pure Vector Search Is Naive

Vector similarity finds *semantically similar* text. But users often know exact terms — product codes, regulation numbers, names.

If a compliance officer asks for *"ASIC RG 271 obligations"*, pure semantic search might return paragraphs about "regulatory compliance requirements" that have similar vector representations — but miss the document that contains the exact string "RG 271".

**Hybrid search** — combining dense (vector) and sparse (BM25/TF-IDF) retrieval — consistently outperforms either alone:

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

def hybrid_search(query, documents, alpha=0.5, top_k=5):
    """
    Combine BM25 and dense retrieval scores.
    alpha: weight for dense scores (1-alpha for BM25)
    """
    # Dense retrieval
    model = SentenceTransformer("BAAI/bge-m3")
    query_emb = model.encode(query)
    doc_embs = model.encode(documents)
    dense_scores = np.dot(doc_embs, query_emb)
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

    # BM25 retrieval
    tokenized = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = np.array(bm25.get_scores(query.split()))
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

    # Combine
    combined = alpha * dense_scores + (1 - alpha) * bm25_scores
    return np.argsort(combined)[::-1][:top_k]
```

The `alpha` parameter is a tunable — I typically start at 0.5 and adjust based on whether queries tend to be keyword-heavy or concept-heavy.

## Problem 4: Metadata Filtering Is Not Optional

Imagine a RAG system for a company with 10 years of policy documents. Without filtering, the LLM might retrieve a deprecated 2019 policy and confidently present it as current.

Every chunk needs metadata, and queries need to filter on it:

```python
# When indexing
chunk_metadata = {
    "document_id": "policy-data-retention-v3",
    "effective_date": "2024-01-01",
    "department": "legal",
    "status": "active",  # or "superseded", "draft"
    "language": "en"
}

# When querying — filter BEFORE semantic search
results = vector_db.query(
    embedding=query_embedding,
    filter={
        "status": {"$eq": "active"},
        "department": {"$in": ["legal", "compliance"]},
        "effective_date": {"$gte": "2023-01-01"}
    },
    top_k=5
)
```

This sounds obvious. You'd be surprised how many production RAG systems skip it.

## Problem 5: Evaluation Is Hard But Non-Negotiable

The hardest question in RAG: *"Is it working?"*

Anecdotal testing ("I asked it a question and it sounded right") is not an answer. You need:

- **A golden dataset**: question-answer pairs where you know the correct answer AND the correct source document
- **Retrieval metrics**: recall@k (did the right chunk appear in the top-k?), NDCG
- **Generation metrics**: faithfulness (did the answer come from the retrieved context?), answer relevance

Frameworks like **deepeval** and **RAGAS** automate a lot of this. But the golden dataset is manual work and there's no way around it.

My rule: **before shipping, you need at least 50 evaluated examples**. For regulated industries (finance, healthcare), more.

## The Architecture That Works

After several production deployments, here's the architecture I default to:

```
User Query
    │
    ▼
Query Preprocessing
(spell check, expand acronyms, decompose compound questions)
    │
    ▼
Hybrid Retrieval (BM25 + Dense)
with metadata filters
    │
    ▼
Re-ranking
(cross-encoder model to re-score top-20 → top-5)
    │
    ▼
Context Assembly
(deduplicate, order by relevance, add metadata citations)
    │
    ▼
LLM Generation
with instructions to cite sources and flag uncertainty
    │
    ▼
Post-processing
(validate citations, extract structured data if needed)
```

Each stage is independently testable. That matters when something breaks.

## Closing Thought

RAG is not a product. It's an architecture. Getting it right requires thinking carefully about each component — and being honest about where the simple version fails.

The tutorials will get you to a demo in an afternoon. Getting it to production-grade takes weeks of evaluation, iteration, and domain-specific tuning.

But it's worth it. A well-built RAG system is transformative for knowledge-intensive work. I've seen compliance officers close audit findings in minutes instead of hours. I've seen engineers find the right SOP on a first search instead of a 30-minute dig through a shared drive.

That's the real benchmark: does it save real people real time? 

---

*Eu Jin Lok — Melbourne, April 2026*
