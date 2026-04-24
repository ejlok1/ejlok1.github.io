---
title: "Episode 1: Using TF-IDF to Identify the Signal from the Noise"
date: 2017-11-25
description: "An introduction to TF-IDF — one of the foundational building blocks of text mining and information retrieval."
tags: ["NLP", "TF-IDF", "text-mining", "classic-series"]
categories: ["Classic Series"]
series: ["NLP from Scratch"]
showToc: true
---

> *This is Episode 1 of the original "New Paradigm" NLP series, first published in November 2017. The techniques here laid the groundwork for everything that came after — including the transformer era. TF-IDF is still very much alive in 2026, playing a key role in hybrid search systems alongside embeddings. — Eu Jin, 2026*

---

## Introduction

Welcome to the first episode of the New Paradigm series! This is the beginning of what I hope will be a long journey exploring Natural Language Processing — extracting meaning from text, one algorithm at a time.

We start with **TF-IDF** — Term Frequency–Inverse Document Frequency. It sounds intimidating, but the core idea is beautifully simple: *a word is important to a document if it appears often in that document, but rarely across all documents.*

Think of it this way. The word "the" appears in every document. It tells you nothing. But the word "neural" appearing frequently in a document, while being rare in your broader corpus? That's a signal.

## The Maths Behind It

TF-IDF is the product of two scores:

**Term Frequency (TF)**: How often does a term appear in a document?

```
TF(t, d) = count of term t in document d / total terms in document d
```

**Inverse Document Frequency (IDF)**: How rare is the term across all documents?

```
IDF(t) = log(N / number of documents containing t)
```

Multiply them together and you get TF-IDF — a score that rewards terms that are distinctive to a document.

## Why It Still Matters in 2026

You might think TF-IDF is a relic. We have embeddings now. We have dense retrievers. We have LLMs.

But here's the thing — **hybrid search** (the backbone of modern RAG pipelines) almost always combines a dense embedding search with a sparse retrieval method. And the most common sparse retrieval method? A variant of TF-IDF called **BM25**.

So that foundational intuition — "what words are distinctive here?" — is baked into every serious production retrieval system running today.

## The Experiment

The original experiment used Python's `scikit-learn` to compute TF-IDF vectors across a small corpus of documents, then measured cosine similarity to find related articles.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are both pets"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Similarity between document 0 and document 2
similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])
print(f"Similarity: {similarity[0][0]:.3f}")
```

The output would show you that documents about animals are more similar to each other than to unrelated text — even with no semantic understanding, just raw term statistics.

## Wrapping Up

TF-IDF is your first lens for understanding text. It doesn't know what words *mean* — it just knows which words are *distinctive*. That's a powerful starting point.

In the next episode, we build on this to tackle sentiment analysis.

---

*[Original post: November 2017 — migrated to New Paradigm v2, 2026]*
