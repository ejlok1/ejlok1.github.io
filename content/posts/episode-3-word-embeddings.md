---
title: "Episode 3: Get Powerful Predictors through Word Embedding"
date: 2018-01-15
description: "Word2Vec changed how we think about representing language. Here's how it works and why it matters — even in the transformer era."
tags: ["NLP", "word-embeddings", "word2vec", "deep-learning", "classic-series"]
categories: ["Classic Series"]
series: ["NLP from Scratch"]
showToc: true
---

> *Episode 3. This one got the most comments of the original series — embeddings were genuinely mind-bending when they first appeared. The core intuition here (that words can be represented as dense vectors that capture meaning) is the same intuition that powers every modern LLM. We just scaled it up by several orders of magnitude. — Eu Jin, 2026*

---

## Introduction

What if a computer could understand that "king" and "queen" are related? That "Paris" is to "France" as "Tokyo" is to "Japan"? That "walked", "walks", and "walking" all refer to the same action?

That's word embedding — and it was a genuine paradigm shift when it arrived.

## The Problem with One-Hot Encoding

Before embeddings, the standard way to represent words was **one-hot encoding**: a vector of zeros with a single `1` at the position corresponding to that word.

If your vocabulary has 50,000 words, every word is a 50,000-dimensional vector with exactly one non-zero value. The word "cat" and "kitten" are completely orthogonal — no notion that they're related. The word "Paris" and "London" are just as different as "Paris" and "photosynthesis".

This is mathematically clean but semantically empty.

## Enter Word2Vec

In 2013, a team at Google published **Word2Vec** — a neural approach to learning dense word representations. The key insight was almost philosophical:

> *"You shall know a word by the company it keeps."* — J.R. Firth, 1957

Train a shallow neural network to predict a word from its context (or predict context from a word), and the internal representations the network learns will encode semantic relationships.

The famous result:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

That's not hand-coded. The model learned it from raw text.

## How It Works

There are two Word2Vec architectures:

**CBOW (Continuous Bag of Words)**: Given context words, predict the centre word.
- Input: ["the", "cat", "on", "mat"] → predict "sat"

**Skip-gram**: Given a centre word, predict the context words.
- Input: "sat" → predict ["the", "cat", "on", "mat"]

Skip-gram works better for rare words. CBOW is faster. In practice, you'd use a library like `gensim`:

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "the", "kingdom"],
    ["paris", "is", "the", "capital", "of", "france"],
    ["tokyo", "is", "the", "capital", "of", "japan"]
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Find most similar words
similar = model.wv.most_similar("king")
print(similar)
```

## The Lineage to Today

Word2Vec gave us 100–300 dimensional dense vectors. Then came:

- **GloVe** (2014) — global co-occurrence statistics
- **FastText** (2016) — subword information, better for morphologically rich languages
- **ELMo** (2018) — context-dependent embeddings from bidirectional LSTMs
- **BERT** (2018) — transformer-based, contextual, the real game-changer
- **Modern embedding models** (2023–2025) — BGE-M3, E5, text-embedding-3-large — 1024–4096 dimensions, multilingual, instruction-tuned

The idea didn't change. The scale did.

In modern RAG systems, when you embed a chunk of text and store it in a vector database, you're doing exactly what Word2Vec pioneered — just with a vastly more powerful encoder.

---

*[Original post: January 2018 — migrated to New Paradigm v2, 2026]*
