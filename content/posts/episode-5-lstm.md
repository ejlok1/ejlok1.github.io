---
title: "Episode 5: Building Castles in the Sky, or a Memory Palace — Part 1 (LSTM)"
date: 2018-03-21
description: "LSTMs gave neural networks a memory. Here's the intuition behind Long Short-Term Memory networks, and why memory architecture matters."
tags: ["NLP", "deep-learning", "LSTM", "RNN", "classic-series"]
categories: ["Classic Series"]
series: ["NLP from Scratch"]
showToc: true
---

> *Episode 5. I used the "memory palace" metaphor here, and looking back, it's actually a great way to think about what attention mechanisms and KV caches do in modern transformers too — they're just much bigger, more flexible memory palaces. — Eu Jin, 2026*

---

## Introduction

Imagine you're reading a novel. You encounter the sentence:

*"The man rushed to the hospital because he had been shot."*

To understand who "he" refers to in that sentence, you need to remember "the man" from earlier. Your brain holds a **memory** across the sequence of words, updating it as you read.

Standard neural networks — feedforward networks — have no such memory. They process each input independently. For text, that's a fundamental limitation.

**Recurrent Neural Networks (RNNs)** were the first attempt to fix this: add a hidden state that passes forward through the sequence. But they suffered from the **vanishing gradient problem** — trying to learn dependencies across long sequences, the gradients would shrink to nothing during backpropagation.

**LSTMs** (Long Short-Term Memory networks) were designed specifically to solve this.

## The Memory Palace Metaphor

Think of an LSTM as a memory palace with a very deliberate curator. At each step, the curator decides:

1. **What to forget** from the existing memory (the Forget Gate)
2. **What new information to store** (the Input Gate)
3. **What to output** from the current memory (the Output Gate)

This gating mechanism is what allows LSTMs to maintain relevant information across hundreds or thousands of timesteps — remembering that the subject of the sentence was "the man" even 50 words later.

## The Architecture

An LSTM cell maintains two states:
- **Cell state** ($C_t$): the long-term memory — a "conveyor belt" that information can travel along with minimal modification
- **Hidden state** ($h_t$): the working memory — what the network outputs at each step

### The Forget Gate

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

A sigmoid output between 0 and 1. `0` = completely forget. `1` = completely remember. The cell state is multiplied by this gate — irrelevant information gets suppressed.

### The Input Gate

```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

Two steps: decide *which* values to update, then create candidate values to add.

### The Output Gate

```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

The output is a filtered version of the cell state, focused on what's relevant right now.

## Why This Matters for Modern AI

LSTMs powered the first wave of production NLP — Google Translate, Siri, Alexa — all ran on LSTM architectures.

Transformers have replaced LSTMs for most language tasks, but the *principle* is the same: **selective memory is the key capability**. What's changed is the mechanism. Instead of learned gates, transformers use **attention** — learning which past tokens to "attend to" based on content, not position. The attention mechanism is, in some ways, a generalised, parallelisable version of what LSTMs do sequentially.

Understanding LSTMs is still the best way to build intuition for why attention works, and why modern LLMs sometimes lose track of long contexts — they're still working within memory constraints, just different ones.

---

*[Original post: March 2018 — migrated to New Paradigm v2, 2026]*
