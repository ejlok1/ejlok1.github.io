---
title: "Episode 6: Building Castles in the Sky, or a Memory Palace — Part 2 (GRU)"
date: 2018-04-13
description: "GRUs streamlined the LSTM. Fewer gates, faster training, comparable accuracy. Here's how — and why it's still relevant."
tags: ["NLP", "deep-learning", "GRU", "RNN", "classic-series"]
categories: ["Classic Series"]
series: ["NLP from Scratch"]
showToc: true
---

> *Episode 6 — the final post of the original series before a long hiatus. I was right that GRUs were underrated. They're still used in production systems today where latency and model size matter. And the 20% training speedup I documented? In 2026, that kind of efficiency thinking is even more critical when you're running inference at scale. — Eu Jin, 2026*

---

## Introduction

So what do you know — more gates... and more memories.

![You shall not pass meme — Gandalf at the gate as a metaphor for GRU update gates](https://mungingdata.wordpress.com/wp-content/uploads/2018/04/gandalf-and-the-gate.jpg)

Following on from Episode 5 on LSTMs, this post explores **Gated Recurrent Units (GRUs)** — one of the most popular LSTM variants, introduced in 2014 by Kyunghyun Cho and colleagues.

The elevator pitch: GRU is a more simplified version of LSTM, trains faster, and is surprisingly competitive on accuracy.

## I Forget — Why Does This Matter?

Google Translate, Amazon Alexa, Apple Siri — all were running on LSTMs when this was first written. GRU would have been equally good. The reason they hadn't switched (in my opinion at the time) was that GRU was so new, teams didn't want to risk it.

That conservatism is very recognisable in enterprise AI. Organisations are still running on older stacks in 2026 for the same reason.

## GRU vs LSTM: What Changed

Where LSTM has **4 gates**, GRU has **3**:

| LSTM | GRU |
|------|-----|
| Forget Gate | Reset Gate |
| Input Gate | (merged into Reset Gate) |
| Candidate Gate | Memory Gate |
| Output Gate | Update Gate |

The two-state design (cell state + hidden state) of LSTM is collapsed into **one state** in GRU. Simpler wiring, fewer parameters.

### The Reset Gate

Controls how much of the previous hidden state $h_{t-1}$ flows into the new memory content.

```
r_t = σ(W_r · [h_{t-1}, x_t])
```

Think of it as: *"How much of what I remembered is relevant to understanding this new input?"*

In the sentence *"The man was shot by a gun"*, when predicting "gun" after "shot", the reset gate suppresses long-ago context (the word "hospital") and focuses on the immediate context (the word "shot"). The gate value approaches 0, zeroing out old memory.

### The Memory Gate (Candidate Hidden State)

```
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])
```

Blends the filtered previous state with the current input. A tanh ensures values stay between -1 and 1.

### The Update Gate

This is what makes GRU special. A **single gate** does double duty — it simultaneously controls:
- How much of the previous memory $h_{t-1}$ to keep
- How much of the new candidate memory $h̃_t$ to adopt

```
z_t = σ(W_z · [h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

If $z_t = 0.8$: 80% new memory content, 20% old memory. Gandalf the update gate decides who passes.

## The Experiment: GRU vs LSTM Speed

I ran a direct comparison. Results across 1 epoch and projected to 100 epochs:

| Architecture | 1 Epoch (min) | 100 Epochs (hrs) |
|---|---|---|
| LSTM | 12–13 | 20.5 |
| CNN + LSTM | 6 | 9.5 |
| Double LSTM | 21–22 | 35.3 |
| **GRU** | **10** | **16.1** |
| **CNN + GRU** | **5** | **7.7** |
| **Double GRU** | **16–17** | **27.4** |

**~20% faster across the board.** For long training runs, that's hours saved per experiment.

Accuracy was comparable. In several Kaggle competitions I observed GRU slightly outperforming LSTM — likely due to better regularisation from the simpler architecture.

## Final Words (Original) + 2026 Postscript

When I first closed this episode in April 2018, I said I was taking a long hiatus. I did not expect it to be 8 years.

A lot has changed. GRUs and LSTMs have been largely supplanted by transformer architectures for NLP. But the core ideas — **selective memory, gating, learning what to keep and what to forget** — are still foundational. Modern attention mechanisms are, in a sense, soft, parallel, content-addressable versions of these gates.

And efficiency still matters. In 2026, where inference costs at scale are real business constraints, the instinct that drove GRU — *"can we get the same result with fewer parameters?"* — is more important than ever.

---

*[Original post: April 2018 — migrated to New Paradigm v2, 2026]*
