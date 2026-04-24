---
title: "Episode 2: Sentiment Analysis as a Service and a Data Product"
date: 2017-12-10
description: "Building a sentiment analysis pipeline that goes beyond accuracy — turning an ML model into an actual data product."
tags: ["NLP", "sentiment-analysis", "data-product", "classic-series"]
categories: ["Classic Series"]
series: ["NLP from Scratch"]
showToc: true
---

> *Episode 2 of the original series. What I found most interesting re-reading this in 2026 is how the "sentiment as a service" framing prefigured what we now call LLM-powered pipelines. The patterns are the same — it's just that the model in the middle got a lot smarter. — Eu Jin, 2026*

---

## Introduction

In Episode 1, we learned to identify *what* words matter using TF-IDF. Now we go a step further: can we identify *how* a piece of text *feels*? That's sentiment analysis.

But more importantly — how do you go from a working model to something that actually delivers value in production? That's the "as a service" part of this post.

## What Is Sentiment Analysis?

Sentiment analysis is the classification of text into emotional polarity — most commonly **positive**, **negative**, or **neutral**. It sounds simple. In practice, it's wonderfully messy:

- *"This is not bad at all"* — double negative, actually positive
- *"The flight was on time"* — neutral for most, deeply positive for frequent flyers
- Sarcasm, slang, domain-specific language — all break naive classifiers

The classical approach uses a **lexicon-based method**: a dictionary of words scored for sentiment. The VADER lexicon, for example, knows that "amazing" scores +3.1 and "terrible" scores -3.4. Sum up the scores, apply some heuristics for negation and punctuation, and you have a sentiment score.

## Building a Sentiment Pipeline

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

reviews = [
    "The model was incredibly accurate and easy to use!",
    "Terrible experience. The system crashed twice.",
    "It was fine. Nothing special."
]

for review in reviews:
    scores = analyzer.polarity_scores(review)
    print(f"{review[:40]}... → compound: {scores['compound']:.2f}")
```

The `compound` score ranges from -1 (most negative) to +1 (most positive). A simple threshold (e.g. > 0.05 = positive) gives you a working classifier.

## From Model to Data Product

The real insight from this episode wasn't the model — it was the framing. A sentiment *model* is a function. A sentiment *data product* is a system that:

1. Accepts text at scale (an API endpoint, a streaming pipeline)
2. Returns structured, usable output (not just a label — a score, a confidence, a breakdown)
3. Can be monitored, versioned, and improved over time
4. Has a clear consumer — a dashboard, a downstream model, a business process

That framing has only become more relevant. In 2026, when we talk about "LLM-powered pipelines", we're still building data products. The model in the middle just happens to be a 70B parameter transformer instead of a lexicon lookup.

## What's Changed Since 2017

Back then, getting > 85% accuracy on a sentiment task required careful feature engineering or fine-tuning a smaller model. Today, you can prompt `claude-sonnet-4` and get nuanced sentiment *with reasoning* in a single API call. The barrier to the model is gone.

The barrier that remains — and the one this episode was really about — is the **productionisation**. Getting it into a pipeline. Making it reliable. Making it useful to a business. That hasn't changed at all.

---

*[Original post: December 2017 — migrated to New Paradigm v2, 2026]*
