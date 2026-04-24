---
title: "Episode 4: A Picture Spoken by 500 Words, According to CNN"
date: 2018-02-24
description: "CNNs are for images, right? Not quite. Here's how Convolutional Neural Networks became surprisingly effective text classifiers."
tags: ["NLP", "deep-learning", "CNN", "text-classification", "classic-series"]
categories: ["Classic Series"]
series: ["NLP from Scratch"]
showToc: true
---

> *Episode 4. CNNs for text was one of those ideas that felt wrong until it worked — and it worked really well. The architecture isn't widely used for NLP anymore (transformers won that battle), but the intuition about local feature detection is still valuable, and the same CNN is quietly doing a lot of work in multimodal models today. — Eu Jin, 2026*

---

## Introduction

When you think of Convolutional Neural Networks, you probably think of images — detecting edges, recognising faces, classifying whether a photo is a cat or a dog.

So what's a CNN doing in an NLP blog?

It turns out that the same mechanism that detects visual patterns in a grid of pixels can detect *linguistic patterns* in a sequence of word vectors. And in 2015–2018, CNNs were state-of-the-art for text classification tasks.

## The Core Idea: Local Feature Detection

A CNN applies small **filters** (kernels) that slide across the input, looking for patterns at different positions. For images, a filter might detect a horizontal edge. For text, a filter slides over n-gram windows — sequences of 2, 3, or 4 consecutive words — and learns to detect meaningful phrases.

The key property: **position invariance**. A CNN doesn't care if the phrase "not bad" appears at the start or end of a sentence — it detects it wherever it is.

## CNN Architecture for Text

1. **Embedding layer**: Convert each word to a dense vector (Word2Vec or trained from scratch)
2. **Convolutional layer**: Apply multiple filters of different sizes (bigrams, trigrams, 4-grams)
3. **Max pooling**: For each filter, keep the strongest activation across the sequence
4. **Dense layer + softmax**: Classify the document

```python
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[2,3,4], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x)).squeeze(3)  # [batch, filters, seq_len - k + 1]
            p = torch.max(c, dim=2)[0]           # [batch, filters]
            pooled.append(p)
        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.fc(x)
```

## Results and Legacy

In the original experiments, CNNs achieved competitive accuracy on sentiment classification benchmarks — often matching or beating LSTMs at a fraction of the training time. The combination of **CNN + LSTM** (feature extraction + sequential memory) was a popular hybrid approach.

The transformer architecture eventually superseded both for most NLP tasks. But CNNs are far from dead:

- **Multimodal models** use CNN-based vision encoders to produce image embeddings that LLMs can reason over
- **Lightweight production models** still use CNNs where latency matters
- **Signal processing** for time-series and audio still heavily relies on CNN intuitions

The phrase "a picture spoken by 500 words" captures something real: CNNs learned that some word sequences *paint a picture* of sentiment as clearly as any pixel pattern.

---

*[Original post: February 2018 — migrated to New Paradigm v2, 2026]*
