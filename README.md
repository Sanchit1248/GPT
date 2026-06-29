# Decoder-Only Transformer from Scratch (GPT)

A PyTorch implementation of a decoder-only Transformer inspired by the GPT architecture. This project explores the complete lifecycle of a modern language model—from building the architecture from scratch to transfer learning and downstream fine-tuning.

---

## Overview

The project is divided into three stages:

1. **Building a GPT-style decoder-only Transformer from scratch**
2. **Loading pretrained GPT-2 weights**
3. **Fine-tuning for downstream tasks**

The objective was to understand the internals of autoregressive language models instead of treating them as black-box APIs.

---

## Features

- Decoder-only Transformer implemented from scratch in PyTorch
- Causal self-attention with masking
- Multi-head attention
- Layer Normalization
- Residual connections
- Feed-forward networks (MLP blocks)
- Token & positional embeddings
- Autoregressive text generation
- Loading pretrained OpenAI GPT-2 weights
- Fine-tuning for sequence classification
- Fine-tuning for instruction following

---

## Repository Structure

```
.
├── BaseGPT.ipynb                    # GPT implementation and language model training
├── ClassifierFinetuningGPT.ipynb    # Spam/Ham classifier fine-tuning
├── InstructionFinetuningGPT.ipynb   # Instruction tuning
├── the-verdict.txt                  # Training corpus
└── README.md
```

---

## Project Pipeline

### 1. GPT Architecture

Implemented a decoder-only Transformer from scratch including:

- Token embeddings
- Positional embeddings
- Masked Multi-Head Self-Attention
- Feed Forward Network
- Residual connections
- Layer Normalization
- Autoregressive decoding

A small language model was then trained on a custom corpus to verify correctness of the implementation.

---

### 2. Transfer Learning

Instead of training a large model from scratch, pretrained GPT-2 weights were loaded into the custom implementation.

This demonstrates how pretrained language models can be adapted to downstream tasks with significantly less training.

---

### 3. Downstream Fine-Tuning

#### Spam Classification

The pretrained model was fine-tuned as a binary classifier for distinguishing:

- Spam
- Ham

---

#### Instruction Following

The same pretrained model was instruction-tuned to improve its ability to follow natural language prompts.

---

## Tech Stack

- Python
- PyTorch
- NumPy
- GPT-2
- Transformer Architecture

---

## Learning Outcomes

This project provided hands-on experience with:

- Transformer architecture
- Autoregressive language modeling
- Attention mechanisms
- Transfer learning
- Language model fine-tuning
- Sequence classification
- Instruction tuning

---

## Future Improvements

- Train on larger datasets
- Mixed precision (FP16/BF16)
- Flash Attention
- KV Cache for faster inference
- LoRA / QLoRA fine-tuning
- RLHF / GRPO based post-training
- Larger context windows

---

## References

- Attention Is All You Need (Vaswani et al.)
- Improving Language Understanding by Generative Pre-Training
- Language Models are Unsupervised Multitask Learners (GPT-2)
- Build a Large Language Model (From Scratch) — Sebastian Raschka

---

## Acknowledgements

This repository was built as a learning project to gain a deeper understanding of modern Large Language Models by implementing their core components from first principles.
