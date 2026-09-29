# Bilingual LLM from Scratch

Building a bilingual (English + Portuguese) large language model from the ground up — from counting character pairs in pure Python to a chat model trained on a rented GPU — and documenting every step in a public article series.

**Start:** September 28, 2026 · **Planned finish:** March 2027 · **Cadence:** one public deliverable every week

## Why

Open models are excellent and free to download. Using one is not the same as understanding one.

This project rebuilds every piece by hand — tokenizer, gradients, attention, the training loop, fine-tuning — so that each decision can be explained, measured and tested. It is written from the perspective of a software engineer with an object-oriented background: every concept is mapped to classes, objects and interfaces, and the math is introduced only at the moment it is needed.

Read the full manifesto: [English](articles/00-manifesto/en.md) · [Português](articles/00-manifesto/pt.md)

## Principles

- **From scratch.** Libraries are introduced only after the thing they replace has been built by hand once.
- **Test-first.** Every component is built with TDD: the test is written before the code. A model built from scratch deserves the same engineering discipline as any other software.
- **Reproducible.** Every number in an article is produced by code in this repository.
- **Open for real.** Code under MIT, articles under CC BY 4.0, and training data restricted to permissive licenses (ODC-By, Apache, MIT, CC0, CC BY).
- **Bilingual.** The final model speaks English and Portuguese, and every article is published in both languages.

## Roadmap

| # | Mission | Code | Article (EN) | Article (PT) |
|---|---|---|---|---|
| 0 | Manifesto and repository | — | [manifesto](articles/00-manifesto/en.md) | [manifesto](articles/00-manifesto/pt.md) |
| 1 | Count-based bigram model (pure Python) | | | |
| 2 | Micrograd: gradients and backpropagation | | | |
| 3 | Tensors and a first neural network (PyTorch) | | | |
| 4 | BPE tokenizer | | | |
| 5 | Attention | | | |
| 6 | Assembling a GPT | | | |
| 7 | Pretraining on a MacBook | | | |
| 8 | Modern architecture | | | |
| 9 | Scale: data, scaling laws, rented GPU | | | |
| 10 | Supervised fine-tuning: from text completer to assistant | | | |
| 11 | Release and retrospective | | | |

Links are filled in as each mission ships.

## Repository layout

```
missions/NN-slug/    code and tests for each mission, with a short README
articles/NN-slug/    en.md and pt.md, the canonical source of each article, plus its covers
tools/covers/        script that generates the article covers as SVG and PNG
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/luizcampos331/bilingual-llm-from-scratch.git
cd bilingual-llm-from-scratch
uv sync
```

Each mission's README explains how to run its code and tests.

## Who writes what

All code and tests are written by hand, test-first. The articles are written with AI assistance, based on the code, results and mistakes recorded in this repository.

## Hardware

Missions 1–8 run locally on an Apple M3 laptop with 8 GB of unified memory. Missions 9–10 use rented GPUs, and the real costs are published in the articles.

## License

- **Code:** [MIT](LICENSE)
- **Articles:** [CC BY 4.0](articles/LICENSE)
- **Model weights:** to be decided in mission 11, since they depend on the licenses of the training data.
