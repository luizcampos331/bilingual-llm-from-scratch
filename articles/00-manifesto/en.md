# I'm building an LLM from scratch. In public, in two languages.

*Series "Bilingual LLM from Scratch" · Article 0 · [Leia em português](pt.md)*

In 2026, one command downloads an excellent language model and runs it on your laptop. So why would anyone spend six months building a worse one?

Because using is not understanding, and I want to understand.

## Why from scratch, and why now

Today's open models are extraordinary. You can fine-tune one in an afternoon and have something useful by evening. But fine-tuning teaches you to turn the knobs of a machine that stays closed. You learn *that* it works, not *why*.

Building from scratch forces the questions the shortcut hides:

- What exactly is a token, and why does Portuguese cost more tokens than English?
- What is the model actually learning when the loss goes down?
- What does attention really do, without the hand-waving?
- When training goes wrong, where is the bug?

I've spent ten years as a software developer, mostly in Node.js, shipping production back-end, front-end and mobile applications and building entire AWS infrastructures. I learned every one of those things by building, not by reading. I'm going to learn language models the same way: by writing every piece myself.

## What I'm building

Twelve missions over 25 weeks, from September 28, 2026 to March 2027:

1. A model that predicts the next letter by counting character pairs, in pure Python.
2. An automatic differentiation engine, the machinery that tells each number in a model which way to move, written from scratch.
3. A first neural network in PyTorch.
4. A BPE tokenizer, the same family of algorithm that splits text into pieces for GPT models.
5. Attention.
6. A complete GPT, assembled piece by piece and validated against the official GPT-2 weights.
7. Pretraining on a MacBook.
8. The architectural upgrades modern models use, each one measured in isolation.
9. Scale: real data, scaling laws and a rented GPU.
10. Supervised fine-tuning: turning a text completer into an assistant.
11. Release and retrospective.

Through mission 8, everything runs on an Apple M3 laptop with 8 GB of memory. That constraint is deliberate: it forces me to understand every byte training consumes. A rented GPU only shows up in mission 9, and I'll publish exactly what it cost.

## The promise to you

**LLMs explained for programmers.** If you know what a class, an interface and an object are, you have enough background. Every new concept is bridged to something a developer already knows, and the math shows up when it's needed, not before.

**Built test-first.** Every component is written with TDD: the test comes before the code. A model built from scratch deserves the same rigor as any other software. Testing machine learning code has its own challenges, like randomness and approximate numbers, and they become part of the series too.

**Everything reproducible.** Every number I publish comes from code in the repository. If I claim a tokenizer spends 30% more tokens on Portuguese, you can run the experiment and check.

**Two languages.** Every article comes out in English and Portuguese, and the final model will be bilingual too. Good technical material about LLMs in Portuguese is still scarce, and small open models that actually speak Portuguese are even scarcer.

**Open for real.** Code under MIT, articles under CC BY 4.0, and training data restricted to permissive licenses, so anyone can use the model without fine print.

## What success looks like in March 2027

Criteria anyone can check, not impressions:

- Twelve missions, all built test-first, in a public repository.
- Twelve articles in English and twelve in Portuguese, plus a public deliverable every week.
- A bilingual model on the Hugging Face Hub with a model card, trained only on permissively licensed data and evaluated in both languages.
- A chat interface you can run on your own machine.
- The total training cost, published to the cent.
- A consolidated guide for anyone who wants to walk the same path.

## What this model will not be

Better to set expectations now.

**It won't compete with GPT, Claude, Llama or Qwen.** The final model will be GPT-2 sized, a few hundred million parameters at most. It will make mistakes, invent facts and lose track of long conversations. The goal isn't to beat anyone. It's to understand every decision that made it work, and fail.

**It's not a product.** It's a study model, documented end to end.

**It's not a course taught by a machine learning expert.** I'm an experienced software engineer learning machine learning rigorously and in public. Every article has a section called "where I got it wrong". It will probably be the most useful part.

## Who writes what

All the code in this project is mine, written test-first, one test at a time. That's where the learning happens, and that's what I want to demonstrate. The articles are written with AI assistance, always from my code, my numbers and my mistakes. No result is made up, and every one of them can be checked in the repository.

## How to follow along

Something ships every week: code, a short progress post or an article. The repository is [bilingual-llm-from-scratch](https://github.com/luizcampos331/bilingual-llm-from-scratch), and the articles are published here on LinkedIn.

The next article starts with the simplest language model possible: counting which letters tend to follow which. No neural networks, no libraries, just Python and a dictionary. It's surprising how much you can understand with so little.

---

*Code: [github.com/luizcampos331/bilingual-llm-from-scratch](https://github.com/luizcampos331/bilingual-llm-from-scratch) · [Versão em português](pt.md) · Text licensed under CC BY 4.0*
