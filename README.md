# 🎬🔊 MAIA-AudioVisual

### Extending Multimodal AI Assessment to the Audio-Visual Domain

[![FBK Research Badge](https://img.shields.io/badge/Research-FBK-8A2BE2?style=for-the-badge)](https://www.fbk.eu/en/) [![](https://img.shields.io/badge/Domain-Multimodal%20AI-7B1FA2?style=for-the-badge)](https://www.fbk.eu/en/) [![](https://img.shields.io/badge/Task-Visual%20Statement%20Verification-6A1B9A?style=for-the-badge)](https://www.fbk.eu/en/) [![](https://img.shields.io/badge/Models-Qwen%20%7C%20Gemma-9C27B0?style=for-the-badge)](https://www.fbk.eu/en/) [![](https://img.shields.io/badge/Focus-Audio--Visual%20Reasoning-AB47BC?style=for-the-badge)](https://www.fbk.eu/en/)

*A research internship project developed in collaboration with [FBK – Fondazione Bruno Kessler](https://www.fbk.eu/en/), extending the [MAIA](https://aclanthology.org/2025.clicit-1.106.pdf) benchmark to genuinely audio-visual reasoning, with a diagnostic, competence-oriented evaluation pipeline.*

---

## 💼 Project Overview

This internship project is carried out in collaboration with **[FBK – Fondazione Bruno Kessler](https://www.fbk.eu/en/)**.

**MAIA-AV** builds on the original **MAIA** benchmark and on the findings of the companion project [**Lost in Translation**](https://github.com/BlackCodingKitten/lost_in_translation), which showed that the original MAIA questions were designed around visual content alone, with no systematic dependency on the audio track. MAIA-AV addresses this limitation directly: it introduces new video items in which **audio genuinely matters**, so that the correct answer cannot always be recovered from vision alone.

The project also introduces a **preliminary, question-independent analysis stage**: before any question is ever shown to a model, each model's raw visual understanding of the video is extracted, structured, and diagnosed on its own terms. This turns MAIA-AV from a purely result-driven benchmark into a **diagnostic framework**, capable of separating failures in perception from failures in reasoning.

---

## 🧠 Research Motivation

High accuracy on a video-language task is not, by itself, evidence of genuine multimodal understanding. Models can exploit statistical shortcuts, linguistic plausibility, or a single dominant modality while still producing correct answers. Real videos, moreover, are intrinsically **audiovisual**: dialogue, music, and environmental sound often carry information that cannot be reconstructed from frames alone.

MAIA-AV starts from a simple but demanding requirement: for a subset of items, **audio and video must be jointly necessary**, each modality insufficient on its own. This makes it possible to distinguish genuine cross-modal integration from unimodal collapse, linguistic shortcuts, or accidental correctness.

---

## 🔍 The Preliminary Analysis Pipeline

Before any question, caption, or foil is introduced, each of the four evaluated multimodal and omnimodal models independently analyzes the video content alone. The pipeline proceeds through four stages:

### 1️⃣ Video Preprocessing

Each video is split into **4-second temporal windows with a 3-second stride**, giving 1 second of overlap between consecutive segments. This preserves continuity across boundaries without discarding chronological order, and every segment keeps its exact `start_time` / `end_time` for later reconstruction.

### 2️⃣ Semantic Representation Extraction

Each temporal window is independently described across **seven semantic categories**: entities, actions, events, spatial relations, state changes, temporal relations, and causal hypotheses. Every extracted element is grounded in specific frames and tagged as `observed` or `inferred`, so that directly visible facts and model inferences never get mixed together.

### 3️⃣ Event & Relation Analysis

Local, overlapping descriptions are consolidated into a single **event-based representation** of the whole video, removing redundancy from window overlap while preserving genuinely repeated events. On top of this, three complementary relational dimensions are built:

- 🗺️ **Spatial** — configurations between entities, tied to specific temporal segments
- ⏱️ **Temporal** — `before` / `after` / `during` / `overlaps` / `simultaneous` relations between events
- 🔗 **Causal** — `causes` / `enables` / `motivates` / `prevents` links, always kept strictly separate from mere temporal succession

### 4️⃣ Representation Integration & NLI-Based Proficiency Assessment

All levels are merged into a single, fixed visual representation, used as the **premise** of a Natural Language Inference task: each caption and its corresponding foil are independently checked for `entailment`, `contradiction`, or `neutral`. A pair is marked `PASS` only when the representation **entails the caption and contradicts the foil** — a deliberately conservative criterion.

---

## 📊 Question Complexity Levels

Questions are classified along three dimensions — **spatial**, **temporal**, and **causal** — on a shared three-level scale:

| Level | Meaning                                                                      |
| ----- | ---------------------------------------------------------------------------- |
| **0** | Information directly retrievable, no integration required                    |
| **1** | One explicit relation or a limited number of evidence units                  |
| **2** | Broader reconstruction across multiple events, phases, or implicit relations |

This classification reflects the *informational* demands of each question, not its surface-level linguistic complexity, and makes it possible to analyze model performance sub-capability by sub-capability rather than as a single accuracy score.

---

## ✅ Caption–Foil Quality Validation

Every generated caption–foil pair goes through a **semi-automatic validation pipeline** before entering the final evaluation set:

1. **Structural check** — verifies that the foil modifies precisely the semantic dimension (spatial, temporal, or causal) the question is meant to test
2. **NLI check** — confirms that the foil genuinely contradicts the caption, rather than merely rephrasing or weakly diverging from it
3. **Manual review** — every uncertain, neutral, or structurally invalid case is isolated and inspected by hand, and corrected pairs are re-validated through the same pipeline

This keeps the benchmark efficient to build while ensuring that no problematic pair silently makes it into the final dataset.

---

## ⚙️ Workflow Summary

1. **Preprocess** videos into overlapping temporal windows
2. **Extract** structured semantic representations per window, per model
3. **Consolidate** local descriptions into global events and relations
4. **Infer** causal dependencies from consolidated events
5. **Finalize** a fixed, question-independent visual representation per model
6. **Generate** category-specific caption–foil pairs
7. **Validate** pairs structurally and semantically, with human-in-the-loop review
8. **Assess** visual proficiency via NLI, then evaluate full VSV performance with audio

---

## 🚀 Main Objectives

- Build a diagnostic, competence-oriented extension of the MAIA benchmark
- Design a **question-independent** visual analysis stage, decoupled from the final task
- Compare four multimodal/omnimodal models (**Qwen2.5-Omni**, **Qwen3-Omni**, **Gemma 3n**, **Gemma 4**) on their raw visual understanding
- Introduce controlled **audio–visual** items where neither modality is sufficient alone
- Provide a rigorous, semi-automatic caption–foil validation pipeline
- Disentangle perception failures from reasoning failures in downstream evaluation

---

## 📁 Repository Structure

```
MAIA-AV/
├── data/     # Datasets, annotations, and generated caption–foil pairs
├── latex/    # Thesis / paper source files
├── src/      # Preprocessing, extraction, and evaluation pipeline code
└── README.md
```

---

## 🤝 Collaboration

**Institution:** [FBK — Fondazione Bruno Kessler](https://www.fbk.eu/en/) **Related work:** [Lost in Translation](https://github.com/BlackCodingKitten/lost_in_translation) · [MAIA Benchmark](https://aclanthology.org/2025.clicit-1.106.pdf)

---

### ✨ Researching grounding, audio-visual integration, and competence-oriented multimodal evaluation
