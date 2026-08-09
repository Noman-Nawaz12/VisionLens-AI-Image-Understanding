# 🖼️ Lightweight Vision-Language Model Research Project

**Reducing Hallucination & the Connector Bottleneck in Multimodal LLMs (VQA / Captioning)**

`Python` `PyTorch` `HuggingFace Transformers` `PEFT (LoRA)` `BLIP`

---

## 📌 Overview

This project implements the proposal *"Multimodal Large Language Models for Vision-Language Tasks"*: a lightweight, laptop-friendly pipeline for fine-tuning and evaluating a small open-source Vision-Language Model (VLM) on **VQA** (Visual Question Answering) or **Image Captioning**, while specifically targeting two research gaps:

- 🔌 **Connector bottleneck** — information loss between the vision encoder and language model
- 🧠 **Hallucination** — the model describing things not actually present in the image

Because this is meant to run on a normal PC (not a research GPU cluster), it uses **BLIP-base** (a small, well-documented open-source VLM) and **LoRA** for parameter-efficient fine-tuning instead of full fine-tuning.

---

## 🧩 How It Maps to the Proposal's Objectives

| Objective | Where it's implemented |
|---|---|
| Study MLLM architecture & connector bottleneck | `src/model_setup.py` → `inspect_architecture()` |
| Fine-tune a small VLM with LoRA | `src/model_setup.py` → `apply_lora()`, `src/train.py` |
| Apply a hallucination-reduction technique | `src/hallucination_reduction.py` (grounding prompts + self-consistency) |
| Evaluate accuracy & hallucination rate | `src/evaluate.py` |
| Measure memory/latency | `src/evaluate.py` (tracemalloc + timing) |

---

## 🏗️ Project Structure

```
vlm_research_project/
│
├── src/
│   ├── config.py                  # All settings in one place
│   ├── model_setup.py             # Loads BLIP + LoRA, inspects architecture
│   ├── hallucination_reduction.py # Grounding prompts + self-consistency decoding
│   ├── train.py                   # LoRA fine-tuning loop
│   └── evaluate.py                # Benchmark: accuracy, hallucination rate, latency, memory
│
├── data/
│   ├── images/                    # <- put your dataset images here
│   ├── annotations.example.json   # training data format example
│   └── eval_annotations.example.json  # evaluation data format example
│
├── results/                       # benchmark_report.json + saved checkpoints land here
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
cd vlm_research_project
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

⚠️ First run will download the BLIP base model (~1GB) from Hugging Face automatically — needs internet the first time only.

⚠️ No GPU? It'll still run, just slowly. Reduce `MAX_TRAIN_SAMPLES` and `EVAL_SAMPLES` in `src/config.py` if it's too slow.

---

## 🚀 Usage

### 1. Inspect the model architecture (Objective 1)
```bash
python -m src.model_setup
```
Prints how many parameters live in the vision encoder vs. the connector vs. the language model — this is where you can literally *see* the connector bottleneck.

### 2. Prepare your dataset
- Put images in `data/images/`
- Copy `data/annotations.example.json` → `data/annotations.json` and fill in your own Q&A/captions
- Copy `data/eval_annotations.example.json` → `data/eval_annotations.json` for testing (needs an extra `objects_present` field used for the hallucination check)

A few hundred image-question-answer triples is enough for a laptop-scale LoRA fine-tune. Public datasets like a small subset of **VQAv2** or **COCO Captions** work well — just reformat them to match the JSON structure above.

### 3. Fine-tune with LoRA (Objective 2)
```bash
python -m src.train
```
Saves the adapted model to `results/checkpoints/`.

### 4. Run the benchmark (Objectives 3, 4, 5)
```bash
python -m src.evaluate
```
Compares **baseline** generation vs. the **hallucination-reduced** generation on:
- ✅ Accuracy
- 🚫 Hallucination rate
- ⏱️ Latency per sample
- 💾 Peak memory usage

Results saved to `results/benchmark_report.json`.

---

## 🛡️ Hallucination-Reduction Technique (Objective 3)

Two techniques are implemented in `src/hallucination_reduction.py` — switch between them via `HALLUCINATION_METHOD` in `config.py`:

1. **Grounding prompts** — rewrites every question into an instruction that tells the model to only answer from visible evidence and say "not visible" instead of guessing.
2. **Self-consistency decoding** — generates the same answer several times with sampling; keeps the majority answer and reports an *agreement score*. Low agreement is itself a hallucination warning sign.

---

## 📊 Expected Outcomes (matches proposal section 4)

- ✅ A fine-tuned/adapted VLM prototype for VQA or captioning
- ✅ A benchmark report comparing accuracy, hallucination rate, and efficiency vs. baseline (`results/benchmark_report.json`)
- ✅ A printed architecture breakdown documenting the connector bottleneck
- ✅ A reusable evaluation pipeline (`src/evaluate.py`) for future experiments

---

## ⚠️ Limitations & Honest Notes

- This is a **scaled-down, laptop-friendly** version of the full research proposal — real benchmarks (VQAv2, POPE, CHAIR) are much larger; this pipeline is structured so you can swap in the real datasets later.
- The hallucination check in `evaluate.py` is a simple heuristic, not a full hallucination benchmark like POPE/CHAIR — good enough to demonstrate the pipeline, but should be replaced with a proper benchmark for a real paper/thesis.
- Training speed on CPU will be slow. If possible, run `src/train.py` on a machine with even a modest GPU, or Google Colab's free GPU tier.
