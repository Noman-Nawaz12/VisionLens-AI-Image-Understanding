# 🖼️VisionLens-AI Image Understanding

**AI Image Understanding, Grounded in What's Actually There**
Lightweight Vision-Language Model Research Project — Reducing Hallucination & the Connector Bottleneck in Multimodal LLMs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![BLIP](https://img.shields.io/badge/BLIP-VisionLanguage-lightgrey)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-orange)

---

## 📌 Overview

**VisionLens** implements the research proposal *"Multimodal Large Language Models for Vision-Language Tasks."* It's a lightweight, laptop-friendly pipeline for fine-tuning and evaluating a small open-source Vision-Language Model (VLM) on **VQA** (Visual Question Answering) or **Image Captioning**, specifically targeting two well-known research gaps in MLLMs:

- 🔌 **Connector bottleneck** — information loss between the vision encoder and language model
- 🧠 **Hallucination** — the model describing things not actually present in the image

Built on **BLIP-base** (a small, well-documented open-source VLM) with **LoRA** for parameter-efficient fine-tuning, so the whole pipeline runs on a normal laptop — no research GPU cluster required.

---

## 🖥️ Application Preview

**Inference Tab**
- Upload an image, ask a question (or generate a caption)
- Side-by-side comparison: baseline output vs. hallucination-reduced output
- Self-consistency agreement score with all sampled answers

**Architecture Tab**
- One-click parameter breakdown across vision encoder / connector / language model
- Visual bottleneck analysis

**Benchmark Tab**
- Run the full evaluation pipeline from the UI
- Accuracy, hallucination rate, latency, and peak memory — baseline vs. grounded

---

## 🚀 Key Features

### 🧩 Connector Bottleneck Analysis
Breaks down the model's parameters into its three MLLM components and reports where information is most compressed — directly visualizing the research gap described in the proposal.

### 🎯 Parameter-Efficient Fine-Tuning (LoRA)
Instead of full fine-tuning (expensive, GPU-heavy), only small LoRA adapters are trained on top of the frozen base model — fast, cheap, and laptop-friendly.

### 🛡️ Hallucination-Reduction Techniques
Two techniques implemented and swappable via config:
1. **Grounding prompts** — instructs the model to answer strictly from visible evidence, or say "not visible"
2. **Self-consistency decoding** — samples multiple generations and keeps the majority-agreed answer, using disagreement as a hallucination signal

### 📊 Full Benchmark Pipeline
Compares baseline vs. hallucination-reduced generation on accuracy, hallucination rate, inference latency, and peak memory — reusable for any future VLM experiment.

---

## 🏗️ System Workflow

```
User uploads image + question
              │
              ▼
     BLIP Vision-Language Model
   (vision encoder → connector → LM)
              │
       ┌──────┴──────┐
       ▼             ▼
   Baseline      Hallucination-Reduced
   Generation    (grounding prompt /
                  self-consistency)
       │             │
       └──────┬──────┘
              ▼
   Side-by-side comparison
   + accuracy / hallucination /
     latency / memory metrics
```

---

## 🧠 Technologies Used

| Layer | Tool |
|---|---|
| Core language | Python 3.10+ |
| Deep learning | PyTorch |
| Model & processor | HuggingFace Transformers (BLIP) |
| Fine-tuning | PEFT (LoRA) |
| Frontend | Streamlit |
| Image handling | Pillow |
| Training acceleration | Accelerate |

---

## 📂 Project Structure

```
vlm_research_project/
│
├── app.py                         # Streamlit UI (Inference / Architecture / Benchmark tabs)
├── src/
│   ├── config.py                  # All settings in one place (task, model, LoRA, hallucination method)
│   ├── model_setup.py             # Loads BLIP + LoRA, inspects architecture
│   ├── hallucination_reduction.py # Grounding prompts + self-consistency decoding
│   ├── train.py                   # LoRA fine-tuning loop
│   └── evaluate.py                # Benchmark: accuracy, hallucination rate, latency, memory
│
├── data/
│   ├── images/                    # Dataset images
│   ├── annotations.json           # Training Q&A / captions
│   └── eval_annotations.json      # Evaluation set (includes objects_present for grounding checks)
│
├── results/                       # benchmark_report.json + saved LoRA checkpoints
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd vlm_research_project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

⚠️ First run downloads the BLIP base model (~1GB) from Hugging Face automatically.
⚠️ No GPU? Still works, just slower — reduce sample counts in `src/config.py` if needed.

---

## 🚀 Usage

### Run the web app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Or run individual pipeline stages from the terminal
```bash
python -m src.model_setup     # architecture breakdown
python -m src.train           # LoRA fine-tuning
python -m src.evaluate        # full benchmark
```

---

## 🎯 Outcomes Delivered

| Proposal Objective | Delivered As |
|---|---|
| Fine-tuned/adapted VLM prototype with reduced hallucination | LoRA checkpoint in `results/checkpoints/` + grounding techniques applied at inference |
| Benchmark report (accuracy, hallucination rate, efficiency vs. baseline) | `results/benchmark_report.json`, viewable in the Benchmark tab |
| Connector-bottleneck documentation & mitigation | Architecture tab breakdown + LoRA/grounding mitigation strategies |
| Reusable evaluation pipeline | `src/evaluate.py` — swap in any dataset via `data/eval_annotations.json` |

---

## ⚠️ Limitations & Honest Notes

- This is a **scaled-down, laptop-friendly** version of the full research proposal — real benchmarks (VQAv2, POPE, CHAIR) are much larger; the pipeline is structured so real datasets can be swapped in later.
- The hallucination check in `evaluate.py` is a lightweight heuristic, not a full benchmark like POPE/CHAIR — sufficient to demonstrate the pipeline, but should be replaced with a proper benchmark for formal research use.
- BLIP-VQA gives short (1–2 word) answers by design — for full-sentence descriptions, set `TASK = "captioning"` in `src/config.py`.
- CPU training/inference is slow. A modest GPU or Google Colab's free tier is recommended for real experiments.

---

## 👨‍💻 Developer

**VisionLens** — built as an AI/ML capstone research project combining vision-language architecture study, parameter-efficient fine-tuning, and hallucination-reduction evaluation.
