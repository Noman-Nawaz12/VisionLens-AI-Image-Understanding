"""
Central configuration for the Vision-Language Model research project.
Edit these values to switch tasks, models, or evaluation settings.
"""

# ---- Task selection ----
TASK = "captioning"  # options: "vqa" or "captioning"

# ---- Model selection ----
# Small/medium open-source VLM that can run on CPU (slowly) or a modest GPU.
# BLIP is a good starting point: lightweight, well-documented, HF-hosted.
BASE_MODEL_VQA = "Salesforce/blip-vqa-base"
BASE_MODEL_CAPTION = "Salesforce/blip-image-captioning-base"

# ---- LoRA fine-tuning settings (parameter-efficient adaptation) ----
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query", "value"]  # attention layers to adapt

# ---- Hallucination-reduction technique ----
# "grounding_prompt": forces the model to only answer from visible evidence
# "self_consistency": generates multiple answers and keeps the majority-agreed one
HALLUCINATION_METHOD = "grounding_prompt"
SELF_CONSISTENCY_SAMPLES = 3

# ---- Training settings ----
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 3
MAX_TRAIN_SAMPLES = 500  # keep small for a laptop-friendly run

# ---- Evaluation settings ----
EVAL_SAMPLES = 100
DEVICE = "cuda"  # falls back to "cpu" automatically if no GPU found

# ---- Paths ----
DATA_DIR = "data"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "results/checkpoints"
