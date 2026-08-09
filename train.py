"""
Objective 2: Fine-tune (via LoRA) a small/medium open-source VLM
for VQA or captioning, kept laptop-friendly with a small sample cap.

Expects a dataset directory structured as:

  data/
    images/
      img_0001.jpg
      img_0002.jpg
      ...
    annotations.json   # see format below

annotations.json format (VQA):
[
  {"image": "img_0001.jpg", "question": "What color is the car?", "answer": "red"},
  ...
]

annotations.json format (captioning):
[
  {"image": "img_0001.jpg", "caption": "A red car parked on the street."},
  ...
]

Usage:
  python -m src.train
"""

import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from src import config
from src.model_setup import load_model_and_processor, apply_lora


class VLMDataset(Dataset):
    def __init__(self, annotations_path, images_dir, processor, task):
        with open(annotations_path, "r") as f:
            self.data = json.load(f)[: config.MAX_TRAIN_SAMPLES]
        self.images_dir = images_dir
        self.processor = processor
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.images_dir, item["image"])).convert("RGB")

        if self.task == "vqa":
            inputs = self.processor(
                image, item["question"], return_tensors="pt", padding="max_length",
                truncation=True, max_length=32,
            )
            labels = self.processor.tokenizer(
                item["answer"], return_tensors="pt", padding="max_length",
                truncation=True, max_length=16,
            ).input_ids
        else:
            inputs = self.processor(
                image, return_tensors="pt", padding="max_length",
                truncation=True, max_length=32,
            )
            labels = self.processor.tokenizer(
                item["caption"], return_tensors="pt", padding="max_length",
                truncation=True, max_length=32,
            ).input_ids

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels.squeeze(0)
        return inputs


def train():
    model, processor, device = load_model_and_processor()
    if config.USE_LORA:
        model = apply_lora(model)

    annotations_path = os.path.join(config.DATA_DIR, "annotations.json")
    images_dir = os.path.join(config.DATA_DIR, "images")

    if not os.path.exists(annotations_path):
        print(f"[!] No annotations found at {annotations_path}.")
        print("    Add your dataset there before running training.")
        print("    See the docstring at the top of this file for the expected format.")
        return

    dataset = VLMDataset(annotations_path, images_dir, processor, config.TASK)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    model.train()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{config.EPOCHS} | Step {step} | Loss {loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"== Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f} ==")

    model.save_pretrained(config.CHECKPOINT_DIR)
    print(f"Model saved to {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
