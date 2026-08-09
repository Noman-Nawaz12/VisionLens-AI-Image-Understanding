"""
Loads the base Vision-Language Model and (optionally) wraps it with LoRA
for parameter-efficient fine-tuning.

This module also implements Objective 1 from the proposal:
"Study the architecture of MLLMs: visual encoders, connector modules,
and language model backbones, and how information loss occurs at the connector."

Run this file directly to print a breakdown of the model's three components
and where the connector bottleneck lives.
"""

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
from src import config


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor():
    """Loads the base BLIP model + processor for the configured task."""
    if config.TASK == "vqa":
        model_name = config.BASE_MODEL_VQA
        model = BlipForQuestionAnswering.from_pretrained(model_name)
    else:
        model_name = config.BASE_MODEL_CAPTION
        model = BlipForConditionalGeneration.from_pretrained(model_name)

    processor = BlipProcessor.from_pretrained(model_name)
    device = get_device()
    model.to(device)
    return model, processor, device


def apply_lora(model):
    """
    Wraps the language-model backbone with LoRA adapters so we only train
    a small number of parameters (parameter-efficient fine-tuning).
    Requires: pip install peft
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def inspect_architecture(model):
    """
    Breaks the model down into its three MLLM components and reports
    parameter counts — this is where the 'connector bottleneck' becomes
    visible: the connector module is almost always the smallest piece,
    meaning it has to compress a lot of visual information into very
    few parameters relative to the vision encoder and language model.
    """
    components = {"vision_encoder": 0, "connector": 0, "language_model": 0, "other": 0}

    for name, param in model.named_parameters():
        n = param.numel()
        lname = name.lower()
        if "vision_model" in lname or "vit" in lname or "visual_encoder" in lname:
            components["vision_encoder"] += n
        elif any(k in lname for k in ["text_decoder", "text_encoder", "language_model", "lm_head"]):
            components["language_model"] += n
        elif any(k in lname for k in [
            "proj", "connector", "bridge", "qformer", "q_former",
            "itm_head", "vision_proj", "text_proj", "cross_attention"
        ]):
            components["connector"] += n
        else:
            components["other"] += n

    total = sum(components.values())
    print("\n=== MLLM Architecture Breakdown ===")
    for part, count in components.items():
        pct = (count / total * 100) if total else 0
        print(f"{part:>16}: {count:>12,} params  ({pct:5.2f}%)")
    print(f"{'TOTAL':>16}: {total:>12,} params")

    if components["connector"] > 0:
        ratio = components["vision_encoder"] / max(components["connector"], 1)
        print(f"\nConnector bottleneck ratio (vision_encoder / connector): {ratio:.1f}x")
        print("A high ratio means the connector must compress a large amount")
        print("of visual detail into a much smaller representation — this is")
        print("the information bottleneck described in the research gap.\n")

    return components


if __name__ == "__main__":
    model, processor, device = load_model_and_processor()
    print(f"Loaded {config.TASK} model on device: {device}")
    inspect_architecture(model)

    if config.USE_LORA:
        model = apply_lora(model)
