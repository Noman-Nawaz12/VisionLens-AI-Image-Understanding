"""
Objectives 4 & 5: Evaluate the model on standard benchmarks, measuring
task accuracy AND hallucination rate, plus inference memory/latency.

This produces results/benchmark_report.json and prints a summary table
comparing the baseline model vs. the hallucination-reduction technique.

Usage:
  python -m src.evaluate
"""

import json
import os
import time
import tracemalloc

from PIL import Image
import torch

from src import config
from src.model_setup import load_model_and_processor
from src.hallucination_reduction import run_hallucination_reduction, build_grounded_prompt


def is_hallucinated(predicted: str, ground_truth_objects: list) -> bool:
    """
    Simple grounding check: flags a prediction as a likely hallucination
    if it mentions an object/attribute that is not in the image's
    known object list (from annotations). This is a lightweight proxy
    for full hallucination benchmarks like POPE or CHAIR.
    """
    predicted = predicted.lower()
    known = " ".join(ground_truth_objects).lower()
    # crude heuristic: if predicted text shares no words with known objects
    pred_words = set(predicted.split())
    known_words = set(known.split())
    return len(pred_words & known_words) == 0


def run_baseline(model, processor, image, question, device):
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True).strip().lower()


def evaluate():
    model, processor, device = load_model_and_processor()

    eval_path = os.path.join(config.DATA_DIR, "eval_annotations.json")
    images_dir = os.path.join(config.DATA_DIR, "images")

    if not os.path.exists(eval_path):
        print(f"[!] No eval set found at {eval_path}.")
        print("    Expected format: list of {image, question, answer, objects_present}")
        return

    with open(eval_path, "r") as f:
        eval_data = json.load(f)[: config.EVAL_SAMPLES]

    results = {"baseline": [], "grounded": []}
    latencies = {"baseline": [], "grounded": []}

    tracemalloc.start()

    for item in eval_data:
        image = Image.open(os.path.join(images_dir, item["image"])).convert("RGB")
        question = item["question"]
        gt_answer = item["answer"].lower()
        objects_present = item.get("objects_present", [gt_answer])

        # --- baseline ---
        t0 = time.time()
        baseline_answer = run_baseline(model, processor, image, question, device)
        latencies["baseline"].append(time.time() - t0)

        results["baseline"].append({
            "question": question,
            "predicted": baseline_answer,
            "ground_truth": gt_answer,
            "correct": gt_answer in baseline_answer,
            "hallucinated": is_hallucinated(baseline_answer, objects_present),
        })

        # --- hallucination-reduced ---
        t0 = time.time()
        grounded_result = run_hallucination_reduction(model, processor, image, question, device)
        latencies["grounded"].append(time.time() - t0)

        results["grounded"].append({
            "question": question,
            "predicted": grounded_result["answer"],
            "agreement_score": grounded_result["agreement_score"],
            "ground_truth": gt_answer,
            "correct": gt_answer in grounded_result["answer"],
            "hallucinated": is_hallucinated(grounded_result["answer"], objects_present),
        })

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {}
    for method in ["baseline", "grounded"]:
        n = len(results[method])
        accuracy = sum(r["correct"] for r in results[method]) / n if n else 0
        hallucination_rate = sum(r["hallucinated"] for r in results[method]) / n if n else 0
        avg_latency = sum(latencies[method]) / n if n else 0
        summary[method] = {
            "accuracy": round(accuracy, 3),
            "hallucination_rate": round(hallucination_rate, 3),
            "avg_latency_sec": round(avg_latency, 3),
        }

    summary["peak_memory_mb"] = round(peak_mem / (1024 * 1024), 2)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    print("\n=== Benchmark Summary ===")
    print(f"{'Method':<10} {'Accuracy':<10} {'Hallucination Rate':<20} {'Avg Latency (s)':<16}")
    for method in ["baseline", "grounded"]:
        s = summary[method]
        print(f"{method:<10} {s['accuracy']:<10} {s['hallucination_rate']:<20} {s['avg_latency_sec']:<16}")
    print(f"\nPeak memory used during eval: {summary['peak_memory_mb']} MB")
    print(f"Full report saved to {report_path}")


if __name__ == "__main__":
    evaluate()
