"""
Implements Objective 3 from the proposal:
"Design and apply at least one hallucination-reduction technique."

Two techniques are implemented here:

1. Grounding prompts — rewrite the question/prompt to explicitly instruct
   the model to only describe what is visibly present, and to say
   "not visible" rather than guess.

2. Self-consistency decoding — generate the same answer multiple times
   with sampling, then keep the majority-agreed answer. If the model
   is hallucinating, its guesses tend to disagree across samples;
   if it is grounded in the image, answers converge.
"""

from collections import Counter
import torch
from src import config


def build_grounded_prompt(question: str) -> str:
    """
    Objective 3 (technique 1): grounding prompt.
    Adds an explicit instruction that discourages guessing.
    """
    return (
        f"Answer strictly based on visible evidence in the image. "
        f"If the answer is not visible, say 'not visible'. "
        f"Question: {question}"
    )


def generate_with_self_consistency(model, processor, image, question, device, n_samples=None):
    """
    Objective 3 (technique 2): self-consistency decoding.
    Runs generation multiple times with sampling enabled, then returns
    the most frequent answer plus an agreement score (0-1).

    A low agreement score is itself a useful hallucination signal —
    the evaluation pipeline logs it alongside every prediction.
    """
    n_samples = n_samples or config.SELF_CONSISTENCY_SAMPLES
    prompt = build_grounded_prompt(question)

    inputs = processor(image, prompt, return_tensors="pt").to(device)

    answers = []
    for _ in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                max_new_tokens=20,
            )
        answers.append(processor.decode(out[0], skip_special_tokens=True).strip().lower())

    counts = Counter(answers)
    best_answer, best_count = counts.most_common(1)[0]
    agreement_score = best_count / n_samples

    return {
        "answer": best_answer,
        "agreement_score": agreement_score,
        "all_samples": answers,
    }


def run_hallucination_reduction(model, processor, image, question, device):
    """Dispatches to whichever technique is set in config.py."""
    if config.HALLUCINATION_METHOD == "self_consistency":
        return generate_with_self_consistency(model, processor, image, question, device)

    # default: grounding_prompt (single generation, but grounded prompt)
    prompt = build_grounded_prompt(question)
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()
    return {"answer": answer, "agreement_score": None, "all_samples": [answer]}
