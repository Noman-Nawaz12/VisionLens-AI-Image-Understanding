"""
Streamlit interface for the Vision-Language Model research project.

streamlit run app.py
"""

import io
import json
import os
import time
import contextlib

import streamlit as st
from PIL import Image

from src import config
from src.model_setup import load_model_and_processor, inspect_architecture
from src.hallucination_reduction import run_hallucination_reduction
from src.evaluate import run_baseline, evaluate

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="VLM Research Console", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
   .metric-card {
    background: #f0f8ff;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    color: #1a1a1a !important;
}
.metric-label { font-size: 0.8rem; color: #666 !important; text-transform: uppercase; letter-spacing: .04em; }
.metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 2px; color: #1a1a1a !important; }
    .answer-box {
    background: #f0f8ff;
    border: 1px solid #e6e6e6;
    border-left: 4px solid #d97757;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 1.05rem;
    color: #1a1a1a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.answer-box.grounded {
    border-left-color: #2f9e44;
}
    .section-title { font-weight: 600; font-size: 1rem; margin-bottom: 6px; color: #333; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar — run configuration (read-only view of config.py + live controls)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Run Configuration")
    st.write(f"**Task:** `{config.TASK}`")
    st.write(f"**Hallucination method:** `{config.HALLUCINATION_METHOD}`")
    st.write(f"**LoRA enabled:** `{config.USE_LORA}`")
    st.divider()
    n_samples = st.slider("Self-consistency samples", 1, 5, config.SELF_CONSISTENCY_SAMPLES)
    st.caption("Only used when method = self_consistency")
    st.divider()
    st.caption("Change TASK / HALLUCINATION_METHOD in `src/config.py`, then restart this app.")

st.markdown("""
<div style="
    background: #f0f8ff;
    border: 1px solid #b6d4fe;
    border-radius: 12px;
    padding: 24px 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    text-align: center;
">
    <div style="font-size: 1.9rem; font-weight: 800; color: #1a1a1a;">
        🖼️ VisionLens — AI Image Understanding
    </div>
    <div style="font-size: 0.95rem; color: #555; margin-top: 4px;">
        Baseline vs. hallucination-reduced generation — with architecture and benchmark inspection.
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model_and_processor()


with st.spinner("Loading model (first run downloads BLIP, ~1GB)..."):
    load_start = time.time()
    model, processor, device = get_model()
    load_time = time.time() - load_start

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Device</div>'
                f'<div class="metric-value">{device.upper()}</div></div>', unsafe_allow_html=True)
with col_b:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Model load time</div>'
                f'<div class="metric-value">{load_time:.1f}s</div></div>', unsafe_allow_html=True)
with col_c:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Task</div>'
                f'<div class="metric-value">{config.TASK.upper()}</div></div>', unsafe_allow_html=True)

st.write("")

tab1, tab2, tab3 = st.tabs(["🔍  Inference", "🧩  Architecture", "📊  Benchmark"])

# ---------------------------------------------------------------------
# TAB 1 — Inference
# ---------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1.3])

    with left:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)

        if config.TASK == "vqa":
            question = st.text_input("Question", "What is happening in this image? Describe what you see.")
            st.caption(
                "Note: BLIP-VQA is trained to give short answers (a word or two) — "
                "this is expected model behavior, not a bug. Set TASK = 'captioning' "
                "in src/config.py for full-sentence descriptions instead."
            )
        else:
            question = None
            st.info("Task is set to captioning — the model will describe the image directly.")

        run_clicked = st.button("▶ Run Inference", type="primary", use_container_width=True)

    with right:
        if run_clicked:
            if not uploaded_file:
                st.warning("Upload an image first.")
            else:
                t0 = time.time()
                with st.spinner("Running baseline generation..."):
                    baseline_answer = run_baseline(model, processor, image, question or "", device)
                baseline_time = time.time() - t0

                t0 = time.time()
                with st.spinner(f"Running hallucination-reduced generation ({config.HALLUCINATION_METHOD})..."):
                    grounded_result = run_hallucination_reduction(model, processor, image, question or "", device)
                grounded_time = time.time() - t0

                st.markdown('<div class="section-title">Baseline output</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{baseline_answer or "(empty response)"}</div>',
                            unsafe_allow_html=True)
                st.caption(f"⏱ {baseline_time:.2f}s")

                st.write("")

                st.markdown('<div class="section-title">Hallucination-reduced output</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box grounded">{grounded_result["answer"] or "(empty response)"}</div>',
                            unsafe_allow_html=True)
                st.caption(f"⏱ {grounded_time:.2f}s")

                if grounded_result["agreement_score"] is not None:
                    st.progress(grounded_result["agreement_score"],
                                text=f"Self-consistency agreement: {grounded_result['agreement_score']*100:.0f}%")
                    with st.expander("See all sampled answers"):
                        for i, a in enumerate(grounded_result["all_samples"], 1):
                            st.write(f"{i}. {a}")

                if baseline_answer.strip() == grounded_result["answer"].strip():
                    st.info("Both methods agree — low hallucination risk for this sample.")
                else:
                    st.warning("Baseline and grounded outputs differ — worth a closer look.")
        else:
            st.markdown(
                "<div style='color:#888; padding-top: 60px; text-align:center;'>"
                "Upload an image and click <b>Run Inference</b> to compare outputs.</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# TAB 2 — Architecture
# ---------------------------------------------------------------------
with tab2:
    st.write(
        "Breaks the loaded model's parameters into the three MLLM components "
        "described in the proposal: **vision encoder**, **connector**, and **language model**."
    )
    if st.button("Inspect architecture"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            components = inspect_architecture(model)

        total = sum(components.values()) or 1
        cols = st.columns(len(components))
        for col, (part, count) in zip(cols, components.items()):
            pct = count / total * 100
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{part.replace("_"," ")}</div>'
                    f'<div class="metric-value">{pct:.1f}%</div>'
                    f'<div style="font-size:0.8rem;color:#888">{count:,} params</div></div>',
                    unsafe_allow_html=True,
                )
        st.write("")
        if components["connector"] == 0 and components["other"] > 0:
            st.info(
                "This model fuses vision and language through cross-attention layers "
                "embedded inside the language-model decoder rather than a separate "
                "connector block — those parameters are counted under **other**. "
                "This is itself a finding worth documenting: the 'connector' isn't "
                "always a distinct module, and when it's diffused like this it's "
                "harder to isolate and optimize independently."
            )
        st.code(buf.getvalue(), language="text")

# ---------------------------------------------------------------------
# TAB 3 — Benchmark
# ---------------------------------------------------------------------
with tab3:
    report_path = os.path.join(config.RESULTS_DIR, "benchmark_report.json")

    run_bench = st.button("▶ Run benchmark now (uses data/eval_annotations.json)")
    if run_bench:
        eval_path = os.path.join(config.DATA_DIR, "eval_annotations.json")
        if not os.path.exists(eval_path):
            st.error(f"No eval file found at {eval_path}. Add one first.")
        else:
            with st.spinner("Running evaluation — this can take a while on CPU..."):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    evaluate()
            st.success("Benchmark complete.")
            st.code(buf.getvalue(), language="text")

    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        summary = report["summary"]

        st.markdown('<div class="section-title">Baseline vs. Grounded</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            ("Accuracy", "accuracy", "{:.0%}"),
            ("Hallucination rate", "hallucination_rate", "{:.0%}"),
            ("Avg latency", "avg_latency_sec", "{:.2f}s"),
        ]
        for col, (label, key, fmt) in zip([c1, c2, c3], metrics):
            base_v = summary["baseline"][key]
            ground_v = summary["grounded"][key]
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div style="font-size:0.9rem;color:#888">baseline</div>'
                    f'<div class="metric-value">{fmt.format(base_v)}</div>'
                    f'<div style="font-size:0.9rem;color:#888;margin-top:6px">grounded</div>'
                    f'<div class="metric-value" style="color:#2f9e44">{fmt.format(ground_v)}</div>'
                    f'</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Peak memory</div>'
                f'<div class="metric-value">{summary["peak_memory_mb"]} MB</div></div>',
                unsafe_allow_html=True)

        with st.expander("Raw report JSON"):
            st.json(report)
    else:
        st.info("No benchmark report yet. Click the button above, or run `python -m src.evaluate` in a terminal.")
