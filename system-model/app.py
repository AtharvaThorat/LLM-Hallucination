import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import streamlit as st
import pandas as pd

from utils import (
    load_dataset,
    pick_random_patient,
    build_qa_from_patient,
    format_patient_summary,
    create_session,
    end_session,
    load_sessions_index,
    update_session_query_count,
    append_result_json,
    load_results_for_sessions,
    save_results_csv,
    generate_all_outputs,
    zip_graphs,
    OUTPUT_DIR,
    RESULTS_PATH,
)
from main import run_question

DATA_PATH = Path(__file__).parent.parent / "data" / "patient_dataset_cleaned.json"
GRAPHS_DIR = OUTPUT_DIR / "graphs"


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Hallucination Evaluator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 LLM Hallucination Evaluation — Medical QA")
st.caption("Compares 4 baselines × 4 models across patient records. All 16 combinations run per question.")

# ── Load dataset ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_dataset():
    return load_dataset(str(DATA_PATH))

dataset = get_dataset()

# ── Session state init ────────────────────────────────────────────────────────
if "active_session" not in st.session_state:
    st.session_state.active_session = None
if "question_results" not in st.session_state:
    st.session_state.question_results = []
if "current_patient" not in st.session_state:
    st.session_state.current_patient = None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Sessions Panel
# ══════════════════════════════════════════════════════════════════════════════
st.header("1 · Sessions")

sessions = load_sessions_index()
if sessions:
    rows = []
    for s in sessions:
        rows.append({
            "Session ID": s["session_id"],
            "Name": s["name"],
            "Started": s["started_at"][:19],
            "Status": s["status"],
            "Queries": s["query_count"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No sessions yet. Start your first session below.")

col1, col2 = st.columns([3, 1])
with col1:
    session_name_input = st.text_input("Session name (optional)", placeholder="e.g. Run 1 — 50 auto queries")
with col2:
    st.write("")
    st.write("")
    if st.button("▶ Start New Session", type="primary", disabled=st.session_state.active_session is not None):
        new_session = create_session(name=session_name_input or None)
        st.session_state.active_session = new_session
        st.session_state.question_results = []
        st.rerun()

if st.session_state.active_session:
    s = st.session_state.active_session
    st.success(f"Active session: **{s['name']}** (ID: {s['session_id']}) — {s.get('query_count', 0)} queries so far")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Run a Batch
# ══════════════════════════════════════════════════════════════════════════════
st.header("2 · Run a Batch")

if not st.session_state.active_session:
    st.warning("Start a session above before running a batch.")
else:
    col_mode, col_k, col_n = st.columns(3)
    with col_mode:
        batch_mode = st.radio("Question mode", ["Auto", "Manual"], horizontal=True)
    with col_k:
        top_k = st.slider("Top-k (RAG retrieval)", min_value=1, max_value=15, value=5,
                          help="Number of facts retrieved for Baseline 3 and Proposed")
    with col_n:
        if batch_mode == "Auto":
            batch_size = st.number_input("Number of questions", min_value=1, value=10, step=1)
            if batch_size < 30:
                st.warning("⚠ Fewer than 30 questions — metrics may not be statistically reliable.")
        else:
            batch_size = st.number_input("Number of questions", min_value=1, value=5, step=1)

    if batch_mode == "Auto":
        if st.button("▶ Run Auto Batch", type="primary"):
            session = st.session_state.active_session
            progress = st.progress(0, text="Running batch...")
            for i in range(int(batch_size)):
                patient = pick_random_patient(dataset)
                qa = build_qa_from_patient(patient, mode="auto")
                question = qa["question"]

                progress.progress((i + 0.5) / batch_size, text=f"Question {i+1}/{int(batch_size)}: {question[:60]}...")
                results_for_q = run_question(
                    patient=patient,
                    question=question,
                    session_id=session["session_id"],
                    session_name=session["name"],
                    batch_mode="auto",
                    top_k=top_k,
                )
                for r in results_for_q:
                    append_result_json(r)
                update_session_query_count(session["session_id"])
                session["query_count"] = session.get("query_count", 0) + 1
                st.session_state.question_results.append({
                    "patient": patient,
                    "question": question,
                    "results": results_for_q,
                })
                progress.progress((i + 1) / batch_size, text=f"Completed {i+1}/{int(batch_size)}")
            progress.empty()
            st.success(f"Batch complete — {int(batch_size)} questions processed.")
            st.rerun()

    else:
        if st.button("📋 Pick a Patient"):
            st.session_state.current_patient = pick_random_patient(dataset)

        if st.session_state.current_patient:
            st.subheader("Patient Summary")
            st.text(format_patient_summary(st.session_state.current_patient))
            manual_question = st.text_input("Your question about this patient")
            if st.button("▶ Submit Question", type="primary", disabled=not manual_question.strip()):
                session = st.session_state.active_session
                patient = st.session_state.current_patient
                with st.spinner("Running all 16 combinations..."):
                    results_for_q = run_question(
                        patient=patient,
                        question=manual_question,
                        session_id=session["session_id"],
                        session_name=session["name"],
                        batch_mode="manual",
                        top_k=top_k,
                    )
                for r in results_for_q:
                    append_result_json(r)
                update_session_query_count(session["session_id"])
                session["query_count"] = session.get("query_count", 0) + 1
                st.session_state.question_results.append({
                    "patient": patient,
                    "question": manual_question,
                    "results": results_for_q,
                })
                st.session_state.current_patient = None
                st.success("Question processed.")
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Live Per-Question Results
# ══════════════════════════════════════════════════════════════════════════════
st.header("3 · Results")

if st.session_state.question_results:
    if st.session_state.active_session:
        if st.button("⏹ End Session", type="secondary"):
            end_session(st.session_state.active_session["session_id"])
            st.session_state.active_session = None
            st.success("Session ended.")
            st.rerun()

    for qi, qdata in enumerate(reversed(st.session_state.question_results)):
        q_label = f"Q{len(st.session_state.question_results) - qi}: {qdata['question']}"
        with st.expander(q_label, expanded=(qi == 0)):
            st.markdown(f"**Patient:** `{qdata['patient'].get('patient_id','')}`")
            st.markdown(f"**Question:** {qdata['question']}")
            st.divider()

            baseline_order = ["baseline1", "baseline2", "baseline3", "proposed"]
            baseline_labels = {
                "baseline1": "Baseline 1 — Full Context",
                "baseline2": "Baseline 2 — Rule-Based",
                "baseline3": "Baseline 3 — Semantic RAG",
                "proposed": "Proposed — Evidence-Constrained RAG",
            }
            model_order = ["claude", "gpt", "gemini", "llama"]

            for bl in baseline_order:
                st.markdown(f"#### {baseline_labels[bl]}")
                cols = st.columns(4)
                bl_results = [r for r in qdata["results"] if r["baseline"] == bl]
                bl_results_sorted = sorted(bl_results, key=lambda r: model_order.index(r["model"]) if r["model"] in model_order else 99)

                for ci, r in enumerate(bl_results_sorted):
                    m = r["metrics"]
                    hallucinated = m["hallucination"] == 1
                    abstained = m["abstained"] == 1
                    border_color = "#ff4b4b" if hallucinated else "#21c354"

                    with cols[ci]:
                        st.markdown(
                            f"""
                            <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                            <b>{r['model'].upper()}</b>
                            {'&nbsp;<span style="background:#ff4b4b;color:white;padding:2px 6px;border-radius:4px;font-size:11px;">HALLUCINATED</span>' if hallucinated else ''}
                            {'&nbsp;<span style="background:#f0ad4e;color:white;padding:2px 6px;border-radius:4px;font-size:11px;">ABSTAINED</span>' if abstained else ''}
                            <hr style="margin:6px 0">
                            <small><b>Answer:</b> {r['predicted_answer'][:200]}</small>
                            <hr style="margin:6px 0">
                            <small>
                            Hallucination: {m['hallucination']} | Abstained: {m['abstained']}<br>
                            Ev. Precision: {m['evidence_precision']} | Ev. F1: {m['evidence_f1']}<br>
                            SAR: {m['sar']} | Over-conf: {m['over_confidence']}
                            </small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.popover("View Prompt"):
                            st.text(r["prompt_sent"][:1000])
                        with st.popover("View Evidence"):
                            for chunk in r["retrieved_chunks"]:
                                st.write(f"• {chunk}")

else:
    st.info("No results yet. Run a batch above.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Report Generator
# ══════════════════════════════════════════════════════════════════════════════
st.header("4 · Generate Report")

all_sessions = load_sessions_index()
if not all_sessions:
    st.info("No sessions available yet.")
else:
    st.markdown("Select sessions to include in the report:")
    selected_ids = []
    total_queries = 0
    for s in all_sessions:
        checked = st.checkbox(
            f"**{s['name']}** — {s['started_at'][:10]} — {s['query_count']} queries ({s['status']})",
            key=f"chk_{s['session_id']}",
        )
        if checked:
            selected_ids.append(s["session_id"])
            total_queries += s.get("query_count", 0)

    if selected_ids:
        st.info(f"Report will use **{total_queries} queries** from {len(selected_ids)} session(s).")
        if st.button("📊 Generate Report", type="primary"):
            with st.spinner("Generating all graphs and CSVs..."):
                agg_df, confusion_dict = generate_all_outputs(selected_ids)
            if agg_df is not None:
                st.success("Report generated! See graphs and exports below.")
                st.session_state["report_ready"] = True
                st.session_state["agg_df"] = agg_df
                st.rerun()
            else:
                st.error("No results found for the selected sessions.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Graphs
# ══════════════════════════════════════════════════════════════════════════════
st.header("5 · Graphs")

if st.session_state.get("report_ready") and GRAPHS_DIR.exists():
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Bar Charts", "🔁 Model Comparison", "🌡 Heatmaps", "🕸 Radar Charts", "🔲 Confusion Matrices"
    ])

    def show_png_grid(folder: Path, cols: int = 3):
        pngs = sorted(folder.glob("*.png"))
        if not pngs:
            st.info("No graphs found.")
            return
        columns = st.columns(cols)
        for i, p in enumerate(pngs):
            columns[i % cols].image(str(p), use_container_width=True)

    with tab1:
        show_png_grid(GRAPHS_DIR / "bar_charts", cols=3)
    with tab2:
        show_png_grid(GRAPHS_DIR / "model_comparison", cols=3)
    with tab3:
        show_png_grid(GRAPHS_DIR / "heatmaps", cols=3)
    with tab4:
        show_png_grid(GRAPHS_DIR / "radar_charts", cols=2)
    with tab5:
        show_png_grid(GRAPHS_DIR / "confusion_matrices", cols=4)

    if st.session_state.get("agg_df") is not None:
        st.subheader("Aggregate Metrics Table")
        st.dataframe(st.session_state["agg_df"], use_container_width=True)
else:
    st.info("Generate a report above to view graphs.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Export
# ══════════════════════════════════════════════════════════════════════════════
st.header("6 · Export")

col_a, col_b, col_c = st.columns(3)

results_csv = OUTPUT_DIR / "results.csv"
agg_csv = OUTPUT_DIR / "aggregate_metrics.csv"

with col_a:
    if results_csv.exists():
        with open(results_csv, "rb") as f:
            st.download_button("⬇ Download results.csv", f, file_name="results.csv", mime="text/csv")
    else:
        st.button("⬇ Download results.csv", disabled=True)

with col_b:
    if agg_csv.exists():
        with open(agg_csv, "rb") as f:
            st.download_button("⬇ Download aggregate_metrics.csv", f,
                               file_name="aggregate_metrics.csv", mime="text/csv")
    else:
        st.button("⬇ Download aggregate_metrics.csv", disabled=True)

with col_c:
    if GRAPHS_DIR.exists() and any(GRAPHS_DIR.rglob("*.png")):
        zip_path = zip_graphs()
        with open(zip_path, "rb") as f:
            st.download_button("⬇ Download All Graphs (ZIP)", f,
                               file_name="graphs_export.zip", mime="application/zip")
    else:
        st.button("⬇ Download All Graphs (ZIP)", disabled=True)
