import json
import os
import random
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation import compute_aggregate_metrics, compute_confusion_matrices

SESSIONS_INDEX_PATH = Path(__file__).parent.parent / "outputs" / "sessions_index.json"
RESULTS_PATH = Path(__file__).parent.parent / "outputs" / "results.json"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

QUESTION_TEMPLATES = {
    "allergy": [
        "Does the patient have any documented allergies?",
        "Is the patient allergic to any medications?",
        "Has the patient ever had an allergic reaction?",
    ],
    "medication": [
        "What medications is the patient currently taking?",
        "Has the patient ever been prescribed antibiotics?",
        "What drugs has the patient been prescribed?",
    ],
    "condition": [
        "What medical conditions has the patient been diagnosed with?",
        "Does the patient have any chronic conditions?",
        "What is the patient's medical history?",
    ],
    "observation": [
        "What are the patient's most recent lab results?",
        "What is the patient's blood pressure?",
        "What does the patient's recent blood work show?",
    ],
}


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_random_patient(dataset: list[dict]) -> dict:
    return random.choice(dataset)


def build_qa_from_patient(patient: dict, mode: str = "auto", question: str = None) -> dict:
    if mode == "manual":
        return {"question": question, "supporting_evidence": _derive_evidence(patient, question)}

    # Auto: pick question category based on what's populated
    categories = []
    if patient.get("allergies") is not None:
        categories.append("allergy")
    if patient.get("medications"):
        categories.append("medication")
    if patient.get("conditions"):
        categories.append("condition")
    if patient.get("observations"):
        categories.append("observation")

    if not categories:
        categories = ["allergy"]

    category = random.choice(categories)
    question = random.choice(QUESTION_TEMPLATES[category])

    return {"question": question, "supporting_evidence": _derive_evidence(patient, question)}


def _derive_evidence(patient: dict, question: str) -> list[str]:
    q = question.lower()
    evidence = []

    if "allerg" in q:
        allergies = patient.get("allergies", [])
        if not allergies:
            evidence.append("No allergies are currently recorded for this patient.")
        else:
            for a in allergies:
                evidence.append(f"Patient has a documented allergy to {a.get('allergy', '')}.")

    if "medication" in q or "drug" in q or "prescri" in q or "antibiotic" in q:
        for m in patient.get("medications", []):
            evidence.append(f"Patient is prescribed {m.get('medication', '')}.")

    if "condition" in q or "diagnos" in q or "history" in q or "chronic" in q:
        for c in patient.get("conditions", []):
            evidence.append(f"Patient has been diagnosed with {c.get('condition', '')}.")

    if "lab" in q or "blood" in q or "result" in q or "pressure" in q or "observation" in q:
        for o in patient.get("observations", [])[:5]:
            evidence.append(
                f"Patient {o.get('test','')}: {o.get('value','')} {o.get('unit','')} "
                f"(recorded {o.get('time','')[:10]})."
            )

    return evidence


def format_patient_summary(patient: dict) -> str:
    pid = patient.get("patient_id", "Unknown")
    demo = patient.get("demographics", {})
    age = ""
    if demo.get("birthdate"):
        try:
            from datetime import date
            birth = datetime.strptime(demo["birthdate"], "%Y-%m-%d").date()
            age = f", Age: {(date.today() - birth).days // 365}"
        except Exception:
            pass
    gender = demo.get("gender", "Unknown")
    conditions = [c.get("condition", "") for c in patient.get("conditions", [])]
    meds = [m.get("medication", "") for m in patient.get("medications", [])]
    allergies = [a.get("allergy", "") for a in patient.get("allergies", [])]

    lines = [
        f"Patient ID: {pid}",
        f"Gender: {gender}{age}",
        f"Conditions: {', '.join(conditions) if conditions else 'None'}",
        f"Medications: {', '.join(meds) if meds else 'None'}",
        f"Allergies: {', '.join(allergies) if allergies else 'None recorded'}",
    ]
    return "\n".join(lines)


# ── Session management ──────────────────────────────────────────────────────

def _ensure_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sessions_index() -> list[dict]:
    _ensure_outputs()
    if not SESSIONS_INDEX_PATH.exists():
        return []
    with open(SESSIONS_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_sessions_index(sessions: list[dict]):
    _ensure_outputs()
    with open(SESSIONS_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)


def create_session(name: str = None) -> dict:
    sessions = load_sessions_index()
    session_id = f"s{len(sessions) + 1}_{uuid.uuid4().hex[:6]}"
    session = {
        "session_id": session_id,
        "name": name or f"Session {len(sessions) + 1}",
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
        "status": "active",
        "query_count": 0,
    }
    sessions.append(session)
    _save_sessions_index(sessions)
    return session


def end_session(session_id: str):
    sessions = load_sessions_index()
    for s in sessions:
        if s["session_id"] == session_id:
            s["status"] = "complete"
            s["ended_at"] = datetime.now().isoformat()
    _save_sessions_index(sessions)


def update_session_query_count(session_id: str, increment: int = 1):
    sessions = load_sessions_index()
    for s in sessions:
        if s["session_id"] == session_id:
            s["query_count"] = s.get("query_count", 0) + increment
    _save_sessions_index(sessions)


# ── Results I/O ─────────────────────────────────────────────────────────────

def append_result_json(result_entry: dict, path: str = None):
    _ensure_outputs()
    fpath = Path(path) if path else RESULTS_PATH
    results = []
    if fpath.exists():
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    results.append(result_entry)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_results_for_sessions(session_ids: list[str], path: str = None) -> list[dict]:
    fpath = Path(path) if path else RESULTS_PATH
    if not fpath.exists():
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    return [r for r in all_results if r.get("session_id") in session_ids]


def save_results_csv(results: list[dict], path: str = None):
    _ensure_outputs()
    fpath = Path(path) if path else OUTPUT_DIR / "results.csv"
    rows = []
    for r in results:
        row = {
            "session_id": r.get("session_id"),
            "session_name": r.get("session_name"),
            "batch_mode": r.get("batch_mode"),
            "timestamp": r.get("timestamp"),
            "model": r.get("model"),
            "baseline": r.get("baseline"),
            "patient_id": r.get("patient_id"),
            "question": r.get("question"),
            "predicted_answer": r.get("predicted_answer"),
            "evidence_is_thin": r.get("evidence_is_thin"),
        }
        row.update(r.get("metrics", {}))
        rows.append(row)
    pd.DataFrame(rows).to_csv(fpath, index=False)


def save_aggregate_csv(agg_df: pd.DataFrame, path: str = None):
    _ensure_outputs()
    fpath = Path(path) if path else OUTPUT_DIR / "aggregate_metrics.csv"
    agg_df.to_csv(fpath, index=False)


# ── Graph generation ─────────────────────────────────────────────────────────

METRICS = ["hallucination", "abstained", "evidence_precision", "evidence_f1", "sar", "over_confidence"]
METRIC_LABELS = {
    "hallucination": "Hallucination Rate",
    "abstained": "Abstention Rate",
    "evidence_precision": "Evidence Precision",
    "evidence_f1": "Evidence F1",
    "sar": "SAR",
    "over_confidence": "Over-confidence Rate",
}
BASELINES_ORDER = ["baseline1", "baseline2", "baseline3", "proposed"]
MODELS_ORDER = ["claude", "gpt", "gemini", "llama"]


def generate_bar_charts(df: pd.DataFrame, output_dir: Path):
    out = output_dir / "bar_charts"
    out.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        col = f"{metric}_mean"
        if col not in df.columns:
            plt.close()
            continue
        pivot = df.pivot(index="baseline", columns="model", values=col).reindex(BASELINES_ORDER)
        pivot.plot(kind="bar", ax=ax, colormap="Set2", width=0.7)
        ax.set_title(f"{METRIC_LABELS[metric]} by Baseline and Model", fontsize=14, fontweight="bold")
        ax.set_xlabel("Baseline", fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_xticklabels(["Baseline 1", "Baseline 2", "Baseline 3", "Proposed"], rotation=0)
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out / f"{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()


def generate_model_comparison_charts(df: pd.DataFrame, output_dir: Path):
    out = output_dir / "model_comparison"
    out.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        col = f"{metric}_mean"
        if col not in df.columns:
            plt.close()
            continue
        pivot = df.pivot(index="model", columns="baseline", values=col).reindex(MODELS_ORDER)
        pivot.plot(kind="bar", ax=ax, colormap="Set1", width=0.7)
        ax.set_title(f"{METRIC_LABELS[metric]} by Model and Baseline", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_xticklabels(MODELS_ORDER, rotation=0)
        ax.legend(title="Baseline", labels=["Baseline 1", "Baseline 2", "Baseline 3", "Proposed"],
                  bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out / f"{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()


def generate_heatmaps(df: pd.DataFrame, output_dir: Path):
    out = output_dir / "heatmaps"
    out.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        col = f"{metric}_mean"
        if col not in df.columns:
            continue
        pivot = df.pivot(index="model", columns="baseline", values=col)
        pivot = pivot.reindex(index=MODELS_ORDER, columns=BASELINES_ORDER)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r" if metric in ["hallucination", "over_confidence"] else "RdYlGn",
                    ax=ax, vmin=0, vmax=1, linewidths=0.5)
        ax.set_title(f"{METRIC_LABELS[metric]} Heatmap", fontsize=14, fontweight="bold")
        ax.set_xticklabels(["Baseline 1", "Baseline 2", "Baseline 3", "Proposed"], rotation=0)
        plt.tight_layout()
        plt.savefig(out / f"{metric}.png", dpi=150, bbox_inches="tight")
        plt.close()


def generate_radar_charts(df: pd.DataFrame, output_dir: Path):
    out = output_dir / "radar_charts"
    out.mkdir(parents=True, exist_ok=True)
    metrics_radar = METRICS
    labels = [METRIC_LABELS[m] for m in metrics_radar]
    N = len(metrics_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for model in MODELS_ORDER:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        model_df = df[df["model"] == model]
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        for i, baseline in enumerate(BASELINES_ORDER):
            row = model_df[model_df["baseline"] == baseline]
            if row.empty:
                continue
            values = [row[f"{m}_mean"].values[0] for m in metrics_radar]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, color=colors[i],
                    label=["Baseline 1", "Baseline 2", "Baseline 3", "Proposed"][i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"{model.upper()} — Performance Profile", size=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(out / f"{model}.png", dpi=150, bbox_inches="tight")
        plt.close()


def generate_confusion_matrix_plots(confusion_dict: dict, output_dir: Path):
    out = output_dir / "confusion_matrices"
    out.mkdir(parents=True, exist_ok=True)
    for (model, baseline), cm in confusion_dict.items():
        matrix = np.array([
            [cm["confusion_TP"], cm["confusion_FP"]],
            [cm["confusion_FN"], cm["confusion_TN"]],
        ])
        labels = [["TP\n(Correct Abstention)", "FP\n(Over-cautious)"],
                  ["FN\n(Missed Hallucination)", "TN\n(Correct Answer)"]]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Evidence Thin", "Evidence Sufficient"],
                    yticklabels=["Abstained", "Did Not Abstain"])
        bl_label = {"baseline1": "Baseline 1", "baseline2": "Baseline 2",
                    "baseline3": "Baseline 3", "proposed": "Proposed"}.get(baseline, baseline)
        ax.set_title(f"Confusion Matrix — {model.upper()} / {bl_label}\n"
                     f"Abstention P={cm['abstention_precision']:.2f} "
                     f"R={cm['abstention_recall']:.2f} "
                     f"F1={cm['abstention_f1']:.2f}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / f"{model}_{baseline}.png", dpi=150, bbox_inches="tight")
        plt.close()


def generate_all_outputs(session_ids: list[str], results_path: str = None, output_dir: Path = None):
    if output_dir is None:
        output_dir = OUTPUT_DIR / "graphs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results_for_sessions(session_ids, path=results_path)
    if not results:
        return None, None

    agg_df = compute_aggregate_metrics(results)
    confusion_dict = compute_confusion_matrices(results)

    save_results_csv(results)
    save_aggregate_csv(agg_df)

    generate_bar_charts(agg_df, output_dir)
    generate_model_comparison_charts(agg_df, output_dir)
    generate_heatmaps(agg_df, output_dir)
    generate_radar_charts(agg_df, output_dir)
    generate_confusion_matrix_plots(confusion_dict, output_dir)

    return agg_df, confusion_dict


def zip_graphs(output_dir: Path = None) -> str:
    if output_dir is None:
        output_dir = OUTPUT_DIR / "graphs"
    zip_path = str(OUTPUT_DIR / "graphs_export.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in Path(output_dir).rglob("*.png"):
            zf.write(fpath, fpath.relative_to(output_dir.parent))
    return zip_path
