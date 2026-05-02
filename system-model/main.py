from datetime import datetime

from baselines import baseline1, baseline2, baseline3
from proposed import proposed_method
from models import query_model
from evaluation import evaluate_all

MODELS = ["claude", "gpt", "gemini", "llama"]
BASELINES = {
    "baseline1": lambda p, q, k: baseline1(p, q),
    "baseline2": lambda p, q, k: baseline2(p, q),
    "baseline3": lambda p, q, k: baseline3(p, q, top_k=k),
    "proposed":  lambda p, q, k: proposed_method(p, q, top_k=k),
}


def run_question(
    patient: dict,
    question: str,
    session_id: str,
    session_name: str,
    batch_mode: str = "auto",
    top_k: int = 5,
) -> list[dict]:
    results = []
    timestamp = datetime.now().isoformat()

    for baseline_name, prompt_fn in BASELINES.items():
        prompt, evidence = prompt_fn(patient, question, top_k)

        for model_name in MODELS:
            answer = query_model(model_name, prompt)
            metrics = evaluate_all(answer, evidence)

            results.append({
                "session_id": session_id,
                "session_name": session_name,
                "batch_mode": batch_mode,
                "timestamp": timestamp,
                "model": model_name,
                "baseline": baseline_name,
                "patient_id": patient.get("patient_id", ""),
                "question": question,
                "prompt_sent": prompt,
                "predicted_answer": answer,
                "retrieved_chunks": evidence,
                "evidence_is_thin": metrics.pop("evidence_is_thin"),
                "metrics": metrics,
            })

    return results
