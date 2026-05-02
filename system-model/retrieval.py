import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_atomic_facts(patient: dict) -> list[str]:
    facts = []

    for c in patient.get("conditions", []):
        name = c.get("condition", "")
        stop = c.get("stop")
        if stop:
            facts.append(f"Patient was previously diagnosed with {name} (resolved).")
        else:
            facts.append(f"Patient has been diagnosed with {name}.")

    for m in patient.get("medications", []):
        name = m.get("medication", "")
        stop = m.get("stop")
        if stop:
            facts.append(f"Patient was previously prescribed {name}.")
        else:
            facts.append(f"Patient is prescribed {name}.")

    allergies = patient.get("allergies", [])
    if not allergies:
        facts.append("No allergies are currently recorded for this patient.")
    else:
        for a in allergies:
            name = a.get("allergy", "")
            facts.append(f"Patient has a documented allergy to {name}.")

    for o in patient.get("observations", []):
        test = o.get("test", "")
        value = o.get("value", "")
        unit = o.get("unit", "")
        time = o.get("time", "")[:10] if o.get("time") else ""
        facts.append(f"Patient {test}: {value} {unit} (recorded {time}).")

    for e in patient.get("encounters", []):
        etype = e.get("type", "")
        start = e.get("start", "")[:10] if e.get("start") else ""
        facts.append(f"Patient had a {etype} encounter on {start}.")

    for p in patient.get("procedures", []):
        proc = p.get("procedure", "")
        date = p.get("date", "")[:10] if p.get("date") else ""
        facts.append(f"Patient underwent {proc} on {date}.")

    for cp in patient.get("careplans", []):
        plan = cp.get("careplan", "")
        facts.append(f"Patient is on care plan: {plan}.")

    return facts


def embed_facts(facts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(facts, convert_to_numpy=True)


def semantic_search(question: str, facts: list[str], embeddings: np.ndarray, top_k: int = 5) -> list[str]:
    if not facts:
        return []
    model = _get_model()
    q_emb = model.encode([question], convert_to_numpy=True)
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_k = min(top_k, len(facts))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [facts[i] for i in top_indices]


def rule_based_retrieval(patient: dict, question: str) -> list[str]:
    q = question.lower()
    facts = generate_atomic_facts(patient)

    if "allerg" in q:
        allergies = patient.get("allergies", [])
        if not allergies:
            return ["No allergies are currently recorded for this patient."]
        return [f"Patient has a documented allergy to {a.get('allergy', '')}." for a in allergies]

    elif "medication" in q or "drug" in q or "prescri" in q:
        meds = patient.get("medications", [])
        if not meds:
            return ["No medications are currently recorded for this patient."]
        return [f"Patient is prescribed {m.get('medication', '')}." for m in meds]

    elif "symptom" in q or "observation" in q or "vital" in q or "lab" in q or "test" in q or "result" in q:
        obs = patient.get("observations", [])
        if not obs:
            return ["No observations or lab results are recorded for this patient."]
        return [
            f"Patient {o.get('test', '')}: {o.get('value', '')} {o.get('unit', '')} (recorded {o.get('time','')[:10]})."
            for o in obs
        ]

    elif "history" in q or "condition" in q or "disease" in q or "diagnos" in q:
        conds = patient.get("conditions", [])
        if not conds:
            return ["No conditions are currently recorded for this patient."]
        return [f"Patient has been diagnosed with {c.get('condition', '')}." for c in conds]

    elif "procedure" in q or "surgery" in q or "operation" in q:
        procs = patient.get("procedures", [])
        if not procs:
            return ["No procedures are recorded for this patient."]
        return [f"Patient underwent {p.get('procedure', '')} on {p.get('date','')[:10]}." for p in procs]

    elif "care" in q or "plan" in q:
        cps = patient.get("careplans", [])
        if not cps:
            return ["No care plans are recorded for this patient."]
        return [f"Patient is on care plan: {cp.get('careplan', '')}." for cp in cps]

    else:
        return facts
