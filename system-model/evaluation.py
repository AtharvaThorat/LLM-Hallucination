import re
import pandas as pd
from collections import defaultdict

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "but", "if", "then", "that", "this", "it", "its", "not",
    "no", "so", "up", "out", "about", "than", "also", "any", "all",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "too", "very", "just", "there",
    "their", "they", "them", "these", "those", "what", "which", "who",
    "whom", "when", "where", "why", "how", "i", "you", "he", "she",
    "we", "my", "your", "his", "her", "our", "us", "me", "him",
}

ABSTENTION_PHRASES = [
    "insufficient information",
    "not documented",
    "no record",
    "cannot determine",
    "not enough information",
    "unable to confirm",
    "not recorded",
    "no documentation",
    "no data available",
    "cannot be confirmed",
]

MEDICAL_CLAIM_MARKERS = [
    "allergic", "allergy", "diagnosed", "diagnosis", "prescribed",
    "has condition", "suffers from", "positive for", "negative for",
    "does not have", "no allergy", "no condition", "no medication",
    "is taking", "was taking", "has been taking", "has no",
    "never had", "confirmed", "documented allergy", "known allergy",
]


def content_words(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 2}


def is_abstention(answer: str) -> bool:
    a = answer.lower().strip()
    return any(phrase in a for phrase in ABSTENTION_PHRASES)


def is_evidence_thin(evidence_chunks: list[str]) -> bool:
    if not evidence_chunks:
        return True
    thin_markers = ["not recorded", "no allergies are currently recorded", "none documented"]
    return all(any(m in chunk.lower() for m in thin_markers) for chunk in evidence_chunks)


def hallucination_flag(answer: str, evidence_chunks: list[str]) -> int:
    if is_abstention(answer):
        return 0

    answer_lower = answer.lower()
    evidence_text = " ".join(evidence_chunks).lower()
    answer_content = content_words(answer)
    evidence_content = content_words(evidence_text)

    has_medical_claim = any(marker in answer_lower for marker in MEDICAL_CLAIM_MARKERS)
    if not has_medical_claim:
        return 0

    unsupported = answer_content - evidence_content
    unsupported_ratio = len(unsupported) / max(len(answer_content), 1)

    if unsupported_ratio > 0.6:
        return 1

    ep = evidence_precision(answer, evidence_chunks)
    if ep < 0.25 and has_medical_claim:
        return 1

    return 0


def abstention_flag(answer: str) -> int:
    return 1 if is_abstention(answer) else 0


def evidence_precision(answer: str, evidence_chunks: list[str]) -> float:
    if not evidence_chunks:
        return 0.0
    pred = content_words(answer)
    evid = content_words(" ".join(evidence_chunks))
    if not pred:
        return 0.0
    return len(pred & evid) / len(pred)


def evidence_recall(answer: str, evidence_chunks: list[str]) -> float:
    if not evidence_chunks:
        return 0.0
    pred = content_words(answer)
    evid = content_words(" ".join(evidence_chunks))
    if not evid:
        return 0.0
    return len(pred & evid) / len(evid)


def evidence_f1(answer: str, evidence_chunks: list[str]) -> float:
    p = evidence_precision(answer, evidence_chunks)
    r = evidence_recall(answer, evidence_chunks)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def sar(answer: str, evidence_chunks: list[str], hallucination: int) -> int:
    if is_abstention(answer):
        return 1
    if hallucination == 1:
        return 0
    return int(evidence_f1(answer, evidence_chunks) >= 0.3)


def over_confidence(answer: str, evidence_chunks: list[str], hallucination: int) -> int:
    if is_evidence_thin(evidence_chunks) and hallucination == 1:
        return 1
    return 0


def evaluate_all(answer: str, evidence_chunks: list[str]) -> dict:
    h = hallucination_flag(answer, evidence_chunks)
    a = abstention_flag(answer)
    ep = evidence_precision(answer, evidence_chunks)
    ef1 = evidence_f1(answer, evidence_chunks)
    s = sar(answer, evidence_chunks, h)
    oc = over_confidence(answer, evidence_chunks, h)
    thin = is_evidence_thin(evidence_chunks)

    return {
        "hallucination": h,
        "abstained": a,
        "evidence_precision": round(ep, 4),
        "evidence_f1": round(ef1, 4),
        "sar": s,
        "over_confidence": oc,
        "evidence_is_thin": thin,
    }


def compute_aggregate_metrics(results: list[dict]) -> pd.DataFrame:
    rows = []
    groups = defaultdict(list)

    for r in results:
        key = (r["model"], r["baseline"])
        groups[key].append(r["metrics"])

    for (model, baseline), metrics_list in groups.items():
        df_m = pd.DataFrame(metrics_list)
        row = {"model": model, "baseline": baseline}
        for col in ["hallucination", "abstained", "evidence_precision", "evidence_f1", "sar", "over_confidence"]:
            row[f"{col}_mean"] = round(df_m[col].mean(), 4)
            row[f"{col}_std"] = round(df_m[col].std(), 4)

        cm = _confusion_matrix_for_group(metrics_list)
        row.update(cm)
        rows.append(row)

    return pd.DataFrame(rows)


def _confusion_matrix_for_group(metrics_list: list[dict]) -> dict:
    TP = FP = FN = TN = 0
    for m in metrics_list:
        thin = m.get("evidence_is_thin", False)
        abstained = m.get("abstained", 0)
        hallucinated = m.get("hallucination", 0)

        if thin and abstained:
            TP += 1
        elif not thin and abstained:
            FP += 1
        elif thin and not abstained:
            FN += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "confusion_TP": TP,
        "confusion_FP": FP,
        "confusion_FN": FN,
        "confusion_TN": TN,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
    }


def compute_confusion_matrices(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        key = (r["model"], r["baseline"])
        groups[key].append(r["metrics"])
    return {key: _confusion_matrix_for_group(mlist) for key, mlist in groups.items()}
