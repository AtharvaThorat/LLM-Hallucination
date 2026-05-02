"""
Evaluation metrics for hallucination-resistant medical QA.
Metrics include:
- Hallucination detection
- Exact match
- F1 score
- Evidence-based metrics (precision, recall, F1)
- SAR (Supporting Answer Rate)
- Abstention rate
"""

from typing import List, Dict, Any
import re
from difflib import SequenceMatcher


class HallucinationDetector:
    """
    Detect hallucination in model responses.
    Hallucination = claim not supported by evidence.
    """
    
    @staticmethod
    def detect(
        predicted_answer: str,
        ground_truth: str,
        supporting_evidence: List[str]
    ) -> int:
        """
        Detect if predicted answer contains hallucinations.

        Args:
            predicted_answer: Generated answer from model
            ground_truth: True answer
            supporting_evidence: List of supporting evidence texts

        Returns:
            1 if hallucination detected, 0 otherwise
        """
        # Tokenize using word boundaries to strip punctuation
        def tokenize(text):
            return set(re.findall(r'\b\w+\b', text.lower()))

        pred_tokens = tokenize(predicted_answer)
        truth_tokens = tokenize(ground_truth)

        evidence_tokens = set()
        for evidence in supporting_evidence:
            evidence_tokens.update(re.findall(r'\b\w+\b', evidence.lower()))

        # Reference = ground truth + evidence
        reference_tokens = truth_tokens | evidence_tokens

        # Short answers like "Yes" / "No" — check if contained in ground truth
        meaningful_pred = {t for t in pred_tokens if len(t) > 2}
        if not meaningful_pred:
            return 0 if predicted_answer.lower().strip() in ground_truth.lower() else 1

        # Primary check: is the prediction grounded in evidence?
        # (Do evidence key terms appear in the prediction?)
        meaningful_evidence = {t for t in evidence_tokens if len(t) > 3}
        if meaningful_evidence:
            pred_text = predicted_answer.lower()
            grounded = sum(1 for t in meaningful_evidence if t in pred_text)
            if grounded / len(meaningful_evidence) >= 0.3:
                return 0  # prediction is grounded in evidence — not hallucinated

        # Fallback: precision check (what fraction of prediction words are in reference)
        supported = len(meaningful_pred & reference_tokens)
        precision = supported / len(meaningful_pred)
        return 0 if precision >= 0.5 else 1
    
    @staticmethod
    def _answers_match(answer1: str, answer2: str, threshold: float = 0.8) -> bool:
        """Check if two answers are similar."""
        ratio = SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()
        return ratio >= threshold


class ExactMatch:
    """Exact match metric."""
    
    @staticmethod
    def calculate(predicted: str, ground_truth: str) -> int:
        """
        Calculate exact match.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1 if exact match, 0 otherwise
        """
        # Normalize: lowercase and strip punctuation/whitespace
        def normalize(text):
            return re.sub(r'[^\w\s]', '', text.lower()).strip()

        norm_pred = normalize(predicted)
        norm_truth = normalize(ground_truth)

        # Full match
        if norm_pred == norm_truth:
            return 1

        # Prefix match: "Yes" matches "Yes The patient's symptom list includes..."
        if norm_truth.startswith(norm_pred) and len(norm_pred) >= 3:
            return 1

        # High similarity match (>=0.85)
        ratio = SequenceMatcher(None, norm_pred, norm_truth).ratio()
        return 1 if ratio >= 0.85 else 0


class F1Score:
    """F1 score based on token overlap."""
    
    @staticmethod
    def calculate(predicted: str, ground_truth: str) -> float:
        """
        Calculate F1 score based on token-level overlap.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0-1)
        """
        pred_tokens = set(re.findall(r'\b\w+\b', predicted.lower()))
        truth_tokens = set(re.findall(r'\b\w+\b', ground_truth.lower()))
        
        if not truth_tokens and not pred_tokens:
            return 1.0
        
        if not truth_tokens or not pred_tokens:
            return 0.0
        
        overlap = len(pred_tokens & truth_tokens)
        
        if overlap == 0:
            return 0.0
        
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        recall = overlap / len(truth_tokens) if truth_tokens else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1


class EvidenceMetrics:
    """
    Metrics for evaluating grounding in evidence.
    """
    
    @staticmethod
    def calculate_precision(predicted: str, evidence: List[str]) -> float:
        """
        Evidence Precision: What fraction of predicted answer is supported by evidence?
        
        Args:
            predicted: Predicted answer
            evidence: List of supporting evidence texts
            
        Returns:
            Precision score (0-1)
        """
        if not predicted:
            return 0.0

        evidence_text = " ".join(evidence).lower()
        meaningful_tokens = [t for t in re.findall(r'\b\w+\b', predicted.lower()) if len(t) > 3]

        if not meaningful_tokens:
            return 1.0

        supported_count = sum(1 for t in meaningful_tokens if t in evidence_text)
        return supported_count / len(meaningful_tokens)
    
    @staticmethod
    def calculate_recall(predicted: str, evidence: List[str]) -> float:
        """
        Evidence Recall: What fraction of evidence is captured in prediction?
        
        Args:
            predicted: Predicted answer
            evidence: List of supporting evidence texts
            
        Returns:
            Recall score (0-1)
        """
        if not evidence:
            return 0.0

        predicted_lower = predicted.lower()
        matched_evidence = 0
        
        pred_word_set = set(re.findall(r'\b\w+\b', predicted_lower))
        for ev in evidence:
            ev_words = {w for w in re.findall(r'\b\w+\b', ev.lower()) if len(w) > 3}
            if not ev_words:
                continue
            overlap = len(ev_words & pred_word_set) / len(ev_words)
            if overlap >= 0.3:
                matched_evidence += 1

        return matched_evidence / len(evidence) if evidence else 1.0
    
    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """
        Evidence F1: Harmonic mean of precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class SAR:
    """
    Supporting Answer Rate (SAR).
    Percentage of answers that are supported by evidence.
    """
    
    @staticmethod
    def calculate(predicted: str, evidence: List[str]) -> int:
        """
        Calculate SAR: Is the answer supported by evidence?
        
        Args:
            predicted: Predicted answer
            evidence: List of supporting evidence
            
        Returns:
            1 if supported, 0 otherwise
        """
        if not predicted or not evidence:
            return 0
        
        evidence_text = " ".join(evidence).lower()
        meaningful_words = [w for w in re.findall(r'\b\w+\b', predicted.lower()) if len(w) > 3]

        if not meaningful_words:
            return 0

        matched = sum(1 for w in meaningful_words if w in evidence_text)

        # If 30% or more of meaningful words match evidence, consider it supported
        return 1 if matched / len(meaningful_words) >= 0.3 else 0


class AbstentionDetector:
    """
    Detect abstention in model responses.
    """
    
    @staticmethod
    def is_abstained(response: str) -> int:
        """
        Check if response indicates abstention.
        
        Args:
            response: Generated response
            
        Returns:
            1 if abstained, 0 otherwise
        """
        abstention_indicators = [
            "insufficient information",
            "not enough information",
            "no information",
            "cannot determine",
            "unclear from the evidence",
            "not stated",
            "not provided",
            "no evidence",
            "cannot answer",
        ]
        
        response_lower = response.lower()
        
        for indicator in abstention_indicators:
            if indicator in response_lower:
                return 1
        
        return 0


def evaluate(
    predicted_answer: str,
    ground_truth: str,
    supporting_evidence: List[str]
) -> Dict[str, Any]:
    """
    Complete evaluation of a prediction.
    
    Args:
        predicted_answer: Model-generated answer
        ground_truth: True answer from dataset
        supporting_evidence: List of evidence texts
        
    Returns:
        Dictionary with all metrics
    """
    # Calculate metrics
    hallucination = HallucinationDetector.detect(
        predicted_answer, ground_truth, supporting_evidence
    )
    
    exact_match = ExactMatch.calculate(predicted_answer, ground_truth)
    f1 = F1Score.calculate(predicted_answer, ground_truth)
    
    evidence_precision = EvidenceMetrics.calculate_precision(
        predicted_answer, supporting_evidence
    )
    evidence_recall = EvidenceMetrics.calculate_recall(
        predicted_answer, supporting_evidence
    )
    evidence_f1 = EvidenceMetrics.calculate_f1(evidence_precision, evidence_recall)
    
    sar = SAR.calculate(predicted_answer, supporting_evidence)
    abstained = AbstentionDetector.is_abstained(predicted_answer)
    
    return {
        "hallucination": hallucination,
        "exact_match": exact_match,
        "f1_score": round(f1, 4),
        "evidence_precision": round(evidence_precision, 4),
        "evidence_recall": round(evidence_recall, 4),
        "evidence_f1": round(evidence_f1, 4),
        "sar": sar,
        "abstained": abstained,
    }


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple evaluations.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Aggregated metrics
    """
    if not results:
        return {}
    
    metrics = {}
    metric_keys = ["hallucination", "exact_match", "f1_score", 
                   "evidence_precision", "evidence_recall", "evidence_f1", 
                   "sar", "abstained"]
    
    for key in metric_keys:
        values = [r["metrics"][key] for r in results if "metrics" in r]
        if values:
            metrics[f"{key}_mean"] = round(sum(values) / len(values), 4)
            metrics[f"{key}_count"] = len(values)
    
    return metrics
