"""
Proposed Method: Evidence-Constrained RAG with Abstention
- Uses same retrieval as Baseline 3 (semantic)
- Adds strict grounding rules
- Enforces abstention when evidence is insufficient
"""

from typing import Dict, Any
from retrieval import retrieve_for_proposed


class ProposedMethod:
    """
    Evidence-Constrained RAG with Abstention.
    
    Key features:
    - Uses semantic retrieval (same as Baseline 3)
    - Adds strict grounding enforcement
    - Forces abstention for unsupported queries
    - Ensures medical safety
    """
    
    @staticmethod
    def build_prompt(patient: Dict[str, Any], question: str) -> str:
        """
        Build prompt with evidence constraints and abstention rules.
        
        Args:
            patient: Patient dictionary
            question: Question to answer
            
        Returns:
            Formatted prompt string with enforced constraints
        """
        # Retrieve data using semantic method (same as Baseline 3)
        retrieved_chunks = retrieve_for_proposed(patient, question)
        
        prompt = f"""Evidence:
{retrieved_chunks}

Rules:
- Answer ONLY if the evidence explicitly supports the answer
- DO NOT make assumptions or infer beyond what is stated
- DO NOT assume absence of information means negative
- If insufficient information to answer, respond with:
  "Insufficient information"
- Be precise and grounded in the provided evidence

Question: {question}
Answer:"""
        
        return prompt


def proposed(patient: Dict[str, Any], question: str) -> str:
    """
    Proposed method prompt builder.
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Prompt string with evidence constraints
    """
    return ProposedMethod.build_prompt(patient, question)


def is_abstention(response: str) -> bool:
    """
    Check if an LLM response indicates abstention.
    
    Args:
        response: Generated response from LLM
        
    Returns:
        True if response indicates insufficient information
    """
    abstention_phrases = [
        "insufficient information",
        "not enough information",
        "no information",
        "cannot determine",
        "unclear from the evidence",
        "not stated",
        "not provided",
        "no evidence",
    ]
    
    response_lower = response.lower().strip()
    
    for phrase in abstention_phrases:
        if phrase in response_lower:
            return True
    
    return False


def extract_answer(response: str) -> str:
    """
    Extract clean answer from generated response.
    
    Args:
        response: Raw generated response
        
    Returns:
        Cleaned answer text
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "Answer:",
        "The answer is:",
        "Based on the evidence:",
        "According to the evidence:",
    ]
    
    cleaned = response.strip()
    
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    
    return cleaned
