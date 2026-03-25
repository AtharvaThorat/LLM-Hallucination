"""
Baseline implementations for medical QA.
Each baseline differs only in prompt construction and retrieval strategy.
"""

from typing import Dict, Any
from retrieval import retrieve_for_baseline2, retrieve_for_baseline3


class BaselinePromptBuilder:
    """Base class for building prompts for baselines."""
    
    @staticmethod
    def build_prompt(retrieval_text: str = "", question: str = "") -> str:
        """Build a prompt (to be overridden by subclasses)."""
        raise NotImplementedError


class Baseline1(BaselinePromptBuilder):
    """
    Baseline 1: Full Context Prompting (Naïve)
    - Uses entire patient record
    - No retrieval or filtering
    - No grounding enforcement
    - High hallucination risk
    """
    
    @staticmethod
    def build_prompt(patient: Dict[str, Any], question: str) -> str:
        """
        Build prompt using full patient data.
        
        Args:
            patient: Complete patient dictionary
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        # Extract all patient information
        history = ", ".join(patient.get("past_medical_history", []))
        medications = ", ".join(patient.get("medications", []))
        allergies = ", ".join(patient.get("allergies", []))
        symptoms = ", ".join(patient.get("symptoms", []))
        
        prompt = f"""Patient Record:
- Past Medical History: {history if history else "None"}
- Medications: {medications if medications else "None"}
- Allergies: {allergies if allergies else "None"}
- Symptoms: {symptoms if symptoms else "None"}

Question: {question}
Answer:"""
        
        return prompt


class Baseline2(BaselinePromptBuilder):
    """
    Baseline 2: Retrieved-Context Prompting (Rule-Based)
    - Uses rule-based keyword matching for retrieval
    - Reduces irrelevant context
    - No grounding enforcement
    - Less noise than Baseline 1
    """
    
    @staticmethod
    def build_prompt(patient: Dict[str, Any], question: str) -> str:
        """
        Build prompt using rule-based retrieved data.
        
        Args:
            patient: Patient dictionary
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        # Retrieve data using rule-based method
        retrieved_data = retrieve_for_baseline2(patient, question)
        
        prompt = f"""Relevant Patient Information:
{retrieved_data}

Question: {question}
Answer:"""
        
        return prompt


class Baseline3(BaselinePromptBuilder):
    """
    Baseline 3: Standard RAG (Semantic Retrieval)
    - Uses embedding-based semantic retrieval
    - Retrieves top-k relevant chunks
    - Encourages (but does NOT enforce) grounding
    - Better retrieval than Baseline 2
    - Handles paraphrased queries
    """
    
    @staticmethod
    def build_prompt(patient: Dict[str, Any], question: str) -> str:
        """
        Build prompt using semantic retrieval.
        
        Args:
            patient: Patient dictionary
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        # Retrieve data using semantic method
        retrieved_chunks = retrieve_for_baseline3(patient, question)
        
        prompt = f"""Context:
{retrieved_chunks}

Instruction:
Use the context above to answer the question.

Question: {question}
Answer:"""
        
        return prompt


def baseline1(patient: Dict[str, Any], question: str) -> str:
    """
    Full-Context Prompting baseline.
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Prompt string
    """
    return Baseline1.build_prompt(patient, question)


def baseline2(patient: Dict[str, Any], question: str) -> str:
    """
    Rule-Based Retrieved-Context baseline.
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Prompt string
    """
    return Baseline2.build_prompt(patient, question)


def baseline3(patient: Dict[str, Any], question: str) -> str:
    """
    Standard RAG (Semantic Retrieval) baseline.
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Prompt string
    """
    return Baseline3.build_prompt(patient, question)


# Mapping of baseline names to functions
BASELINES = {
    "baseline1": baseline1,
    "baseline2": baseline2,
    "baseline3": baseline3,
}


def get_baseline(baseline_name: str):
    """
    Get a baseline function by name.
    
    Args:
        baseline_name: Name of the baseline
        
    Returns:
        Baseline function
        
    Raises:
        ValueError: If baseline not found
    """
    if baseline_name not in BASELINES:
        raise ValueError(
            f"Baseline '{baseline_name}' not found. "
            f"Available baselines: {list(BASELINES.keys())}"
        )
    return BASELINES[baseline_name]
