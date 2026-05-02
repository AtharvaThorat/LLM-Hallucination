"""
Retrieval systems for baselines and proposed method.
Implements:
- Rule-based retrieval (Baseline 2)
- Semantic/embedding-based retrieval (Baseline 3, Proposed)
"""

from typing import List, Dict, Any
import re


class RuleBasedRetrieval:
    """
    Rule-based keyword matching retrieval for Baseline 2.
    """
    
    @staticmethod
    def retrieve(patient: Dict[str, Any], question: str) -> str:
        """
        Retrieve relevant patient data using keyword matching.
        
        Args:
            patient: Patient dictionary
            question: Question to answer
            
        Returns:
            Formatted retrieved data as text
        """
        question_lower = question.lower()
        retrieved_data = {}
        
        # Check for allergies-related questions
        if any(word in question_lower for word in ["allerg", "reaction", "intolerant"]):
            if "allergies" in patient and patient["allergies"]:
                retrieved_data["allergies"] = patient["allergies"]
        
        # Check for medication-related questions
        if any(word in question_lower for word in ["medication", "drug", "medicine", "taking", "prescribed"]):
            if "medications" in patient and patient["medications"]:
                retrieved_data["medications"] = patient["medications"]
        
        # Check for symptom-related questions
        if any(word in question_lower for word in ["symptom", "pain", "ache", "discomfort", "experiencing"]):
            if "symptoms" in patient and patient["symptoms"]:
                retrieved_data["symptoms"] = patient["symptoms"]
        
        # Check for medical history questions
        if any(word in question_lower for word in ["history", "condition", "disease", "had", "suffer", "past"]):
            if "past_medical_history" in patient and patient["past_medical_history"]:
                retrieved_data["past_medical_history"] = patient["past_medical_history"]
        
        # If no specific match, return all fields
        if not retrieved_data:
            retrieved_data = {
                "past_medical_history": patient.get("past_medical_history", []),
                "medications": patient.get("medications", []),
                "allergies": patient.get("allergies", []),
                "symptoms": patient.get("symptoms", []),
            }
        
        # Format as text
        return RuleBasedRetrieval._format_retrieved_data(retrieved_data)
    
    @staticmethod
    def _format_retrieved_data(data: Dict[str, List[str]]) -> str:
        """Format retrieved data as readable text."""
        text_parts = []
        
        if "past_medical_history" in data and data["past_medical_history"]:
            text_parts.append(f"Past Medical History: {', '.join(data['past_medical_history'])}")
        
        if "medications" in data and data["medications"]:
            text_parts.append(f"Medications: {', '.join(data['medications'])}")
        
        if "allergies" in data and data["allergies"]:
            text_parts.append(f"Allergies: {', '.join(data['allergies'])}")
        
        if "symptoms" in data and data["symptoms"]:
            text_parts.append(f"Symptoms: {', '.join(data['symptoms'])}")
        
        return "\n".join(text_parts)


class SemanticRetrieval:
    """
    Semantic retrieval using embeddings for Baseline 3 and Proposed method.
    """
    
    def __init__(self):
        """Initialize semantic retrieval with embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load sentence-transformers: {e}")
            print("Falling back to simple keyword matching")
            self.encoder = None
    
    def retrieve(
        self,
        patient: Dict[str, Any],
        question: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve relevant patient data using semantic similarity.
        
        Args:
            patient: Patient dictionary
            question: Question to answer
            top_k: Number of top chunks to retrieve
            
        Returns:
            Formatted retrieved chunks as text
        """
        # Extract chunks from patient data
        chunks = self._create_chunks(patient)
        
        if not chunks:
            return "No relevant information found."
        
        if self.encoder is None:
            # Fallback to keyword matching if no encoder available
            return self._keyword_based_retrieve(patient, question)
        
        # Retrieve top-k chunks using semantic similarity
        retrieved_chunks = self._semantic_retrieve(chunks, question, top_k)
        
        return "\n".join(retrieved_chunks)
    
    def _create_chunks(self, patient: Dict[str, Any]) -> List[str]:
        """
        Create atomic chunks from patient data.
        
        Args:
            patient: Patient dictionary
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Create chunks from past medical history
        for condition in patient.get("past_medical_history") or []:
            chunks.append(f"Patient has a history of {condition}.")

        # Create chunks from medications
        for medication in patient.get("medications") or []:
            chunks.append(f"Patient is taking {medication}.")

        # Create chunks from allergies
        for allergy in patient.get("allergies") or []:
            chunks.append(f"Patient has an allergy: {allergy}.")

        # Create chunks from symptoms
        for symptom in patient.get("symptoms") or []:
            chunks.append(f"Patient is experiencing {symptom}.")
        
        return chunks
    
    def _semantic_retrieve(
        self,
        chunks: List[str],
        question: str,
        top_k: int
    ) -> List[str]:
        """
        Retrieve top-k chunks using semantic similarity.
        
        Args:
            chunks: List of text chunks
            question: Question to answer
            top_k: Number of chunks to retrieve
            
        Returns:
            Top-k most relevant chunks
        """
        # Embed question
        question_embedding = self.encoder.encode(question)
        
        # Embed all chunks
        chunk_embeddings = self.encoder.encode(chunks)
        
        # Compute similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return top-k chunks
        return [chunks[idx] for idx in top_indices if similarities[idx] > 0.0]
    
    def _keyword_based_retrieve(
        self,
        patient: Dict[str, Any],
        question: str
    ) -> str:
        """
        Fallback keyword-based retrieval.
        
        Args:
            patient: Patient dictionary
            question: Question to answer
            
        Returns:
            Retrieved information as text
        """
        chunks = self._create_chunks(patient)
        
        if not chunks:
            return "No relevant information found."
        
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(question_words & chunk_words)
            if overlap > 0:
                scored_chunks.append((chunk, overlap))
        
        # Sort by overlap score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top chunks
        retrieved = [chunk for chunk, _ in scored_chunks[:5]]
        
        return "\n".join(retrieved) if retrieved else "\n".join(chunks[:5])


def retrieve_for_baseline2(patient: Dict[str, Any], question: str) -> str:
    """
    Retrieve data for Baseline 2 (Rule-Based).
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Retrieved data as text
    """
    retrieval = RuleBasedRetrieval()
    return retrieval.retrieve(patient, question)


def retrieve_for_baseline3(patient: Dict[str, Any], question: str) -> str:
    """
    Retrieve data for Baseline 3 (Semantic RAG).
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Retrieved data as text
    """
    retrieval = SemanticRetrieval()
    return retrieval.retrieve(patient, question, top_k=5)


def retrieve_for_proposed(patient: Dict[str, Any], question: str) -> str:
    """
    Retrieve data for Proposed method (Same as Baseline 3).
    
    Args:
        patient: Patient dictionary
        question: Question to answer
        
    Returns:
        Retrieved data as text
    """
    return retrieve_for_baseline3(patient, question)
