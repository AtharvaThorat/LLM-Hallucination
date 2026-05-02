"""
Utility functions for the medical QA system.
"""

import json
import os
from typing import List, Dict, Any


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load the JSON dataset containing patient records.
    
    Args:
        dataset_path: Path to the medical_qa_dataset.json file
        
    Returns:
        Dictionary containing patients and their QA interactions
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def load_env_variables():
    """
    Load environment variables from .env file.
    """
    from dotenv import load_dotenv
    load_dotenv()


def get_api_key(key_name: str) -> str:
    """
    Retrieve API key from environment variables.
    
    Args:
        key_name: Name of the environment variable
        
    Returns:
        API key value
        
    Raises:
        ValueError: If the key is not found
    """
    key_value = os.getenv(key_name)
    if not key_value:
        raise ValueError(f"Environment variable '{key_name}' not found in .env file")
    return key_value


def format_patient_data(patient: Dict[str, Any]) -> str:
    """
    Convert patient data to readable text format.
    
    Args:
        patient: Patient dictionary
        
    Returns:
        Formatted text representation of patient data
    """
    text = ""
    
    if "past_medical_history" in patient and patient["past_medical_history"]:
        text += f"Past Medical History: {', '.join(patient['past_medical_history'])}\n"
    
    if "medications" in patient and patient["medications"]:
        text += f"Medications: {', '.join(patient['medications'])}\n"
    
    if "allergies" in patient and patient["allergies"]:
        text += f"Allergies: {', '.join(patient['allergies'])}\n"
    
    if "symptoms" in patient and patient["symptoms"]:
        text += f"Symptoms: {', '.join(patient['symptoms'])}\n"
    
    return text.strip()


def extract_patient_by_id(data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
    """
    Extract a specific patient from the dataset.
    
    Args:
        data: Full dataset
        patient_id: Patient identifier
        
    Returns:
        Patient dictionary
        
    Raises:
        ValueError: If patient not found
    """
    for patient in data["patients"]:
        if patient["patient_id"] == patient_id:
            return patient
    
    raise ValueError(f"Patient with ID {patient_id} not found in dataset")


def extract_qa_pairs(patient: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all QA interactions from a patient record.
    
    Args:
        patient: Patient dictionary
        
    Returns:
        List of QA pairs with supporting evidence
    """
    if "qa_interactions" not in patient:
        return []
    
    return patient["qa_interactions"]


def get_evidence_text(evidence_list: List[Dict[str, Any]]) -> List[str]:
    """
    Extract evidence text from supporting evidence.
    
    Args:
        evidence_list: List of evidence dictionaries
        
    Returns:
        List of evidence text strings
    """
    evidence_texts = []
    for evidence in evidence_list:
        if "evidence_text" in evidence:
            evidence_texts.append(evidence["evidence_text"])
    return evidence_texts


def create_output_dir(output_dir: str = "outputs"):
    """
    Create output directory structure.
    
    Args:
        output_dir: Base output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)


def save_results_json(results: List[Dict[str, Any]], output_dir: str = "outputs"):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory path
    """
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def save_results_csv(results: List[Dict[str, Any]], output_dir: str = "outputs"):
    """
    Save evaluation results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory path
    """
    import csv
    
    output_file = os.path.join(output_dir, "results.csv")
    
    if not results:
        print("No results to save")
        return
    
    # Get all possible metric keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    fieldnames = sorted(list(all_keys))
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_file}")
