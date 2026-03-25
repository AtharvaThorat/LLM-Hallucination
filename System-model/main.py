"""
Main experiment runner for hallucination-resistant medical QA system.
Orchestrates:
- Dataset loading
- Model querying
- Baseline evaluation
- Proposed method evaluation
- Metric computation
- Results storage
- Visualization generation
"""

import os
import sys
from typing import List, Dict, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import all system modules
from utils import (
    load_dataset, load_env_variables, extract_patient_by_id,
    extract_qa_pairs, get_evidence_text, create_output_dir,
    save_results_json, save_results_csv
)
from models import query_model
from baselines import get_baseline
from proposed import proposed
from evaluation import evaluate, aggregate_metrics
from proposed import is_abstention, extract_answer


class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "outputs",
        models: List[str] = None,
        baselines: List[str] = None,
        sample_size: int = None
    ):
        """
        Initialize experiment runner.
        
        Args:
            dataset_path: Path to medical_qa_dataset.json
            output_dir: Where to save results
            models: List of models to test
            baselines: List of baselines to test
            sample_size: Number of patients to evaluate (None for all)
        """
        # Load environment
        load_env_variables()
        
        # Load dataset
        self.dataset = load_dataset(dataset_path)
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Set models
        self.models = models or ["gemini"]
        
        # Set baselines
        self.baselines = baselines or ["baseline1", "baseline2", "baseline3", "proposed"]
        
        # Sample size
        self.sample_size = sample_size
        
        # Results storage
        self.results = []
        
        # Create output directory
        create_output_dir(output_dir)
        
        print(f"Experiment Runner Initialized")
        print(f"  Dataset: {dataset_path}")
        print(f"  Models: {self.models}")
        print(f"  Baselines: {self.baselines}")
        print(f"  Sample size: {self.sample_size if self.sample_size else 'All'}")
        print(f"  Output directory: {output_dir}")
    
    def run(self):
        """Run complete experiment."""
        print("\n" + "="*60)
        print("STARTING EXPERIMENT")
        print("="*60 + "\n")
        
        # Get patients to evaluate
        patients = self.dataset["patients"]
        if self.sample_size:
            patients = patients[:self.sample_size]
        
        print(f"Evaluating {len(patients)} patients...\n")
        
        # Iterate through patients
        for patient_idx, patient in enumerate(patients):
            patient_id = patient["patient_id"]
            print(f"[{patient_idx + 1}/{len(patients)}] Processing patient {patient_id}...")
            
            # Get QA pairs for this patient
            qa_pairs = extract_qa_pairs(patient)
            
            if not qa_pairs:
                print(f"  No QA pairs found, skipping...")
                continue
            
            # Evaluate each QA pair
            for qa_pair in qa_pairs:
                question = qa_pair["question"]
                ground_truth = qa_pair["answer"]
                supporting_evidence = get_evidence_text(qa_pair["supporting_evidence"])
                
                # Evaluate all baselines
                for baseline_name in self.baselines:
                    self._evaluate_baseline(
                        patient, baseline_name, question,
                        ground_truth, supporting_evidence
                    )
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60 + "\n")
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        print(f"Total evaluations: {len(self.results)}")
        print(f"Results saved to {self.output_dir}/")
    
    def _evaluate_baseline(
        self,
        patient: Dict[str, Any],
        baseline_name: str,
        question: str,
        ground_truth: str,
        supporting_evidence: List[str]
    ):
        """
        Evaluate a baseline on a single QA pair.
        
        Args:
            patient: Patient dictionary
            baseline_name: Name of baseline
            question: Question to answer
            ground_truth: Ground truth answer
            supporting_evidence: Supporting evidence
        """
        try:
            # Build prompt
            if baseline_name == "proposed":
                prompt = proposed(patient, question)
            else:
                baseline_func = get_baseline(baseline_name)
                prompt = baseline_func(patient, question)
            
            # Query models
            for model_name in self.models:
                try:
                    print(f"    Querying {model_name} for {baseline_name}...")
                    
                    # Get model response
                    predicted_answer = query_model(model_name, prompt)

                    # Skip evaluation if model returned an error
                    if predicted_answer.startswith("Error generating response"):
                        print(f"      Skipping — model returned an error")
                        continue

                    predicted_answer = extract_answer(predicted_answer)

                    # Evaluate
                    metrics = evaluate(predicted_answer, ground_truth, supporting_evidence)
                    
                    # Store result
                    result = {
                        "model": model_name,
                        "baseline": baseline_name,
                        "patient_id": patient["patient_id"],
                        "question": question,
                        "predicted_answer": predicted_answer,
                        "ground_truth": ground_truth,
                        "metrics": metrics
                    }
                    
                    self.results.append(result)
                
                except Exception as e:
                    print(f"      Error querying {model_name}: {e}")
        
        except Exception as e:
            print(f"    Error evaluating baseline: {e}")
    
    def _save_results(self):
        """Save results to JSON and CSV."""
        # Flatten results for CSV
        flattened_results = []
        for result in self.results:
            flat_result = {
                "model": result["model"],
                "baseline": result["baseline"],
                "patient_id": result["patient_id"],
                "question": result["question"],
                "predicted_answer": result["predicted_answer"],
                "ground_truth": result["ground_truth"],
            }
            flat_result.update(result["metrics"])
            flattened_results.append(flat_result)
        
        # Save JSON
        save_results_json(flattened_results, self.output_dir)
        
        # Save CSV
        save_results_csv(flattened_results, self.output_dir)
    
    def _generate_visualizations(self):
        """Generate performance visualizations."""
        if not self.results:
            print("No results to visualize")
            return
        
        # Convert to DataFrame for easy analysis
        flattened = []
        for result in self.results:
            flat = {
                "model": result["model"],
                "baseline": result["baseline"],
            }
            flat.update(result["metrics"])
            flattened.append(flat)
        
        df = pd.DataFrame(flattened)
        
        print("\nGenerating visualizations...")
        
        # Metrics to plot
        metrics = ["hallucination", "exact_match", "f1_score", 
                   "evidence_f1", "sar", "abstained"]
        
        # Create bar plots
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Group by baseline and model
            pivot_data = df.pivot_table(
                values=metric,
                index="baseline",
                columns="model",
                aggfunc="mean",
                fill_value=0
            )

            ax = plt.gca()
            pivot_data.plot(kind="bar", ax=ax, rot=30)

            # Add value labels on top of every bar (including 0-height bars)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

            plt.title(f"{metric.replace('_', ' ').title()} by Baseline and Model",
                     fontsize=14, fontweight='bold')
            plt.xlabel("Baseline", fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylim(0, max(pivot_data.values.max() * 1.3, 0.1))  # ensure y-axis not collapsed
            plt.tight_layout()
            
            # Save
            output_path = os.path.join(
                self.output_dir, "graphs",
                f"{metric}.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Group by baseline
        grouped = df.groupby("baseline")[metrics].mean()
        print("\nBy Baseline:")
        print(grouped.round(4))
        
        # Group by model
        grouped = df.groupby("model")[metrics].mean()
        print("\nBy Model:")
        print(grouped.round(4))
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("SUMMARY STATISTICS\n")
            f.write("="*60 + "\n\n")
            
            grouped = df.groupby("baseline")[metrics].mean()
            f.write("By Baseline:\n")
            f.write(grouped.round(4).to_string())
            f.write("\n\n")
            
            grouped = df.groupby("model")[metrics].mean()
            f.write("By Model:\n")
            f.write(grouped.round(4).to_string())
        
        print(f"\n  Saved: {summary_file}")


def main():
    """Main entry point."""
    # Configuration
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "medical_qa_dataset.json"
    )
    
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs"
    )
    
    # Initialize experiment
    experiment = ExperimentRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        models=["gemini", "chatgpt"],
        baselines=["baseline1", "baseline2", "baseline3", "proposed"],
        sample_size=2  # Limit to 2 patients; set to None for full dataset
    )
    
    # Run experiment
    experiment.run()


if __name__ == "__main__":
    main()
