# Output and Visualization Specifications

## Objective

The system must generate structured outputs and visualizations that allow clear comparison of:

- Different baselines
- Different models
- Hallucination-related performance metrics

The outputs must be suitable for:
- Research analysis
- Paper writing
- Presentation

---

# Output Directory Structure

All outputs must be saved in:
outputs/
├── results.json
├── results.csv
├── graphs/
│ ├── hallucination_rate.png
│ ├── exact_match.png
│ ├── f1_score.png
│ ├── supported_facts.png
│ ├── sar.png
│ ├── abstention.png

---

---

# 🧾 1. Structured Results Output

## Format: JSON

Each evaluation instance must include:

```json
{
  "model": "gemini",
  "baseline": "baseline2",
  "patient_id": "P001",
  "question": "...",
  "predicted_answer": "...",
  "ground_truth": "...",
  "metrics": {
    "hallucination": 0,
    "exact_match": 1,
    "f1_score": 0.92,
    "evidence_precision": 1.0,
    "evidence_recall": 1.0,
    "evidence_f1": 1.0,
    "sar": 1,
    "abstained": 0
  }
}
```
---
Format: CSV
Columns must include:

model, baseline, patient_id, question, predicted_answer ground_truth, hallucination, exact_match, f1_score evidence_precision, evidence_recall, evidence_f1, sar, abstained

---
### Visualization Requirements

All graphs must be generated automatically after evaluation.

- Graph Type 1: Metric-wise Bar Charts (MANDATORY)

For each metric:

- X-axis: Baselines (baseline1, baseline2, baseline3, proposed)
- Y-axis: Metric value
- Hue: Model (gemini, llama, biomedlm)

Metrics to plot:
- Hallucination Rate
- Exact Match
- F1 Score
- Evidence F1
- SAR
- Abstention Rate

- Graph Type 2: Model Comparison Charts
- X-axis: Models
- Y-axis:

Metric value
- Hue: Baselines

Purpose:
Compare how each model performs across baselines

- Graph Type 3: Radar Plot (OPTIONAL)
Each model shown across all metrics
Useful for presentation

---
### Graph Output Location

All graphs must be saved in:

outputs/graphs/

Example:
outputs/graphs/hallucination_rate.png
outputs/graphs/f1_score.png
outputs/graphs/sar.png

---
### Visualization Implementation Rules

Graphs must:
- Have titles
- Include legends
- Be clearly labeled

### When to Generate Graphs

Graphs must be generated:

AFTER all evaluations are complete

Automatically within main.py

---
### Key Insights Expected

The output system must allow analysis of:

- Hallucination Reduction
    - Proposed method should show lowest hallucination rate
    - Model Comparison

Which model is safest (lowest hallucination)
Which model is most accurate (highest F1)

Trade-offs
- Accuracy vs Abstention
- Retrieval vs Grounding

---
### Strict Requirements

DO NOT:
- Skip saving results
- Skip visualization
- Generate partial outputs
- Mix metrics incorrectly

---
### Final Goal

The output system must:

Enable full comparison across:
- models
- baselines
- Provide clear visual evidence of improvements

---