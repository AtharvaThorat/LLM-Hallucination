# Baselines and Proposed Method Implementation Guide

---
## Objective

This document defines the exact implementation rules for all baselines and the proposed method in the **hallucination-resistant medical question answering system**

The goal is to ensure:
* Consistent implementation across all methods
* Fair comparison across models
* Clear separation between **retrieval**, **prompting**, and **evaluation**

This project treats hallucination as a **system design problem**, not a model training problem. 

---

# Core Principle

> All methods MUST use the SAME:

* Dataset
* LLM interface
* Evaluation pipeline

> Methods differ ONLY in:

* Retrieval strategy
* Prompt construction
* Grounding & abstention enforcement

NOT allowed:

* Changing model behavior
* Using different datasets
* Injecting external knowledge
* Fine-tuning per baseline

---

# Input Specification

Each method receives:

```python
patient   # dictionary containing patient data
question  # string
```

### Patient Structure

Each patient contains:

* `past_medical_history`
* `medications`
* `allergies`
* `symptoms`
* `qa_interactions` (with supporting evidence)

---

# Output Specification

Each method MUST return:

```python
prompt  # string to send to LLM
```

The LLM call is handled **outside** the baseline.

---

# Baseline 1: Full-Context Prompting (Naïve)

## Description

* Uses the **entire patient record**
* No retrieval or filtering
* No grounding enforcement

Relies entirely on the LLM to self-regulate correctness.

## Implementation Steps

1. Extract ALL fields:

   * medical history
   * medications
   * allergies
   * symptoms

2. Convert into readable text

## Prompt Template

```
Patient Record:
- Past Medical History: {history}
- Medications: {medications}
- Allergies: {allergies}
- Symptoms: {symptoms}

Question: {question}
Answer:
```

## Behavior

* Model has full visibility of data
* Model is free to infer

## Limitations

* High hallucination risk
* Cannot detect missing information
* Does not scale for long records

---

# Baseline 2: Retrieved-Context Prompting (Rule-Based)

## Description

* Retrieves relevant patient data using **keyword matching**
* Reduces irrelevant context
* No grounding enforcement

## Retrieval Logic

```python
q = question.lower()

if "allerg" in q:
    retrieve allergies
elif "medication" in q or "drug" in q:
    retrieve medications
elif "symptom" in q:
    retrieve symptoms
elif "history" in q or "condition" in q:
    retrieve past_medical_history
else:
    retrieve all fields
```

## Prompt Template

```
Relevant Patient Information:
{retrieved_data}

Question: {question}
Answer:
```

## Behavior

* Less noise than Baseline 1
* Model still allowed to infer


---

# Baseline 3: Standard RAG (Semantic Retrieval)

## Description

* Uses **embedding-based retrieval**
* Retrieves top-k relevant chunks
* Encourages (but does NOT enforce) grounding

## Retrieval Pipeline

1. Convert patient data into atomic facts:

   * "Patient has hypertension"
   * "Patient takes metformin"
   * "Patient allergic to penicillin"

2. Embed each chunk

3. Embed question

4. Perform similarity search

5. Retrieve top-k chunks

## Prompt Template

```
Context:
{retrieved_chunks}

Instruction:
Use the context above to answer the question.

Question: {question}
Answer:
```

## Behavior

* Better retrieval than Baseline 2
* Handles paraphrased queries


---

# Proposed Method: Evidence-Constrained RAG with Abstention

## Description

* Uses SAME retrieval as Baseline 3
* Adds **strict grounding rules**
* Enforces **abstention when evidence is insufficient**

This aligns with medical safety requirements by preventing unsupported claims. 

## Additional Rules

The model MUST:

* Use ONLY provided evidence
* NOT infer missing information
* NOT assume absence = negative
* Abstain if evidence is insufficient

## Prompt Template

```
Evidence:
{retrieved_chunks}

Rules:
- Answer ONLY if the evidence explicitly supports the answer
- DO NOT make assumptions
- If insufficient information, respond with:
  "Insufficient information"

Question: {question}
Answer:
```

## Expected Behavior

* No unsupported inference
* Safe responses
* Explicit abstention

## Example Output

```
Insufficient information to determine aspirin allergy.
```

## Advantages

* Eliminates hallucination from missing data
* Improves reliability
* Enforces system-level correctness

---

# Optional Baseline: Chain-of-Thought RAG (CoT-RAG)

## Description

* Same retrieval as Baseline 3
* Adds reasoning steps

## Prompt Template

```
Context:
{retrieved_chunks}

Think step-by-step using the context.

Question: {question}
Answer:
```

## Behavior

* May improve reasoning

---

# Implementation Rules

## DO

* Implement each method as:

```python
def baseline_name(patient, question):
    return prompt
```

* Keep logic modular
* Reuse retrieval components
* Keep prompts consistent

## DO NOT

* Change model behavior per baseline
* Mix patient data
* Skip retrieval where required
* Hardcode answers

---

# Key Comparison Summary

| Method     | Retrieval Type | Grounding | Abstention |
| ---------- | -------------- | --------- | ---------- |
| Baseline 1 | None           | ❌         | ❌          |
| Baseline 2 | Rule-based     | ❌         | ❌          |
| Baseline 3 | Semantic       | Soft      | ❌          |
| Proposed   | Semantic       | Strict    | ✅          |

---

# Final Goal

The system must:

* Run all baselines across all models
* Enable fair comparison
* Measure hallucination behavior
* Demonstrate improvement of the proposed method

---

# Summary

This file defines:

* Exact behavior of each baseline
* Prompt structures
* Retrieval logic
* Grounding and abstention rules

All implementations MUST strictly follow this specification.

---

# Next Steps

You can now generate:

* `baselines.py`
* `retrieval.py`
* `models.py`
* `evaluation.py`

---
