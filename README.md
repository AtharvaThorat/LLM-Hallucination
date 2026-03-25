# LLM-Hallucination

You must build a complete Python-based system for evaluating hallucination in medical question answering.

You MUST strictly follow the rules defined in:
`copilot-instructions.md`

This file defines:
- architecture
- coding rules
- evaluation logic
- constraints

Do NOT deviate from it.

---

# Objective

Build a modular system that:

1. Loads patient dataset from JSON
2. Implements multiple QA baselines
3. Integrates multiple LLMs
4. Runs experiments across:
   - models
   - baselines
5. Evaluates hallucination using defined metrics
6. Stores structured results

---

# Required Project Structure

Create the following files inside `system-model/`:
system-model/
├── main.py
├── models.py
├── baselines.py
├── proposed.py
├── retrieval.py
├── evaluation.py
├── utils.py


---

# Dataset

Use dataset from: data/medical_qa_dataset.json

Dataset structure:

- patients[]
  - patient_id
  - medical fields
  - qa_interactions[]
    - question
    - answer
    - supporting_evidence[]

---

# 🤖 Model Integration

Implement 3 models:

### 1. Gemini (API)
- Load API key from `.env`
- Use `google.generativeai`

### 2. BioMedLM (Hugging Face)
- Use transformers

### 3. LLaMA 3.1 (Hugging Face)
- Use transformers pipeline

---

## Unified Model Interface

Implement:
query_model(model_name, prompt)
This must route to:
- Gemini
- BioMedLM
- LLaMA

---
### Baselines

Implement in baselines.py:

Baseline 1: Full Context
- Use full patient data

Baseline 2: Rule-Based Retrieval
- Retrieve data using keywords

Baseline 3: RAG
- Use semantic retrieval from retrieval.py

---
### Proposed Method

In proposed.py:
- Same as RAG BUT:

    - Add constraint: "Answer ONLY if evidence exists"

    - If not: → return "Insufficient information"

---
### Retrieval System

In retrieval.py:

- Baseline 2
    - Rule-based (keywords)

- Baseline 3 / Proposed
    - Semantic search
    - Use embeddings
    - Store chunks per patient

---
### Constraints (STRICT)
- DO NOT hardcode API keys
- DO NOT mix patient data
- DO NOT skip evaluation
- DO NOT use LLM for evaluation
- DO NOT write monolithic code