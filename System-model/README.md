# Medical QA System - System Model

Complete research-grade implementation for hallucination-resistant medical question answering with patient-specific memory.

---

## 📋 Project Structure

```
System-model/
├── utils.py              # Utility functions (data loading, I/O)
├── models.py             # Unified LLM interface (Gemini, BioMedLM, LLaMA)
├── retrieval.py          # Retrieval systems (rule-based & semantic)
├── baselines.py          # Three baseline prompt builders
├── proposed.py           # Proposed method with abstention
├── evaluation.py         # Hallucination metrics & evaluation
├── main.py              # Experiment runner & visualization
└── requirements.txt      # Python dependencies
```

---

## 🎯 Core Components

### 1. **models.py** - Unified LLM Interface
Provides unified access to multiple models:
- **Gemini** (API-based via Google)
- **BioMedLM** (Hugging Face)
- **LLaMA 3.1** (Hugging Face)

**Usage:**
```python
from models import query_model

response = query_model("gemini", prompt)
response = query_model("biomedlm", prompt)
response = query_model("llama", prompt)
```

### 2. **retrieval.py** - Retrieval Systems

#### Rule-Based Retrieval (Baseline 2)
- Keyword-based matching
- Filters irrelevant data
- Fast, interpretable

#### Semantic Retrieval (Baseline 3 & Proposed)
- Embedding-based similarity
- Top-k chunk retrieval
- Better handling of paraphrases

**Usage:**
```python
from retrieval import retrieve_for_baseline2, retrieve_for_baseline3

# Rule-based
retrieved = retrieve_for_baseline2(patient, question)

# Semantic
retrieved = retrieve_for_baseline3(patient, question)
```

### 3. **baselines.py** - Three Baselines

| Baseline | Strategy | Grounding | Hallucination Risk |
|----------|----------|-----------|-------------------|
| **Baseline 1** | Full context | None | High |
| **Baseline 2** | Rule-based retrieval | Weak | Medium |
| **Baseline 3** | Semantic RAG | Encouraged | Low |

**Usage:**
```python
from baselines import baseline1, baseline2, baseline3

prompt1 = baseline1(patient, question)
prompt2 = baseline2(patient, question)
prompt3 = baseline3(patient, question)
```

### 4. **proposed.py** - Evidence-Constrained RAG with Abstention

Proposed method that:
- Uses semantic retrieval (like Baseline 3)
- Enforces strict evidence grounding
- Forces abstention for unsupported queries
- **Aligns with medical safety requirements**

**Key Features:**
```
- Answer ONLY if evidence explicitly supports
- DO NOT make assumptions
- If insufficient info → respond "Insufficient information"
- Prevents hallucination from missing data
```

**Usage:**
```python
from proposed import proposed, is_abstention

prompt = proposed(patient, question)
response = query_model("gemini", prompt)

if is_abstention(response):
    print("Model abstained due to insufficient evidence")
```

### 5. **evaluation.py** - Comprehensive Metrics

**Metrics Implemented:**
- **Hallucination Rate**: Did model claim facts not in evidence?
- **Exact Match**: Does answer perfectly match ground truth?
- **F1 Score**: Token-level overlap with ground truth
- **Evidence Precision**: What % of answer is supported?
- **Evidence Recall**: What % of evidence is captured?
- **Evidence F1**: Harmonic mean of precision & recall
- **SAR** (Supporting Answer Rate): Is answer grounded?
- **Abstention Rate**: How often does model abstain?

**Usage:**
```python
from evaluation import evaluate, aggregate_metrics

metrics = evaluate(
    predicted_answer="Yes, patient allergic to penicillin",
    ground_truth="Yes, documented allergy to penicillin",
    supporting_evidence=["Penicillin – anaphylaxis"]
)

print(metrics)
# {
#   "hallucination": 0,
#   "exact_match": 0,
#   "f1_score": 0.80,
#   "evidence_precision": 1.0,
#   "evidence_recall": 0.8,
#   "evidence_f1": 0.89,
#   "sar": 1,
#   "abstained": 0
# }
```

### 6. **main.py** - Experiment Runner

Orchestrates complete evaluation pipeline:
1. Load dataset
2. Iterate through patients & QA pairs
3. Generate prompts for all baselines
4. Query all models
5. Evaluate all metrics
6. Save structured results (JSON, CSV)
7. Generate visualizations

**Usage:**
```bash
python main.py
```

**Configuration (in main.py):**
```python
experiment = ExperimentRunner(
    dataset_path="data/medical_qa_dataset.json",
    output_dir="outputs",
    models=["gemini"],  # ["gemini", "biomedlm", "llama"]
    baselines=["baseline1", "baseline2", "baseline3", "proposed"],
    sample_size=None  # None = all patients
)
experiment.run()
```

### 7. **utils.py** - Utilities

Provides:
- Dataset loading from JSON
- Environment variable management
- Patient data extraction
- Evidence extraction
- Output directory creation
- Results saving (JSON/CSV)

---

## 🚀 Getting Started

### 1. Installation

```bash
cd System-model
pip install -r requirements.txt
```

### 2. Environment Setup

Create `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
# Optional alias also supported by code:
# HF_TOKEN=your_hf_token_here
```

Authenticate Hugging Face (recommended before running BioMedLM/LLaMA):
```bash
huggingface-cli login
# or: hf auth login
```

### 3. Run Experiments

```bash
python main.py
```

This will:
- Evaluate all patients in the dataset
- Run all baselines with specified models
- Save results to `outputs/results.json` and `outputs/results.csv`
- Generate performance graphs in `outputs/graphs/`

---

## 📊 Output Structure

```
outputs/
├── results.json                  # Detailed evaluation results
├── results.csv                   # Tabular format
├── summary.txt                   # Summary statistics
└── graphs/
    ├── hallucination.png         # Hallucination rate comparison
    ├── exact_match.png           # Exact match comparison
    ├── f1_score.png              # F1 score comparison
    ├── evidence_f1.png           # Evidence F1 comparison
    ├── sar.png                   # SAR comparison
    └── abstained.png             # Abstention rate comparison
```

---

## 📈 Expected Results

**Proposed Method should demonstrate:**
1. **Lowest hallucination rate** (due to abstention)
2. **Highest evidence precision** (strict grounding)
3. **Better SAR** (supported answer rate)
4. **Acceptable accuracy trade-off** (some questions abstained)

---

## 🔬 Design Principles

### Modular Architecture
- Each module has single responsibility
- Interchangeable components
- Easy to extend

### LLM is NOT the System
- LLM = text generator only
- System controls:
  - Data retrieval
  - Prompt construction
  - Evaluation logic
  - Visualization

### Fair Comparison
- Same dataset for all methods
- Same LLMs for all baselines
- Methods differ ONLY in prompt construction
- No fine-tuning per baseline

### Medical Safety Focus
- Proposed method enforces abstention
- Prevents unsupported claims
- Hallucination-resistant design

---

## 🔧 Customization

### Add New Model

1. Create class in `models.py`:
```python
class MyModel(LLMInterface):
    def generate(self, prompt: str) -> str:
        # Implementation
        pass
```

2. Register in `ModelFactory`:
```python
class ModelFactory:
    _models = {
        "mymodel": MyModel,
        ...
    }
```

### Add New Baseline

1. Create function in `baselines.py`:
```python
def my_baseline(patient: Dict[str, Any], question: str) -> str:
    # Custom retrieval logic
    retrieved = custom_retrieve(patient, question)
    prompt = f"Context: {retrieved}\nQuestion: {question}\nAnswer:"
    return prompt
```

2. Register in `BASELINES`:
```python
BASELINES = {
    "my_baseline": my_baseline,
    ...
}
```

### Add New Metric

1. Create class in `evaluation.py`:
```python
class MyMetric:
    @staticmethod
    def calculate(predicted: str, ground_truth: str) -> float:
        # Implementation
        pass
```

2. Use in `evaluate()` function

---

## 📝 Dataset Format

Expected JSON structure:
```json
{
  "patients": [
    {
      "patient_id": "PT-001",
      "first_name": "John",
      "last_name": "Doe",
      "date_of_birth": "1980-01-01",
      "past_medical_history": ["Hypertension"],
      "medications": ["Lisinopril 10mg"],
      "allergies": ["Penicillin"],
      "symptoms": ["Headache"],
      "qa_interactions": [
        {
          "question": "Does the patient have hypertension?",
          "answer": "Yes",
          "supporting_evidence": [
            {
              "source_field": "past_medical_history",
              "evidence_text": "Hypertension"
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 🐛 Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'google'
```
Solution: `pip install google-generativeai`

### API Key Issues
```
ValueError: GOOGLE_API_KEY not found in environment
```
Solution: Add to `.env` file and ensure it's loaded

### Hugging Face Access / Gated Model Issues
```
401 Client Error / You must be authenticated to access model
```
Solution:
1. Run `hf auth login`.
2. Ensure `.env` includes `HUGGINGFACE_TOKEN=...`.
3. Request/accept access on model cards if prompted:
  - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - https://huggingface.co/stanford-crfm/BioMedLM

### Memory Issues (with large models)
```python
# In main.py, reduce sample size:
sample_size=5  # Instead of None
```

### Slow Semantic Retrieval
- First run downloads embedding model (~100MB)
- Subsequent runs are faster
- Consider reducing `top_k` parameter

---

## 📚 References

- **Copilot Instructions**: `../.github/copilot-instructions.md`
- **Baseline Definitions**: `../baselines.md`
- **Output Specifications**: `../output.md`
- **Project Description**: `../README.md`

---

## 📄 License & Citation

This project is part of CSCI-566 research at USC.

---

## 👨‍💻 Development Notes

### Hugging Face Model IDs Used
- LLaMA: `meta-llama/Llama-3.1-8B-Instruct`
- BioMedLM: `stanford-crfm/BioMedLM`

### Architecture Constraints (STRICT)
- ✅ DO: Keep code modular
- ✅ DO: Use same LLM interface
- ✅ DO: Evaluate all metrics
- ✅ DO: Store all results

- ❌ DON'T: Hardcode API keys
- ❌ DON'T: Mix patient data across evaluation
- ❌ DON'T: Use LLM for evaluation
- ❌ DON'T: Create monolithic code

---

## 📞 Support

For issues, questions, or improvements, refer to the main project README and documentation.
