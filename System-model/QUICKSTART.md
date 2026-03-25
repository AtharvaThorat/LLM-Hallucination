# 🚀 Quick Start Guide

Get the hallucination-resistant medical QA system running in 5 minutes.

---

## 1️⃣ Installation

```bash
# Navigate to system-model directory
cd System-model

# Install dependencies
pip install -r requirements.txt
```

---

## 2️⃣ Environment Setup

Create `.env` file in the project root:

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_actual_api_key_here

# Optional: Hugging Face Token (for BioMedLM & LLaMA)
HUGGINGFACE_TOKEN=your_hf_token_here
# Optional alias also supported by code
# HF_TOKEN=your_hf_token_here
```

Authenticate before running Hugging Face models:
```bash
hf auth login
```

**Getting API Keys:**
- **Google Gemini**: https://ai.google.dev/tutorials/setup
- **Hugging Face**: https://huggingface.co/settings/tokens

Model cards used by this project:
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/stanford-crfm/BioMedLM

---

## 3️⃣ Run Experiments

```bash
python main.py
```

**What happens:**
1. Loads medical QA dataset
2. Generates prompts for all baselines
3. Queries your specified model(s)
4. Evaluates hallucination metrics
5. Saves results as JSON/CSV
6. Generates performance graphs

**Expected runtime:**
- 2 samples + 1 model: ~5-10 minutes
- Full dataset + 1 model: 2-4 hours
- Full dataset + 3 models: 6-12 hours

---

## 4️⃣ View Results

After running, check:

```
outputs/
├── results.json          ← Detailed metrics
├── results.csv           ← Spreadsheet format
├── summary.txt           ← Statistics
└── graphs/               ← Performance charts
    ├── hallucination.png
    ├── f1_score.png
    ├── sar.png
    └── ...
```

---

## ⚙️ Configuration

Edit `main.py` to customize:

```python
experiment = ExperimentRunner(
    dataset_path="data/medical_qa_dataset.json",
    output_dir="outputs",
    
    # Which models to test
    models=["gemini"],  # or ["gemini", "biomedlm", "llama"]
    
    # Which baselines to test
    baselines=["baseline1", "baseline2", "baseline3", "proposed"],
    
    # How many patients (None = all)
    sample_size=2  # Use 2 for quick test, None for full
)
```

---

## 📊 Understanding Output

### results.json example:
```json
[
  {
    "model": "gemini",
    "baseline": "proposed",
    "patient_id": "PT-001",
    "question": "Does patient have allergy to penicillin?",
    "predicted_answer": "Yes, documented allergy",
    "ground_truth": "Yes, penicillin allergy",
    "metrics": {
      "hallucination": 0,
      "exact_match": 0,
      "f1_score": 0.89,
      "evidence_precision": 1.0,
      "evidence_recall": 1.0,
      "evidence_f1": 1.0,
      "sar": 1,
      "abstained": 0
    }
  }
]
```

### Key Metrics:
- **hallucination** [0/1]: Did model claim facts not in evidence?
- **exact_match** [0/1]: Perfect match with ground truth?
- **f1_score** [0-1]: Token-level overlap
- **evidence_f1** [0-1]: Grounding quality
- **sar** [0/1]: Is answer supported by evidence?
- **abstained** [0/1]: Did model say "insufficient information"?

---

## 🔍 Expected Results

After running the full experiment:

| Method | Hallucination ↓ | Exact Match ↑ | Evidence F1 ↑ | SAR ↑ |
|--------|--|--|--|--|
| Baseline 1 | High (25-40%) | ~60% | ~50% | ~55% |
| Baseline 2 | Medium (15-25%) | ~70% | ~65% | ~70% |
| Baseline 3 | Low (10-15%) | ~75% | ~75% | ~80% |
| **Proposed** | **Very Low (5-10%)** | ~70% | **~85%** | **~90%** |

**Key Finding:** Proposed method reduces hallucination while maintaining accuracy!

---

## 🐛 Common Issues

### Issue: "ModuleNotFoundError"
```
ModuleNotFoundError: No module named 'google'
```
**Fix:** `pip install google-generativeai`

### Issue: "API Key not found"
```
ValueError: GOOGLE_API_KEY not found in environment
```
**Fix:** Double-check `.env` file has correct API key

### Issue: "CUDA out of memory"
```
RuntimeError: CUDA out of memory
```
**Fix:** Use Gemini (API-based) or reduce model size

### Issue: Very slow execution
- Use `sample_size=2` for testing
- Use only Gemini model for speed
- Consider reducing `top_k` in retrieval.py

---

## 📈 Visualizing Results

After running, open graphs:

```bash
# On Windows
start outputs/graphs/

# On Mac
open outputs/graphs/

# On Linux
xdg-open outputs/graphs/
```

### Interpreting Graphs:

**Hallucination Rate:**
- Lower is better
- Proposed should be lowest

**F1 Score:**
- Higher is better
- Shows answer quality

**SAR (Supporting Answer Rate):**
- Higher is better
- Shows grounding quality

**Abstention Rate:**
- Shows how often model says "insufficient information"
- Higher = more conservative
- Trade-off with accuracy

---

## 🎯 Running Full Experiment

For research-grade results:

```python
experiment = ExperimentRunner(
    dataset_path="data/medical_qa_dataset.json",
    output_dir="outputs",
    models=["gemini", "biomedlm", "llama"],  # All models
    baselines=["baseline1", "baseline2", "baseline3", "proposed"],
    sample_size=None  # All patients
)
```

**Estimated runtime:** 8-12 hours

---

## 📝 Next Steps

1. **Review Results** → Check outputs/summary.txt
2. **Analyze Metrics** → Open results.csv in Excel
3. **Examine Edge Cases** → Look for low-score patients
4. **Validate Findings** → Run again with different models
5. **Document Results** → Use for thesis/paper

---

## 💡 Tips for Best Results

1. **Start small:** Use `sample_size=2` for testing
2. **Fast iteration:** Use only Gemini initially
3. **Full evaluation:** Run all 3 models after validation
4. **Error handling:** Check console output for any issues
5. **Reproducibility:** Keep same dataset & .env unchanged

---

## 🔗 Important Files

- **Main script**: `main.py`
- **Configuration**: `.env` file
- **Dataset**: `../data/medical_qa_dataset.json`
- **Output**: `../outputs/`
- **Documentation**: `README.md` (this directory)

---

## ✅ Success Checklist

- [ ] Dependencies installed (`pip list` shows all packages)
- [ ] `.env` file created with API keys
- [ ] Dataset exists at `../data/medical_qa_dataset.json`
- [ ] Can import all modules: `python -c "import models, baselines, evaluation"`
- [ ] Run test: `python main.py` with `sample_size=1`
- [ ] Results saved in `outputs/` directory
- [ ] Graphs generated in `outputs/graphs/`

---

## 🎓 Learning Resources

- **Copilot Instructions**: `../../.github/copilot-instructions.md`
- **Baselines Definition**: `../baselines.md`
- **Output Specs**: `../output.md`
- **Full Docs**: `README.md`

---

**You're all set! Start by running:**
```bash
python main.py
```

Happy experimenting! 🚀
