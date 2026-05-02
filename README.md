# LLM Hallucination Evaluation System — Medical QA

A research system that measures how much large language models hallucinate when answering questions about patient medical records. Four prompting strategies are compared across four LLMs, with all 16 combinations evaluated per question.

---

## What We Are Testing

**The core problem:** When a patient record contains no documented allergy but includes a medication like Penicillin, does the LLM incorrectly assert an allergy exists?

A hallucinating model says:
> "Yes, the patient has a documented penicillin allergy."

The correct response (when no allergy is recorded):
> "Insufficient information — no allergy has been documented."

### 4 Prompting Strategies (Baselines)

| Method | Description |
|--------|-------------|
| **Baseline 1** | Full patient record sent to LLM — no filtering, no rules |
| **Baseline 2** | Rule-based keyword matching selects only the relevant record section |
| **Baseline 3** | Semantic RAG — top-k most relevant facts retrieved via embedding similarity |
| **Proposed** | Same retrieval as Baseline 3 + strict prompt rules forbidding inference + abstention instruction |

### 4 LLMs Tested

| Model | Source |
|-------|--------|
| Claude | Anthropic API (`claude-3-haiku-20240307`) |
| GPT | OpenAI API (`gpt-3.5-turbo`) |
| Gemini | Google Generative AI (`gemini-1.5-flash`) |
| LLaMA (Qwen) | HuggingFace local (`Qwen/Qwen2.5-0.5B-Instruct`) |

Every question runs through all **16 combinations** (4 methods × 4 models).

### 6 Evaluation Metrics

| Metric | Direction | What it measures |
|--------|-----------|-----------------|
| Hallucination Rate | ↓ lower = better | Did the model assert something not in the evidence? |
| Abstention Rate | ↑ (in context) | Did the model correctly say "Insufficient information"? |
| Evidence Precision | ↑ higher = better | Fraction of the answer that came from the retrieved evidence |
| Evidence F1 | ↑ higher = better | Overall grounding quality (precision + recall harmonic mean) |
| SAR (Supported Answer Rate) | ↑ higher = better | Is the answer either grounded or a correct abstention? |
| Over-confidence Rate | ↓ lower = better | Confident wrong claim when evidence was missing |

---

## Project Structure

```
project_v2/
├── data/
│   └── patient_dataset_cleaned.json    ← 227 patient records
├── system-model/
│   ├── retrieval.py      ← atomic facts, embeddings, semantic search, rule-based retrieval
│   ├── baselines.py      ← prompt builders for B1, B2, B3
│   ├── proposed.py       ← prompt builder for Proposed method
│   ├── models.py         ← ONLY file that calls LLMs
│   ├── evaluation.py     ← all 6 metrics (pure string logic, no LLM calls)
│   ├── utils.py          ← session management, I/O, graph generation
│   ├── main.py           ← orchestration: runs one question through all 16 combinations
│   └── app.py            ← Streamlit UI
├── outputs/              ← auto-created on first run
│   ├── results.json
│   ├── sessions_index.json
│   ├── results.csv
│   ├── aggregate_metrics.csv
│   └── graphs/
│       ├── bar_charts/
│       ├── model_comparison/
│       ├── heatmaps/
│       ├── radar_charts/
│       └── confusion_matrices/
├── .env                  ← API keys (never commit this)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10 or higher
- pip
- Internet connection (for API calls to Claude, GPT, Gemini)
- At least 4 GB RAM (for local Qwen model)
- API keys for Anthropic, OpenAI, and Google Generative AI

---

## Step-by-Step Setup and Run Guide

### Step 1 — Clone / navigate to the project

```bash
cd path/to/project_v2
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `transformers` may take a few minutes to install. If you have a GPU, PyTorch will use it automatically for the local Qwen model.

### Step 4 — Add your API keys

Open the `.env` file in the project root and fill in your keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
HF_TOKEN=hf_...
```

> Get keys from:
> - Claude: https://console.anthropic.com
> - GPT: https://platform.openai.com/api-keys
> - Gemini: https://aistudio.google.com/app/apikey
> - HuggingFace: https://huggingface.co/settings/tokens

### Step 5 — Run the Streamlit app

```bash
cd system-model
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## How to Use the App

### Starting a Session

1. In **Section 1 — Sessions**, optionally type a session name (e.g. "Run 1 — 50 auto")
2. Click **▶ Start New Session**

### Running a Batch

3. In **Section 2 — Run a Batch**:
   - Choose **Auto** (system picks patients and questions) or **Manual** (you type the question)
   - Set **Top-k** (number of facts retrieved for Baseline 3 and Proposed) — default is 5
   - Set the **number of questions** — minimum 10, recommended 30+ for reliable metrics
   - Click **▶ Run Auto Batch** or pick a patient and submit a manual question

4. Results appear in **Section 3** in real-time after each question:
   - Red border = hallucinated
   - Orange badge = abstained
   - Click "View Prompt" or "View Evidence" to inspect details

5. You can run multiple batches (auto and manual) within the same session.

### Ending a Session and Generating a Report

6. Click **⏹ End Session** when done
7. In **Section 4 — Generate Report**, check the sessions you want to include:
   - Select one session for isolated results
   - Select multiple sessions for cumulative results
8. Click **📊 Generate Report**

### Viewing and Exporting Results

9. **Section 5 — Graphs** shows all generated visualizations in tabs:
   - Bar Charts, Model Comparison, Heatmaps, Radar Charts, Confusion Matrices
10. **Section 6 — Export** lets you download:
    - `results.csv` — every individual LLM response with metrics
    - `aggregate_metrics.csv` — mean ± std per model × baseline, ready for LaTeX
    - All graphs as a ZIP file

---

## Output Files Explained

### `results.json` / `results.csv`
One row per question × model × baseline combination. Contains the prompt sent, the model's response, retrieved evidence chunks, and all 6 metric values.

### `aggregate_metrics.csv`
Summary table with mean and standard deviation of each metric for every (model, baseline) pair. Includes confusion matrix values (TP, FP, FN, TN) and abstention precision/recall/F1. Suitable for direct use in research papers.

### Graphs

| Graph type | What it shows |
|-----------|---------------|
| Bar charts | Each metric by baseline, coloured by model |
| Model comparison | Each metric by model, coloured by baseline |
| Heatmaps | Model vs baseline grid for each metric |
| Radar charts | Full 6-metric profile for each model across all baselines |
| Confusion matrices | Per-model-baseline abstention quality (TP/FP/FN/TN) |

---

## Common Issues

**`ModuleNotFoundError` when running app.py**
Make sure you are inside the `system-model/` directory when running streamlit:
```bash
cd system-model
streamlit run app.py
```

**API key errors**
Verify your `.env` file has the correct keys. Do not add quotes around the key values.

**Slow Qwen model on first run**
The model downloads (~1 GB) on first use. Subsequent runs load from cache and are faster.

**Out of memory with Qwen**
The `Qwen/Qwen2.5-0.5B-Instruct` model requires ~2 GB RAM. Close other applications if needed.

**Streamlit session resets**
Results are saved to `outputs/results.json` after every question. Even if the browser refreshes, your data is safe. Re-select your session in the Report Generator to regenerate graphs.

---

## Research Notes

- Hallucination detection uses pure string heuristics — no LLM is used in evaluation (prevents circular bias)
- `allergies = []` is treated as "not recorded", never as "confirmed no allergies"
- Baseline 3 and Proposed use identical retrieval — the only variable is the prompt, isolating the grounding mechanism as the cause of any improvement
- For statistically reliable metrics, run at least 30 questions per session
- The `aggregate_metrics.csv` file is formatted for direct import into LaTeX tables
