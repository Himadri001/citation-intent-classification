# Citation Intent Classification with Zero-Shot, Few-Shot, and RAG

I compare three prompting strategies — zero-shot, few-shot, and retrieval-augmented generation (RAG) — for classifying why a scientific paper cites prior work. I use two public benchmarks: SciCite (3 classes) and ACL-ARC (6 classes), combined into a single corpus of 12,910 citation sentences. All three conditions run on the same language model so that any performance difference comes from the prompting strategy, not model capacity.

---

## Project Structure

```
citation_rag/
├── corpus/                     # combined_corpus.json saved here after data_prep.py runs
├── embeddings/                 # FAISS index and numpy arrays (built on Colab)
├── eval/
│   └── qa_pairs.json           # 10 hand-crafted research questions
├── results/
│   ├── zero_shot/              # zero-shot predictions
│   ├── few_shot/               # few-shot predictions
│   └── rag/                    # RAG QA answers and classification results
├── src/
│   ├── data_prep.py            # dataset download, merge, corpus build
│   ├── llm_client.py           # OpenRouter API wrapper
│   ├── prompts.py              # prompt builders for all three conditions
│   ├── zero_shot.py            # zero-shot classification runner
│   ├── few_shot.py             # few-shot classification runner
│   ├── retriever.py            # FAISS index build and retrieval
│   ├── rag_qa.py               # RAG QA pipeline
│   └── evaluate.py             # evaluation metrics and plots
├── notebooks/
│   └── rag_colab.ipynb         # full pipeline notebook (run on Google Colab)
├── report/
│   ├── report.tex              # LaTeX final report
│   └── references.bib          # BibTeX references
├── .env                        # OpenRouter API key — never commit this
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- A free or paid [OpenRouter](https://openrouter.ai) API key
- A Google account for running the Colab notebook

> Always use `python3` and `pip3`. The system `python` on macOS may resolve to Python 2.7.

---

## Step 1 — Clone the Repository

```bash
git clone git@github.com:himadrichowdhury/citation-rag.git
cd citation-rag
```

---

## Step 2 — Install Dependencies

```bash
pip3 install -r requirements.txt
```

Key packages and what I use them for:

| Package | Purpose |
|---|---|
| `openai` | OpenRouter API client (OpenAI-compatible) |
| `python-dotenv` | Load API key from `.env` |
| `datasets==2.18.0` | Download SciCite and ACL-ARC from Hugging Face |
| `sentence-transformers` | SPECTER encoder for building FAISS embeddings |
| `faiss-cpu` | Fast vector similarity search |
| `scikit-learn` | Macro-F1, classification reports |
| `pandas`, `matplotlib`, `seaborn` | Results tables and plots |
| `tqdm` | Progress bars |

> `datasets` must be pinned to `2.18.0`. Newer versions dropped support for the custom loading scripts that SciCite uses.

---

## Step 3 — Add Your OpenRouter API Key

```bash
echo "OPENROUTER_API_KEY=your-key-here" > .env
```

Get a key at [openrouter.ai](https://openrouter.ai) → Dashboard → Keys → Create key.

I use `google/gemini-2.0-flash-001` by default. To change the model, edit one line in `src/llm_client.py`:

```python
DEFAULT_MODEL = "google/gemini-2.0-flash-001"
```

---

## Step 4 — Download the Datasets

I use two datasets hosted on Hugging Face:

- **SciCite** (`allenai/scicite`) — ~10,969 citation sentences from biomedical and CS papers, labeled with 3 intent classes: `background`, `method`, `result`
- **ACL-ARC** (`zapsdcn/citation_intent`) — ~1,941 citation sentences from NLP papers, labeled with 6 classes: `background`, `method`, `result`, `extends`, `motivation`, `future`

Run the data preparation script to download both datasets:

```bash
python3 src/data_prep.py
```

This normalizes labels across both datasets, merges all splits, and saves the corpus to `corpus/combined_corpus.json`.

Expected output:

```
Saved 12910 entries to corpus/combined_corpus.json
Total corpus size: 12910
Label distribution: {'background': 7371, 'method': 3489, 'result': 1821,
                     'motivation': 88, 'extends': 72, 'future': 69}
```

---

## Step 5 — Build the FAISS Index

I build the FAISS index on Google Colab because loading the SPECTER encoder (440 MB) alongside FAISS exhausts memory on a local machine.

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `notebooks/rag_colab.ipynb`
2. Run all cells from the top — the notebook will prompt you to upload `corpus/combined_corpus.json`
3. At the end of Step 3 in the notebook, the index files download automatically

Place the four downloaded files into `citation_rag/embeddings/`:

```
embeddings/
├── corpus.index          # FAISS vector index (~19 MB)
├── corpus_texts.npy      # raw text of each passage
├── corpus_labels.npy     # intent label of each passage
└── corpus_ids.npy        # unique ID of each passage
```

---

## Step 6 — Run Zero-Shot Classification

This classifies 300 stratified SciCite test sentences and all 139 ACL-ARC test sentences using zero-shot prompting. Predictions are cached after every call so the run can be safely interrupted and resumed.

```bash
python3 src/zero_shot.py
```

Results saved to:
```
results/zero_shot/predictions_scicite.json
results/zero_shot/predictions_acl_arc.json
```

---

## Step 7 — Run Few-Shot Classification

Same as zero-shot but with fixed demonstration examples prepended to each prompt — 3 examples for SciCite and 6 for ACL-ARC (one per class).

```bash
python3 src/few_shot.py
```

Results saved to:
```
results/few_shot/predictions_scicite.json
results/few_shot/predictions_acl_arc.json
```

---

## Step 8 — Run RAG Classification and QA on Colab

The RAG pipeline runs in `notebooks/rag_colab.ipynb`. It covers two tracks:

- **Classification**: classifies the same test sentences using dynamically retrieved training examples from the FAISS index
- **QA**: answers 10 hand-crafted research questions by retrieving relevant passages from the full corpus and generating grounded answers

Results download automatically at the end of each track.

---

## How to Add Your Own Questions

To run the RAG QA system on your own research questions, add entries to `eval/qa_pairs.json`:

```json
{
  "id": "q011",
  "question": "Find citations where attention mechanisms are used as a method.",
  "expected_labels": ["method"],
  "reference_answer": "Method citations adopting attention include sentences like 'We apply the attention mechanism of [X]...'"
}
```

| Field | Description |
|---|---|
| `id` | Unique identifier — continue the numbering (q011, q012, ...) |
| `question` | Research question in plain English |
| `expected_labels` | Intent labels the retriever should surface |
| `reference_answer` | What a correct answer looks like, for manual review |

After editing, upload the updated `qa_pairs.json` to Colab and re-run in the notebook. For each question, the system retrieves the five most similar passages from the corpus, generates a grounded answer using only those passages, and scores it for faithfulness.

---

## How to Compute Classification Results

Compute Macro-F1:

```bash
python3 - <<'EOF'
import json
from sklearn.metrics import f1_score

for source, labels in [
    ("scicite", ["background", "method", "result"]),
    ("acl_arc", ["background", "method", "result", "extends", "motivation", "future"]),
]:
    for condition in ["zero_shot", "few_shot"]:
        with open(f"results/{condition}/predictions_{source}.json") as f:
            data = list(json.load(f).values())
        golds = [r["label"] for r in data]
        preds = [r["pred"]  for r in data]
        f1 = f1_score(golds, preds, average="macro", labels=labels, zero_division=0)
        print(f"{condition} | {source} | Macro-F1 = {f1:.3f}")
EOF
```

RAG results print automatically inside the Colab notebook. 
