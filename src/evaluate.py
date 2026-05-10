import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)

from llm_client import call_llm, DEFAULT_MODEL


# ── Classification evaluation (zero-shot / few-shot) ──────────────────────────

def evaluate_classification(results, labels, condition_name):
    golds = [r["label"] for r in results]
    preds = [r["pred"]  for r in results]

    macro_f1  = f1_score(golds, preds, average="macro",  labels=labels, zero_division=0)
    accuracy  = accuracy_score(golds, preds)
    per_class = f1_score(golds, preds, average=None, labels=labels, zero_division=0)
    cm        = confusion_matrix(golds, preds, labels=labels)
    report    = classification_report(golds, preds, labels=labels, zero_division=0)

    print(f"\n=== {condition_name} ===")
    print(f"Macro F1: {macro_f1:.4f}  |  Accuracy: {accuracy:.4f}")
    print(report)

    return {
        "condition":    condition_name,
        "macro_f1":     macro_f1,
        "accuracy":     accuracy,
        "per_class_f1": dict(zip(labels, per_class.tolist())),
        "confusion_matrix": cm.tolist(),
        "report":       report,
    }


def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=labels, yticklabels=labels,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


# ── Retrieval evaluation ───────────────────────────────────────────────────────

def evaluate_retrieval(rag_results):
    label_precisions, known_recalls = [], []

    for r in rag_results:
        retrieved_labels = [p["label"] for p in r["retrieved"]]
        expected         = set(r.get("expected_labels", []))
        known_ids        = set(r.get("known_relevant_ids", []))
        retrieved_ids    = {p["id"] for p in r["retrieved"]}
        k                = len(retrieved_labels)

        if expected:
            lp = sum(1 for l in retrieved_labels if l in expected) / k
            label_precisions.append(lp)

        if known_ids:
            kr = len(retrieved_ids & known_ids) / len(known_ids)
            known_recalls.append(kr)

    results = {}
    if label_precisions:
        results["label_precision@k"] = float(np.mean(label_precisions))
        print(f"Label Precision@k: {results['label_precision@k']:.4f}")
    if known_recalls:
        results["known_recall@k"] = float(np.mean(known_recalls))
        print(f"Known-Relevant Recall@k: {results['known_recall@k']:.4f}")

    return results


# ── Faithfulness evaluation (LLM-as-judge) ────────────────────────────────────

def faithfulness_prompt(question, retrieved_passages, answer):
    passage_block = "\n".join(
        f"[{i+1}] {p['text']}" for i, p in enumerate(retrieved_passages)
    )
    return [
        {"role": "system", "content": (
            "You are an evaluator. Given retrieved passages and a generated answer, "
            "rate whether every claim in the answer is supported by the passages. "
            "Reply with a score from 1 (completely unsupported) to 5 (fully grounded) "
            "and one sentence of explanation. Format: 'Score: X. Reason: ...' "
        )},
        {"role": "user", "content": (
            f"Passages:\n{passage_block}\n\n"
            f"Answer:\n{answer}\n\n"
            "Faithfulness score:"
        )},
    ]


def evaluate_faithfulness(rag_results, model=DEFAULT_MODEL):
    scores = []
    for r in rag_results:
        prompt = faithfulness_prompt(r["question"], r["retrieved"], r["answer"])
        raw    = call_llm(prompt, model=model, max_tokens=64)
        score  = _parse_score(raw)
        scores.append({"id": r["id"], "raw": raw, "score": score})

    avg = float(np.mean([s["score"] for s in scores if s["score"] is not None]))
    print(f"Avg Faithfulness: {avg:.2f}/5")
    return scores, avg


def _parse_score(raw):
    import re
    m = re.search(r"(\d)", raw)
    if m:
        s = int(m.group(1))
        return s if 1 <= s <= 5 else None
    return None
