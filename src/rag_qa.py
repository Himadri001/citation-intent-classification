import json
import os
from tqdm import tqdm

from llm_client import call_llm, DEFAULT_MODEL
from prompts import rag_qa_prompt
from retriever import retrieve

RESULTS_PATH = "results/rag/answers.json"
EVAL_PATH    = "eval/qa_pairs.json"


def answer_question(question, k=5, model=DEFAULT_MODEL):
    retrieved = retrieve(question, k=k)
    prompt    = rag_qa_prompt(question, retrieved)
    answer    = call_llm(prompt, model=model, max_tokens=256)
    return {
        "question":  question,
        "retrieved": retrieved,
        "answer":    answer,
    }


def run_rag_eval(k=5, model=DEFAULT_MODEL):
    with open(EVAL_PATH) as f:
        qa_pairs = json.load(f)

    done = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            done = json.load(f)
        print(f"Resuming from cache: {len(done)}/{len(qa_pairs)} done")

    for qa in tqdm(qa_pairs, desc="RAG QA"):
        qid = qa["id"]
        if qid in done:
            continue
        result = answer_question(qa["question"], k=k, model=model)
        result["id"]               = qid
        result["reference_answer"] = qa.get("reference_answer", "")
        result["expected_labels"]  = qa.get("expected_labels", [])
        result["known_relevant_ids"] = qa.get("known_relevant_ids", [])
        done[qid] = result
        with open(RESULTS_PATH, "w") as f:
            json.dump(done, f, indent=2)

    print(f"RAG QA complete: {len(done)} answers saved to {RESULTS_PATH}")
    return list(done.values())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # demo: answer one question interactively
    question = input("Enter a research question: ")
    result   = answer_question(question, k=5)
    print("\nRetrieved passages:")
    for i, p in enumerate(result["retrieved"], 1):
        print(f"  [{i}] ({p['label']}, score={p['score']:.3f}) {p['text'][:120]}")
    print(f"\nAnswer:\n{result['answer']}")
