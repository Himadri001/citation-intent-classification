from data_prep import SCICITE_TAXONOMY, ACL_ARC_TAXONOMY

SCICITE_DEMOS = [
    {"text": "We follow the preprocessing steps of [CITATION].",       "label": "method"},
    {"text": "[CITATION] first introduced the task of ...",             "label": "background"},
    {"text": "Our model outperforms [CITATION] by 3 F1 points.",        "label": "result"},
]

ACL_ARC_DEMOS = [
    {"text": "We use the parser of [CITATION] to extract dependencies.", "label": "method"},
    {"text": "[CITATION] laid the groundwork for this line of research.", "label": "background"},
    {"text": "Our results are significantly better than [CITATION].",     "label": "result"},
    {"text": "We extend the framework proposed by [CITATION].",           "label": "extends"},
    {"text": "As noted by [CITATION], this problem remains open.",        "label": "motivation"},
    {"text": "We leave exploration of [CITATION]'s approach to future work.", "label": "future"},
]


def zero_shot_prompt(sentence, taxonomy):
    label_list = "\n".join(f"- {k}: {v}" for k, v in taxonomy.items())
    return [
        {"role": "system", "content": (
            "You are a scientific document analyst. "
            "Classify the citation intent of the sentence below. "
            f"Respond with exactly one label from this list:\n{label_list}"
        )},
        {"role": "user", "content": f'Citation: "{sentence}"\nLabel:'},
    ]


def few_shot_prompt(sentence, taxonomy, demos):
    label_list = "\n".join(f"- {k}: {v}" for k, v in taxonomy.items())
    demo_block = "\n\n".join(
        f'Citation: "{d["text"]}"\nLabel: {d["label"]}' for d in demos
    )
    return [
        {"role": "system", "content": (
            "You are a scientific document analyst. "
            "Classify the citation intent of the sentence below. "
            f"Respond with exactly one label from this list:\n{label_list}"
        )},
        {"role": "user", "content": (
            f"{demo_block}\n\n"
            f'Citation: "{sentence}"\nLabel:'
        )},
    ]


def rag_qa_prompt(question, retrieved_passages):
    passage_block = ""
    for i, p in enumerate(retrieved_passages, 1):
        passage_block += f'[{i}] ({p["label"]}) "{p["text"]}"\n\n'

    return [
        {"role": "system", "content": (
            "You are a research assistant with access to a corpus of scientific citation sentences. "
            "Answer the user's question using ONLY the retrieved passages below. "
            "Reference passages by their number [1], [2], etc. "
            "If the passages do not contain enough information to answer, say so explicitly. "
            "Do not use knowledge outside the provided passages."
        )},
        {"role": "user", "content": (
            f"Question: {question}\n\n"
            f"Retrieved passages:\n{passage_block}"
            "Answer:"
        )},
    ]
