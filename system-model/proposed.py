from retrieval import generate_atomic_facts, embed_facts, semantic_search


def proposed_method(patient: dict, question: str, top_k: int = 5) -> tuple[str, list[str]]:
    facts = generate_atomic_facts(patient)
    embeddings = embed_facts(facts)
    retrieved = semantic_search(question, facts, embeddings, top_k=top_k)
    retrieved_str = "\n".join(f"- {fact}" for fact in retrieved)

    prompt = f"""Evidence:
{retrieved_str}

Rules:
- Answer ONLY if the evidence above explicitly and directly supports the answer.
- DO NOT infer, assume, or use any prior knowledge.
- DO NOT assume that an empty list or absence of a record means the absence of a condition.
- If the evidence is insufficient to answer, respond with EXACTLY:
  "Insufficient information"

Question: {question}
Answer:"""

    return prompt, retrieved
