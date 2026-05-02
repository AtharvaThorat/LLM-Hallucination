from retrieval import (
    generate_atomic_facts,
    embed_facts,
    semantic_search,
    rule_based_retrieval,
)


def _format_field(label: str, items: list, key: str) -> str:
    if not items:
        return f"- {label}: None documented"
    values = ", ".join(str(i.get(key, "")) for i in items if i.get(key))
    return f"- {label}: {values}"


def baseline1(patient: dict, question: str) -> tuple[str, list[str]]:
    conditions = _format_field("Conditions", patient.get("conditions", []), "condition")
    medications = _format_field("Medications", patient.get("medications", []), "medication")

    allergies_raw = patient.get("allergies", [])
    if not allergies_raw:
        allergies_str = "- Allergies: None documented"
    else:
        allergy_names = ", ".join(a.get("allergy", "") for a in allergies_raw)
        allergies_str = f"- Allergies: {allergy_names}"

    obs = patient.get("observations", [])
    if obs:
        obs_str = "- Observations:\n" + "\n".join(
            f"  * {o.get('test','')}: {o.get('value','')} {o.get('unit','')} ({o.get('time','')[:10]})"
            for o in obs[:10]
        )
    else:
        obs_str = "- Observations: None documented"

    encounters = _format_field("Encounters", patient.get("encounters", []), "type")
    procedures = _format_field("Procedures", patient.get("procedures", []), "procedure")
    careplans = _format_field("Care Plans", patient.get("careplans", []), "careplan")

    evidence = generate_atomic_facts(patient)

    prompt = f"""Patient Record:
{conditions}
{medications}
{allergies_str}
{obs_str}
{encounters}
{procedures}
{careplans}

Question: {question}
Answer:"""

    return prompt, evidence


def baseline2(patient: dict, question: str) -> tuple[str, list[str]]:
    retrieved = rule_based_retrieval(patient, question)
    retrieved_str = "\n".join(f"- {fact}" for fact in retrieved)

    prompt = f"""Relevant Patient Information:
{retrieved_str}

Question: {question}
Answer:"""

    return prompt, retrieved


def baseline3(patient: dict, question: str, top_k: int = 5) -> tuple[str, list[str]]:
    facts = generate_atomic_facts(patient)
    embeddings = embed_facts(facts)
    retrieved = semantic_search(question, facts, embeddings, top_k=top_k)
    retrieved_str = "\n".join(f"- {fact}" for fact in retrieved)

    prompt = f"""Context:
{retrieved_str}

Use the context above to answer the question.

Question: {question}
Answer:"""

    return prompt, retrieved
