import json
import re

from prompt_list import extract_relation_prompt


def parse_json_object(response):
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string")
    text = response.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON response") from error
    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")
    return data


def parse_selected_ids(response, candidate_count, limit):
    selected = parse_json_object(response).get("selected_ids")
    if not isinstance(selected, list):
        raise ValueError("JSON must contain a selected_ids list")
    if len(selected) > limit:
        raise ValueError("Too many selected candidates")
    if any(type(item) is not int or not 1 <= item <= candidate_count for item in selected):
        raise ValueError("Invalid candidate ID")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate candidate ID")
    return selected


def clean_relations(response, entity_id, relation_candidates):
    scores = parse_json_object(response).get("scores")
    if not isinstance(scores, list):
        raise ValueError("JSON must contain a scores list")
    relations, seen = [], set()
    for item in scores:
        if not isinstance(item, dict):
            raise ValueError("Invalid relation score item")
        candidate_id, score = item.get("id"), item.get("score")
        if type(candidate_id) is not int or not 1 <= candidate_id <= len(relation_candidates):
            raise ValueError("Invalid relation ID")
        if candidate_id in seen or type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("Invalid or duplicate relation score")
        seen.add(candidate_id)
        candidate = relation_candidates[candidate_id - 1]
        relations.append({"entity": entity_id, "relation": candidate["relation"],
                          "head": candidate["head"], "score": float(score)})
    return bool(relations), relations


def extract_answer(response):
    data = parse_json_object(response)
    if not isinstance(data.get("reason"), str) or "answer" not in data:
        raise ValueError("Answer JSON must contain reason and answer")
    answer = data["answer"]
    if answer is not None and not isinstance(answer, str):
        raise ValueError("Answer must be a string or null")
    return answer.strip() if answer is not None else ""


def construct_triple(source_name, relation, target_name, is_head):
    if is_head:
        return (source_name, relation, target_name)
    return (target_name, relation, source_name)


def format_reasoning_path(triples, fallback_sep=" | "):
    if not triples:
        return ""
    if (
        isinstance(triples, tuple)
        and len(triples) == 3
        and not isinstance(triples[0], (list, tuple))
    ):
        triples = [triples]

    triples = [tuple(triple) for triple in triples]
    return fallback_sep.join(f"{h} -> {r} -> {t}" for h, r, t in triples)


def build_relation_prune_prompt(question, entity_name, total_relations, retain_topk, path_history=None):
    relations = "\n".join(
        f"[{index}] [{'outgoing' if candidate['head'] else 'incoming'}] {candidate['relation']}"
        for index, candidate in enumerate(total_relations, start=1)
    )
    return extract_relation_prompt.substitute(
        relation_num=retain_topk, question=question,
        entity_name=entity_name, relations=relations,
        path_context=(
            "Path History: " + (format_reasoning_path(path_history) or "None") +
            "\nUse the path history when assessing which relations can continue the reasoning.\n"
            if path_history is not None else ""
        ),
    )
