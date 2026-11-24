import os
import re
import json



def prepare_dataset(dataset_name):
    datasets = {"cwq": ("cwq.json", "question"), "webqsp": ("webqsp.json", "RawQuestion"), "grailqa": ("graliqa.json", "question")}
    filename, question_key = datasets[dataset_name]
    path = os.path.join("/home/liyuyang/lyy/kgqa/data/test", filename)
    with open(path, encoding="utf-8") as stream:
        return json.load(stream), question_key


def extract_json(text):
    """Extract the first JSON object or array from an LLM response."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    starts = [index for index, character in enumerate(cleaned) if character in "[{"]
    for start in starts:
        close_character = "]" if cleaned[start] == "[" else "}"
        end = cleaned.rfind(close_character)
        if end <= start:
            continue
        try:
            return json.loads(cleaned[start:end + 1])
        except (TypeError, ValueError):
            continue
    return None


def disable_net_proxy():
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(var, None)


def topic_entity_names(top_ent):
    """Return non-empty topic-entity display names, falling back to IDs."""
    if isinstance(top_ent, dict):
        return [
            str(name or entity_id).strip()
            for entity_id, name in top_ent.items()
            if str(name or entity_id).strip()
        ]
    if isinstance(top_ent, list):
        return [str(item).strip() for item in top_ent if str(item).strip()]
    value = str(top_ent or "").strip()
    return [value] if value else []


def format_top_entities(top_ent):
    """Format top entities for logs and progress messages."""
    names = topic_entity_names(top_ent)
    return ", ".join(names) if names else "None"


def extract_candidate_indices(text, valid_indices):
    """Read one ``[index]`` selection per line."""
    if not isinstance(text, str):
        return []

    selected_indices = []
    seen = set()
    for line in text.splitlines():
        match = re.fullmatch(r"\[(\d+)\]", line.strip())
        if not match:
            continue
        index = int(match.group(1))
        if index in valid_indices and index not in seen:
            seen.add(index)
            selected_indices.append(index)
    return selected_indices


def ensure_triple_list(triples):
    if not triples:
        return []

    if (
        isinstance(triples, (list, tuple))
        and len(triples) == 3
        and not isinstance(triples[0], (list, tuple))
    ):
        return [tuple(triples)]

    return [tuple(triple) for triple in triples]


def _split_reverse_relation(relation):
    relation = str(relation or "").strip()
    if relation.startswith("reverse "):
        return True, relation[len("reverse "):].strip()
    if relation.endswith(" [reverse]"):
        return True, relation[:-len(" [reverse]")].strip()
    return False, relation


def freebase_rel_to_text(rel: str) -> str:
    return str(rel or "").replace(".", " ")


def format_relation_name(relation, style="readable"):
    _, relation = _split_reverse_relation(relation)
    if style == "raw":
        return relation
    return freebase_rel_to_text(relation)


def format_entity_name(entity):
    entity_text = str(entity or "")
    normalized = entity_text.strip()
    if re.fullmatch(r"\[?CVT(?::[^\]]+)?\]?", normalized, flags=re.IGNORECASE):
        return "[CVT]"
    return entity_text


def format_reasoning_path(triples, fallback_sep=" | ", relation_style="readable"):
    if not triples:
        return ""
    triples = ensure_triple_list(triples)

    pieces = []
    current_tail = None
    for idx, (head, relation, tail) in enumerate(triples):
        head = format_entity_name(head)
        tail = format_entity_name(tail)
        is_reverse, relation = _split_reverse_relation(relation)
        relation_text = format_relation_name(relation, relation_style)

        if idx == 0 or current_tail != head:
            if pieces:
                pieces.append(fallback_sep)
            pieces.append(head)

        if is_reverse:
            pieces.append(f" <- {relation_text} <- {tail}")
        else:
            pieces.append(f" -> {relation_text} -> {tail}")
        current_tail = tail

    return "".join(pieces)


def is_valid_answer(answer: str) -> bool:
    if not answer:
        return False

    answer = answer.strip()
    answer_lower = answer.lower()

    if answer_lower in {"error", "none", "unknown", "n/a"}:
        return False
    if answer_lower.startswith(("m.", "g.", "cvt:", "[cvt:")):
        return False
    if "[cvt:" in answer_lower:
        return False
    return True
