import os
import re
import json
from pathlib import Path
import logging
from datetime import datetime



def prepare_dataset(dataset_name):
    datasets = {"cwq": ("cwq.json", "question"), "webqsp": ("webqsp.json", "RawQuestion"), "grailqa": ("graliqa.json", "question")}
    filename, question_key = datasets[dataset_name]
    path = Path(__file__).resolve().parents[3] / "data" / "test" / filename
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


def format_relation_path(topic_entity, relation_path):
    text = str(topic_entity.get("entity_name") or topic_entity["entity_id"])
    for index, hop in enumerate(relation_path, start=1):
        arrow = " -> " if hop["head"] else " <- "
        text += arrow + hop["relation"] + arrow + f"?e{index}"
    return text


def node_path_key(node):
    return (
        node.topic_entity.get("entity_id", ""),
        tuple((hop["relation"], hop["head"]) for hop in node.relation_path),
    )


def _normalize_path_score(score):
    try:
        score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    return round(score, 1)


def parse_batch_evaluation_output(output, expected_indices):
    """Parse LLM path-evaluation scores from JSON or a loose text fallback."""
    expected = set(expected_indices)
    parsed = extract_json(output)
    if isinstance(parsed, dict):
        for key in ("results", "scores", "evaluations", "candidates"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    results = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index not in expected:
                continue
            results[index] = _normalize_path_score(item.get("score"))

    if results:
        return results

    pattern = re.compile(
        r"(?is)(?:candidate|path|index)\s*\[?#?\s*(\d+)\s*\]?\s*[:\n].*?"
        r"score\s*:\s*((?:0(?:\.\d+)?|1(?:\.0+)?))\b"
        r"(?=(?:candidate|path|index)\s*\[?#?\s*\d+\s*\]?\s*[:\n]|\Z)"
    )
    for match in pattern.finditer(output or ""):
        index = int(match.group(1))
        if index not in expected:
            continue
        results[index] = _normalize_path_score(match.group(2))
    return results


logger = logging.getLogger(__name__)


USAGE_KEYS = ("num_calls", "prompt_tokens", "completion_tokens", "total_tokens")


def setup_logging(args):
    os.makedirs(args.run_dir, exist_ok=True)
    log_path = os.path.join(args.run_dir, args.run_name + ".log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Run started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    safe_args = {key: value for key, value in vars(args).items() if key != "api_key"}
    logger.info("Effective args: %s", json.dumps(safe_args, ensure_ascii=False, sort_keys=True))
    return log_path


class UsageTracker:
    def __init__(self, model):
        self.model = model
        self.before = model.get_usage_snapshot()

    def delta(self):
        after = self.model.get_usage_snapshot()
        return {key: max(0, int(after.get(key, 0) or 0) - int(self.before.get(key, 0) or 0)) for key in USAGE_KEYS}
