"""Structural stopping, final relation-path selection and delayed KG grounding."""
import logging
import re
from mcts_utils import (
    FREEBASE_NS, build_grounding_query, format_relation_path, node_path_key,
)
from sparql_utils import execute_sparql, get_entity_names
from utils import extract_json, format_reasoning_path, is_valid_answer
from prompt_list import (
    verify_answer_system_prompt, verify_answer_user_prompt,
    final_answer_from_paths_system_prompt, final_answer_from_paths_user_prompt,
    missing_topic_cot_system_prompt, missing_topic_cot_user_prompt,
)

logger = logging.getLogger(__name__)


def extract_answer(text):
    """Extract the answer payload emitted by final-answer prompts."""
    if not text:
        return ""

    match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    marker = "the final answer is"
    text_lower = text.lower()
    start_index = text_lower.find(marker)
    if start_index != -1:
        answer = text[start_index + len(marker):].strip()
        for separator in [".", "\n", "<"]:
            separator_index = answer.find(separator)
            if separator_index != -1:
                answer = answer[:separator_index]
        return answer.strip(" :\t\r\n")

    text = text.strip()
    if "\n" not in text and len(text) < 200:
        return text
    return ""


def make_path_record(node):
    return {
        "topic_entity": dict(node.topic_entity),
        "relation_path": [dict(hop) for hop in node.relation_path],
        "path_value": float(node.path_value),
    }


def _rank_unique_nodes(root, minimum_score=None):
    stack = [root]
    seen = set()
    selected = []
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        if node.is_virtual_root or not node.relation_path:
            continue
        if minimum_score is not None and node.path_value < minimum_score:
            continue
        key = node_path_key(node)
        if key not in seen:
            seen.add(key)
            selected.append(node)
    return sorted(selected, key=lambda n: (n.path_value, len(n.relation_path)), reverse=True)


def get_final_answer_nodes(root, good_threshold=0.7, path_topk=10):
    # 对落地路径数统一设上限，避免最终回答上下文无限扩大。
    qualified = _rank_unique_nodes(root, good_threshold)
    return (qualified or _rank_unique_nodes(root))[:path_topk]


def collect_candidate_records(root, selected_nodes=None):
    selected_keys = {node_path_key(node) for node in selected_nodes or []}
    records = []
    for node in _rank_unique_nodes(root):
        records.append({
            **make_path_record(node), "candidate_id": len(records),
            "uct_visits": node.uct_visits, "total_reward": node.total_reward,
            "in_final_evidence": node_path_key(node) in selected_keys,
        })
    return records


def parse_early_stop_verdict(output, path_count):
    parsed = extract_json(output)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("stop"), bool):
        return None
    if not parsed["stop"]:
        return {"stop": False, "paths": []}
    paths = parsed.get("paths")
    if not isinstance(paths, list) or not paths:
        return None
    selected = []
    for index in paths:
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= path_count:
            return None
        if index not in selected:
            selected.append(index)
    return {"stop": True, "paths": selected}


def try_early_stop(*, root, question, reasoner_model, good_threshold,
                   answer_threshold, path_topk=10, verification_cache=None):
    nodes = _rank_unique_nodes(root, good_threshold)[:path_topk]
    if not nodes or max(node.path_value for node in nodes) < answer_threshold:
        return {"status": "not_triggered", "nodes": []}
    signature = tuple((node_path_key(node), node.path_value) for node in nodes)
    cache = verification_cache if verification_cache is not None else {}
    if signature not in cache:
        paths = "\n".join(
            f"Path {i} (score: {node.path_value:.1f}): "
            + format_relation_path(node.topic_entity, node.relation_path)
            for i, node in enumerate(nodes, start=1)
        )
        raw = reasoner_model(
            system_prompt=verify_answer_system_prompt,
            user_prompt=verify_answer_user_prompt.substitute(question=question, paths=paths),
            max_tokens=256,
        )[0]
        verdict = parse_early_stop_verdict(raw, len(nodes))
        if verdict is None:
            return {"status": "invalid", "nodes": []}
        cache[signature] = verdict
    verdict = cache[signature]
    if verdict["stop"]:
        return {"status": "accepted", "nodes": [nodes[i - 1] for i in verdict["paths"]]}
    return {"status": "rejected", "nodes": []}


def _normalize_binding(binding):
    """Preserve literals and Freebase IDs; omit foreign URIs and blank nodes."""
    kind, value = binding.get("type"), binding.get("value", "")
    if kind == "uri" and value.startswith(FREEBASE_NS):
        return {"id": value[len(FREEBASE_NS):], "kind": "entity"}
    if kind in {"literal", "typed-literal"} and value != "":
        return {"id": value, "kind": "literal",
                "datatype": binding.get("datatype"), "lang": binding.get("xml:lang")}
    return None


def ground_relation_paths(paths, step_history=None, query_limit=1000, rows_per_path=30):
    """Only invoked after MCTS returns. Each row preserves a full KG binding."""
    grounded = []
    for record in paths:
        topic = record["topic_entity"]
        chain = record["relation_path"]
        query = build_grounding_query(topic["entity_id"], chain, query_limit)
        rows = execute_sparql(query)
        bindings = []
        seen = set()
        for row in rows:
            values = [_normalize_binding(row.get(f"e_{i}", {})) for i in range(1, len(chain) + 1)]
            if any(value is None for value in values):
                continue
            key = tuple((v["kind"], v["id"], v.get("datatype"), v.get("lang")) for v in values)
            if key not in seen:
                seen.add(key)
                bindings.append(values)
        # 优先保留不同末端的绑定；再补充同一末端的其他中间绑定。
        diverse, remaining, endpoints = [], [], set()
        for values in bindings:
            endpoint = values[-1]
            key = (endpoint["kind"], endpoint["id"], endpoint.get("datatype"), endpoint.get("lang"))
            if key in endpoints:
                remaining.append(values)
            else:
                endpoints.add(key)
                diverse.append(values)
        selected = (diverse + remaining)[:rows_per_path]
        ids = list(dict.fromkeys(v["id"] for row in selected for v in row if v["kind"] == "entity"))
        names = get_entity_names(ids)
        instances = []
        for values in selected:
            current = topic.get("entity_name") or topic["entity_id"]
            walk = []
            for hop, binding in zip(chain, values):
                label = names.get(binding["id"], binding["id"]) if binding["kind"] == "entity" else binding["id"]
                if not label or label == "Unknown":
                    label = binding["id"]
                relation = hop["relation"] if hop["head"] else "reverse " + hop["relation"]
                walk.append((current, relation, label))
                current = label
            instances.append({"bindings": values, "walk_path": walk, "endpoint": current})
        entry = {
            "relation_path": record, "query_result_count": len(rows),
            "valid_binding_count": len(bindings), "grounded_count": len(instances),
            "query_limit_reached": len(rows) >= query_limit,
            "instances": instances,
        }
        grounded.append(entry)
        if step_history is not None:
            step_history.append({"event": "entity_grounding", **entry})
    return grounded


def build_path_evidence_text(grounded):
    lines = []
    for group in grounded:
        if not group["instances"]:
            continue
        record = group["relation_path"]
        lines.append("Relation pattern: " + format_relation_path(record["topic_entity"], record["relation_path"]))
        lines.append(f"Showing {group['grounded_count']} of {group['query_result_count']} returned binding rows (query limit may apply).")
        for i, instance in enumerate(group["instances"], start=1):
            lines.append(f"  Binding {i}: " + format_reasoning_path(instance["walk_path"], relation_style="raw"))
    return "\n".join(lines)


def generate_direct_cot_answer(question, reasoner_model):
    """Answer directly with the shared answer-first CoT protocol."""
    user_prompt = missing_topic_cot_user_prompt.substitute(question=question)
    raw_answer = reasoner_model(
        system_prompt=missing_topic_cot_system_prompt,
        user_prompt=user_prompt,
        max_tokens=256,
    )[0]
    return extract_answer(raw_answer) or "none", raw_answer


def generate_final_answer(question, grounded, reasoner_model):
    evidence = build_path_evidence_text(grounded)
    if evidence:
        raw = reasoner_model(
            system_prompt=final_answer_from_paths_system_prompt,
            user_prompt=final_answer_from_paths_user_prompt.substitute(question=question, evidence_paths=evidence),
            max_tokens=512,
        )[0]
        return extract_answer(raw) or "none", "grounded_paths"
    answer, _ = generate_direct_cot_answer(question, reasoner_model)
    return answer, "direct_cot_no_grounding"
