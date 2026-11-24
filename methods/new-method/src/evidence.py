"""evidence functions for SPAR-MCTS."""

import re
from utils import ensure_triple_list, extract_json, format_reasoning_path
from mcts_utils import node_path_key
import logging
from prompt_list import (
    verify_answer_system_prompt,
    verify_answer_user_prompt,
)
from utils import (
    is_valid_answer,
)
from prompt_list import (
    final_answer_from_paths_system_prompt,
    final_answer_from_paths_user_prompt,
    missing_topic_cot_system_prompt,
    missing_topic_cot_user_prompt,
)

logger = logging.getLogger(__name__)


# --- evidence ---

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


def unpack_path_record(item):
    """Normalize an MCTS path record to paths plus a floating-point score."""
    walk_path, score, evidence_path = item
    return (
        ensure_triple_list(walk_path),
        float(score),
        ensure_triple_list(evidence_path),
    )


def build_path_evidence_text(paths):
    """Format MCTS records as path-level evidence for answer prompts."""
    if not paths:
        return ""

    lines = ["KG Evidence Paths:"]
    for index, item in enumerate(paths, start=1):
        walk_path, score, _ = unpack_path_record(item)
        if not walk_path:
            continue
        path_text = format_reasoning_path(
            walk_path,
            fallback_sep=", ",
            relation_style="readable",
        )
        lines.append(f"Path {index} (path_value: {score:.1f}): {path_text}")

    return "\n".join(lines) if len(lines) > 1 else ""


def make_path_record(node):
    return (
        node.walk_path,
        float(getattr(node, "path_value", 0.0) or 0.0),
        node.evidence_path,
    )


def format_path_record(item, idx=None):
    walk_path, score, _ = unpack_path_record(item)
    label = f"Path {idx}" if idx is not None else "Path"
    walk_path_text = format_reasoning_path(walk_path, fallback_sep=", ")
    return f"{label} (path_value: {score:.1f}): {walk_path_text}"


def parse_early_stop_verdict(output, path_count):
    """Parse the strict, compact verdict used by early-stop verification."""
    parsed = extract_json(output)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("stop"), bool):
        return None

    raw_answer = parsed.get("answer", "")
    verdict = {
        "stop": parsed["stop"],
        "answer": raw_answer.strip() if isinstance(raw_answer, str) else "",
        "paths": [],
    }
    if not verdict["stop"]:
        return verdict

    raw_paths = parsed.get("paths")
    if not isinstance(raw_paths, list):
        return None

    valid_paths = []
    seen = set()
    for raw_index in raw_paths:
        if isinstance(raw_index, bool):
            return None
        if isinstance(raw_index, int):
            path_index = raw_index
        elif isinstance(raw_index, str) and raw_index.strip().isdigit():
            path_index = int(raw_index.strip())
        else:
            return None
        if path_index < 1 or path_index > path_count:
            return None
        if path_index not in seen:
            seen.add(path_index)
            valid_paths.append(path_index)

    if not verdict["answer"] or not valid_paths:
        return None
    verdict["paths"] = valid_paths
    return verdict


# --- evidence ---

def collect_nodes(node, include_empty=False):
    if node is None:
        return []

    nodes = []
    stack = [node]
    while stack:
        current = stack.pop()
        if include_empty or current.walk_path:
            nodes.append(current)
        if current.children:
            stack.extend(current.children)
    return nodes


def real_tree_nodes(root, require_path=False):
    return [
        node
        for node in collect_nodes(root, include_empty=not require_path)
        if not getattr(node, "is_virtual_root", False)
        and (not require_path or getattr(node, "walk_path", None))
    ]


def collect_candidate_records(root, selected_nodes=None):
    """Serialize only the endpoint facts needed for E009 error analysis."""
    selected_keys = {
        node_path_key(node)
        for node in selected_nodes or []
        if node_path_key(node)
    }
    records = []
    seen = set()
    for node in real_tree_nodes(root, require_path=True):
        path_key = node_path_key(node)
        if not path_key or path_key in seen:
            continue
        seen.add(path_key)
        records.append({
            "candidate_id": len(records),
            "entity_id": str(getattr(node, "entity_id", "") or "").strip(),
            "entity_name": str(getattr(node, "entity_name", "") or "").strip(),
            "path_value": round(
                float(getattr(node, "path_value", 0.0) or 0.0),
                6,
            ),
            "in_final_evidence": path_key in selected_keys,
        })
    return records


def _rank_unique_nodes(node, minimum_score=None, nodes=None):
    nodes = list(nodes) if nodes is not None else collect_nodes(node)
    selected = []
    seen = set()
    for item in nodes:
        path_score = float(getattr(item, "path_value", 0.0) or 0.0)
        if minimum_score is not None and path_score < minimum_score:
            continue
        identity = node_path_key(item)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(item)
    selected.sort(
        key=lambda item: (
            float(getattr(item, "path_value", 0.0) or 0.0),
            len(item.walk_path),
        ),
        reverse=True,
    )
    return selected


def get_fallback_nodes(node, path_topk=10, nodes=None):
    """Select a bounded set of the globally best available paths."""
    return _rank_unique_nodes(
        node,
        nodes=nodes,
    )[:max(1, int(path_topk or 1))]


def get_final_answer_nodes(
    node,
    good_threshold=0.7,
    path_topk=10,
    nodes=None,
):
    """Select every unique good path for summarization and final answering.

    Final evidence is filtered only by its score for the original question,
    without a path_topk limit when qualified paths exist. If no path reaches
    the threshold, the globally best path_topk paths provide bounded fallback
    evidence instead of leaving the answer prompt empty.
    """
    qualified_nodes = _rank_unique_nodes(
        node,
        minimum_score=good_threshold,
        nodes=nodes,
    )
    if qualified_nodes:
        return qualified_nodes
    return get_fallback_nodes(
        node,
        path_topk=path_topk,
        nodes=nodes,
    )


def get_early_stop_nodes(node, good_threshold=0.7, path_topk=10, nodes=None):
    """Select the highest-scoring good paths for early-stop verification."""
    return _rank_unique_nodes(
        node,
        minimum_score=good_threshold,
        nodes=nodes,
    )[:max(1, int(path_topk or 1))]


# --- evidence ---

def _verification_signature(nodes):
    path_signature = tuple(
        (
            node_path_key(node),
            round(float(getattr(node, "path_value", 0.0) or 0.0), 3),
        )
        for node in nodes
    )
    return path_signature


def _result(status, paths=None, answer="", nodes=None):
    return {
        "status": status,
        "paths": list(paths or []),
        "answer": str(answer or "").strip(),
        "nodes": list(nodes or []),
    }


def _accepted_result(verdict, paths, nodes):
    selected = [index - 1 for index in verdict["paths"]]
    return _result(
        "accepted",
        [paths[index] for index in selected],
        verdict.get("answer", ""),
        [nodes[index] for index in selected],
    )


def try_early_stop(*, root, question, reasoner_model,
                   good_threshold, answer_threshold, path_topk=10,
                   verification_cache=None,
                   tree_nodes=None,
):
    selected_nodes = get_early_stop_nodes(
        root,
        good_threshold=good_threshold,
        path_topk=path_topk,
        nodes=tree_nodes,
    )

    # Static subgoals have no completion status; only strong paths trigger verification.
    if not selected_nodes:
        return _result("not_triggered")

    best_score = max(float(getattr(node, "path_value", 0.0) or 0.0) for node in selected_nodes)
    if best_score < answer_threshold:
        return _result("not_triggered")

    logger.debug(
        "  Early-stop verifier triggered: best_score=%.3f, paths=%s.",
        best_score,
        len(selected_nodes),
    )

    early_paths = [
        make_path_record(node)
        for node in selected_nodes
    ]
    early_paths = [p for p in early_paths if p and p[0]]
    if not early_paths:
        return _result("not_triggered")

    formatted_paths = [format_path_record(item, idx=idx) for idx, item in enumerate(early_paths, start=1)]
    paths_text = "\n".join(formatted_paths) if formatted_paths else "None"

    signature = _verification_signature(selected_nodes)
    verification_cache = verification_cache if verification_cache is not None else {}
    if signature in verification_cache:
        cached_verdict = verification_cache[signature]
        if cached_verdict.get("stop"):
            logger.debug("  Reuse accepted early-stop verification for unchanged evidence.")
            return _accepted_result(
                cached_verdict,
                early_paths,
                selected_nodes,
            )
        logger.debug("  Reuse rejected early-stop verification for unchanged evidence.")
        return _result("rejected")

    quick_user_prompt = verify_answer_user_prompt.substitute(question=question, paths=paths_text)
    raw_verdict = reasoner_model(system_prompt=verify_answer_system_prompt, user_prompt=quick_user_prompt, max_tokens=128)[0]
    verdict = parse_early_stop_verdict(raw_verdict, path_count=len(early_paths))

    if verdict and verdict["stop"]:
        if is_valid_answer(verdict["answer"]):
            verification_cache[signature] = verdict
            return _accepted_result(
                verdict,
                early_paths,
                selected_nodes,
            )
        logger.debug("Early stop returned an invalid answer, continuing search...")
        return _result("invalid")

    if verdict is not None:
        verification_cache[signature] = verdict
        logger.debug("Early stop rejected (insufficient evidence), continuing search...")
        return _result("rejected")

    logger.debug("Early stop returned invalid verifier output, continuing search...")
    return _result("invalid")


# --- evidence ---

def generate_direct_cot_answer(question, reasoner_model):
    """Answer directly with the shared answer-first CoT protocol."""
    user_prompt = missing_topic_cot_user_prompt.substitute(question=question)
    raw_answer = reasoner_model(
        system_prompt=missing_topic_cot_system_prompt,
        user_prompt=user_prompt,
        max_tokens=256,
    )[0]
    return extract_answer(raw_answer) or "none", raw_answer


def generate_final_answer(
    question,
    paths,
    reasoner_model,
):
    """Generate the final answer from selected KG paths, or use direct CoT if none exist."""
    evidence_paths_text = build_path_evidence_text(paths)

    if evidence_paths_text:
        logger.debug("Path-based final answer generation...")
        user_prompt = final_answer_from_paths_user_prompt.substitute(
            question=question,
            evidence_paths=evidence_paths_text,
        )
        raw_answer = reasoner_model(
            system_prompt=final_answer_from_paths_system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
        )[0]
        answer = extract_answer(raw_answer)
        return answer, "paths"

    answer, _ = generate_direct_cot_answer(question, reasoner_model)
    return answer, "direct_cot_no_paths"
