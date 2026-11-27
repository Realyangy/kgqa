"""replan functions for SPAR-MCTS."""

import re
from utils import topic_entity_names
import json
import logging
from mcts_utils import node_path_key
from utils import format_reasoning_path
from collections import deque
import copy
from mcts_core import reopen_nodes
from mcts_core import MCTSNode
from subgoal import (
    can_expand_after_progress,
    incremental_subgoal_hits,
)
from subgoal import anchor_values
import hashlib
from prompt_list import (
    subgoal_plan_diagnosis_system_prompt,
    subgoal_plan_diagnosis_user_prompt,
    subgoal_plan_reconstruct_system_prompt,
    subgoal_plan_reconstruct_user_prompt,
    subgoal_plan_refine_system_prompt,
    subgoal_plan_refine_user_prompt,
)
from subgoal import (
    repair_missing_constraint_anchors,
    unused_topic_entities,
    validate_executable_plan,
)
from utils import extract_json

logger = logging.getLogger(__name__)


# --- replan ---

_STEP_REFERENCE = re.compile(r"step:(\d+)$")


def topic_reference_records(topic_entities):
    """Return stable, prompt-visible references for topic entities."""
    return [
        {"ref": f"topic:t{index}", "entity": name}
        for index, name in enumerate(
            topic_entity_names(topic_entities),
            start=1,
        )
    ]


def compile_refine_updates(semantic_updates, topic_entities):
    """Translate semantic refine fields without changing plan structure."""
    topic_by_ref = _topic_reference_map(topic_entities)
    compiled = []
    for update in semantic_updates or []:
        subgoal_id = str(update.get("id", "") or "").strip()
        internal = {"id": subgoal_id}
        for field in ("goal", "answer_type"):
            if field in update:
                internal[field] = update[field]

        resolve_mode = update.get("resolve_mode")
        if resolve_mode is not None:
            internal["resolve_mode"] = resolve_mode
        if "topic_sources" in update:
            anchors, error = _resolve_topic_references(
                update["topic_sources"],
                topic_by_ref,
            )
            if error:
                return None, error
            internal["anchor_entity"] = _anchor_value(anchors)
            if resolve_mode is None and anchors:
                internal["resolve_mode"] = "retrieval"
        if resolve_mode == "auto":
            internal["anchor_entity"] = None
        compiled.append(internal)
    return compiled, ""


def compile_reconstructed_plan(semantic_steps, topic_entities):
    """Generate IDs, dependencies, anchors, and modes from semantic steps."""
    topic_by_ref = _topic_reference_map(topic_entities)
    plan = []
    for position, step in enumerate(semantic_steps or [], start=1):
        dependencies = []
        anchors = []
        for source in step.get("sources", []) or []:
            if source in topic_by_ref:
                name = topic_by_ref[source]
                if name not in anchors:
                    anchors.append(name)
                continue
            match = _STEP_REFERENCE.fullmatch(source)
            if not match:
                return None, f"step {position} has unknown source: {source}"
            source_position = int(match.group(1))
            if source_position < 1 or source_position >= position:
                return None, (
                    f"step {position} references non-prior step: {source}"
                )
            dependency = f"g{source_position}"
            if dependency not in dependencies:
                dependencies.append(dependency)

        resolve_mode = step["resolve_mode"]
        if resolve_mode == "auto":
            if not dependencies:
                return None, (
                    f"auto step {position} has no prior-step source"
                )
            anchors = []
        elif not dependencies and not anchors:
            return None, (
                f"retrieval step {position} has no executable source"
            )

        plan.append({
            "id": f"g{position}",
            "goal": step["goal"],
            "depend": dependencies,
            "answer_type": step["answer_type"],
            "anchor_entity": _anchor_value(anchors),
            "resolve_mode": resolve_mode,
        })
    return plan, ""


def _topic_reference_map(topic_entities):
    return {
        record["ref"]: record["entity"]
        for record in topic_reference_records(topic_entities)
    }


def _resolve_topic_references(references, topic_by_ref):
    anchors = []
    for reference in references or []:
        if reference not in topic_by_ref:
            return None, f"unknown topic source: {reference}"
        name = topic_by_ref[reference]
        if name not in anchors:
            anchors.append(name)
    return anchors, ""


def _anchor_value(anchors):
    if not anchors:
        return None
    return anchors[0] if len(anchors) == 1 else anchors


# --- replan ---

def normalize_plan_diagnosis_with_error(parsed, refinable_ids):
    """Normalize the first-call keep/refine/rebuild diagnosis."""
    if not isinstance(parsed, dict):
        return None, "diagnosis is not a JSON object"

    operation = str(parsed.get("operation", "") or "").strip().lower()
    if operation not in {"keep", "refine", "rebuild"}:
        return None, "operation must be keep, refine, or rebuild"

    reason = str(parsed.get("reason", "") or "").strip()
    targets, error = _normalize_string_list(parsed.get("targets", []), "targets")
    if error:
        return None, error
    defects, error = _normalize_string_list(parsed.get("defects", []), "defects")
    if error:
        return None, error
    missing, error = _normalize_string_list(
        parsed.get("missing_constraints", []),
        "missing_constraints",
    )
    if error:
        return None, error

    valid_ids = {
        str(subgoal_id or "").strip()
        for subgoal_id in refinable_ids or []
        if str(subgoal_id or "").strip()
    }
    unknown_targets = [target for target in targets if target not in valid_ids]
    if unknown_targets:
        return None, (
            "diagnosis targets unknown subgoal(s): "
            f"{', '.join(unknown_targets)}"
        )

    if operation == "keep" and (targets or defects or missing):
        return None, "keep diagnosis must not contain repair details"
    if operation == "refine":
        if not targets:
            return None, "refine diagnosis requires at least one target"
        if missing:
            return None, (
                "missing constraints require rebuild, not refine"
            )

    return {
        "operation": operation,
        "reason": reason,
        "targets": targets,
        "defects": defects,
        "missing_constraints": missing,
    }, ""


def normalize_refine_repair_with_error(parsed, target_ids):
    """Normalize the second-call semantic patch for a refine diagnosis."""
    if not isinstance(parsed, dict):
        return None, "refine repair is not a JSON object"
    updates = parsed.get("updates")
    if not isinstance(updates, list) or not updates:
        return None, "refine repair requires a non-empty updates list"

    allowed_targets = {
        str(target or "").strip()
        for target in target_ids or []
        if str(target or "").strip()
    }
    allowed_fields = {
        "id",
        "goal",
        "answer_type",
        "resolve_mode",
        "topic_sources",
    }
    mutable_fields = allowed_fields - {"id"}
    normalized = []
    seen_ids = set()
    for position, raw_update in enumerate(updates, start=1):
        if not isinstance(raw_update, dict):
            return None, f"refine update {position} is not an object"
        unknown_fields = set(raw_update) - allowed_fields
        if unknown_fields:
            return None, (
                f"refine update {position} contains forbidden field(s): "
                f"{', '.join(sorted(unknown_fields))}"
            )

        subgoal_id = str(raw_update.get("id", "") or "").strip()
        if subgoal_id not in allowed_targets:
            return None, (
                "refine update targets an undiagnosed subgoal: "
                f"{subgoal_id or 'missing id'}"
            )
        if subgoal_id in seen_ids:
            return None, f"duplicate refine update id: {subgoal_id}"
        seen_ids.add(subgoal_id)
        if not (set(raw_update) & mutable_fields):
            return None, f"refine update {subgoal_id} has no mutable field"

        update = {"id": subgoal_id}
        for field in ("goal", "answer_type"):
            if field not in raw_update:
                continue
            value = raw_update[field]
            if not isinstance(value, str) or not value.strip():
                return None, f"refine update {subgoal_id} has invalid {field}"
            update[field] = value.strip()

        if "resolve_mode" in raw_update:
            resolve_mode = str(
                raw_update["resolve_mode"] or ""
            ).strip().lower()
            if resolve_mode not in {"retrieval", "auto"}:
                return None, (
                    f"refine update {subgoal_id} has invalid resolve_mode"
                )
            update["resolve_mode"] = resolve_mode

        if "topic_sources" in raw_update:
            sources, error = _normalize_string_list(
                raw_update["topic_sources"],
                f"refine update {subgoal_id} topic_sources",
            )
            if error:
                return None, error
            update["topic_sources"] = sources
        normalized.append(update)
    return normalized, ""


def normalize_reconstruct_repair_with_error(parsed):
    """Normalize second-call semantic steps for a rebuild diagnosis."""
    if not isinstance(parsed, dict):
        return None, "rebuild repair is not a JSON object"
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        return None, "rebuild repair requires a non-empty steps list"

    allowed_fields = {"goal", "answer_type", "resolve_mode", "sources"}
    steps = []
    for position, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            return None, f"rebuild step {position} is not an object"
        unknown_fields = set(raw_step) - allowed_fields
        if unknown_fields:
            return None, (
                f"rebuild step {position} contains forbidden field(s): "
                f"{', '.join(sorted(unknown_fields))}"
            )

        goal = raw_step.get("goal")
        answer_type = raw_step.get("answer_type")
        resolve_mode = str(
            raw_step.get("resolve_mode", "") or ""
        ).strip().lower()
        if not isinstance(goal, str) or not goal.strip():
            return None, f"rebuild step {position} has invalid goal"
        if not isinstance(answer_type, str) or not answer_type.strip():
            return None, (
                f"rebuild step {position} has invalid answer_type"
            )
        if resolve_mode not in {"retrieval", "auto"}:
            return None, (
                f"rebuild step {position} has invalid resolve_mode"
            )
        sources, error = _normalize_string_list(
            raw_step.get("sources", []),
            f"rebuild step {position} sources",
        )
        if error:
            return None, error
        steps.append({
            "goal": goal.strip(),
            "answer_type": answer_type.strip(),
            "resolve_mode": resolve_mode,
            "sources": sources,
        })
    return steps, ""


def _normalize_string_list(value, field_name):
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        return None, f"{field_name} must be a list of strings"
    normalized = []
    for item in value:
        item = item.strip()
        if not item:
            return None, f"{field_name} contains an empty value"
        if item not in normalized:
            normalized.append(item)
    return normalized, ""


# --- replan ---

def _node_value_sort_key(node):
    return (
        float(node.path_value),
        len(node.walk_path),
    )


def _matched_subgoal_ids(node):
    matched_subgoals = []
    for hit in node.subgoal_answer_hits:
        subgoal_id = str(hit.get("subgoal_id", "") or "").strip()
        if subgoal_id and subgoal_id not in matched_subgoals:
            matched_subgoals.append(subgoal_id)
    return matched_subgoals


def current_plan_records(global_state):
    if not global_state:
        return []

    active_ids = {
        str(subgoal.get("id", "") or "").strip()
        for subgoal in global_state.get_active_subgoals()
    }
    records = []
    for subgoal in global_state.subgoals:
        subgoal_id = str(subgoal.get("id", "") or "").strip()
        records.append({
            "id": subgoal_id,
            "goal": str(subgoal.get("goal", "") or "").strip(),
            "depend": list(subgoal.get("depend", []) or []),
            "answer_type": str(
                subgoal.get("answer_type", "entity") or "entity"
            ).strip(),
            "anchor_entity": subgoal.get("anchor_entity"),
            "resolve_mode": str(
                subgoal.get("resolve_mode", "retrieval") or "retrieval"
            ).strip(),
            "status": str(
                subgoal.get("status", "unresolved") or "unresolved"
            ).strip(),
            "answers": [
                str(answer.get("answer", "") or "").strip()
                for answer in subgoal.get("answers", []) or []
                if str(answer.get("answer", "") or "").strip()
            ],
            "is_active": subgoal_id in active_ids,
        })
    return records


def plan_definition_records(global_state):
    """Return the compact, runtime-state-free plan used in E007/E011 traces."""
    fields = (
        "id",
        "goal",
        "depend",
        "answer_type",
        "anchor_entity",
        "resolve_mode",
    )
    return [
        {field: record[field] for field in fields}
        for record in current_plan_records(global_state)
    ]


def format_current_plan(global_state):
    records = current_plan_records(global_state)
    return json.dumps(records, ensure_ascii=False, indent=2) if records else "[]"


def format_plan_edit_constraints(global_state):
    records = current_plan_records(global_state)
    constraints = {
        "refinable_ids": [
            record["id"]
            for record in records
            if record["status"] in {
                "resolved",
                "unresolved",
                "partial",
            }
        ],
        "refine_mutable_fields": [
            "goal",
            "answer_type",
            "resolve_mode",
            "topic_sources",
        ],
        "refine_immutable_fields": [
            "id",
            "depend",
            "status",
            "answers",
            "is_active",
        ],
    }
    return json.dumps(constraints, ensure_ascii=False, indent=2)


def format_topic_references(topic_entities):
    return json.dumps(
        topic_reference_records(topic_entities),
        ensure_ascii=False,
        indent=2,
    )


def format_diagnosis(diagnosis):
    return json.dumps(diagnosis or {}, ensure_ascii=False, indent=2)


def format_refine_context(global_state, target_ids):
    """Show only diagnosed nodes and their direct graph neighborhood."""
    records = current_plan_records(global_state)
    targets = set(target_ids or [])
    relevant_ids = set(targets)
    for record in records:
        if record["id"] in targets:
            relevant_ids.update(record["depend"])
        if targets.intersection(record["depend"]):
            relevant_ids.add(record["id"])
    selected = [
        record for record in records if record["id"] in relevant_ids
    ]
    return json.dumps(selected, ensure_ascii=False, indent=2)


def format_refine_evidence(evidence, relevant_ids):
    """Keep unmatched paths and paths matched to the local repair context."""
    relevant_ids = set(relevant_ids or [])
    records = [
        record
        for record in (evidence or {}).get("records", []) or []
        if (
            not record.get("matched_subgoals")
            or relevant_ids.intersection(record.get("matched_subgoals", []))
        )
    ]
    return json.dumps(records, ensure_ascii=False, indent=2) if records else "[]"


def collect_replan_evidence(nodes, path_topk=10):
    unique_nodes = []
    seen_paths = set()
    for node in nodes or []:
        key = node_path_key(node)
        if not key or key in seen_paths:
            continue
        seen_paths.add(key)
        unique_nodes.append(node)

    unique_nodes.sort(key=_node_value_sort_key, reverse=True)
    limit = max(1, int(path_topk or 1))

    # A global top-k can be dominated by only one evidence type. Replanning
    # needs both evidence already explained by the current plan and promising
    # evidence the plan cannot yet explain. Select balanced quotas first, then
    # fill any unused slots with the remaining highest-valued unique paths.
    matched_nodes = [
        node for node in unique_nodes if _matched_subgoal_ids(node)
    ]
    unmatched_nodes = [
        node for node in unique_nodes if not _matched_subgoal_ids(node)
    ]
    matched_quota = (limit + 1) // 2
    unmatched_quota = limit // 2
    selected_nodes = (
        matched_nodes[:matched_quota]
        + unmatched_nodes[:unmatched_quota]
    )
    selected_ids = {id(node) for node in selected_nodes}
    for node in unique_nodes:
        if len(selected_nodes) >= limit:
            break
        if id(node) in selected_ids:
            continue
        selected_nodes.append(node)
        selected_ids.add(id(node))
    selected_nodes.sort(key=_node_value_sort_key, reverse=True)

    records = []
    for index, node in enumerate(selected_nodes):
        matched_subgoals = _matched_subgoal_ids(node)
        records.append({
            "index": index,
            "terminal_entity": str(
                node.entity_name or ""
            ).strip(),
            "path_value": round(
                float(node.path_value),
                3,
            ),
            "matched_subgoals": matched_subgoals,
            "path": format_reasoning_path(
                node.walk_path,
                fallback_sep=", ",
                relation_style="readable",
            ),
        })
    return {
        "records": records,
        "matched_count": sum(
            bool(record["matched_subgoals"]) for record in records
        ),
        "unmatched_count": sum(
            not record["matched_subgoals"] for record in records
        ),
    }


def format_replan_evidence(evidence):
    records = (evidence or {}).get("records", []) or []
    return json.dumps(records, ensure_ascii=False, indent=2) if records else "[]"


def log_replan_event(event):
    log = logger.info if event.get("payload_valid") else logger.warning
    log(
        "  Replan decision: operation=%s | diagnosis_valid=%s | "
        "repair_attempted=%s | repair_valid=%s | payload_valid=%s | "
        "changed=%s | modified_subgoals=%s | reason=%s",
        event.get("decision_operation", ""),
        event.get("diagnosis_valid", ""),
        event.get("repair_attempted", False),
        event.get("repair_valid", ""),
        event.get("payload_valid", ""),
        event.get("plan_changed", ""),
        ", ".join(event.get("modified_subgoal_ids", []) or []) or "None",
        event.get("reason", "") or "None",
    )


# --- replan ---

def subgoal_definition_signature(subgoal):
    return (
        str(subgoal.get("id", "") or "").strip(),
        str(subgoal.get("goal", "") or "").strip(),
        tuple(str(dep).strip() for dep in subgoal.get("depend", []) or []),
        str(subgoal.get("answer_type", "") or "").strip(),
        tuple(anchor_values(subgoal.get("anchor_entity"))),
        str(
            subgoal.get("resolve_mode", "retrieval") or "retrieval"
        ).strip().lower(),
    )


def build_refined_plan(global_state, refined_subgoals):
    """Apply local field updates and invalidate all downstream runtime state."""
    if not global_state:
        return [], [], "no current subgoal state"
    if not refined_subgoals:
        return [], [], "refine updates are empty"

    refined_ids = [
        str(subgoal.get("id", "") or "").strip()
        for subgoal in refined_subgoals
    ]
    duplicate_ids = {
        subgoal_id
        for subgoal_id in refined_ids
        if refined_ids.count(subgoal_id) > 1
    }
    if duplicate_ids:
        return [], [], (
            f"duplicate refined subgoal id(s): "
            f"{', '.join(sorted(duplicate_ids))}"
        )

    merged_plan = copy.deepcopy(global_state.subgoals)
    positions = {
        str(subgoal.get("id", "") or "").strip(): index
        for index, subgoal in enumerate(merged_plan)
    }
    if len(positions) != len(merged_plan) or "" in positions:
        return [], [], (
            "current plan contains missing or duplicate subgoal ids"
        )

    modified_ids = []
    mutable_fields = {
        "goal",
        "answer_type",
        "anchor_entity",
        "resolve_mode",
    }
    for update in refined_subgoals:
        subgoal_id = str(update.get("id", "") or "").strip()
        if subgoal_id not in positions:
            return [], [], (
                f"refine cannot add unknown subgoal: "
                f"{subgoal_id or 'missing id'}"
            )

        position = positions[subgoal_id]
        old_subgoal = merged_plan[position]
        if old_subgoal.get("status") not in {
            "resolved",
            "unresolved",
            "partial",
        }:
            return [], [], (
                f"refine cannot replace subgoal {subgoal_id} "
                "with invalid status"
            )

        replacement = copy.deepcopy(old_subgoal)
        for field in mutable_fields:
            if field in update:
                replacement[field] = copy.deepcopy(update[field])
        if replacement.get("resolve_mode", "retrieval") == "auto":
            replacement["anchor_entity"] = None

        replacement["id"] = subgoal_id
        replacement["depend"] = copy.deepcopy(
            old_subgoal.get("depend", []) or []
        )
        replacement["status"] = "unresolved"
        replacement["answers"] = []

        if (
            subgoal_definition_signature(replacement)
            == subgoal_definition_signature(old_subgoal)
        ):
            return [], [], (
                f"refine subgoal has no content change: {subgoal_id}"
            )
        merged_plan[position] = replacement
        modified_ids.append(subgoal_id)

    invalidated_ids = set(modified_ids)
    for subgoal in merged_plan:
        subgoal_id = str(subgoal.get("id", "") or "").strip()
        if subgoal_id in invalidated_ids:
            continue
        if any(
            dependency in invalidated_ids
            for dependency in subgoal.get("depend", []) or []
        ):
            subgoal["status"] = "unresolved"
            subgoal["answers"] = []
            invalidated_ids.add(subgoal_id)
    return merged_plan, modified_ids, ""


def reproject_after_plan_change(search, tree_nodes=None):
    """Keep revised-plan evidence and rebuild its ancestor-closed MCTS tree."""
    common_state = copy.deepcopy(search.global_subgoal_state.__dict__)

    def fallback(mode):
        search.global_subgoal_state.__dict__.clear()
        search.global_subgoal_state.__dict__.update(copy.deepcopy(common_state))
        result = restart_after_plan_change(search, tree_mode=mode, tree_reuse_enabled=True)
        result.update(original_node_count=original_node_count,
                      discarded_node_count=original_node_count,
                      attempted_recovered_answer_count=recovered_answer_count)
        return result

    old_nodes = search._all_real_tree_nodes(require_path=False)
    original_node_count = len(old_nodes)
    answer_count_before = _grounded_answer_count(
        search.global_subgoal_state
    )

    _reset_plan_conditioned_tree_state(search, old_nodes)
    tree_state_updated = search._check_subgoals_from_full_tree(
        rebuild_tree_statistics=False,
        reopen_answer_nodes=False,
        tree_nodes=tree_nodes,
    )
    search._sync_subgoal_answer_hits_to_tree(old_nodes)

    matched_nodes = [
        node
        for node in old_nodes
        if node.walk_path
        and incremental_subgoal_hits(node)
    ]
    answer_count_after = _grounded_answer_count(
        search.global_subgoal_state
    )
    recovered_answer_count = max(
        0,
        answer_count_after - answer_count_before,
    )

    if not matched_nodes:
        return fallback("restart_no_match")

    retained = _ancestor_closure(matched_nodes)
    _prune_to_nodes(search.root, retained)
    retained_nodes = _breadth_first_real_nodes(search.root)

    # Every retained node is closed first. Under the revised plan, every
    # eligible retained KG node may resume exploration, not only the direct
    # evidence matches. Its relation history is reset by reopen_nodes so that
    # pruning is performed again with the revised subgoal context.
    for node in retained_nodes:
        node.is_fully_expanded = True

    search._rebuild_tree_statistics_for_current_subgoals(
        sync_hits=False,
        nodes=retained_nodes,
    )
    frontier_nodes = reopen_nodes(
        search,
        [
            node
            for node in retained_nodes
            if can_expand_after_progress(
                node,
                search.global_subgoal_state,
            )
        ],
    )
    search._refresh_root_activity()

    early_stop_paths = None
    if not search._has_expandable_node(search.root):
        logger.info("  Reprojected tree has no expandable frontier; checking early stop before restart.")
        early_result = search._attempt_early_stop()
        if early_result["status"] == "accepted":
            early_stop_paths = early_result["paths"]
        else:
            logger.info("  Reprojection did not yield a verified answer; restart the revised plan from topic entities.")
            result = fallback("restart_no_expandable_match")
            # Restart occurs after pruning; report disposal of the whole old tree.
            result.update(
                original_node_count=original_node_count,
                discarded_node_count=original_node_count,
                direct_match_node_count=len(matched_nodes),
                rematched_node_count=len(matched_nodes),
            )
            return result

    entities = _node_entities(frontier_nodes)
    frontier_summary = {
        "source": (
            "retained_evidence_nodes"
            if frontier_nodes
            else "no_expandable_match"
        ),
        "node_count": len(frontier_nodes),
        "entities": entities,
    }
    evaluated_path_count = sum(
        bool(node.walk_path)
        for node in retained_nodes
    )
    reuse_summary = {
        "tree_reuse_enabled": True,
        "tree_mode": "evidence_reprojection",
        "original_node_count": original_node_count,
        "discarded_node_count": original_node_count - len(retained_nodes),
        "retained_node_count": len(retained_nodes),
        "evaluated_path_count": evaluated_path_count,
        "replayed_node_count": evaluated_path_count,
        "rematched_node_count": len(matched_nodes),
        "direct_match_node_count": len(matched_nodes),
        "reopened_node_count": len(frontier_nodes),
        "recovered_answer_count": recovered_answer_count,
        "tree_statistics_rebuilt": True,
    }
    logger.info(
        "  Evidence-preserving tree reprojection: original=%s | "
        "matched=%s | retained=%s | pruned=%s | replayed=%s | "
        "reopened=%s | recovered_answers=%s",
        reuse_summary["original_node_count"],
        reuse_summary["direct_match_node_count"],
        reuse_summary["retained_node_count"],
        reuse_summary["discarded_node_count"],
        reuse_summary["replayed_node_count"],
        reuse_summary["reopened_node_count"],
        reuse_summary["recovered_answer_count"],
    )
    return {
        "tree_state_updated": tree_state_updated,
        "frontier_summary": frontier_summary,
        "early_stop_paths": early_stop_paths,
        **reuse_summary,
    }


def restart_after_plan_change(
    search,
    *,
    tree_mode="restart",
    tree_reuse_enabled=False,
    tree_state_updated=False,
    recovered_answer_count=0,
):
    """Discard the old evidence tree and restart from fresh topic roots."""
    old_real_nodes = search._all_real_tree_nodes(require_path=False)
    topic_entities = list(
        getattr(search, "initial_topic_entities", None) or [
            dict(getattr(node, "entity_info", None) or {})
            for node in search.root.children
            if getattr(node, "entity_info", None)
        ]
    )

    search.root = MCTSNode(entity_info={}, is_virtual_root=True)
    for entity_info in topic_entities:
        entity_id = str(entity_info.get("entity_id", "") or "").strip()
        if not entity_id:
            continue
        root = MCTSNode(
            entity_info=entity_info,
            parent=search.root,
            visited_ids={entity_id},
        )
        root.root_active = False
        search.root.children.append(root)

    search.early_stop_verification_cache.clear()
    search.early_stop_answer = ""
    search.final_answer_nodes = []
    search._refresh_root_activity()

    active_roots = search._get_active_roots() or list(search.root.children)
    entities = _node_entities(active_roots)
    logger.info(
        "  Replan tree restart: discarded=%s | fresh_roots=%s | "
        "entities=%s",
        len(old_real_nodes),
        len(active_roots),
        ", ".join(entities) if entities else "None",
    )
    return {
        "tree_state_updated": tree_state_updated,
        "frontier_summary": {
            "source": "fresh_topic_roots",
            "node_count": len(active_roots),
            "entities": entities,
        },
        "tree_reuse_enabled": tree_reuse_enabled,
        "tree_mode": tree_mode,
        "original_node_count": len(old_real_nodes),
        "discarded_node_count": len(old_real_nodes),
        "retained_node_count": 0,
        "evaluated_path_count": 0,
        "replayed_node_count": 0,
        "rematched_node_count": 0,
        "direct_match_node_count": 0,
        "reopened_node_count": 0,
        "recovered_answer_count": recovered_answer_count,
        "tree_statistics_rebuilt": False,
    }


def _grounded_answer_count(global_state):
    return sum(
        len(subgoal.get("answers", []) or [])
        for subgoal in getattr(global_state, "subgoals", []) or []
    )


def _reset_plan_conditioned_tree_state(search, nodes):
    for node in nodes:
        node.subgoal_answer_hits = []
        node.progress_credit = 0.0
        node.fused_value = 0.0
        node.total_reward = 0.0
        node.value_weight = 0.0
        node.uct_visits = 0
    search.root.total_reward = 0.0
    search.root.value_weight = 0.0
    search.root.uct_visits = 0
    search.early_stop_verification_cache.clear()
    search.early_stop_answer = ""


def _ancestor_closure(nodes):
    retained = set()
    for node in nodes:
        current = node
        while current and not current.is_virtual_root:
            retained.add(current)
            current = current.parent
    return retained


def _prune_to_nodes(root, retained):
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        parent.children = [
            child
            for child in parent.children
            if child in retained
        ]
        queue.extend(parent.children)


def _breadth_first_real_nodes(root):
    nodes = []
    queue = deque(root.children)
    while queue:
        node = queue.popleft()
        nodes.append(node)
        queue.extend(node.children)
    return nodes


def _node_entities(nodes):
    entities = []
    for node in nodes:
        entity = (
            str(node.entity_name or "").strip()
            or str(node.entity_id or "").strip()
        )
        if entity and entity not in entities:
            entities.append(entity)
    return entities


# --- replan ---

class Replanner:
    """Diagnose first, repair once when needed, then compile in code."""

    def __init__(self, search=None):
        self.search = search
        self.replan_count = 0
        self.replan_events = []
        self._last_terminal_input_signature = None
        self._terminal_repair_outcomes = {}
        self.enable_replan_tree_reuse = True
        if search is not None:
            self.enable_subgoal_replan = search.enable_subgoal_replan
            self.enable_replan_tree_reuse = bool(
                getattr(search, "enable_replan_tree_reuse", True)
            )
            self.planner_model = search.planner_model
            self.question = search.question
            self.top_entities = search.top_entities

    @property
    def global_subgoal_state(self):
        if self.search is not None:
            return self.search.global_subgoal_state
        return getattr(self, "_global_subgoal_state", None)

    @global_subgoal_state.setter
    def global_subgoal_state(self, value):
        if self.search is not None:
            self.search.global_subgoal_state = value
        else:
            self._global_subgoal_state = value

    def try_replan(
        self,
        iteration,
        tree_nodes=None,
        checkpoint_id=None,
    ):
        if (
            not self.enable_subgoal_replan
            or not self.global_subgoal_state
        ):
            return {
                "action": "none",
                "success": False,
                "checked": False,
            }
        if self.replan_count >= 1:
            logger.info(
                "  Skip plan repair: this sample already applied one "
                "successful repair."
            )
            return {
                "action": "skip_replan_limit",
                "success": False,
                "checked": True,
                "plan_changed": False,
                "skipped_replan_limit": True,
            }

        diagnosis, evidence, request_context = (
            self._request_plan_diagnosis(tree_nodes=tree_nodes)
        )
        if request_context.get("duplicate_terminal_input"):
            logger.info(
                "  Skip repeated plan repair: the plan state and "
                "LLM-visible tree evidence are unchanged since the last "
                "no-change decision."
            )
            return {
                "action": "skip_duplicate_replan",
                "success": False,
                "checked": True,
                "plan_changed": False,
                "skipped_duplicate": True,
            }

        input_signature = request_context.get("input_signature")
        if not diagnosis:
            operation = request_context.get("parsed_operation", "")
            error = request_context.get(
                "normalization_error",
                "invalid diagnosis",
            )
            event = self._base_event(
                iteration,
                operation,
                evidence,
                reason=f"Invalid plan diagnosis: {error}",
            )
            event["checkpoint_id"] = checkpoint_id
            event["diagnosis_valid"] = False
            return self._reject_replan(event, input_signature, error)

        operation = diagnosis["operation"]
        repair_attempt_key = self._repair_attempt_key(diagnosis)
        event = self._base_event(
            iteration,
            operation,
            evidence,
            reason=diagnosis["reason"],
        )
        event["checkpoint_id"] = checkpoint_id
        event.update({
            "diagnosis_valid": True,
            "diagnosed_targets": diagnosis["targets"],
            "diagnosed_defects": diagnosis["defects"],
            "missing_constraints": diagnosis["missing_constraints"],
            "repair_attempted": operation != "keep",
        })

        if operation == "keep":
            self._last_terminal_input_signature = input_signature
            event.update({
                "repair_type": "none",
                "payload_valid": True,
                "repair_valid": None,
            })
            self._record_replan_event(event)
            return {
                "action": "none",
                "success": False,
                "checked": True,
                "plan_changed": False,
                "repair_type": "none",
            }

        cached_outcome = self._terminal_repair_outcomes.get(
            repair_attempt_key
        )
        if cached_outcome:
            self._last_terminal_input_signature = input_signature
            logger.info(
                "  Skip repeated %s repair for the unchanged plan and "
                "targets: prior %s outcome (%s).",
                operation,
                cached_outcome["type"],
                cached_outcome["reason"],
            )
            return {
                "action": "skip_cached_repair_outcome",
                "success": False,
                "checked": True,
                "plan_changed": False,
                "skipped_cached_repair_outcome": True,
            }

        semantic_repair, repair_error = self._request_plan_repair(
            diagnosis,
            evidence,
        )
        if semantic_repair is None:
            event["repair_valid"] = False
            return self._reject_replan(
                event,
                input_signature,
                repair_error,
                prefix=f"Invalid {operation} repair",
            )

        decision, compile_error = self._compile_repair_decision(
            diagnosis,
            semantic_repair,
        )
        if decision is None:
            event["repair_valid"] = False
            self._cache_terminal_repair_outcome(
                repair_attempt_key,
                "compile_failure",
                compile_error,
            )
            return self._reject_replan(
                event,
                input_signature,
                compile_error,
                prefix="Non-executable semantic repair",
            )

        event.update({
            "repair_type": operation,
            "repair_valid": True,
            "payload_valid": True,
            "repair_payload_size": len(semantic_repair),
        })
        old_plan = plan_definition_records(self.global_subgoal_state)
        apply_info = self._apply_replan_decision(
            decision,
            tree_nodes=tree_nodes,
        )
        if not apply_info["success"]:
            if operation == "refine" and apply_info.get("no_change"):
                self._cache_terminal_repair_outcome(
                    repair_attempt_key,
                    "no_change",
                    apply_info["reason"],
                )
                self._last_terminal_input_signature = input_signature
                event.update({
                    "diagnosed_operation": operation,
                    "decision_operation": "keep",
                    "need_replan": False,
                    "repair_type": "none",
                    "repair_valid": True,
                    "payload_valid": True,
                    "plan_changed": False,
                    "no_change": True,
                    "fallback_operation": "keep",
                    "reason": (
                        "Refine produced no semantic change; "
                        "treating the result as KEEP."
                    ),
                    "no_change_reason": apply_info["reason"],
                })
                self._record_replan_event(event)
                return {
                    "action": "none",
                    "success": False,
                    "checked": True,
                    "plan_changed": False,
                    "repair_type": "none",
                    "no_change": True,
                }
            event["repair_valid"] = False
            self._cache_terminal_repair_outcome(
                repair_attempt_key,
                "structural_failure",
                apply_info["reason"],
            )
            return self._reject_replan(
                event,
                input_signature,
                apply_info["reason"],
                prefix="Non-executable replanned plan",
            )

        apply_info.update(
            self._finish_plan_change(
                tree_nodes,
                repair_type=decision["repair_type"],
            )
        )
        event.update({
            "payload_valid": True,
            "plan_changed": True,
            "old_plan_size": apply_info["old_plan_size"],
            "new_plan_size": len(
                self.global_subgoal_state.subgoals
            ),
            "tree_state_updated": apply_info["tree_state_updated"],
            "frontier_source": apply_info[
                "frontier_summary"
            ]["source"],
            "frontier_node_count": apply_info[
                "frontier_summary"
            ]["node_count"],
            "frontier_entities": apply_info[
                "frontier_summary"
            ]["entities"],
            "tree_reuse_enabled": apply_info.get(
                "tree_reuse_enabled",
                self.enable_replan_tree_reuse,
            ),
            "tree_mode": apply_info.get(
                "tree_mode",
                (
                    "reuse"
                    if self.enable_replan_tree_reuse
                    else "restart"
                ),
            ),
            "original_node_count": apply_info.get(
                "original_node_count",
                0,
            ),
            "discarded_node_count": apply_info.get(
                "discarded_node_count",
                0,
            ),
            "retained_node_count": apply_info.get(
                "retained_node_count",
                0,
            ),
            "evaluated_path_count": apply_info.get(
                "evaluated_path_count",
                0,
            ),
            "rematched_node_count": apply_info.get(
                "rematched_node_count",
                0,
            ),
            "direct_match_node_count": apply_info.get(
                "direct_match_node_count",
                0,
            ),
            "replayed_node_count": apply_info.get(
                "replayed_node_count",
                0,
            ),
            "reopened_node_count": apply_info.get(
                "reopened_node_count",
                0,
            ),
            "recovered_answer_count": apply_info.get(
                "recovered_answer_count",
                0,
            ),
            "modified_subgoal_ids": apply_info.get(
                "modified_subgoal_ids",
                [],
            ),
            "old_plan": old_plan,
            "new_plan": plan_definition_records(
                self.global_subgoal_state
            ),
        })
        self._record_replan_event(event)
        return {
            "action": "repair_plan",
            "success": True,
            "checked": True,
            "plan_changed": True,
            "tree_statistics_rebuilt": apply_info.get(
                "tree_statistics_rebuilt",
                False,
            ),
            "repair_type": operation,
            "early_stop_paths": apply_info.get("early_stop_paths"),
        }

    def _base_event(self, iteration, operation, evidence, reason):
        return {
            "iteration": iteration,
            "stage": "repair",
            "need_replan": operation in {"refine", "rebuild"},
            "repair_type": operation if operation else "invalid",
            "decision_operation": operation,
            "payload_valid": True,
            "plan_changed": False,
            "reason": reason,
            "evidence_count": len(
                (evidence or {}).get("records", []) or []
            ),
        }

    def _reject_replan(
        self,
        event,
        input_signature,
        error,
        prefix=None,
    ):
        self._last_terminal_input_signature = input_signature
        event.update({
            "repair_type": "invalid",
            "payload_valid": False,
            "plan_changed": False,
            "fallback_operation": "keep",
            "validation_error": error,
        })
        if prefix:
            event["reason"] = f"{prefix}: {error}"
        self._record_replan_event(event)
        return {
            "action": "none",
            "success": False,
            "checked": True,
            "plan_changed": False,
        }

    def _compile_repair_decision(self, diagnosis, semantic_repair):
        operation = diagnosis["operation"]
        if operation == "refine":
            updates, error = compile_refine_updates(
                semantic_repair,
                self.top_entities,
            )
            if updates is None:
                return None, error
            return {
                "repair_type": "refine",
                "refined_subgoals": updates,
                "new_plan": [],
            }, ""

        plan, error = compile_reconstructed_plan(
            semantic_repair,
            self.top_entities,
        )
        if plan is None:
            return None, error
        return {
            "repair_type": "rebuild",
            "refined_subgoals": [],
            "new_plan": plan,
        }, ""

    def _apply_replan_decision(
        self,
        decision,
        tree_nodes=None,
    ):
        if decision["repair_type"] == "refine":
            return self._apply_refinement(
                decision["refined_subgoals"],
            )
        if decision["repair_type"] == "rebuild":
            return self._apply_reconstruction(
                decision["new_plan"],
            )
        return {
            "success": False,
            "reason": "keep has no plan payload to apply",
        }

    def _apply_refinement(
        self,
        refined_subgoals,
    ):
        merged_plan, modified_ids, reason = build_refined_plan(
            self.global_subgoal_state,
            refined_subgoals,
        )
        if not merged_plan:
            return {
                "success": False,
                "reason": reason,
                "no_change": "no content change" in reason,
            }

        merged_plan, reason = self._repair_and_validate_plan(
            merged_plan
        )
        if not merged_plan:
            return {"success": False, "reason": reason}
        if all(
            subgoal_definition_signature(merged_plan[index])
            == subgoal_definition_signature(
                self.global_subgoal_state.subgoals[index]
            )
            for index in range(len(merged_plan))
        ):
            return {
                "success": False,
                "reason": (
                    "refine has no content change after anchor repair"
                ),
                "no_change": True,
            }

        old_plan_size = len(
            self.global_subgoal_state.subgoals
        )
        self.global_subgoal_state.replace_subgoals(merged_plan)
        return {
            "success": True,
            "old_plan_size": old_plan_size,
            "modified_subgoal_ids": modified_ids,
        }

    def _apply_reconstruction(
        self,
        new_plan,
    ):
        new_plan, reason = self._repair_and_validate_plan(new_plan)
        if not new_plan:
            return {"success": False, "reason": reason}

        current_plan = self.global_subgoal_state.subgoals
        if (
            len(new_plan) == len(current_plan)
            and all(
                subgoal_definition_signature(new_subgoal)
                == subgoal_definition_signature(current_subgoal)
                for new_subgoal, current_subgoal in zip(
                    new_plan,
                    current_plan,
                )
            )
        ):
            return {
                "success": False,
                "reason": (
                    "rebuild has no content change after anchor repair"
                ),
            }

        old_plan_size = len(current_plan)
        self.global_subgoal_state.replace_subgoals(new_plan)
        return {
            "success": True,
            "old_plan_size": old_plan_size,
            "modified_subgoal_ids": [],
        }

    def _finish_plan_change(self, tree_nodes=None, repair_type="refine"):
        # Commit the common intervention prefix before either tree policy runs.
        self.replan_count += 1
        self._last_terminal_input_signature = None
        self.search.no_subgoal_progress_rounds = 0
        return self._update_tree_after_plan_change(
            tree_nodes,
            repair_type=repair_type,
        )

    def _update_tree_after_plan_change(self, tree_nodes=None, repair_type="refine"):
        # A rebuild replaces the plan structure, so old tree topology and
        # plan-conditioned search history are not reused. Refinement preserves
        # the plan skeleton and can safely attempt evidence reprojection.
        if repair_type == "rebuild":
            return restart_after_plan_change(
                self.search,
                tree_mode="restart_rebuild",
                tree_reuse_enabled=False,
            )
        if not self.enable_replan_tree_reuse:
            return restart_after_plan_change(self.search)
        return reproject_after_plan_change(
            self.search,
            tree_nodes=tree_nodes,
        )

    def _repair_and_validate_plan(self, candidate_plan):
        repaired = repair_missing_constraint_anchors(
            copy.deepcopy(candidate_plan),
            self.top_entities,
        )
        valid, reason = validate_executable_plan(
            repaired,
            self.top_entities,
        )
        if not valid:
            return None, reason
        unused_topics = unused_topic_entities(
            repaired,
            self.top_entities,
        )
        if unused_topics:
            logger.info(
                "  Replanned plan leaves topic entities unused "
                "(allowed): %s",
                ", ".join(unused_topics),
            )
        return repaired, ""

    def _repair_attempt_key(self, diagnosis):
        plan_signature = tuple(
            subgoal_definition_signature(subgoal)
            for subgoal in self.global_subgoal_state.subgoals
        )
        return (
            plan_signature,
            diagnosis["operation"],
            tuple(diagnosis.get("targets", []) or []),
        )

    def _cache_terminal_repair_outcome(
        self,
        repair_attempt_key,
        outcome_type,
        reason,
    ):
        self._terminal_repair_outcomes[repair_attempt_key] = {
            "type": outcome_type,
            "reason": str(reason or "unknown"),
        }

    def _request_plan_diagnosis(self, tree_nodes=None):
        nodes = (
            self.search._all_real_tree_nodes(require_path=True)
            if tree_nodes is None
            else tree_nodes
        )
        evidence = collect_replan_evidence(
            nodes,
            path_topk=self.search.path_topk,
        )
        user_prompt = subgoal_plan_diagnosis_user_prompt.substitute(
            question=self.question,
            topic_references=format_topic_references(
                self.top_entities
            ),
            current_plan=format_current_plan(
                self.global_subgoal_state
            ),
            plan_edit_constraints=format_plan_edit_constraints(
                self.global_subgoal_state
            ),
            compressed_tree_evidence=format_replan_evidence(
                evidence
            ),
        )
        input_signature = hashlib.sha256(
            (
                subgoal_plan_diagnosis_system_prompt
                + "\n"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest()
        request_context = {
            "input_signature": input_signature,
            "duplicate_terminal_input": (
                input_signature == self._last_terminal_input_signature
            ),
        }
        if request_context["duplicate_terminal_input"]:
            return None, evidence, request_context

        output = self.planner_model(
            system_prompt=subgoal_plan_diagnosis_system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
        )[0]
        parsed = extract_json(output)
        diagnosis, error = normalize_plan_diagnosis_with_error(
            parsed,
            self._refinable_subgoal_ids(),
        )
        request_context.update({
            "normalization_error": error,
            "parsed_operation": (
                str(parsed.get("operation", "") or "").strip().lower()
                if isinstance(parsed, dict)
                else ""
            ),
        })
        return diagnosis, evidence, request_context

    def _request_plan_repair(self, diagnosis, evidence):
        if diagnosis["operation"] == "refine":
            user_prompt = subgoal_plan_refine_user_prompt.substitute(
                question=self.question,
                diagnosis=format_diagnosis(diagnosis),
                topic_references=format_topic_references(
                    self.top_entities
                ),
                local_plan_context=format_refine_context(
                    self.global_subgoal_state,
                    diagnosis["targets"],
                ),
                plan_edit_constraints=format_plan_edit_constraints(
                    self.global_subgoal_state
                ),
                relevant_tree_evidence=format_refine_evidence(
                    evidence,
                    diagnosis["targets"],
                ),
            )
            output = self.planner_model(
                system_prompt=subgoal_plan_refine_system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
            )[0]
            return normalize_refine_repair_with_error(
                extract_json(output),
                diagnosis["targets"],
            )

        user_prompt = subgoal_plan_reconstruct_user_prompt.substitute(
            question=self.question,
            diagnosis=format_diagnosis(diagnosis),
            topic_references=format_topic_references(
                self.top_entities
            ),
            current_plan=format_current_plan(
                self.global_subgoal_state
            ),
            compressed_tree_evidence=format_replan_evidence(
                evidence
            ),
        )
        output = self.planner_model(
            system_prompt=subgoal_plan_reconstruct_system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048,
        )[0]
        return normalize_reconstruct_repair_with_error(
            extract_json(output)
        )

    def _refinable_subgoal_ids(self):
        return [
            str(subgoal.get("id", "") or "").strip()
            for subgoal in self.global_subgoal_state.subgoals
            if str(
                subgoal.get("status", "unresolved") or "unresolved"
            ).strip() in {"resolved", "unresolved", "partial"}
        ]

    def _record_replan_event(self, event):
        event.setdefault("event_id", len(self.replan_events) + 1)
        self.replan_events.append(event)
        log_replan_event(event)
