"""subgoal functions for SPAR-MCTS."""

import re
import copy
import logging
from prompt_list import (
    complexity_judge_system_prompt,
    complexity_judge_user_prompt,
    decompose_system_prompt,
    decompose_user_prompt,
    simple_plan_system_prompt,
    simple_plan_user_prompt,
)
from utils import extract_json, format_top_entities, topic_entity_names

logger = logging.getLogger(__name__)


# --- subgoal ---


def anchor_values(anchor):
    """Return normalized non-empty anchor references from a scalar or list."""
    if isinstance(anchor, list):
        values = [str(item).strip() for item in anchor]
    elif anchor is None:
        values = []
    else:
        values = [str(anchor).strip()]
    return [value for value in values if value and value.lower() not in {"none", "null"}]


# --- subgoal ---

class SubgoalState:
    """Static initial plan used to guide the transitional entity search."""
    def __init__(self, subgoals):
        self.subgoals = copy.deepcopy(subgoals or [])
        for sg in self.subgoals:
            sg["status"] = "unresolved"
            sg["answers"] = []

    def format_for_prompt(self):
        if not self.subgoals:
            return "None"
        lines = ["[active]"]
        for sg in self.subgoals:
            deps = ", ".join(sg.get("depend", [])) or "None"
            lines.append(f"{sg.get('id', '')}: {sg.get('goal', '')} | depend: {deps} | answer_type: {sg.get('answer_type', 'entity')}")
        return "\n".join(lines + ["[required_answers]", "None", "[resolved_background]", "None"])

    def format_for_early_stop_prompt(self):
        return "None"

    def format_answer_context(self):
        return "None"

    def all_resolved(self):
        return False


def _normalize_anchor_text(text):
    text = re.sub(r"[^\w]+", " ", str(text or "").casefold())
    return re.sub(r"\s+", " ", text).strip()


def _topic_entity_lookup(topic_entities):
    """Map normalized topic IDs and names to the canonical display name."""
    lookup = {}
    if isinstance(topic_entities, dict):
        items = topic_entities.items()
    elif isinstance(topic_entities, list):
        items = ((item, item) for item in topic_entities)
    else:
        items = ((topic_entities, topic_entities),) if str(topic_entities or "").strip() else ()

    for entity_id, entity_name in items:
        canonical_name = str(entity_name or entity_id or "").strip()
        if not canonical_name:
            continue
        for alias in (entity_id, entity_name):
            alias_norm = _normalize_anchor_text(alias)
            if alias_norm:
                lookup[alias_norm] = canonical_name
    return lookup


def _goal_contains_topic_entity(goal, topic_name):
    goal_norm = _normalize_anchor_text(goal)
    topic_norm = _normalize_anchor_text(topic_name)
    if not goal_norm or not topic_norm:
        return False
    return f" {topic_norm} " in f" {goal_norm} "


def repair_missing_constraint_anchors(subgoals, topic_entities):
    """Normalize anchors and recover missing topic-entity anchors from goals."""
    if not isinstance(subgoals, list):
        return subgoals

    topic_names = topic_entity_names(topic_entities)
    if not topic_names:
        return subgoals

    topic_by_norm = _topic_entity_lookup(topic_entities)
    used_norms = set()
    for subgoal in subgoals:
        if not isinstance(subgoal, dict):
            continue
        if str(subgoal.get("resolve_mode", "retrieval")).strip().lower() == "auto":
            if anchor_values(subgoal.get("anchor_entity")):
                logger.debug("  Cleared anchor_entity for auto subgoal %s.", subgoal.get("id", "unknown"))
            subgoal["anchor_entity"] = None
            continue

        valid_anchors = []
        invalid_anchors = []
        for anchor in anchor_values(subgoal.get("anchor_entity")):
            anchor_norm = _normalize_anchor_text(anchor)
            if anchor_norm in topic_by_norm:
                canonical = topic_by_norm[anchor_norm]
                if canonical not in valid_anchors:
                    valid_anchors.append(canonical)
                used_norms.add(_normalize_anchor_text(canonical))
            else:
                invalid_anchors.append(str(anchor))

        if invalid_anchors:
            logger.debug(
                "  Discarded invalid anchor_entity value(s) for %s: %s",
                subgoal.get("id", "unknown"),
                invalid_anchors,
            )
        subgoal["anchor_entity"] = (
            valid_anchors[0]
            if len(valid_anchors) == 1
            else (valid_anchors or None)
        )

    sorted_topics = sorted(topic_names, key=lambda item: len(_normalize_anchor_text(item)), reverse=True)
    for subgoal in subgoals:
        if not isinstance(subgoal, dict):
            continue
        if str(subgoal.get("resolve_mode", "retrieval")).strip().lower() == "auto":
            continue
        if anchor_values(subgoal.get("anchor_entity")):
            continue

        matches = [
            topic_name
            for topic_name in sorted_topics
            if _goal_contains_topic_entity(subgoal.get("goal", ""), topic_name)
        ]
        if not matches:
            continue

        unused_matches = [
            topic_name
            for topic_name in matches
            if _normalize_anchor_text(topic_name) not in used_norms
        ]
        selected = unused_matches + [
            topic_name
            for topic_name in matches
            if topic_name not in unused_matches
        ]
        subgoal["anchor_entity"] = selected[0] if len(selected) == 1 else selected
        used_norms.update(_normalize_anchor_text(item) for item in selected)
        logger.debug(
            "  Repaired anchor_entity for %s: %s",
            subgoal.get("id", "unknown"),
            subgoal["anchor_entity"],
        )

    return subgoals


def unused_topic_entities(subgoals, topic_entities):
    """Return topic entities that the plan does not explicitly reference.

    Topic entities are available KG search anchors, not a guarantee that every
    item must appear as an independent plan constraint.  Dataset annotations
    may contain auxiliary types or implicit constraints such as ``Male`` for
    "father".  Callers may log this information, but it must not determine
    structural executability.
    """
    topic_names = topic_entity_names(topic_entities)
    anchored = {
        _normalize_anchor_text(anchor)
        for subgoal in subgoals or []
        for anchor in anchor_values(subgoal.get("anchor_entity"))
    }
    unused = []
    for topic_name in topic_names:
        topic_norm = _normalize_anchor_text(topic_name)
        if topic_norm in anchored:
            continue
        if any(_goal_contains_topic_entity(subgoal.get("goal", ""), topic_name) for subgoal in subgoals or []):
            continue
        unused.append(topic_name)
    return unused


def validate_executable_plan(subgoals, topic_entities):
    """Validate only deterministic plan structure and executable references."""
    if not isinstance(subgoals, list) or not subgoals:
        return False, "plan is empty"

    topic_by_norm = _topic_entity_lookup(topic_entities)
    prior_ids = set()
    for position, subgoal in enumerate(subgoals, start=1):
        if not isinstance(subgoal, dict):
            return False, f"subgoal {position} is not an object"

        subgoal_id = str(subgoal.get("id", "") or "").strip()
        goal = str(subgoal.get("goal", "") or "").strip()
        answer_type = str(subgoal.get("answer_type", "") or "").strip()
        if not subgoal_id:
            return False, f"subgoal {position} has no id"
        if subgoal_id in prior_ids:
            return False, f"duplicate subgoal id: {subgoal_id}"
        if not goal:
            return False, f"subgoal {subgoal_id} has no goal"
        if not answer_type:
            return False, f"subgoal {subgoal_id} has no answer_type"

        raw_dependencies = subgoal.get("depend", [])
        if raw_dependencies is None:
            raw_dependencies = []
        if not isinstance(raw_dependencies, list) or not all(
            isinstance(dependency, str) for dependency in raw_dependencies
        ):
            return False, f"subgoal {subgoal_id} has invalid dependencies"
        dependencies = [dependency.strip() for dependency in raw_dependencies]
        if any(not dependency for dependency in dependencies):
            return False, f"subgoal {subgoal_id} has an empty dependency"
        if len(set(dependencies)) != len(dependencies):
            return False, f"subgoal {subgoal_id} has duplicate dependencies"
        non_prior_dependencies = [
            dependency for dependency in dependencies if dependency not in prior_ids
        ]
        if non_prior_dependencies:
            return False, (
                f"{subgoal_id} depends on non-prior subgoal(s): "
                f"{', '.join(non_prior_dependencies)}"
            )

        resolve_mode = str(
            subgoal.get("resolve_mode", "retrieval") or "retrieval"
        ).strip().lower()
        if resolve_mode not in {"retrieval", "auto"}:
            return False, f"subgoal {subgoal_id} has invalid resolve_mode: {resolve_mode}"

        anchors = anchor_values(subgoal.get("anchor_entity"))
        if resolve_mode == "auto":
            if anchors:
                return False, f"auto subgoal {subgoal_id} must not have anchor_entity"
            if not dependencies:
                return False, f"auto subgoal {subgoal_id} has no prior dependencies"
        else:
            if not dependencies and not anchors:
                return False, (
                    f"retrieval subgoal {subgoal_id} has neither anchor_entity "
                    "nor prior dependencies"
                )
            illegal_anchors = [
                anchor
                for anchor in anchors
                if _normalize_anchor_text(anchor) not in topic_by_norm
            ]
            if illegal_anchors:
                return False, (
                    f"retrieval subgoal {subgoal_id} has non-topic anchor(s): "
                    f"{', '.join(illegal_anchors)}"
                )

        prior_ids.add(subgoal_id)

    return True, ""


# --- subgoal ---

class SubgoalDecomposer:
    def __init__(self, model):
        self.model = model

    def _simple_fallback(self, question, topic_entities):
        """Return a safe one-subgoal plan when the one-shot planner is invalid."""
        return [{
            "id": "g1",
            "goal": question,
            "depend": [],
            "answer_type": "entity",
            "anchor_entity": topic_entity_names(topic_entities),
            "resolve_mode": "retrieval",
        }]

    def _normalize_subgoals(self, raw_subgoals):
        """Validate the minimum schema required by SubgoalState and search."""
        if not isinstance(raw_subgoals, list) or not raw_subgoals:
            return None

        normalized = []
        seen_ids = set()
        for position, raw_subgoal in enumerate(raw_subgoals, start=1):
            if not isinstance(raw_subgoal, dict):
                return None

            goal = str(raw_subgoal.get("goal", "") or "").strip()
            if not goal:
                return None

            subgoal_id = str(raw_subgoal.get("id", f"g{position}") or f"g{position}").strip()
            if not subgoal_id or subgoal_id in seen_ids:
                return None
            seen_ids.add(subgoal_id)

            dependencies = raw_subgoal.get("depend", [])
            if dependencies is None:
                dependencies = []
            if not isinstance(dependencies, list):
                return None
            dependencies = [str(dep).strip() for dep in dependencies if str(dep).strip()]
            if any(dep not in seen_ids for dep in dependencies):
                # A plan must be topologically ordered so active-subgoal tracking is deterministic.
                return None

            resolve_mode = str(raw_subgoal.get("resolve_mode", "retrieval") or "retrieval").strip().lower()
            if resolve_mode not in {"retrieval", "auto"}:
                return None

            normalized.append({
                "id": subgoal_id,
                "goal": goal,
                "depend": dependencies,
                "answer_type": str(raw_subgoal.get("answer_type", "entity") or "entity").strip() or "entity",
                "anchor_entity": raw_subgoal.get("anchor_entity"),
                "resolve_mode": resolve_mode,
            })
        return normalized

    def decompose(self, question, topic_entities):
        topic_text = format_top_entities(topic_entities)

        judge_prompt = complexity_judge_user_prompt.substitute(
            question=question,
            topic_entities=topic_text,
        )
        judge_output = self.model(
            system_prompt=complexity_judge_system_prompt,
            user_prompt=judge_prompt,
            max_tokens=256,
        )[0]
        judge_result = extract_json(judge_output)

        complexity = "simple"
        reason = ""
        if isinstance(judge_result, dict):
            label = str(judge_result.get("complexity", "") or "").strip().lower()
            if label in {"simple", "complex"}:
                complexity = label
                reason = str(judge_result.get("reason", "") or "").strip()
            else:
                logger.warning("Complexity judgment returned an invalid label; using simple fallback.")
        else:
            logger.warning("Complexity judgment returned invalid JSON; using simple fallback.")

        if complexity == "simple":
            prompt = simple_plan_user_prompt.substitute(
                question=question,
                topic_entities=topic_text,
            )
            output = self.model(
                system_prompt=simple_plan_system_prompt,
                user_prompt=prompt,
                max_tokens=512,
            )[0]
            parsed_plan = extract_json(output)
            plan = parsed_plan.get("subgoals") if isinstance(parsed_plan, dict) else None
        else:
            prompt = decompose_user_prompt.substitute(
                question=question,
                topic_entities=topic_text,
                decomposition_reason=reason or "The question requires multiple semantic constraints.",
            )
            output = self.model(
                system_prompt=decompose_system_prompt,
                user_prompt=prompt,
                max_tokens=1024,
            )[0]
            plan = extract_json(output)

        subgoals = None
        if isinstance(plan, list):
            subgoals = self._normalize_subgoals(plan)
            if subgoals:
                subgoals = repair_missing_constraint_anchors(subgoals, topic_entities)
            if subgoals:
                is_valid, validation_reason = validate_executable_plan(
                    subgoals,
                    topic_entities,
                )
                if not is_valid:
                    logger.warning("%s planning output is not executable: %s", complexity.capitalize(), validation_reason)
                    subgoals = None
                else:
                    unused_topics = unused_topic_entities(
                        subgoals,
                        topic_entities,
                    )
                    if unused_topics:
                        logger.info(
                            "%s planning output leaves topic entities "
                            "unused (allowed): %s",
                            complexity.capitalize(),
                            ", ".join(unused_topics),
                        )

        if not subgoals:
            logger.warning("%s planning output was invalid; using a simple fallback plan.", complexity.capitalize())
            subgoals = self._simple_fallback(question, topic_entities)

        return subgoals, complexity == "complex"


# --- subgoal ---

