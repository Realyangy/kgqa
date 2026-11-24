"""subgoal functions for SPAR-MCTS."""

import re
from utils import format_reasoning_path
import copy
import logging
from utils import topic_entity_names
from prompt_list import (
    complexity_judge_system_prompt,
    complexity_judge_user_prompt,
    decompose_system_prompt,
    decompose_user_prompt,
    simple_plan_system_prompt,
    simple_plan_user_prompt,
)
from utils import extract_json, format_top_entities, topic_entity_names
from collections import defaultdict
from mcts_utils import node_path_key
from utils import extract_candidate_indices, format_reasoning_path, is_valid_answer
from prompt_list import (
    subgoal_answer_selection_system_prompt, subgoal_answer_selection_user_prompt,
    subgoal_auto_resolve_system_prompt, subgoal_auto_resolve_user_prompt,
)
from sparql_utils import is_freebase_entity_id

logger = logging.getLogger(__name__)


# --- subgoal ---

MAX_SUPPORT_PATHS_PER_ANSWER = 2


def normalize_name(name):
    """Normalize entity or answer text for deterministic matching."""
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def progress_signature(global_state):
    """Return the plan-state changes that count as search progress.

    A new answer key or a derived coverage/status transition means the current
    plan has advanced. By contrast, stronger support for an already selected
    answer changes confidence only: it must not indefinitely postpone the
    full-tree check or replanning.
    """
    if not global_state:
        return ()

    signature = []
    for subgoal in global_state.subgoals:
        answers = [
            (
                str(answer.get("answer", "") or "").strip(),
                str(answer.get("entity_id", "") or "").strip(),
            )
            for answer in subgoal.get("answers", []) or []
        ]
        coverage = global_state.dependency_coverage(subgoal)
        signature.append((
            str(subgoal.get("id", "") or "").strip(),
            str(subgoal.get("status", "") or "").strip(),
            tuple(sorted(answers)),
            int(coverage.get("covered", 0) or 0),
            int(coverage.get("total", 0) or 0),
        ))
    return tuple(signature)


def anchor_values(anchor):
    """Return normalized non-empty anchor references from a scalar or list."""
    if isinstance(anchor, list):
        values = [str(item).strip() for item in anchor]
    elif anchor is None:
        values = []
    else:
        values = [str(anchor).strip()]
    return [value for value in values if value and value.lower() not in {"none", "null"}]


def parse_auto_resolve_answers(text):
    """Parse strict auto-resolve answer lines with prerequisite references."""
    if not isinstance(text, str) or not text.strip():
        return []

    cleaned = text.strip()
    if cleaned.upper() == "INSUFFICIENT":
        return []

    answers = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(
            r"answer:\s*(.+?)\s*\|\s*supported_by:\s*(.+)",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            return []

        answer = match.group(1).strip()
        if not answer or answer.lower() == "none":
            return []

        support_refs = []
        seen_refs = set()
        for raw_ref in match.group(2).split(";"):
            ref_match = re.fullmatch(r"([A-Za-z][\w-]*)\[(\d+)\]", raw_ref.strip())
            if not ref_match:
                return []
            ref = (ref_match.group(1), int(ref_match.group(2)))
            if ref not in seen_refs:
                seen_refs.add(ref)
                support_refs.append(ref)
        if not support_refs:
            return []
        answers.append({"answer": answer, "supported_by": support_refs})
    return answers


def format_target_subgoal(sg):
    """Format one subgoal for the candidate-answer selection prompt."""
    dep_str = ", ".join(sg.get("depend", [])) if sg.get("depend") else "None"
    refs = []
    for ans in sg.get("answers", []) or []:
        answer = str(ans.get("answer", "") or "").strip()
        if not answer:
            continue
        entity_id = str(ans.get("entity_id", "") or "").strip()
        refs.append(f"{answer}(id={entity_id})" if entity_id else answer)
    existing_answers = "; ".join(refs) if refs else "None"

    return (
        f"id: {sg.get('id', '')} | "
        f"status: {sg.get('status', 'unresolved')} | "
        f"goal: {sg.get('goal', '')} | "
        f"depend: {dep_str} | "
        f"answer_type: {sg.get('answer_type', 'entity')} | "
        f"existing_answers: {existing_answers}"
    )


def format_auto_target_subgoal(sg):
    """Format one auto subgoal with only the fields needed for derivation."""
    dependencies = "; ".join(sg.get("depend", []) or []) or "None"
    return (
        f"ID: {sg.get('id', '')} | "
        f"goal: {sg.get('goal', '')} | "
        f"depend: {dependencies} | "
        f"answer_type: {sg.get('answer_type', 'entity')}"
    )


def format_dependency_answers(global_state, active_sg):
    """Format resolved dependency subgoals for candidate-answer selection."""
    lines = []
    for dep_id in active_sg.get("depend", []):
        dep_sg = get_subgoal_by_id(global_state, dep_id) or {}
        dep_goal = str(dep_sg.get("goal", "") or dep_id).strip()

        compact_answers = []
        for ans in global_state.get_subgoal_answers(dep_id):
            answer = str(ans.get("answer", "") or "").strip()
            if not answer:
                continue

            entity_id = str(ans.get("entity_id", "") or "").strip()
            answer_ref = format_answer_ref(answer, entity_id)

            parents = []
            for parent in ans.get("parents", []) or []:
                parent_ref = format_answer_ref(parent.get("answer", ""), parent.get("entity_id", ""))
                if parent_ref:
                    parents.append(parent_ref)
            if parents:
                answer_ref = f"{answer_ref}({'; '.join(parents)})"

            compact_answers.append(answer_ref)

        answers_text = "; ".join(compact_answers) if compact_answers else "None"
        lines.append(f"{dep_id}: {dep_goal} | answers: {answers_text}")

    return "\n".join(lines) if lines else "None"


def format_bound_answer(ans, fallback="None"):
    """Format an answer together with its concrete dependency bindings."""
    answer = str(ans.get("answer", "") or "").strip()
    if not answer:
        return fallback
    parents = []
    for parent in ans.get("parents", []) or []:
        parent_answer = str(parent.get("answer", "") or "").strip()
        if parent_answer:
            parents.append(parent_answer)
    if parents:
        answer = f"{answer}({'; '.join(parents)})"
    return answer


def format_answer_for_log(ans):
    """Format one subgoal answer for concise progress logging."""
    answer = format_bound_answer(ans)

    confidence = float(ans.get("confidence", 0.0) or 0.0)
    return f"{answer}({confidence:.1f})"


def format_coverage_for_log(global_state, sg):
    """Format dependency coverage for concise progress logging."""
    coverage = global_state.dependency_coverage(sg)
    total = int(coverage.get("total", 0) or 0)
    if total <= 0:
        return ""
    covered = int(coverage.get("covered", 0) or 0)
    return f" | coverage={covered}/{total}"


def parents_as_list(parents):
    """Convert dependency binding dicts into serializable parent-answer records."""
    parent_items = []
    for subgoal_id, parent in (parents or {}).items():
        parent_items.append({
            "subgoal_id": subgoal_id,
            "answer": parent.get("answer", ""),
            "entity_id": parent.get("entity_id", ""),
        })
    return parent_items


def answer_identity(answer, entity_id=""):
    """Build a stable identity string for one answer."""
    entity_id = str(entity_id or "").strip()
    if entity_id:
        return f"id:{entity_id}"
    answer = normalize_name(answer)
    return f"name:{answer}" if answer else ""


def dependency_answer_key(ans):
    """Build the stable key used for dependency-answer coverage."""
    return answer_identity(ans.get("answer", ""), ans.get("entity_id", ""))


def answer_key(answer):
    """Build the stable key used for subgoal-answer deduplication and tree hits."""
    base_key = answer_identity(answer.get("answer", ""), answer.get("entity_id", ""))
    if not base_key:
        return ""

    parent_keys = []
    for parent in answer.get("parents", []) or []:
        parent_key = dependency_answer_key(parent)
        if parent_key:
            parent_keys.append(f"{parent.get('subgoal_id', '')}:{parent_key}")
    if parent_keys:
        return f"{base_key}|parents:{';'.join(sorted(parent_keys))}"
    return base_key


def format_answer_ref(answer, entity_id=None, fallback=""):
    """Format answer text with an optional entity-id fallback."""
    answer = str(answer or "").strip()
    entity_id = str(entity_id or "").strip()
    return answer or entity_id or fallback


def format_dependency_bindings(parents):
    """Format dependency bindings attached to one candidate answer."""
    refs = []
    for dep_id, parent in (parents or {}).items():
        answer = str(parent.get("answer", "") or "").strip()
        entity_id = str(parent.get("entity_id", "") or "").strip()
        if answer:
            refs.append(f"{dep_id}={answer}")
        elif entity_id:
            refs.append(f"{dep_id}=id:{entity_id}")
    return "; ".join(refs) if refs else "None"


def format_auto_dependency_answers(global_state, active_sg):
    """Format prerequisite answers as indexed single-line records for auto resolution."""
    answer_indices = {}
    for subgoal in getattr(global_state, "subgoals", []) or []:
        subgoal_id = str(subgoal.get("id", "") or "").strip()
        if not subgoal_id:
            continue
        answer_indices[subgoal_id] = {}
        for answer_index, answer in enumerate(subgoal.get("answers", []) or []):
            key = dependency_answer_key(answer)
            if key and key not in answer_indices[subgoal_id]:
                answer_indices[subgoal_id][key] = answer_index

    lines = []
    for dep_id in active_sg.get("depend", []) or []:
        dep_sg = get_subgoal_by_id(global_state, dep_id) or {}
        dep_goal = str(dep_sg.get("goal", "") or dep_id).strip()
        formatted_answers = []

        for answer_index, answer in enumerate(global_state.get_subgoal_answers(dep_id)):
            answer_text = str(answer.get("answer", "") or "").strip()
            if not answer_text:
                continue

            answer_text = f"[{answer_index}] {answer_text}"
            bindings = []
            for parent in answer.get("parents", []) or []:
                parent_id = str(parent.get("subgoal_id", "") or "").strip()
                parent_answer = str(parent.get("answer", "") or "").strip()
                if not parent_id or not parent_answer:
                    continue
                parent_key = dependency_answer_key(parent)
                parent_index = answer_indices.get(parent_id, {}).get(parent_key)
                parent_ref = (
                    f"{parent_id}[{parent_index}]"
                    if parent_index is not None
                    else parent_id
                )
                bindings.append(f"{parent_ref}={parent_answer}")
            if bindings:
                answer_text += f" (depends: {' & '.join(bindings)})"
            formatted_answers.append(answer_text)

        answers_text = "; ".join(formatted_answers) if formatted_answers else "None"
        lines.append(f"ID: {dep_id} | goal: {dep_goal} | answers: {answers_text}")

    return "\n".join(lines) if lines else "None"


def get_subgoal_by_id(global_state, sg_id):
    """Find one subgoal record by id."""
    for sg in getattr(global_state, "subgoals", []) or []:
        if sg.get("id") == sg_id:
            return sg
    return None


def answer_evidence_matches_node(ans, child_node):
    """Check whether an answer's evidence path appears on a tree node."""
    evidence_texts = answer_evidence_texts(ans)
    if not evidence_texts:
        return True

    node_texts = [text for text in node_path_texts(child_node) if text]
    return any(
        evidence_text in node_text
        for evidence_text in evidence_texts
        for node_text in node_texts
    )


def answer_evidence_texts(ans):
    texts = []
    for evidence_path in ans.get("evidence_paths", []) or []:
        evidence_text = str(evidence_path or "").strip()
        if evidence_text and evidence_text != "None":
            texts.append(evidence_text)

    evidence_text = str(ans.get("evidence_path", "") or "").strip()
    if evidence_text and evidence_text != "None":
        texts.append(evidence_text)

    return list(dict.fromkeys(texts))


def rank_support_evidence(items, default_score=0.0):
    """Return the two strongest distinct path records for one answer."""
    ranked = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "") or "").strip()
        evidence_path = str(item.get("evidence_path", "") or "").strip()
        if (not path or path == "None") and (not evidence_path or evidence_path == "None"):
            continue
        raw_score = item["score"] if "score" in item else default_score
        try:
            score = float(raw_score or 0.0)
        except (TypeError, ValueError):
            score = float(default_score or 0.0)
        ranked.append({
            "path": path if path and path != "None" else evidence_path,
            "evidence_path": evidence_path if evidence_path and evidence_path != "None" else path,
            "score": max(0.0, min(1.0, score)),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    seen = set()
    for item in ranked:
        path_key = normalize_name(item["path"] or item["evidence_path"])
        if not path_key or path_key in seen:
            continue
        seen.add(path_key)
        selected.append(item)
        if len(selected) >= MAX_SUPPORT_PATHS_PER_ANSWER:
            break
    return selected


def set_answer_support_evidence(answer, items, default_score=0.0):
    """Normalize support paths and write their derived fields to an answer."""
    selected = rank_support_evidence(items, default_score=default_score)
    evidence_paths = list(dict.fromkeys(
        item["evidence_path"] for item in selected if item["evidence_path"]
    ))
    answer["support_evidence"] = selected
    answer["evidence_paths"] = evidence_paths
    answer["evidence_path"] = evidence_paths[0] if evidence_paths else ""
    answer["confidence"] = max(
        (item["score"] for item in selected),
        default=max(0.0, min(1.0, float(default_score or 0.0))),
    )
    return selected


def node_path_texts(child_node):
    walk_path = getattr(child_node, "walk_path", None) or []
    evidence_path = getattr(child_node, "evidence_path", None) or walk_path
    return [
        format_reasoning_path(walk_path, fallback_sep=", ") if walk_path else "",
        format_reasoning_path(evidence_path, fallback_sep=", ") if evidence_path else "",
    ]


# --- subgoal ---

class SubgoalState:
    """Subgoal runtime state machine.

    The only persisted runtime fields are ``answers`` and ``status``:
    ``unresolved`` (no evidence-grounded answer), ``partial`` (answer evidence exists
    but current dependencies are incomplete), and ``resolved`` (sufficiently
    supported for the current dependency-answer set). Active eligibility, dependency
    coverage, and downstream consumption are derived from those fields and
    answer bindings; they are never stored as competing mutable state.

    Each subgoal also contains these plan/data fields:
        id: 子目标ID，如 "g1", "g2"
        goal: 子目标任务描述文本
        answer_type: 答案类型
        depend: 依赖的前置子目标ID列表
        answers: 答案列表，每个答案包含 answer, entity_id, evidence_path
        anchor_entity: 锚点实体名称（该子目标激活时，此实体参与 MCTS 扩展）
    """

    def __init__(self, subgoals):
        self.replace_subgoals(subgoals)

    def _initialize_subgoals(self, subgoals):
        for sg in subgoals:
            sg.setdefault("status", "unresolved")
            sg.setdefault("answers", [])
            sg.setdefault("anchor_entity", None)
            sg.setdefault("resolve_mode", "retrieval")
            # Cached fields from older runs are derived, not runtime state.
            sg.pop("coverage", None)
            sg.pop("answer_set_status", None)
        # Plans are topologically ordered, so one forward pass restores status
        # from answer evidence and dependency bindings.
        for sg in subgoals:
            self._refresh_subgoal_status(sg)

    def replace_subgoals(self, subgoals):
        """Install a complete validated plan and initialize its runtime fields."""
        self.subgoals = copy.deepcopy(subgoals) if subgoals else []
        self._initialize_subgoals(self.subgoals)

    def has_consumed_answers(self, sg_id):
        """Whether a resolved subgoal answer is used by a descendant."""
        source = next(
            (sg for sg in self.subgoals if sg.get("id") == sg_id),
            None,
        )
        if not source or source.get("status") != "resolved":
            return False

        source_keys = {
            dependency_answer_key(answer)
            for answer in source.get("answers", []) or []
            if dependency_answer_key(answer)
        }
        if not source_keys:
            return False

        for consumer in self.subgoals:
            if consumer.get("status") not in {"partial", "resolved"}:
                continue
            if sg_id not in (consumer.get("depend", []) or []):
                continue
            for answer in consumer.get("answers", []) or []:
                for parent in answer.get("parents", []) or []:
                    if parent.get("subgoal_id") != sg_id:
                        continue
                    parent_key = dependency_answer_key(parent)
                    if parent_key and parent_key in source_keys:
                        return True
        return False

    def dependency_coverage(self, sg):
        """Return retrieval coverage derived from current answer bindings."""
        if not self._uses_dependency_coverage(sg):
            return {}
        return self._compute_dependency_coverage(sg)

    def _dependencies_ready(self, sg):
        """Allow retrieval to consume partial answers; keep auto strict."""
        allowed_statuses = (
            {"resolved"}
            if sg.get("resolve_mode", "retrieval") == "auto"
            else {"partial", "resolved"}
        )
        by_id = {
            subgoal.get("id"): subgoal
            for subgoal in self.subgoals
            if subgoal.get("id")
        }
        for dependency_id in sg.get("depend", []) or []:
            dependency = by_id.get(dependency_id)
            if (
                not dependency
                or dependency.get("status") not in allowed_statuses
                or not dependency.get("answers")
            ):
                return False
        return True

    def get_active_subgoals(self):
        """Return unfinished subgoals whose dependencies are ready."""
        active = []
        for sg in self.subgoals:
            if sg.get("status") == "resolved":
                continue
            if self._dependencies_ready(sg):
                active.append(sg)
        return active

    def _format_answer_refs(self, answers):
        refs = []
        for ans in answers or []:
            answer = format_bound_answer(ans, fallback="")
            if not answer:
                continue
            entity_id = str(ans.get("entity_id", "") or "").strip()
            if entity_id:
                refs.append(f"{answer}(id={entity_id})")
            else:
                refs.append(answer)
        return "; ".join(refs) if refs else "None"

    def _format_subgoal_line(self, sg, include_answers=False):
        dep_str = ", ".join(sg.get("depend", [])) if sg.get("depend") else "None"
        line = (
            f"{sg.get('id', '')}: {sg.get('goal', '')} | "
            f"depend: {dep_str} | answer_type: {sg.get('answer_type', 'entity')}"
        )
        if include_answers:
            line += f" | existing_answers: {self._format_answer_refs(sg.get('answers', []))}"
            coverage = self.dependency_coverage(sg)
            if coverage.get("total", 0) > 0:
                line += f" | coverage: {coverage.get('covered', 0)}/{coverage.get('total', 0)}"
        return line

    def format_for_prompt(self, all_active=False):
        if not self.subgoals:
            return "None"

        # 静态分解不维护运行时状态，因此把完整初始计划作为active目标展示。
        active = self.subgoals if all_active else self.get_active_subgoals()
        available = [] if all_active else [
            sg
            for sg in self.subgoals
            if sg.get("status") in {"partial", "resolved"}
            and sg.get("answers")
        ]

        required_dep_ids = []
        seen_dep_ids = set()
        for sg in active:
            for dep_id in sg.get("depend", []) or []:
                if dep_id in seen_dep_ids:
                    continue
                seen_dep_ids.add(dep_id)
                required_dep_ids.append(dep_id)

        required_answers = []
        for sg in available:
            if sg.get("id") in seen_dep_ids:
                required_answers.append(sg)
        resolved_background = [
            sg
            for sg in available
            if sg.get("status") == "resolved"
            and sg.get("id") not in seen_dep_ids
        ]

        lines = []
        lines.append("[active]")
        if active:
            for sg in active:
                lines.append(self._format_subgoal_line(
                    sg,
                    include_answers=sg.get("status") == "partial",
                ))
        else:
            lines.append("None")

        lines.append("[required_answers]")
        if required_answers:
            sg_by_id = {sg.get("id"): sg for sg in required_answers}
            for dep_id in required_dep_ids:
                sg = sg_by_id.get(dep_id)
                if sg:
                    lines.append(self._format_subgoal_line(sg, include_answers=True))
        else:
            lines.append("None")

        lines.append("[resolved_background]")
        if resolved_background:
            for sg in resolved_background:
                lines.append(self._format_subgoal_line(sg, include_answers=True))
        else:
            lines.append("None")

        return "\n".join(lines)

    def format_for_early_stop_prompt(self):
        """Format evidence-grounded resolved/partial answers for early stopping."""
        if not self.subgoals:
            return "None"

        sections = []
        for status in ("resolved", "partial"):
            lines = []
            for subgoal in self.subgoals:
                if subgoal.get("status") != status:
                    continue

                answers = []
                seen = set()
                for answer in subgoal.get("answers", []) or []:
                    answer_text = format_bound_answer(answer, fallback="")
                    if not answer_text or answer_text in seen:
                        continue
                    seen.add(answer_text)
                    answers.append(answer_text)

                goal = str(subgoal.get("goal", "") or "").strip()
                subgoal_id = str(subgoal.get("id", "") or "").strip()
                if not goal or not answers:
                    continue
                label = f"{subgoal_id}: " if subgoal_id else ""
                lines.append(
                    f"{label}{goal} | answers: {'; '.join(answers)}"
                )

            if lines:
                sections.append("\n".join([f"[{status}]", *lines]))

        return "\n".join(sections) if sections else "None"

    def format_answer_context(self):
        """Format resolved and partial subgoal answers for evidence prompts."""
        if not self.subgoals:
            return "None"

        lines = []
        for subgoal in self.subgoals:
            if subgoal.get("status") not in {"resolved", "partial"}:
                continue
            answers = []
            seen = set()
            for answer in subgoal.get("answers", []) or []:
                answer_text = str(answer.get("answer", "") or "").strip()
                if not answer_text or answer_text in seen:
                    continue
                seen.add(answer_text)
                answers.append(answer_text)
            goal = str(subgoal.get("goal", "") or "").strip()
            if goal and answers:
                lines.append(f"{goal}: {', '.join(answers)}")
        return "\n".join(lines) if lines else "None"

    def record_answers(self, sg_id, answers):
        """Record evidence-grounded answers and transition the state machine."""
        for sg in self.subgoals:
            if sg["id"] == sg_id:
                sg.setdefault("answers", [])
                added_new_answer = False
                answer_by_key = {}
                for existing in sg["answers"]:
                    key = answer_key(existing)
                    if not key:
                        continue
                    self._normalize_support_evidence(existing)
                    answer_by_key[key] = existing
                for ans in answers:
                    key = answer_key(ans)
                    if not key:
                        continue
                    if key in answer_by_key:
                        self.merge_answer_evidence(answer_by_key[key], ans)
                        continue
                    if self.has_consumed_answers(sg_id):
                        continue
                    self._normalize_support_evidence(ans)
                    answer_by_key[key] = ans
                    sg["answers"].append(ans)
                    added_new_answer = True
                self._refresh_subgoal_status(sg)
                if added_new_answer:
                    self._refresh_downstream_after_answer_growth(sg_id)
                return True
        return False

    def _refresh_downstream_after_answer_growth(self, changed_sg_id):
        """Invalidate descendants whose dependency answer set has grown.

        Retrieval descendants retain their valid old evidence but become partial
        until their dependency coverage includes the newly added answer. Auto
        descendants are derived from complete dependency sets, so their old
        results are cleared and recomputed once their dependencies recover.
        The plan is topologically ordered, allowing a single forward pass to
        propagate the invalidation through all descendants.
        """
        affected_ids = {str(changed_sg_id)}

        for sg in self.subgoals:
            sg_id = str(sg.get("id", "") or "")
            dependencies = sg.get("depend", []) or []
            if not sg_id or not any(dep in affected_ids for dep in dependencies):
                continue

            affected_ids.add(sg_id)
            if sg.get("resolve_mode", "retrieval") == "auto":
                sg["answers"] = []
                sg["status"] = "unresolved"
                continue

            # Retrieval answers remain useful evidence, but an affected
            # descendant cannot stay fully resolved until coverage is restored.
            if sg.get("answers"):
                self._refresh_subgoal_status(sg)
                if not self._dependencies_are_resolved(sg):
                    sg["status"] = "partial"
            else:
                sg["status"] = "unresolved"

    def _normalize_support_evidence(self, ans):
        """Keep at most two highest-scoring, distinct support paths per answer."""
        confidence = max(0.0, min(1.0, float(ans.get("confidence", 0.0) or 0.0)))
        records = list(ans.get("support_evidence", []) or [])
        if not records:
            records = [
                {
                    "path": str(path).strip(),
                    "evidence_path": str(path).strip(),
                    "score": confidence,
                }
                for path in answer_evidence_texts(ans)
                if str(path or "").strip()
            ]

        return set_answer_support_evidence(
            ans,
            records,
            default_score=confidence,
        )

    def merge_answer_evidence(self, target, source):
        """Merge new support into an existing evidence-grounded answer record."""
        target_support = copy.deepcopy(self._normalize_support_evidence(target))
        source_support = copy.deepcopy(self._normalize_support_evidence(source))
        target["support_evidence"] = target_support + source_support
        self._normalize_support_evidence(target)

        # 没有路径证据时也保留答案级最高置信度。
        target_conf = float(target.get("confidence", 0.0) or 0.0)
        source_conf = float(source.get("confidence", 0.0) or 0.0)
        target["confidence"] = max(target_conf, source_conf)
        target_parents = target.setdefault("parents", [])
        seen_parent_keys = {
            (parent.get("subgoal_id", ""), dependency_answer_key(parent))
            for parent in target_parents
        }
        for parent in source.get("parents", []) or []:
            parent_key = (parent.get("subgoal_id", ""), dependency_answer_key(parent))
            if parent_key in seen_parent_keys:
                continue
            seen_parent_keys.add(parent_key)
            target_parents.append(parent)

    def _uses_dependency_coverage(self, sg):
        return bool(sg.get("depend")) and sg.get("resolve_mode", "retrieval") != "auto"

    def _dependencies_are_resolved(self, sg):
        return all(self.has_subgoal_resolved(dep_id) for dep_id in sg.get("depend", []) or [])

    def _compute_dependency_coverage(self, sg):
        by_dependency = {}
        total_required = 0
        total_covered = 0

        for dep_id in sg.get("depend", []) or []:
            required = {
                dependency_answer_key(ans)
                for ans in self.get_subgoal_answers(dep_id)
                if dependency_answer_key(ans)
            }
            covered = set()
            for ans in sg.get("answers", []) or []:
                for parent in ans.get("parents", []) or []:
                    if parent.get("subgoal_id") != dep_id:
                        continue
                    parent_key = dependency_answer_key(parent)
                    if parent_key in required:
                        covered.add(parent_key)

            by_dependency[dep_id] = {
                "covered": len(covered),
                "total": len(required),
            }
            total_required += len(required)
            total_covered += len(covered)

        return {
            "covered": total_covered,
            "total": total_required,
            "by_dependency": by_dependency,
        }

    def _refresh_subgoal_status(self, sg):
        if not sg.get("answers"):
            sg["status"] = "unresolved"
            return

        if not self._uses_dependency_coverage(sg):
            sg["status"] = "resolved"
            return

        coverage = self._compute_dependency_coverage(sg)
        if not self._dependencies_are_resolved(sg):
            sg["status"] = "partial"
        elif coverage["total"] == 0 or coverage["covered"] >= coverage["total"]:
            sg["status"] = "resolved"
        else:
            sg["status"] = "partial"

    def has_subgoal_resolved(self, sg_id):
        for sg in self.subgoals:
            if sg["id"] == sg_id and sg.get("status") == "resolved":
                return True
        return False


    def get_subgoal_answers(self, sg_id):
        for sg in self.subgoals:
            if sg["id"] == sg_id:
                return sg["answers"]
        return []

    def is_subgoal_active(self, sg_id):
        """检查子目标在当前状态中是否active"""
        for sg in self.subgoals:
            if sg["id"] == sg_id:
                if sg.get("status") == "resolved":
                    return False
                return self._dependencies_ready(sg)
        return False

    def all_resolved(self):
        if not self.subgoals:
            return True
        return all(sg.get("status") == "resolved" for sg in self.subgoals)


# --- subgoal ---

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

def _dependency_match_rank(candidate, dep_answer):
    """Rank an exact dependency match by match type and path proximity."""
    dep_eid = str(dep_answer.get("entity_id", "") or "").strip()
    dep_name = normalize_name(dep_answer.get("answer", ""))
    best_rank = None

    # A larger path position is closer to the candidate endpoint. Entity-ID
    # matches take precedence over exact normalized-name matches.
    for position, entity in enumerate(candidate.get("path_entities", []) or []):
        entity_id = str(entity.get("entity_id", "") or "").strip()
        entity_name = normalize_name(entity.get("entity_name", ""))
        if dep_eid and entity_id == dep_eid:
            rank = (2, position)
        elif dep_name and entity_name == dep_name:
            rank = (1, position)
        else:
            continue
        if best_rank is None or rank > best_rank:
            best_rank = rank

    return best_rank


def _find_dependency_bindings(global_state, sg, candidate):
    # 提取sg的依赖子目标
    deps = sg["depend"]
    if not deps:
        return {}

    # 遍历依赖子目标，优先绑定精确ID匹配；同类匹配中选择最靠近候选端点的答案。
    parents = {}
    for dep_id in deps:
        dep_answers = global_state.get_subgoal_answers(dep_id)
        matches = []
        for dep_ans in dep_answers:
            rank = _dependency_match_rank(candidate, dep_ans)
            if rank is not None:
                matches.append((rank, dep_ans))
        if not matches:
            return None
        _, matched_answer = max(matches, key=lambda item: item[0])
        matched_dep = {
            "answer": matched_answer.get("answer", ""),
            "entity_id": matched_answer.get("entity_id", ""),
        }
        parents[dep_id] = matched_dep

    return parents


def _candidate_support_path(candidate):
    return (
        candidate.get("support_path")
        or candidate.get("walk_path")
        or "None"
    )


def _candidate_support_record(candidate):
    confidence = float(candidate.get("path_value", 0.0) or 0.0)
    return {
        "path": _candidate_support_path(candidate),
        "evidence_path": candidate.get("evidence_path") or candidate.get("walk_path") or "",
        "score": max(0.0, min(1.0, confidence)),
    }


def _auto_answer_confidence(bound_answers):
    """Propagate the weakest prerequisite confidence into a derived answer."""
    return min(
        max(
            0.0,
            min(
                1.0,
                float(bound_answer.get("confidence", 0.0) or 0.0),
            ),
        )
        for _, _, bound_answer in bound_answers
    )


def _refresh_candidate_item_evidence(item):
    record = item["record"]
    selected = set_answer_support_evidence(
        record,
        item.get("support_paths", []),
        default_score=record.get("confidence", 0.0),
    )
    item["support_paths"] = copy.deepcopy(selected)


def _make_candidate_answer_item(candidate, answer, entity_id, answer_source, parents):
    candidate_index = candidate.get("candidate_index")
    confidence = float(candidate.get("path_value", 0.0) or 0.0)
    parents = parents or {}

    record = {
        "answer": str(answer or "").strip(),
        "entity_id": str(entity_id or "").strip(),
        "answer_source": answer_source,
        "parents": parents_as_list(parents),
        "candidate_index": candidate_index,
        "evidence_path": "",
        "evidence_paths": [],
        "confidence": max(0.0, min(1.0, confidence)),
    }

    item = {
        "key": answer_key(record),
        "record": record,
        "support_paths": [_candidate_support_record(candidate)],
    }
    _refresh_candidate_item_evidence(item)
    return item


def _iter_candidate_answer_items(candidate, parents):
    name = str(candidate.get("entity_name", "") or "").strip()
    entity_id = str(candidate.get("entity_id", "") or "").strip()
    yield _make_candidate_answer_item(
        candidate=candidate,
        answer=name,
        entity_id=entity_id,
        answer_source="candidate",
        parents=parents,
    )

    for parent in (parents or {}).values():
        parent_answer = str(parent.get("answer", "") or "").strip()
        parent_entity_id = str(parent.get("entity_id", "") or "").strip()
        yield _make_candidate_answer_item(
            candidate=candidate,
            answer=parent_answer,
            entity_id=parent_entity_id,
            answer_source="dependency",
            parents=parents,
        )


def _merge_candidate_answer_item(item, grouped_items):
    record = item["record"]
    if not is_valid_answer(record.get("answer", "")):
        return

    existing = grouped_items.get(item["key"])
    if existing is None:
        grouped_items[item["key"]] = item
        return

    existing["support_paths"].extend(item["support_paths"])
    _refresh_candidate_item_evidence(existing)


def _format_candidate_answer_items(grouped_items, existing_answer_keys):
    item_lines = []
    item_lookup = {}

    for key, item in grouped_items.items():
        if key in existing_answer_keys:
            continue

        record = item["record"]
        candidate_answer_index = len(item_lookup)
        item_lookup[candidate_answer_index] = record
        answer_ref = format_answer_ref(record["answer"], record["entity_id"], fallback="None")
        parents = {
            parent.get("subgoal_id", ""): {
                "answer": parent.get("answer", ""),
                "entity_id": parent.get("entity_id", ""),
            }
            for parent in record.get("parents", [])
            if parent.get("subgoal_id")
        }
        dependency_text = format_dependency_bindings(parents)
        item_lines.append(
            f"[{candidate_answer_index}] answer: {answer_ref} | depends: {dependency_text}"
        )
        item_lines.append("supported_by_paths:")
        for path_index, support in enumerate(item["support_paths"], start=1):
            item_lines.append(f"({path_index}) {support['path']}")

    return item_lines, item_lookup


def build_new_answer_candidates(children):
    candidates = []
    seen_record_keys = set()

    for child in children or []:
        evidence_path = list(getattr(child, "evidence_path", []) or [])
        record_key = node_path_key(child)
        if record_key in seen_record_keys:
            continue
        seen_record_keys.add(record_key)

        entity = dict(getattr(child, "entity_info", None) or {})
        entity_id = str(entity.get("entity_id", "")).strip()
        entity_name = str(entity.get("entity_name", "")).strip()
        if not entity_id or not entity_name or entity_name == "Unknown":
            continue

        walk_path = list(getattr(child, "walk_path", []) or [])
        evidence_path = evidence_path or walk_path
        path_value = float(getattr(child, "path_value", 0.0) or 0.0)
        walk_path_text = format_reasoning_path(walk_path, fallback_sep=", ") if walk_path else "None"
        evidence_path_text = format_reasoning_path(evidence_path, fallback_sep=", ") if evidence_path else "None"

        path_entities = []
        seen_entity_ids = set()
        node = child
        while node is not None:
            ne = getattr(node, "entity_info", None) or {}
            node_entity_id = str(ne.get("entity_id", "")).strip()
            node_entity_name = str(ne.get("entity_name", "")).strip()

            if node_entity_id and node_entity_name and node_entity_name != "Unknown" and node_entity_id not in seen_entity_ids:
                seen_entity_ids.add(node_entity_id)
                path_entities.append({
                    "entity_id": node_entity_id,
                    "entity_name": node_entity_name,
                })
            node = getattr(node, "parent", None)
        path_entities.reverse()

        candidates.append({
            "candidate_index": len(candidates),
            "entity_id": entity_id,
            "entity_name": entity_name,
            "path_value": round(path_value, 1),
            "walk_path": walk_path_text,
            "evidence_path": evidence_path_text,
            "support_path": walk_path_text if walk_path_text != "None" else evidence_path_text,
            "path_entities": path_entities,
        })

    return candidates


class SubgoalUpdater:
    """负责子目标状态更新和自动解析。"""

    def __init__(self, model):
        self.model = model

    def update(
        self,
        global_state,
        answer_candidates,
    ):
        if not self.model or not answer_candidates:
            return global_state

        # 保留格式合法的候选路径；是否选择为子目标答案交给LLM判断。
        filtered_candidates = [cand for cand in answer_candidates if isinstance(cand.get("candidate_index"), int)]
        if not filtered_candidates:
            return global_state

        # Freeze the eligible retrieval targets before recording any answers.
        # Newly activated downstream retrieval subgoals wait for the next
        # search iteration and its newly expanded evidence.
        targets = (
            [
                sg
                for sg in global_state.get_active_subgoals()
                if sg["resolve_mode"] != "auto"
            ]
            + [
                sg
                for sg in global_state.subgoals
                if (
                    sg["status"] == "resolved"
                    and sg["resolve_mode"] != "auto"
                )
            ]
        )
        if not targets:
            return global_state

        # 遍历active和resolved子目标，检查是否有新节点能够回答它
        grouped = defaultdict(list)
        for sg in targets:
            sg_id = sg.get("id", "")
            if not sg_id:
                continue

            # 格式化当前子目标
            format_goal = format_target_subgoal(sg)
            dependency_answers = format_dependency_answers(global_state, sg)

            # 合并相同“答案+依赖绑定”的所有支持路径，再过滤已经保存的答案。
            grouped_candidate_answers = {}

            for cand in filtered_candidates:
                parents = _find_dependency_bindings(global_state, sg, cand)
                if parents is None:
                    continue

                for item in _iter_candidate_answer_items(cand, parents):
                    _merge_candidate_answer_item(item, grouped_candidate_answers)

            existing_answers_by_key = {}
            for existing_answer in sg.get("answers", []) or []:
                existing_key = answer_key(existing_answer)
                if existing_key:
                    existing_answers_by_key[existing_key] = existing_answer

            # The answer is already grounded in KG evidence. If a later search path
            # supports the same answer and dependency binding, merge the stronger
            # evidence directly instead of asking the LLM to select it again.
            for existing_key, existing_answer in existing_answers_by_key.items():
                candidate_item = grouped_candidate_answers.get(existing_key)
                if candidate_item:
                    global_state.merge_answer_evidence(
                        existing_answer,
                        candidate_item["record"],
                    )

            # Once a descendant consumed this subgoal, only strengthen an
            # existing answer; do not ask the LLM to add a new answer key.
            if global_state.has_consumed_answers(sg_id):
                continue

            candidate_answer_lines, candidate_answer_lookup = _format_candidate_answer_items(
                grouped_candidate_answers,
                set(existing_answers_by_key),
            )

            if not candidate_answer_lines:
                continue

            candidate_answers = "\n".join(candidate_answer_lines)

            user_prompt = subgoal_answer_selection_user_prompt.substitute(
                target_subgoal=format_goal,
                dependency_answers=dependency_answers,
                candidate_answers=candidate_answers,
            )
            output = self.model(system_prompt=subgoal_answer_selection_system_prompt, user_prompt=user_prompt, max_tokens=256)[0]
            selected_indices = extract_candidate_indices(output, set(candidate_answer_lookup))
            if not selected_indices:
                continue

            # 解析LLM输出的候选答案索引，提取能够回答sg的候选答案记录
            for candidate_answer_index in selected_indices:
                answer_record = candidate_answer_lookup.get(candidate_answer_index)
                if not answer_record:
                    continue
                grouped[sg_id].append(copy.deepcopy(answer_record))

        ## 3.将答案写入全局状态，并更新
        for sg_id, ans_list in grouped.items():
            if not ans_list:
                continue

            # 把active子目标标记为resolved，或补充resolved子目标的答案。
            if (
                global_state.is_subgoal_active(sg_id)
                or global_state.has_subgoal_resolved(sg_id)
            ):
                global_state.record_answers(sg_id, ans_list)

        return global_state

    def auto_resolve(self, global_state):
        if not self.model:
            return global_state

        # 提取active状态且resolve_mode为"auto"的子目标，这些可以根据依赖子目标直接回答，无需KG检索
        active = global_state.get_active_subgoals()
        if not active:
            return global_state

        auto_sgs = [sg for sg in active if sg.get("resolve_mode", "retrieval") == "auto"]
        if not auto_sgs:
            return global_state

        new_answers = defaultdict(list)
        for sg in auto_sgs:
            sg_id = sg.get("id", "")
            sg_deps = sg.get("depend", [])

            # 若无依赖子目标，那么无法直接回答，可能是需要检索但是误标记了，直接跳过
            if not sg_deps:
                continue

            # 只有所有依赖子目标都完整回答后，auto子目标才可以推理。
            # 依赖仍为partial/unresolved时，避免基于不完整证据做比较、计数或筛选。
            if not all(global_state.has_subgoal_resolved(dep_id) for dep_id in sg_deps):
                continue

            # 收集依赖子目标的答案
            dep_answer_lookup = {}
            for dep_id in sg_deps:
                dep_answer_lookup[dep_id] = {}
                for answer_index, ans in enumerate(global_state.get_subgoal_answers(dep_id)):
                    answer_text = normalize_name(ans.get("answer", ""))
                    if not answer_text:
                        continue
                    dep_answer_lookup[dep_id][answer_index] = ans
            if not any(dep_answer_lookup.values()):
                continue

            target_subgoal_text = format_auto_target_subgoal(sg)
            dependency_text = format_auto_dependency_answers(global_state, sg)
            user_prompt = subgoal_auto_resolve_user_prompt.substitute(
                target_subgoal=target_subgoal_text,
                dependency_answers=dependency_text,
            )
            output = self.model(system_prompt=subgoal_auto_resolve_system_prompt, user_prompt=user_prompt, max_tokens=512)[0]
            resolved_list = parse_auto_resolve_answers(output)
            if not resolved_list:
                continue

            for item in resolved_list:
                answer_text = str(item.get("answer", "") or "").strip()
                if not answer_text:
                    continue

                bound_answers = []
                invalid_reference = False
                for dep_id, answer_index in item.get("supported_by", []):
                    bound_answer = dep_answer_lookup.get(dep_id, {}).get(answer_index)
                    if bound_answer is None:
                        invalid_reference = True
                        break
                    bound_answers.append((dep_id, answer_index, bound_answer))
                if invalid_reference or not bound_answers:
                    continue

                referenced_dep_ids = {dep_id for dep_id, _, _ in bound_answers}
                missing_dep_ids = [dep_id for dep_id in sg_deps if dep_id not in referenced_dep_ids]
                if missing_dep_ids:
                    continue

                source_answer = next(
                    (
                        bound_answer
                        for _, _, bound_answer in bound_answers
                        if normalize_name(bound_answer.get("answer", "")) == normalize_name(answer_text)
                    ),
                    None,
                )
                answer = str(source_answer.get("answer", "") if source_answer else answer_text).strip()
                if not is_valid_answer(answer):
                    continue

                parents = []
                evidence_paths = []
                for dep_id, _, bound_answer in bound_answers:
                    parents.append({
                        "subgoal_id": dep_id,
                        "answer": str(bound_answer.get("answer", "") or "").strip(),
                        "entity_id": str(bound_answer.get("entity_id", "") or "").strip(),
                    })
                    for evidence_path in answer_evidence_texts(bound_answer):
                        if evidence_path and evidence_path not in evidence_paths:
                            evidence_paths.append(evidence_path)

                entity_id = str(source_answer.get("entity_id", "") if source_answer else "").strip()
                new_answers[sg_id].append({
                    "answer": answer,
                    "entity_id": entity_id,
                    "answer_source": "auto_resolve",
                    "parents": parents,
                    "supported_by": [
                        f"{dep_id}[{answer_index}]"
                        for dep_id, answer_index, _ in bound_answers
                    ],
                    "candidate_index": -1,
                    "evidence_path": evidence_paths[0] if evidence_paths else "",
                    "evidence_paths": evidence_paths,
                    "confidence": _auto_answer_confidence(bound_answers),
                })

        # 写入全局状态
        for sg_id, ans_list in new_answers.items():
            if not ans_list:
                continue
            if global_state.is_subgoal_active(sg_id):
                global_state.record_answers(sg_id, ans_list)

        return global_state


def update_until_stable(global_state, updater, answer_candidates):
    """Apply answer selection and auto resolution until the state stabilizes."""
    if not global_state or not updater:
        return global_state, False


    initial_signature = progress_signature(global_state)
    for _ in range(max(1, len(global_state.subgoals))):
        before_round = progress_signature(global_state)
        global_state = updater.update(global_state, answer_candidates)
        global_state = updater.auto_resolve(global_state)
        if progress_signature(global_state) == before_round:
            break
    return global_state, progress_signature(global_state) != initial_signature


# --- subgoal ---

def _retrieval_answer_hit_keys(global_state):
    keys = set()
    for subgoal in getattr(global_state, "subgoals", []) or []:
        if (
            subgoal.get("status") not in {"partial", "resolved"}
            or subgoal.get("resolve_mode", "retrieval") == "auto"
        ):
            continue
        for answer in subgoal.get("answers", []) or []:
            key = answer_key(answer)
            if key:
                keys.add((subgoal.get("id"), key))
    return keys


def _retrieval_dependency_ids(global_state, dependency_ids):
    """Resolve auto dependencies to the retrieval subgoals that produced them."""
    by_id = {
        subgoal.get("id"): subgoal
        for subgoal in getattr(global_state, "subgoals", []) or []
        if subgoal.get("id")
    }
    retrieval_ids = set()
    pending = list(dependency_ids or [])
    visited = set()
    while pending:
        subgoal_id = pending.pop()
        if not subgoal_id or subgoal_id in visited:
            continue
        visited.add(subgoal_id)
        subgoal = by_id.get(subgoal_id)
        if not subgoal:
            continue
        if subgoal.get("resolve_mode", "retrieval") != "auto":
            retrieval_ids.add(subgoal_id)
            continue
        pending.extend(subgoal.get("depend", []) or [])
    return retrieval_ids


def _root_matches_anchor(root, anchors):
    root_name = normalize_name(root.entity_name)
    root_id = normalize_name(root.entity_id)
    for anchor in anchors:
        anchor_text = normalize_name(anchor)
        if not anchor_text:
            continue
        if anchor_text in {root_name, root_id}:
            return True
        if root_name and (
            anchor_text in root_name or root_name in anchor_text
        ):
            return True
    return False


def _root_has_dependency_hit(root, dependency_ids):
    dependency_ids = set(dependency_ids or [])
    stack = [root]
    while stack:
        node = stack.pop()
        if any(
            hit.get("subgoal_id") in dependency_ids
            for hit in getattr(node, "subgoal_answer_hits", []) or []
        ):
            return True
        stack.extend(node.children)
    return False


def active_root_ids(root, global_state):
    if not global_state:
        return set()

    active_ids = set()
    for subgoal in global_state.get_active_subgoals():
        if subgoal.get("resolve_mode", "retrieval") == "auto":
            continue

        anchors = anchor_values(subgoal.get("anchor_entity"))
        for candidate_root in root.children:
            if anchors and _root_matches_anchor(candidate_root, anchors):
                active_ids.add(candidate_root.entity_id)

        dependencies = _retrieval_dependency_ids(
            global_state,
            subgoal.get("depend") or [],
        )
        for candidate_root in root.children:
            if dependencies and _root_has_dependency_hit(
                candidate_root,
                dependencies,
            ):
                active_ids.add(candidate_root.entity_id)
    return active_ids


def refresh_root_activity(root, global_state, enable_dynamic_update):
    if not root.children:
        return

    if enable_dynamic_update:
        active_ids = active_root_ids(root, global_state)
    else:
        active_ids = {candidate.entity_id for candidate in root.children}

    if enable_dynamic_update and not active_ids:
        active_ids = {
            candidate.entity_id
            for candidate in root.children
            if getattr(candidate, "root_active", False)
        }
        if not active_ids:
            active_ids = {candidate.entity_id for candidate in root.children}
            logger.debug(
                "  No root matched active subgoals; activate all topic entities."
            )

    for candidate in root.children:
        candidate.root_active = candidate.entity_id in active_ids

def check_full_tree_answers(
    nodes,
    global_state,
    updater,
):
    """Use every retained path to advance the current plan to a stable state."""
    previous_answer_keys = _retrieval_answer_hit_keys(global_state)
    candidates = build_new_answer_candidates(nodes)
    if not candidates:
        return global_state, False, []

    logger.debug(
        "  Checking current subgoals against %s full-tree candidates.",
        len(candidates),
    )
    global_state, updated = update_until_stable(
        global_state,
        updater,
        candidates,
    )
    if updated:
        attach_subgoal_answer_hits(
            nodes,
            global_state,
        )
    new_answer_keys = (
        _retrieval_answer_hit_keys(global_state) - previous_answer_keys
    )
    answer_nodes = []
    for node in (nodes if updated and new_answer_keys else []):
        node_hit_keys = {
            (hit.get("subgoal_id"), hit.get("answer_key"))
            for hit in getattr(node, "subgoal_answer_hits", []) or []
        }
        parent_hit_keys = {
            (hit.get("subgoal_id"), hit.get("answer_key"))
            for hit in getattr(
                getattr(node, "parent", None),
                "subgoal_answer_hits",
                [],
            ) or []
        }
        if (node_hit_keys - parent_hit_keys) & new_answer_keys:
            answer_nodes.append(node)
    return global_state, updated, answer_nodes


def match_path(subgoal_answers, child_node):
    if not subgoal_answers or not child_node:
        return []

    entity_ids, entity_names = set(), set()
    node = child_node
    while node is not None:
        entity = getattr(node, "entity_info", None) or {}
        entity_id = str(entity.get("entity_id", "") or "").strip()
        entity_name = str(entity.get("entity_name", "") or "").strip()
        if entity_id:
            entity_ids.add(entity_id)
        if entity_name and entity_name != "Unknown":
            entity_names.add(normalize_name(entity_name))
        node = getattr(node, "parent", None)

    entity_ids.update(
        str(entity_id).strip()
        for entity_id in getattr(child_node, "visited_ids", None) or set()
        if entity_id
    )

    evidence_path = (
        getattr(child_node, "evidence_path", None)
        or getattr(child_node, "walk_path", None)
        or []
    )
    for head, _, tail in evidence_path:
        entity_names.update(
            name for name in (normalize_name(head), normalize_name(tail))
            if name
        )

    matched = []
    for answer in subgoal_answers:
        if not answer_evidence_matches_node(answer, child_node):
            continue

        answer_name = normalize_name(answer.get("answer", ""))
        answer_id = str(answer.get("entity_id", "")).strip()
        if answer_id and answer_id in entity_ids:
            matched.append(answer)
            continue
        if answer_name and any(
            answer_name == entity_name
            or answer_name in entity_name
            or entity_name in answer_name
            for entity_name in entity_names
        ):
            matched.append(answer)
            continue
    return matched


def attach_subgoal_answer_hits(
    nodes,
    global_state,
):
    """Synchronize each node with currently valid subgoal-answer hits."""
    if not global_state:
        return

    current_retrieval_answers = {}
    for subgoal in global_state.subgoals:
        if (
            subgoal.get("status") not in {"partial", "resolved"}
            or subgoal.get("resolve_mode", "retrieval") == "auto"
        ):
            continue
        for answer in subgoal.get("answers", []) or []:
            key = answer_key(answer)
            if key:
                current_retrieval_answers[(subgoal["id"], key)] = (
                    subgoal,
                    answer,
                )

    def hit_record(subgoal, answer, key):
        return {
            "subgoal_id": subgoal["id"],
            "goal": subgoal["goal"],
            "resolve_mode": subgoal.get("resolve_mode", "retrieval"),
            "answer_key": key,
            "answer": answer["answer"],
            "entity_id": answer["entity_id"],
        }

    for child in nodes:
        current_hits = {}
        for old_hit in getattr(child, "subgoal_answer_hits", []) or []:
            key = (old_hit.get("subgoal_id"), old_hit.get("answer_key"))
            current = current_retrieval_answers.get(key)
            if current:
                subgoal, answer = current
                current_hits[key] = hit_record(subgoal, answer, key[1])

        for subgoal in global_state.subgoals:
            if (
                subgoal.get("status") not in {"partial", "resolved"}
                or subgoal.get("resolve_mode", "retrieval") == "auto"
            ):
                continue
            answers = subgoal.get("answers", [])
            for answer in match_path(answers, child):
                key = answer_key(answer)
                if key:
                    current_hits[(subgoal["id"], key)] = hit_record(
                        subgoal,
                        answer,
                        key,
                    )
        child.subgoal_answer_hits = list(current_hits.values())


# --- subgoal ---

def _hit_key(hit):
    subgoal_id = str(hit.get("subgoal_id", "") or "").strip()
    answer_key = str(hit.get("answer_key", "") or "").strip()
    if not subgoal_id or not answer_key:
        return None
    return (subgoal_id, answer_key)


def _downstream_consumers(subgoal_id, global_state, unresolved_only=False):
    if not subgoal_id or not global_state:
        return []

    consumers = []
    for sg in global_state.subgoals:
        if unresolved_only and sg.get("status") == "resolved":
            continue
        if subgoal_id in (sg.get("depend") or []):
            consumers.append(sg)
    return consumers


def incremental_subgoal_hits(child):
    """Return evidence-grounded hits introduced here rather than inherited."""
    parent = getattr(child, "parent", None)
    parent_hit_keys = {
        key
        for key in (
            _hit_key(hit)
            for hit in getattr(parent, "subgoal_answer_hits", []) or []
        )
        if key
    }

    child_hit_by_key = {}
    for hit in getattr(child, "subgoal_answer_hits", []) or []:
        key = _hit_key(hit)
        if key:
            child_hit_by_key[key] = hit

    new_hit_keys = set(child_hit_by_key) - parent_hit_keys
    return [child_hit_by_key[key] for key in sorted(new_hit_keys)]


def can_expand_after_progress(child, global_state):
    """Dynamically gate expansion without mutating KG expansion state."""
    new_hits = incremental_subgoal_hits(child)
    if not new_hits:
        return True

    # A non-entity answer can provide progress credit but cannot seed KG search.
    if not is_freebase_entity_id(getattr(child, "entity_id", "")):
        return False

    subgoals = _subgoal_by_id(global_state)
    if any(
        subgoals.get(hit.get("subgoal_id"))
        and subgoals[hit.get("subgoal_id")].get("status") != "resolved"
        for hit in new_hits
    ):
        return True

    pending = [
        hit.get("subgoal_id")
        for hit in new_hits
        if hit.get("subgoal_id")
    ]
    visited = set()
    while pending:
        subgoal_id = pending.pop()
        if subgoal_id in visited:
            continue
        visited.add(subgoal_id)
        for consumer in _downstream_consumers(
            subgoal_id,
            global_state,
        ):
            if consumer.get("resolve_mode", "retrieval") != "auto":
                if consumer.get("status") != "resolved":
                    return True
                continue
            if (
                consumer.get("status") != "resolved"
                and len(consumer.get("depend") or []) > 1
            ):
                return True
            pending.append(consumer.get("id"))
    return False


def _subgoal_by_id(global_state):
    return {
        str(subgoal.get("id", "") or "").strip(): subgoal
        for subgoal in getattr(global_state, "subgoals", []) or []
        if str(subgoal.get("id", "") or "").strip()
    }


def _progress_credit_for_new_hits(new_hits, global_state):
    """Assign three-level credit from the stable post-update plan state.

    0.0: no newly evidence-grounded retrieval hit;
    0.5: the hit adds evidence but its retrieval subgoal remains partial;
    1.0: the hit completes its retrieval subgoal.

    Downstream activation is deliberately not rewarded again. It is a plan-state
    consequence that already changes the next search target, and in a
    multi-dependency plan it would otherwise give order-dependent extra credit
    to whichever prerequisite happens to complete last.
    """
    if not new_hits or not global_state:
        return 0.0

    subgoals = _subgoal_by_id(global_state)
    credit = 0.0
    for hit in new_hits:
        subgoal_id = str(hit.get("subgoal_id", "") or "").strip()
        subgoal = subgoals.get(subgoal_id)
        if not subgoal:
            continue
        if subgoal.get("status") == "resolved":
            return 1.0
        elif subgoal.get("status") == "partial":
            credit = max(credit, 0.5)
    return credit


def _rewardable_retrieval_hits(hits, global_state):
    """Exclude derived auto answers from KG-node progress rewards."""
    modes_by_id = {
        sg.get("id"): sg.get("resolve_mode", "retrieval")
        for sg in getattr(global_state, "subgoals", []) or []
    }
    return [
        hit
        for hit in hits or []
        if modes_by_id.get(
            hit.get("subgoal_id"),
            hit.get("resolve_mode", "retrieval"),
        ) != "auto"
    ]


def assign_subgoal_progress_credits(nodes, global_state):

    # 比较父子节点的命中信息，只奖励当前节点新增且有KG证据支撑的 retrieval 答案。
    # auto 命中是全局依赖推导的结果，可用于路由和扩展门控，但不重复奖励 KG 节点。
    for child in nodes or []:
        new_hits = incremental_subgoal_hits(child)
        retrieval_hits = _rewardable_retrieval_hits(new_hits, global_state)
        child.progress_credit = _progress_credit_for_new_hits(retrieval_hits, global_state)


def fused_value(path_value, progress_credit, progress_alpha=0.0):
    """Return the path-preserving node value used everywhere after simulation.

    `path_value` is the LLM path-progress score in [0, 1]. `progress_credit`
    uses the stable plan state: 0.0 for no new evidence-grounded retrieval hit, 0.5 for
    adding evidence to a partial retrieval subgoal, and 1.0 for completing it.
    `progress_alpha` scales an additive progress bonus within the unused part
    of the path-score range:

        path_value + alpha * (1 - path_value) * progress_credit

    Thus alpha=0 exactly recovers the simulation score. Progress can only
    increase a path's value, never reduce its original-question score, and
    the result remains in [0, 1].
    """
    path_score = max(0.0, min(1.0, float(path_value or 0.0)))
    progress_score = max(0.0, min(1.0, float(progress_credit or 0.0)))
    alpha = max(0.0, min(1.0, float(progress_alpha or 0.0)))
    return path_score + alpha * (1.0 - path_score) * progress_score


def refresh_node_fused_value(node, progress_alpha=0.0):
    """Synchronize a node's explicit fused value after score/credit updates."""
    node.fused_value = fused_value(
        getattr(node, "path_value", 0.0),
        getattr(node, "progress_credit", 0.0),
        progress_alpha,
    )
    return node.fused_value


def assign_node_rewards(nodes, global_state, progress_alpha):
    """Refresh plan-conditioned progress credits and fused node values."""
    if not nodes:
        return
    if progress_alpha > 0 and global_state:
        assign_subgoal_progress_credits(nodes, global_state)
    else:
        for node in nodes:
            node.progress_credit = 0.0
    for node in nodes:
        refresh_node_fused_value(node, progress_alpha)
