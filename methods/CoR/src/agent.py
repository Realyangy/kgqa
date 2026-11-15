import re
import json
import logging
from string import Template
from typing import Any, Dict, List, Optional, Tuple, Union

import freebase_func
from utils import LLMAPI
from prompt_list import (
    relation_pruning_system_prompt, relation_pruning_user_prompt,
    reasoning_system_prompt, reasoning_user_prompt,
    filter_system_prompt, filter_user_prompt,
    generate_directly_system_prompt, generate_directly_user_prompt,
)
from schema import Entity, Relation


class CoRAgent:
    def __init__(self, model_name: str, base_url: Optional[str], api_key: Optional[str], args):
        self.model_name = model_name
        self.llm_api = LLMAPI(model_name=self.model_name, base_url=base_url, api_key=api_key)
        self.temperature_exploration = args.temperature_exploration
        self.temperature_reasoning = args.temperature_reasoning
        self.max_token = args.max_length

        self.depth = args.depth
        self.relation_width = max(1, int(args.width))
        self.sample_relation_threshold = max(1, int(args.sample_relation_threshold))
        self.remove_unnecessary_rel = args.remove_unnecessary_rel

        self._trace_seq = 0

    def _reset_trace(self) -> None:
        self._trace_seq = 0

    def _with_trace(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._trace_seq += 1
        payload["trace_id"] = self._trace_seq
        return payload

    @staticmethod
    def _sparql_event_to_step(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "search_type" in item and "sparql" in item:
            return {
                "step_type": "sparql",
                "search_type": item.get("search_type", ""),
                "sparql": item.get("sparql", ""),
                "result_count": item.get("result_count", -1),
                "status": item.get("status", "ok"),
                "timed_out": item.get("timed_out", False),
                "filter_caused_empty": item.get("filter_caused_empty", False),
            }

        for operation, query in item.items():
            if operation in ("result_count", "trace_id"):
                continue
            return {
                "step_type": "sparql",
                "search_type": operation,
                "sparql": query,
                "result_count": item.get("result_count", -1),
                "status": item.get("status", "ok"),
                "timed_out": item.get("timed_out", False),
                "filter_caused_empty": item.get("filter_caused_empty", False),
            }
        return None

    @staticmethod
    def _llm_event_to_step(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step_type": "llm",
            "operation_type": item.get("type", ""),
            "prompt": item.get("prompt", ""),
            "input_tokens": item.get("input_tokens", -1),
            "output_tokens": item.get("output_tokens", -1),
            "candidate_size": item.get("candidate_size", item.get("candidates_size", -1)),
            "filtered_candidate_size": item.get("filtered_candidate_size", -1),
            "response": item.get("response", ""),
            "parsed_result": item.get("parsed_result", ""),
        }



    @staticmethod
    def _split_answers(text: str) -> List[str]:
        if not text:
            return []
        return [part.strip() for part in re.split(r",|;", text) if part.strip()]

    @staticmethod
    def _build_result_items(
        entity_ids: List[str],
        entity_names: List[str],
        literal_candidates: List[str],
    ) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        seen: set = set()

        for entity_id, entity_name in zip(entity_ids, entity_names):
            name = str(entity_name).strip()
            if not name:
                continue
            key = (str(entity_id).strip(), name)
            if key in seen:
                continue
            seen.add(key)
            results.append({"id": str(entity_id).strip(), "name": name})

        for literal in literal_candidates:
            name = str(literal).strip()
            if not name:
                continue
            key = ("", name)
            if key in seen:
                continue
            seen.add(key)
            results.append({"id": "", "name": name})

        return results

    @staticmethod
    def _map_answer_names_to_results(
        answer_names: List[str],
        entity_ids: List[str],
        entity_names: List[str],
    ) -> List[Dict[str, str]]:
        name_to_id: Dict[str, str] = {}
        for entity_id, entity_name in zip(entity_ids, entity_names):
            name = str(entity_name).strip()
            if name and name not in name_to_id:
                name_to_id[name] = str(entity_id).strip()

        results: List[Dict[str, str]] = []
        seen: set = set()
        for answer in answer_names:
            name = str(answer).strip()
            if not name:
                continue
            item_id = name_to_id.get(name, "")
            key = (item_id, name)
            if key in seen:
                continue
            seen.add(key)
            results.append({"id": item_id, "name": name})

        return results

    # 归一化所有起点实体
    @staticmethod
    def _normalize_topic_entities(topic_entities: List[Any]) -> List[Entity]:
        normalized_entities: List[Entity] = []
        for item in topic_entities:
            if isinstance(item, Entity):
                normalized_entities.append(item)
                continue
            if isinstance(item, dict):
                entity_id = item.get("id", "")
                entity_name = item.get("name", "")
                if entity_id and entity_name:
                    normalized_entities.append(Entity(entity_id, entity_name, 0.0))
        return normalized_entities

    @staticmethod
    def _expand_bidirectional_relations(
        selected_relations: List[Relation],
        head_relations: List[str],
        tail_relations: List[str],
    ) -> List[Relation]:
        head_set = set(head_relations)
        tail_set = set(tail_relations)
        expanded: List[Relation] = []
        seen: set = set()

        for relation in selected_relations:
            relation_id = relation.id
            in_head = relation_id in head_set
            in_tail = relation_id in tail_set

            if in_head and in_tail:
                primary_left = bool(relation.left)
                ordered_directions = [primary_left, not primary_left]
                for direction_left in ordered_directions:
                    key = (relation_id, direction_left)
                    if key in seen:
                        continue
                    expanded.append(Relation(id=relation_id, score=relation.score, left=direction_left))
                    seen.add(key)
                continue

            key = (relation_id, bool(relation.left))
            if key in seen:
                continue
            expanded.append(relation)
            seen.add(key)

        return expanded

    @staticmethod
    def _cap_relations_by_threshold(
        head_relations: List[str],
        tail_relations: List[str],
        threshold: int,
    ) -> Tuple[List[str], List[str]]:
        limit = max(1, int(threshold))
        if len(head_relations) + len(tail_relations) <= limit:
            return list(head_relations), list(tail_relations)
        combined: List[Tuple[str, bool]] = [(relation, True) for relation in head_relations]
        combined.extend((relation, False) for relation in tail_relations)
        truncated = combined[:limit]
        new_head: List[str] = []
        new_tail: List[str] = []
        for relation, is_head in truncated:
            if is_head:
                new_head.append(relation)
            else:
                new_tail.append(relation)
        return new_head, new_tail

    @staticmethod
    def _relation_chain_signature(relation_chain: List[Relation]) -> str:
        if not relation_chain:
            return "<root>"
        parts: List[str] = []
        for relation in relation_chain:
            direction = "->" if relation.left else "<-"
            parts.append(f"{direction}{relation.id}")
        return " ".join(parts)


    @staticmethod
    def _reasoning_path_to_str(
        topic_entity: Entity,
        relation_chain: List[Relation],
        target_entity_name_str: Optional[str] = None,
        relation_display_names: Optional[Dict[str, str]] = None,
    ) -> str:
        lines: List[str] = []
        current_node = f"TopicEntity({topic_entity.name})"
        relation_display_names = relation_display_names or {}

        for index, relation in enumerate(relation_chain, start=1):
            next_node = f"e{index}"
            relation_text = relation_display_names.get(str(relation.id), str(relation.id))
            if relation.left:
                lines.append(f"{current_node} {relation_text} {next_node} .")
            else:
                lines.append(f"{next_node} {relation_text} {current_node} .")
            current_node = next_node

        if target_entity_name_str:
            lines.append(f"{current_node} candidate.target TargetEntity({target_entity_name_str}) .")

        return "\n".join(lines)




    @staticmethod
    def build_relation_prune_prompt(
        question: str,
        topic_entity: Entity,
        relation_chain: List[Relation],
        head_relations: List[str],
        tail_relations: List[str],
        relation_display_names: Optional[Dict[str, str]],
        prompt_template: Template,
        render_mode: str = "directional",
    ) -> str:
        mode = (render_mode or "directional").strip().lower()
        if mode not in ("directional", "unified"):
            mode = "directional"

        def _display_name(relation_id: str) -> str:
            if relation_display_names is None:
                return relation_id
            name = str(relation_display_names.get(relation_id, "")).strip()
            return name or relation_id

        if mode == "unified":
            all_relations: List[str] = []
            seen = set()
            for relation in head_relations + tail_relations:
                if relation in seen:
                    continue
                seen.add(relation)
                all_relations.append(_display_name(relation))
            relations_str = "Relations: " + "; ".join(all_relations)
            if all_relations:
                relations_str += ";"
        else:
            forward_snippet, backward_snippet = freebase_func.build_relation_snippets(topic_entity, relation_chain, relation_display_names)
            relations_str = ""
            if head_relations:
                relations_str += (
                    f"# {forward_snippet}\n"
                    f"{'; '.join([_display_name(relation) for relation in head_relations])};\n"
                )
            if tail_relations:
                relations_str += (
                    f"# {backward_snippet}\n"
                    f"{'; '.join([_display_name(relation) for relation in tail_relations])};\n"
                )
        return prompt_template.substitute(question=question, relations=relations_str)


    @staticmethod
    def _clean_relations(
        text: str,
        head_relations: List[str],
        tail_relations: List[str],
        relation_display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Union[List[Relation], str]]:
        payload = (text or "").strip()
        if not payload:
            return False, "empty output"

        if not payload.startswith("[") and not payload.startswith("{"):
            obj_start = payload.find("{")
            obj_end = payload.rfind("}")
            arr_start = payload.find("[")
            arr_end = payload.rfind("]")
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                payload = payload[obj_start : obj_end + 1]
            elif arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                payload = payload[arr_start : arr_end + 1]

        try:
            items = json.loads(payload)
        except Exception:
            return False, "invalid json output"

        parsed_relation_ids: List[Tuple[Optional[int], int, str, Optional[float]]] = []
        counter = 0

        def _normalize_rank(rank_raw: Any) -> Optional[int]:
            if rank_raw is None:
                return None
            try:
                rank = int(rank_raw)
            except Exception:
                return None
            if rank <= 0:
                return None
            return rank

        def _normalize_score(score_raw: Any) -> Optional[float]:
            if score_raw is None:
                return None
            try:
                score = float(score_raw)
            except Exception:
                return None
            if score < 0:
                return 0.0
            if score > 1:
                return 1.0
            return score

        def _append_relation_id(
            relation_id_raw: Any,
            rank_raw: Any = None,
            score_raw: Any = None,
        ) -> None:
            nonlocal counter
            relation_id = str(relation_id_raw).strip()
            if not relation_id:
                return
            rank = _normalize_rank(rank_raw)
            score = _normalize_score(score_raw)
            parsed_relation_ids.append((rank, counter, relation_id, score))
            counter += 1

        if isinstance(items, dict):
            relation_by_relevance = items.get("relation_by_relevance")
            relevance_score = items.get("relevance_score")

            score_by_index: Dict[int, Optional[float]] = {}
            if isinstance(relevance_score, list):
                for index, score in enumerate(relevance_score):
                    score_by_index[index] = _normalize_score(score)

            if isinstance(relation_by_relevance, list):
                for index, relation_id in enumerate(relation_by_relevance):
                    if isinstance(relation_id, (list, tuple)):
                        if len(relation_id) >= 1:
                            _append_relation_id(relation_id[0], score_raw=score_by_index.get(index))
                        continue
                    if isinstance(relation_id, dict):
                        _append_relation_id(
                            relation_id.get("relation") or relation_id.get("relation_id") or relation_id.get("id"),
                            score_raw=score_by_index.get(index),
                        )
                        continue
                    _append_relation_id(relation_id, score_raw=score_by_index.get(index))
            elif isinstance(relation_by_relevance, str):
                _append_relation_id(relation_by_relevance, score_raw=score_by_index.get(0))
            if not parsed_relation_ids:
                for relation_id in items.keys():
                    _append_relation_id(relation_id)
        elif isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue

                rank_raw = item.get("rank") if "rank" in item else None

                score_raw = item.get("score") if "score" in item else item.get("relevance_score")
                if "relation_id" in item:
                    _append_relation_id(item.get("relation_id"), rank_raw, score_raw)
                    continue

                if "relation" in item:
                    _append_relation_id(item.get("relation"), rank_raw, score_raw)
                    continue

                if len(item) == 1:
                    relation_id, _rationale = next(iter(item.items()))
                    _append_relation_id(relation_id, rank_raw, score_raw)
                    continue

                if "rank" in item and len(item) == 2:
                    for key, _value in item.items():
                        if key == "rank":
                            continue
                        _append_relation_id(key, item.get("rank"), score_raw)
                    continue
        else:
            return False, "json output is not an object or list"

        if not parsed_relation_ids:
            return False, "no relations found"

        parsed_relation_ids.sort(key=lambda x: (x[0] if x[0] is not None else 10**9, x[1]))

        relations: List[Relation] = []
        seen = set()
        display_to_ids: Dict[str, List[str]] = {}
        if relation_display_names:
            for relation_id, display_name in relation_display_names.items():
                name = str(display_name or "").strip()
                if not name:
                    continue
                display_to_ids.setdefault(name, []).append(relation_id)

        def _resolve_relation_id(candidate_id: str) -> str:
            if candidate_id in head_relations or candidate_id in tail_relations:
                return candidate_id
            matched_ids = display_to_ids.get(candidate_id, [])
            if len(matched_ids) == 1:
                return matched_ids[0]
            return candidate_id

        for _rank, _idx, relation_id, parsed_score in parsed_relation_ids:
            relation_id = _resolve_relation_id(relation_id)
            if relation_id in seen:
                continue
            score = parsed_score if parsed_score is not None else 0.0
            in_head = relation_id in head_relations
            in_tail = relation_id in tail_relations
            if in_head:
                relations.append(Relation(id=relation_id, score=score, left=True))
                seen.add(relation_id)
            elif in_tail:
                relations.append(Relation(id=relation_id, score=score, left=False))
                seen.add(relation_id)

        if not relations:
            return False, "no relations found"
        return True, relations

    @staticmethod
    def _try_parse_reasoning_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        candidates: List[str] = [text.strip()]
        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())
        candidates.extend(match.group(0) for match in re.finditer(r"\{[\s\S]*?\}", text))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def build_reasoning_prompt(
        question: str,
        topic_entity: Entity,
        relation_chain: List[Relation],
        target_entity_candidates: List[str],
        prompt_template: Template,
        relation_display_names: Optional[Dict[str, str]] = None,
        candidate_maxsize: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        candidate_size = len(target_entity_candidates)
        if candidate_maxsize is not None and candidate_size > candidate_maxsize:
            filtered_candidate_size = candidate_maxsize
            target_entity_name_str = ", ".join(target_entity_candidates[:candidate_maxsize]) + ", ..."
        else:
            filtered_candidate_size = candidate_size
            target_entity_name_str = ", ".join(target_entity_candidates)

        reasoning_path_str = CoRAgent._reasoning_path_to_str(
            topic_entity=topic_entity,
            relation_chain=relation_chain,
            target_entity_name_str=target_entity_name_str,
            relation_display_names=relation_display_names,
        )
        prompt = prompt_template.substitute(question=question, reasoning_path=reasoning_path_str)
        return prompt, candidate_size, filtered_candidate_size

    @staticmethod
    def _normalize_decision(decision_raw: Any) -> Optional[str]:
        if decision_raw is None:
            return None
        decision = str(decision_raw).strip().lower()
        if decision in {"forward", "backtrack", "constraint", "filter", "stop", "yes", "no"}:
            return decision
        return None

    @staticmethod
    def _extract_reasoning_answer(text: str) -> Tuple[Optional[str], List[str]]:
        parsed_json = CoRAgent._try_parse_reasoning_json(text)
        if parsed_json is not None:
            raw_decision = parsed_json.get("decision")
            decision = CoRAgent._normalize_decision(raw_decision)

            raw_answers = parsed_json.get("answer", [])
            answers: List[str] = []
            if isinstance(raw_answers, list):
                answers = [str(item).strip() for item in raw_answers if str(item).strip()]
            return decision, answers

        return None, []

    @staticmethod
    def _decision_to_action(decision: Optional[str]) -> str:
        if decision == "stop":
            return "Stop"
        if decision == "forward":
            return "Forward"
        if decision == "constraint":
            return "Constraint"
        if decision == "filter":
            return "Constraint"
        if decision == "backtrack":
            return "Backtrack"
        if decision == "yes":
            return "Stop"
        if decision == "no":
            return "Forward"
        return "UnknownAction"

    @staticmethod
    def _extract_filter_answer(text: str) -> Tuple[bool, str]:
        if not text:
            return False, ""

        # Preferred format: strict JSON object with `answer: list[str]`
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                answers = parsed.get("answer", [])
                if isinstance(answers, list):
                    clean_answers = [str(item).strip() for item in answers if str(item).strip()]
                    return True, ", ".join(clean_answers)
        except Exception:
            pass

        # Backward compatibility: legacy "Therefore, the answer is {a, b}" style
        start_index = text.find("{")
        end_index = text.find("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return True, text[start_index + 1 : end_index].strip()

        return False, ""


    def relation_search(self, topic_entity, relation_chain, sparql_history):
        extract, last = freebase_func.build_relation_extract(topic_entity.id, relation_chain)
        head_query = freebase_func.build_sparql_relations(extract + f"{last} ?relation ?x .")
        tail_query = freebase_func.build_sparql_relations(extract + f"?x ?relation {last} .")
        head_meta = freebase_func.execurte_sparql_with_meta(head_query)
        tail_meta = freebase_func.execurte_sparql_with_meta(tail_query)
        head_rows, tail_rows = head_meta.get("rows", []) or [], tail_meta.get("rows", []) or []
        head = freebase_func.replace_relation_prefix(head_rows) if head_rows else []
        tail = freebase_func.replace_relation_prefix(tail_rows) if tail_rows else []
        for direction, query, meta, relations in (("head", head_query, head_meta, head), ("tail", tail_query, tail_meta, tail)):
            sparql_history.append(self._with_trace({"search_type": f"relation_search_{direction}", "sparql": query,
                "result_count": len(relations), "status": str(meta.get("status", "ok")), "timed_out": bool(meta.get("timed_out", False))}))
        if self.remove_unnecessary_rel:
            head = [r for r in head if not freebase_func.abandon_rels(r)]
            tail = [r for r in tail if not freebase_func.abandon_rels(r)]
        if relation_chain:
            previous = relation_chain[-1]
            if previous.left:
                tail = [r for r in tail if r != previous.id]
            else:
                head = [r for r in head if r != previous.id]
        return sorted(set(head)), sorted(set(tail))

    def relation_prune(self, question, topic_entity, relation_chain, head_relations, tail_relations, prompt_history):
        head_relations, tail_relations = self._cap_relations_by_threshold(head_relations, tail_relations, self.sample_relation_threshold)
        names = {str(r): str(r) for r in head_relations + tail_relations}
        prompt = self.build_relation_prune_prompt(question, topic_entity, relation_chain, head_relations, tail_relations, names, relation_pruning_user_prompt)
        response, usage = self.llm_api.generate(prompt, self.temperature_exploration, self.max_token, relation_pruning_system_prompt.substitute())
        candidate_size = len(head_relations) + len(tail_relations)
        filtered_size = candidate_size
        warnings = []
        selected = []
        if not response:
            warnings.append("LLM returned empty response.")
        else:
            flag, parsed = self._clean_relations(response, head_relations, tail_relations, names)
            if not flag or not isinstance(parsed, list):
                warnings.append(f"Failed to parse relation prune result: {parsed}")
            else:
                selected = [r for r in parsed if (r.left and r.id in head_relations) or (not r.left and r.id in tail_relations)]
                if relation_chain:
                    previous = relation_chain[-1]
                    selected = [r for r in selected if not (r.id == previous.id and r.left != previous.left)]
                selected = selected[:self.relation_width]
                filtered_size = len(selected)
                if not selected:
                    warnings.append("No valid relations after validation/acyclic filtering.")
        prompt_history.append(self._with_trace({"type": "relation_prune", "prompt": prompt,
            "input_tokens": usage.get("input_tokens", -1), "output_tokens": usage.get("output_tokens", -1),
            "candidates_size": candidate_size, "filtered_candidate_size": filtered_size,
            "response": response, "parsed_result": [r.id for r in selected], "warnings": warnings}))
        return selected

    def entity_search(self, topic_entity, relation_chain, sparql_history):
        extract, err = freebase_func.build_entities_extract(topic_entity, relation_chain)
        if err:
            sparql_history.append(self._with_trace({"search_type": "entity_search", "sparql": "", "result_count": 0,
                "status": "error", "timed_out": False, "filter_caused_empty": False}))
            return [], [], []
        query = freebase_func.build_sparql_entities(extract)
        meta = freebase_func.execurte_sparql_with_meta(query)
        rows = meta.get("rows", []) or []
        literals, ids = freebase_func.replace_entities_prefix(rows) if rows else ([], [])
        labels = freebase_func.extract_inline_entity_labels(rows) if rows else {}
        ids = [entity for entity in ids if entity.startswith("m.")]
        missing = [entity for entity in dict.fromkeys(ids) if entity not in labels]
        names_by_id = {}
        if missing:
            logging.info("[EntitySearch] need resolve names | total_entities=%s", len(missing))
            try:
                names_by_id = freebase_func.id2entity_names(missing)
            except Exception:
                pass
        names = [str(labels.get(entity, "")).strip() or str(names_by_id.get(entity, "")).strip()
                 or freebase_func.id2entity_name_or_type(entity) for entity in ids]
        filtered = [(entity, name) for entity, name in zip(ids, names) if name != "UnName_Entity"]
        ids, names = [x[0] for x in filtered], [x[1] for x in filtered]
        sparql_history.append(self._with_trace({"search_type": "entity_search", "sparql": query,
            "result_count": len(rows), "status": str(meta.get("status", "ok")),
            "timed_out": bool(meta.get("timed_out", False)),
            "filter_caused_empty": len(rows) > 0 and len(ids) + len(literals) == 0}))
        return ids, names, literals

    def reasoning(self, question, topic_entity, relation_chain, target_entity_candidates, prompt_history, candidate_maxsize=None):
        names = {str(r.id): str(r.id) for r in relation_chain}
        prompt, size, filtered = self.build_reasoning_prompt(question, topic_entity, relation_chain, target_entity_candidates,
            reasoning_user_prompt, names, candidate_maxsize)
        response, usage = self.llm_api.generate(prompt, self.temperature_reasoning, self.max_token, reasoning_system_prompt.substitute())
        warnings = []
        decision, answers = self._extract_reasoning_answer(response) if response else (None, [])
        action = self._decision_to_action(decision)
        if not response:
            warnings.append("LLM returned empty response.")
        else:
            if decision is None:
                warnings.append("Failed to parse decision from reasoning response JSON.")
            if action == "UnknownAction":
                warnings.append("Failed to map reasoning output to valid action.")
        prompt_history.append(self._with_trace({"type": "reasoning", "prompt": prompt, "response": response,
            "decision": decision, "answers": answers, "parsed_result": action, "candidate_size": size,
            "filtered_candidate_size": filtered, "input_tokens": usage.get("input_tokens", -1),
            "output_tokens": usage.get("output_tokens", -1), "warnings": warnings}))
        return (action, response, answers) if response else (None, None, [])

    def validate(self, question, topic_entity, relation_chain, target_entity_names, prompt_history):
        names = {str(r.id): str(r.id) for r in relation_chain}
        prompt, size, filtered = self.build_reasoning_prompt(question, topic_entity, relation_chain, target_entity_names, filter_user_prompt, names)
        response, usage = self.llm_api.generate(prompt, self.temperature_reasoning, self.max_token, filter_system_prompt.substitute())
        warnings = []
        if not response:
            warnings.append("LLM returned empty response.")
            result = ""
        else:
            success, result = self._extract_filter_answer(response)
            if not success:
                warnings.append("Failed to parse answer from response (expected JSON answer list or legacy braced answer).")
        prompt_history.append(self._with_trace({"type": "filter_answer", "prompt": prompt,
            "input_tokens": usage.get("input_tokens", -1), "output_tokens": usage.get("output_tokens", -1),
            "candidate_size": size, "filtered_candidate_size": filtered, "response": response,
            "parsed_result": result, "warnings": warnings}))
        return result if result else None

    def generate_directly(self, question, prompt_history):
        prompt = generate_directly_user_prompt.substitute(question=question)
        response, usage = self.llm_api.generate(prompt, self.temperature_reasoning, self.max_token, generate_directly_system_prompt.substitute())
        result = self._extract_filter_answer(response)[1] if response else ""
        warnings = []
        if not response:
            warnings.append("LLM returned empty response.")
        elif not result:
            warnings.append("Failed to parse answer from response (expected JSON answer list or legacy braced answer).")
        prompt_history.append(self._with_trace({"type": "generate_without_explored_paths", "prompt": prompt,
            "input_tokens": usage.get("input_tokens", -1), "output_tokens": usage.get("output_tokens", -1),
            "response": response, "parsed_result": result, "warnings": warnings}))
        return result


    # 根据问题和起点实体回答
    def answer(self, question: str, topic_entities: List[Entity], **kwargs) -> Dict[str, Any]:
        if not topic_entities:
            return {
                "action": "ERROR",
                "question": question,
                "results": [],
                "reasoning_chains": "No topic_entities provided",
            }

        normalized_entities = self._normalize_topic_entities(topic_entities)
        if not normalized_entities:
            return {
                "action": "ERROR",
                "question": question,
                "results": [],
                "reasoning_chains": "Invalid topic_entities format",
            }

        prompt_history = kwargs.get("prompt_history", [])
        sparql_history = kwargs.get("sparql_history", [])
        step_history = kwargs.get("step_history", [])
        self._reset_trace()

        root_entity = normalized_entities[0]
        logging.info(
            "[CoR] Start question | root=%s(%s) | width=%s depth=%s",
            root_entity.name,
            root_entity.id,
            self.relation_width,
            self.depth,
        )
        stack = [(root_entity, [root_entity], 0)]

        while stack:
            node, current_path, current_depth = stack.pop()

            if isinstance(node, Entity):
                relation_chain: List[Relation] = [item for item in current_path[1:] if isinstance(item, Relation)]
                chain_sig = self._relation_chain_signature(relation_chain)
                logging.info("[CoR][depth=%s] Entity node | chain=%s", current_depth, chain_sig)
                sparql_before = len(sparql_history)
                head_relations, tail_relations = self.relation_search(
                    topic_entity=root_entity,
                    relation_chain=relation_chain,
                    sparql_history=sparql_history,
                )
                logging.info(
                    "[CoR][depth=%s] relation_search done | head=%s tail=%s | chain=%s",
                    current_depth,
                    len(head_relations),
                    len(tail_relations),
                    chain_sig,
                )
                for event in sparql_history[sparql_before:]:
                    step = self._sparql_event_to_step(event)
                    if step is not None:
                        step_history.append(step)
                if not head_relations and not tail_relations:
                    logging.info("[CoR][depth=%s] no relations, skip | chain=%s", current_depth, chain_sig)
                    continue

                prompt_before = len(prompt_history)
                selected_relations = self.relation_prune(
                    question=question,
                    topic_entity=root_entity,
                    relation_chain=relation_chain,
                    head_relations=head_relations,
                    tail_relations=tail_relations,
                    prompt_history=prompt_history,
                )
                for event in prompt_history[prompt_before:]:
                    step_history.append(self._llm_event_to_step(event))
                selected_relations = selected_relations[: self.relation_width]
                selected_relations = self._expand_bidirectional_relations(
                    selected_relations=selected_relations,
                    head_relations=head_relations,
                    tail_relations=tail_relations,
                )
                logging.info(
                    "[CoR][depth=%s] relation_prune selected=%s | chain=%s",
                    current_depth,
                    [f"{item.id}({'head' if item.left else 'tail'})" for item in selected_relations],
                    chain_sig,
                )
                if current_depth < self.depth:
                    for relation in reversed(selected_relations):
                        stack.append((relation, current_path + [relation], current_depth + 1))
                continue

            if isinstance(node, Relation):
                relation_chain: List[Relation] = [item for item in current_path[1:] if isinstance(item, Relation)]
                chain_sig = self._relation_chain_signature(relation_chain)
                logging.info("[CoR][depth=%s] Relation node | chain=%s", current_depth, chain_sig)
                sparql_before = len(sparql_history)
                reasoning_answers: List[str] = []
                entity_ids, entity_names, literal_candidates = self.entity_search(
                    topic_entity=root_entity,
                    relation_chain=relation_chain,
                    sparql_history=sparql_history,
                )
                logging.info(
                    "[CoR][depth=%s] entity_search done | entities=%s literals=%s | chain=%s",
                    current_depth,
                    len(entity_ids),
                    len(literal_candidates),
                    chain_sig,
                )
                for event in sparql_history[sparql_before:]:
                    step = self._sparql_event_to_step(event)
                    if step is not None:
                        step_history.append(step)

                if len(entity_ids) + len(literal_candidates) == 0:
                    action = "Backtrack" if current_depth >= self.depth else "Forward"
                    response = None
                    logging.info(
                        "[CoR][depth=%s] empty candidates => action=%s | chain=%s",
                        current_depth,
                        action,
                        chain_sig,
                    )
                else:
                    joint_entity_candidates = entity_names + literal_candidates
                    prompt_before = len(prompt_history)
                    action, response, reasoning_answers = self.reasoning(
                        question=question,
                        topic_entity=root_entity,
                        relation_chain=relation_chain,
                        target_entity_candidates=joint_entity_candidates,
                        prompt_history=prompt_history,
                    )
                    for event in prompt_history[prompt_before:]:
                        step_history.append(self._llm_event_to_step(event))
                    logging.info(
                        "[CoR][depth=%s] reasoning action=%s | candidates=%s | chain=%s",
                        current_depth,
                        action,
                        len(joint_entity_candidates),
                        chain_sig,
                    )

                if action is None:
                    return {
                        "action": "ERROR",
                        "question": question,
                        "results": [],
                        "reasoning_chains": "LLM API call failed",
                        "prompt_history": prompt_history,
                        "sparql_history": sparql_history,
                        "step_history": step_history,
                    }

                if action.lower() == "stop":
                    if reasoning_answers:
                        answer = reasoning_answers
                        result_items = self._map_answer_names_to_results(
                            answer_names=reasoning_answers,
                            entity_ids=entity_ids,
                            entity_names=entity_names,
                        )
                    else:
                        answer = entity_names + literal_candidates
                        result_items = self._build_result_items(
                            entity_ids=entity_ids,
                            entity_names=entity_names,
                            literal_candidates=literal_candidates,
                        )
                    reasoning_chain_str = self._reasoning_path_to_str(
                        topic_entity=root_entity,
                        relation_chain=relation_chain,
                        target_entity_name_str=", ".join(answer),
                    )
                    return {
                        "action": action,
                        "question": question,
                        "results": result_items,
                        "reasoning_chains": reasoning_chain_str,
                        "prompt_history": prompt_history,
                        "sparql_history": sparql_history,
                        "step_history": step_history,
                    }

                if action.lower() == "constraint":
                    logging.info("[CoR][depth=%s] enter constraint filter | chain=%s", current_depth, chain_sig)
                    joint_entity_candidates = entity_names + literal_candidates
                    filter_response = self.validate(
                        # filter step is appended immediately after call
                        question=question,
                        topic_entity=root_entity,
                        relation_chain=relation_chain,
                        target_entity_names=joint_entity_candidates,
                        prompt_history=prompt_history,
                    )
                    if len(prompt_history) > 0:
                        last_event = prompt_history[-1]
                        if last_event.get("type") == "filter_answer":
                            step_history.append(self._llm_event_to_step(last_event))
                    result = self._split_answers(filter_response or "")
                    if result:
                        logging.info("[CoR][depth=%s] filter accepted answers=%s | chain=%s", current_depth, len(result), chain_sig)
                        mapped_results = self._map_answer_names_to_results(
                            answer_names=result,
                            entity_ids=entity_ids,
                            entity_names=entity_names,
                        )
                        reasoning_chain_str = self._reasoning_path_to_str(
                            topic_entity=root_entity,
                            relation_chain=relation_chain,
                            target_entity_name_str=", ".join(joint_entity_candidates),
                        )
                        return {
                            "action": action,
                            "question": question,
                            "results": mapped_results,
                            "reasoning_chains": reasoning_chain_str,
                            "prompt_history": prompt_history,
                            "sparql_history": sparql_history,
                            "step_history": step_history,
                        }
                    logging.info("[CoR][depth=%s] filter returned empty, break search | chain=%s", current_depth, chain_sig)
                    break

                if action.lower() == "forward":
                    logging.info("[CoR][depth=%s] continue forward | chain=%s", current_depth, chain_sig)
                    if current_depth >= self.depth:
                        logging.info("[CoR][depth=%s] reached max depth, skip forward | chain=%s", current_depth, chain_sig)
                        continue
                    sparql_before_forward = len(sparql_history)
                    head_relations, tail_relations = self.relation_search(
                        # relation_search steps are appended immediately after call
                        topic_entity=root_entity,
                        relation_chain=relation_chain,
                        sparql_history=sparql_history,
                    )
                    for event in sparql_history[sparql_before_forward:]:
                        step = self._sparql_event_to_step(event)
                        if step is not None:
                            step_history.append(step)
                    if not head_relations and not tail_relations:
                        logging.info("[CoR][depth=%s] forward relation_search empty | chain=%s", current_depth, chain_sig)
                        continue

                    prompt_before = len(prompt_history)
                    selected_relations = self.relation_prune(
                        question=question,
                        topic_entity=root_entity,
                        relation_chain=relation_chain,
                        head_relations=head_relations,
                        tail_relations=tail_relations,
                        prompt_history=prompt_history,
                    )
                    for event in prompt_history[prompt_before:]:
                        step_history.append(self._llm_event_to_step(event))
                    selected_relations = selected_relations[: self.relation_width]
                    selected_relations = self._expand_bidirectional_relations(
                        selected_relations=selected_relations,
                        head_relations=head_relations,
                        tail_relations=tail_relations,
                    )
                    if current_depth < self.depth:
                        for relation in reversed(selected_relations):
                            stack.append((relation, current_path + [relation], current_depth + 1))
                    continue

                if action.lower() == "backtrack":
                    logging.info("[CoR][depth=%s] backtrack | chain=%s", current_depth, chain_sig)
                    continue

                logging.warning(f"Unknown action from reasoning: {action}, raw={response}")
                continue

        prompt_before = len(prompt_history)
        logging.info("[CoR] DFS ended without final answer, fallback to generate_directly")
        generated = self.generate_directly(question=question, prompt_history=prompt_history)
        for event in prompt_history[prompt_before:]:
            step_history.append(self._llm_event_to_step(event))
        results = self._split_answers(generated)
        return {
            "action": "generate_directly",
            "question": question,
            "results": results,
            "reasoning_chains": [],
            "prompt_history": prompt_history,
            "sparql_history": sparql_history,
            "step_history": step_history,
        }


