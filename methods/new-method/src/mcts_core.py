"""mcts_core functions for SPAR-MCTS."""

import logging
import math
from mcts_utils import construct_id_triple
from utils import ensure_triple_list
from mcts_utils import (
    build_candidate,
    deduplicate_cvt_candidates,
    node_path_key,
    path_key,
)
from sparql_utils import get_entity_name, is_freebase_entity_id
from prompt_list import (
    cvt_relation_system_prompt,
    cvt_relation_user_prompt,
    entity_prune_system_prompt,
    entity_prune_user_prompt,
    extract_relation_system_prompt,
    extract_relation_user_prompt,
)
from mcts_utils import (
    SPARQL_SEARCH_HEAD_ENTITIES,
    SPARQL_SEARCH_IN_REL,
    SPARQL_SEARCH_OUT_REL,
    SPARQL_SEARCH_TAIL_ENTITIES,
    construct_evidence_triple,
    construct_id_triple,
    construct_walk_triple,
    extract_entity_ids,
    is_noise_relation,
    is_valid_candidate_entity,
    normalize_relation,
    parse_pruned_relations,
)
from sparql_utils import execute_sparql, get_entity_name, is_freebase_entity_id
from utils import extract_candidate_indices, format_relation_name, format_reasoning_path, freebase_rel_to_text
from mcts_utils import parse_batch_evaluation_output
from prompt_list import evaluate_system_prompt, evaluate_user_prompt
from sparql_utils import is_freebase_entity_id
from utils import ensure_triple_list, format_reasoning_path

logger = logging.getLogger(__name__)


# --- mcts_core ---

class MCTSNode:
    """
    MCTS节点类。

    新增 is_virtual_root 标记，用于虚拟根节点（纯容器，不参与扩展）。
    每个非虚拟节点只有一个实体，entity_info 为此实体的字典 {"entity_id": ..., "entity_name": ...}。
    walk_path 始终首尾相连，evidence_path 保留KG原始方向证据，
    id_path 保留同一路径的实体ID表示，仅用于程序内部身份判断。
    """
    def __init__(self, entity_info, parent=None, walk_path=None, evidence_path=None,
                 id_path=None,
                 visited_ids=None, entity_incoming=None, is_virtual_root=False):

        self.is_virtual_root = is_virtual_root

        ## 基本信息
        self.parent = parent
        self.children = []
        self.walk_path = ensure_triple_list(walk_path)
        self.evidence_path = (
            ensure_triple_list(evidence_path)
            if evidence_path is not None
            else list(self.walk_path)
        )
        self.id_path = ensure_triple_list(id_path)
        self.path_value = 0.0
        self.total_reward = 0.0
        self.value_weight = 0.0
        self.uct_visits = 0

        ## 实体及扩展信息
        self.entity_info = dict(entity_info) if entity_info else {}
        self.entity_id = self.entity_info.get("entity_id", "")
        self.entity_name = self.entity_info.get("entity_name", "")
        self.visited_ids = visited_ids if visited_ids else set()
        self.is_fully_expanded = False
        self.expanded_relations = []  # 单个实体，直接用列表存 (relation, head) 元组

        if is_virtual_root or not self.entity_info:
            self.is_fully_expanded = True

        self.entity_incoming = entity_incoming

    def add_child(self, new_entity_info, new_walk_path, relation, head,
                  new_evidence_path=None, new_id_path=None):
        """
        添加子节点。子节点只保留路径末端实体，不再继承父节点中的其他实体。
        """
        new_entity_info = dict(new_entity_info)

        new_walk_steps = ensure_triple_list(new_walk_path)
        new_evidence_steps = ensure_triple_list(
            new_evidence_path if new_evidence_path is not None else new_walk_path
        )
        new_id_steps = ensure_triple_list(new_id_path)
        if not new_id_steps:
            new_id_steps = [
                tuple(construct_id_triple(
                    self.entity_id,
                    relation,
                    new_entity_info["entity_id"],
                    head,
                ))
            ]
        new_full_walk_path = self.walk_path + new_walk_steps
        new_full_evidence_path = self.evidence_path + new_evidence_steps
        new_full_id_path = self.id_path + new_id_steps

        new_visited_ids = self.visited_ids | {new_entity_info["entity_id"]}
        new_incoming = (relation, head)

        child = MCTSNode(
            entity_info=new_entity_info,
            parent=self,
            walk_path=new_full_walk_path,
            evidence_path=new_full_evidence_path,
            id_path=new_full_id_path,
            visited_ids=new_visited_ids,
            entity_incoming=new_incoming,
        )
        self.children.append(child)
        return child

    def get_uct_value(self, c_uct):
        if self.is_virtual_root or self.parent is None:
            return 0.0

        if self.uct_visits <= 0 or self.value_weight <= 0:
            return float("inf")

        q_value = self.total_reward / self.value_weight
        parent_visits = max(2, self.parent.uct_visits)
        u_value = c_uct * math.sqrt(
            math.log(parent_visits) / self.uct_visits
        )
        return q_value + u_value

    def is_terminal(self, max_depth):
        if self.is_virtual_root:
            return False
        return len(self.walk_path) >= max_depth

# --- mcts_core ---

def _make_expansion_candidate(node, candidate):
    next_walk_path = ensure_triple_list(candidate["walk_path"])
    next_evidence_path = ensure_triple_list(
        candidate.get("evidence_path", next_walk_path)
    )
    next_id_path = ensure_triple_list(candidate.get("id_path"))
    return {
        "parent": node,
        "next_entity_info": {
            "entity_id": candidate["entity_id"],
            "entity_name": candidate["entity_name"],
        },
        "candidate": candidate,
        "next_walk_path": next_walk_path,
        "next_evidence_path": next_evidence_path,
        "next_id_path": next_id_path,
        "eval_walk_path": node.walk_path + next_walk_path,
        "eval_id_path": node.id_path + next_id_path,
    }


def expand_node(search, node):
    if (
        node.is_virtual_root
        or node.is_terminal(search.max_depth)
        or node.is_fully_expanded
    ):
        return []

    subgoal_context = search._get_subgoal_context()
    entity_id, entity_name = node.entity_id, node.entity_name
    incoming_relation = node.entity_incoming
    relation_history = [incoming_relation] if incoming_relation else []
    candidates = search.kg_retrieval.relations_search(
        entity_id=entity_id,
        relation_history=relation_history,
    )
    relations = search.kg_retrieval.relations_prune(
        entity_id=entity_id,
        entity_name=entity_name,
        candidate_relations=candidates,
        walk_path_history=node.walk_path,
        subgoal_context=subgoal_context,
    )

    if not relations:
        node.is_fully_expanded = True
        return []

    remaining_relations = [
        relation
        for relation in relations
        if (relation["relation"], relation["head"]) not in node.expanded_relations
    ]
    logger.debug(
        "  %s has %s relations that have not been expanded yet.",
        entity_name,
        len(remaining_relations),
    )
    if not remaining_relations:
        node.is_fully_expanded = True
        return []

    expansion_candidates = []
    existing_paths = {node_path_key(child) for child in node.children}
    for relation_info in remaining_relations:
        relation_name = relation_info["relation"]
        relation_direction = relation_info["head"]
        node.expanded_relations.append((relation_name, relation_direction))
        logger.info(
            "  Try expand relation: %s | direction=%s",
            relation_name,
            relation_direction,
        )

        targets = search.kg_retrieval.entity_search(
            entity_id,
            relation_name,
            relation_direction,
        )
        if not targets:
            continue

        candidate_pool, cvt_ids = [], []
        for target_id in targets:
            if target_id == entity_id or target_id in node.visited_ids:
                continue
            target_name = get_entity_name(target_id)
            if target_name == "Unknown":
                cvt_ids.append(target_id)
            else:
                candidate_pool.append(build_candidate(
                    source_name=entity_name,
                    source_id=entity_id,
                    relation=relation_name,
                    relation_direction=relation_direction,
                    entity_id=target_id,
                    entity_name=target_name,
                ))

        if cvt_ids:
            resolved = search.kg_retrieval.resolve_unknown_entities(
                source_id=entity_id,
                source_name=entity_name,
                unknown_ids=cvt_ids,
                first_rel=relation_name,
                first_is_head=relation_direction,
                visited_ids=node.visited_ids,
                walk_path_history=node.walk_path,
                topk=search.relation_topk,
                subgoal_context=subgoal_context,
            )
            candidate_pool.extend(
                build_candidate(
                    source_name=entity_name,
                    source_id=entity_id,
                    relation=candidate["rel2"],
                    relation_direction=candidate["rel2_direction"],
                    entity_id=candidate["entity_id"],
                    entity_name=candidate["entity_name"],
                    walk_path=candidate["walk_path"],
                    evidence_path=candidate["evidence_path"],
                    id_path=candidate["id_path"],
                )
                for candidate in resolved
            )

        if len(candidate_pool) > search.entity_topk:
            indices = search.kg_retrieval.entity_prune(
                cand_pool=candidate_pool,
                entity_info=node.entity_info,
                walk_path=node.walk_path,
                relation=relation_name,
                subgoal_context=subgoal_context,
            )
            candidate_pool = [
                candidate_pool[index]
                for index in indices
                if 0 <= index < len(candidate_pool)
            ]
        if not candidate_pool:
            continue

        for candidate in candidate_pool:
            next_path = path_key(
                node.id_path + ensure_triple_list(candidate["id_path"])
            )
            if next_path in existing_paths:
                continue
            existing_paths.add(next_path)
            expansion_candidates.append(
                _make_expansion_candidate(node, candidate)
            )

    if len(node.expanded_relations) >= len(relations):
        node.is_fully_expanded = True

    deduplicated = deduplicate_cvt_candidates(expansion_candidates)
    if len(deduplicated) < len(expansion_candidates):
        logger.debug(
            "  Deduplicated CVT-equivalent candidate paths: %s -> %s",
            len(expansion_candidates),
            len(deduplicated),
        )
    return deduplicated


# --- mcts_core ---

class KgSearchUtils:
    def __init__(self, question, rerank_model, model, relation_topk, entity_topk, cvt_relation_topk=2):
        self.question = question
        self.rerank_model = rerank_model
        self.model = model
        self.relation_topk = relation_topk
        self.cvt_relation_topk = cvt_relation_topk
        self.entity_topk = entity_topk
        self.cvt_relation_cache = {}
        self.cvt_target_cache = {}

    def _rerank_relations_for_fallback(self, candidate_relations, entity_name, walk_path_history_text,
                                       subgoal_context, topk):
        if not candidate_relations:
            return []

        if not self.rerank_model:
            return candidate_relations[:topk]

        query_text = (
            f"Question: {self.question}\n"
            f"Current entity: {entity_name}\n"
            f"Known path: {walk_path_history_text}\n"
            f"Subgoal state:\n{subgoal_context}"
        )
        relation_docs = []
        for rel_info in candidate_relations:
            direction = "outgoing" if rel_info["head"] else "incoming"
            relation_text = format_relation_name(rel_info["relation"], "readable")
            relation_docs.append(f"Candidate relation: {relation_text} ({direction})")

        try:
            scores = self.rerank_model.predict([(query_text, doc) for doc in relation_docs])
        except Exception:
            logger.exception("Relation fallback rerank failed; using original relation order.")
            return candidate_relations[:topk]

        scored_relations = list(zip(candidate_relations, scores))
        scored_relations.sort(key=lambda item: float(item[1]), reverse=True)

        return [rel_info for rel_info, _ in scored_relations[:topk]]

    def relations_search(self, entity_id, relation_history=None):
        # 去除上一跳的反向关系，避免 A-r-B 后立刻 B-r^-1-A 成环。
        blocked_reverse = None
        if relation_history:
            last_relation, last_head = relation_history[-1]
            blocked_reverse = (last_relation, not last_head)

        if not is_freebase_entity_id(entity_id):
            return []

        sparql_head = SPARQL_SEARCH_OUT_REL % entity_id
        head_res = execute_sparql(sparql_head)
        head_relations = [normalize_relation(r["relation"]["value"]) for r in head_res]

        sparql_tail = SPARQL_SEARCH_IN_REL % entity_id
        tail_res = execute_sparql(sparql_tail)
        tail_relations = [normalize_relation(r["relation"]["value"]) for r in tail_res]

        total_relations = []
        seen = set()
        for relations, is_head in ((head_relations, True), (tail_relations, False)):
            for rel in relations:
                key = (rel, is_head)
                if blocked_reverse is not None and key == blocked_reverse:
                    continue
                if is_noise_relation(rel) or key in seen:
                    continue
                seen.add(key)
                total_relations.append({"relation": rel, "head": is_head})

        return total_relations

    def relations_prune(self, entity_id, entity_name, candidate_relations, walk_path_history,
                        topk=None, subgoal_context=None):
        if not candidate_relations:
            return []

        retain_topk = topk if topk else self.relation_topk
        if len(candidate_relations) <= retain_topk:
            return [
                {
                    "entity": entity_id,
                    "relation": rel_info["relation"],
                    "head": rel_info["head"],
                }
                for rel_info in candidate_relations
            ]

        relation_lines = []
        index_relation = {}
        for idx, rel_info in enumerate(candidate_relations):
            direction = "outgoing" if rel_info["head"] else "incoming"
            relation_text = format_relation_name(rel_info["relation"], "readable")
            relation_lines.append(f"[{idx}] {relation_text} ({direction})")
            index_relation[idx] = rel_info

        walk_path_history_text = (format_reasoning_path(walk_path_history, fallback_sep=", ") if walk_path_history else "None")
        system_prompt = extract_relation_system_prompt
        subgoal_ctx = subgoal_context if subgoal_context else "None"
        user_prompt = extract_relation_user_prompt.substitute(
            topk=retain_topk,
            question=self.question,
            path_history=walk_path_history_text,
            entity=entity_name,
            subgoal_context=subgoal_ctx,
            relations="\n".join(relation_lines),
        )
        llm_output = self.model(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)[0]
        success, pruned_relations = parse_pruned_relations(llm_output, entity_id, index_relation)

        if not success:
            logger.warning("LLM relation prune failed; falling back to reranked top %s relations.", retain_topk)
            fallback_relations = self._rerank_relations_for_fallback(
                candidate_relations=candidate_relations,
                entity_name=entity_name,
                walk_path_history_text=walk_path_history_text,
                subgoal_context=subgoal_ctx,
                topk=retain_topk,
            )
            pruned_relations = [
                {
                    "entity": entity_id,
                    "relation": rel_info["relation"],
                    "head": rel_info["head"],
                }
                for rel_info in fallback_relations
            ]

        if len(pruned_relations) > retain_topk:
            pruned_relations = pruned_relations[:retain_topk]
        return pruned_relations

    def entity_search(self, entity, relation, head=True):
        if not is_freebase_entity_id(entity):
            return []

        if head:
            entities = execute_sparql(SPARQL_SEARCH_TAIL_ENTITIES % (entity, relation))
        else:
            entities = execute_sparql(SPARQL_SEARCH_HEAD_ENTITIES % (relation, entity))

        entity_ids = extract_entity_ids(entities)
        return [entity_id for entity_id in entity_ids if is_valid_candidate_entity(entity_id)]


    def resolve_unknown_entities(self, source_id, source_name, unknown_ids, first_rel, first_is_head, visited_ids,
                                 walk_path_history=None, topk=2, subgoal_context=None):
        logger.debug("      [CVT]Processing %s unknown entities...", len(unknown_ids))
        unknown_with_relations = {}

        for cvt_id in unknown_ids:
            if cvt_id in self.cvt_relation_cache:
                second_relations = self.cvt_relation_cache[cvt_id]
            else:
                second_relations = self.relations_search(cvt_id)
                self.cvt_relation_cache[cvt_id] = second_relations
            if second_relations:
                unknown_with_relations[cvt_id] = second_relations

        if not unknown_with_relations:
            return []

        grouped_relation_candidates = {}
        for cvt_id, second_relations in unknown_with_relations.items():
            for rel2_info in second_relations:
                rel2_name, rel2_is_head = rel2_info["relation"], rel2_info["head"]
                # 第一跳的直接反向边只能回到 source，后续也会被 visited_ids 丢弃；
                # 在 relation-group 选择前去掉它，避免占用 CVT relation top-k。
                if (rel2_name, rel2_is_head) == (first_rel, not first_is_head):
                    continue
                if is_noise_relation(rel2_name):
                    continue
                group_key = (rel2_name, rel2_is_head)
                group = grouped_relation_candidates.setdefault(group_key, {
                    "cvt_ids": [], "rel2": rel2_name, "head": rel2_is_head
                })
                group["cvt_ids"].append(cvt_id)

        relations_groups = list(grouped_relation_candidates.values())
        if not relations_groups:
            return []

        cvt_relations_topk = self.cvt_relation_topk
        first_rel_text = freebase_rel_to_text(first_rel)
        first_triple = construct_walk_triple(source_name, first_rel, "[CVT]", first_is_head)
        if walk_path_history:
            full_walk_path_history = walk_path_history + [first_triple]
        else:
            full_walk_path_history = [first_triple]
        walk_path_history_text = format_reasoning_path(full_walk_path_history, fallback_sep="; ")

        relation_lines = []
        index_to_group = {}
        for idx, group in enumerate(relations_groups):
            rel2_name = group["rel2"]
            direction = "outgoing" if group["head"] else "incoming"
            relation_lines.append(f"[{idx}] {format_relation_name(rel2_name, 'readable')} ({direction})")
            index_to_group[idx] = group

        subgoal_ctx = subgoal_context if subgoal_context else "None"
        user_cvt_prompt = cvt_relation_user_prompt.substitute(
            topk=cvt_relations_topk,
            question=self.question,
            path_history=walk_path_history_text,
            current_entity=source_name,
            first_relation=first_rel_text,
            subgoal_context=subgoal_ctx,
            relations="\n".join(relation_lines),
        )
        llm_output = self.model(system_prompt=cvt_relation_system_prompt, user_prompt=user_cvt_prompt, temperature=0.0)[0]

        selected_indices = extract_candidate_indices(llm_output, set(index_to_group.keys()))
        if not selected_indices:
            logger.warning("LLM CVT relation prune failed; using reranked fallback.")
            fallback_relations = self._rerank_relations_for_fallback(
                candidate_relations=[
                    {"relation": group["rel2"], "head": group["head"]}
                    for group in relations_groups
                ],
                entity_name=source_name,
                walk_path_history_text=walk_path_history_text,
                subgoal_context=subgoal_ctx,
                topk=cvt_relations_topk,
            )
            fallback_keys = {
                (item["relation"], item["head"])
                for item in fallback_relations
            }
            selected_indices = [
                idx for idx, group in index_to_group.items()
                if (group["rel2"], group["head"]) in fallback_keys
            ]

        selected_groups = [
            index_to_group[idx]
            for idx in selected_indices[:cvt_relations_topk]
        ]

        relation_candidates = []
        seen_keys = set()
        for group in selected_groups:
            for entity in group["cvt_ids"]:
                key = (entity, group["rel2"], group["head"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                relation_candidates.append(key)

        if not relation_candidates:
            return []

        path_candidates = []
        path_documents = []
        seen_path_keys = set()
        for cvt_id, rel2_name, rel2_direction in relation_candidates:
            part_path_key = (cvt_id, rel2_name, rel2_direction)
            if part_path_key in self.cvt_target_cache:
                second_targets = self.cvt_target_cache[part_path_key]
            else:
                second_targets = self.entity_search(cvt_id, rel2_name, rel2_direction)
                self.cvt_target_cache[part_path_key] = second_targets
            if not second_targets:
                continue
            for tgt_id in second_targets:
                if tgt_id in visited_ids:
                    continue
                tgt_name = get_entity_name(tgt_id)
                if tgt_name == "Unknown":
                    continue
                path_key = (cvt_id, rel2_name, rel2_direction, tgt_id)
                if path_key in seen_path_keys:
                    continue
                seen_path_keys.add(path_key)
                cvt_name = f"[CVT:{cvt_id}]"
                walk_path = [
                    construct_walk_triple(source_name, first_rel, cvt_name, first_is_head),
                    construct_walk_triple(cvt_name, rel2_name, tgt_name, rel2_direction),
                ]
                evidence_path = [
                    construct_evidence_triple(source_name, first_rel, cvt_name, first_is_head),
                    construct_evidence_triple(cvt_name, rel2_name, tgt_name, rel2_direction),
                ]
                id_path = [
                    construct_id_triple(source_id, first_rel, cvt_id, first_is_head),
                    construct_id_triple(cvt_id, rel2_name, tgt_id, rel2_direction),
                ]
                path_documents.append(
                    "Possible reasoning path: "
                    f"{format_reasoning_path(walk_path, fallback_sep='; ')}"
                )
                path_candidates.append({
                    "rel2": rel2_name,
                    "rel2_direction": rel2_direction,
                    "entity_id": tgt_id,
                    "entity_name": tgt_name,
                    "walk_path": walk_path,
                    "evidence_path": evidence_path,
                    "id_path": id_path,
                })

        if not path_candidates:
            return []

        logger.debug("      [CVT]Length of path candidates before reranking: %s", len(path_candidates))
        path_pairs = [
            (self.question, document)
            for document in path_documents
        ]
        path_scores = self.rerank_model.predict(path_pairs) if self.rerank_model else [0.0] * len(path_pairs)
        ranked_candidates = sorted(
            zip(path_candidates, path_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [candidate for candidate, _ in ranked_candidates[:topk]]

    def entity_prune(self, cand_pool, entity_info, walk_path, relation, topk=None, subgoal_context=None):
        if not cand_pool:
            return []

        # 实体剪枝的topk
        entity_topk = topk if topk else self.entity_topk

        # 1.提取各实体及其推理路径，保存推理路径是因为通过CVT处理的节点的推理路径可能不同
        candidate_records = []
        for candidate_index, cand_item in enumerate(cand_pool):
            entity_id = cand_item["entity_id"]
            entity_name = cand_item["entity_name"] if cand_item["entity_name"] else cand_item["entity_id"]
            candidate_walk_path = cand_item["walk_path"] if isinstance(cand_item["walk_path"], list) else [cand_item["walk_path"]]

            candidate_records.append({
                "candidate_index": candidate_index,
                "entity_id": entity_id,
                "entity_name": entity_name,
                "walk_path": candidate_walk_path,
            })

        if not candidate_records:
            return []

        # 2.构造查询
        current_entities = [entity_info["entity_name"]] if entity_info and entity_info.get("entity_name") else []
        current_entities_text = ", ".join(current_entities) if current_entities else "None"
        walk_path_history_text = (format_reasoning_path(walk_path, fallback_sep=", ") if walk_path else "None")
        relation_text = freebase_rel_to_text(relation)
        subgoal_ctx = subgoal_context if subgoal_context else "None"

        query_text = (
            f"Question: {self.question}\n"
            f"Current search entities: {current_entities_text}\n"
            f"Current relation: {relation_text}\n"
            f"Known path: {walk_path_history_text}\n"
            f"Subgoal state:\n{subgoal_ctx}"
        )

        # 3.先用rerank模型对候选实体及其推理路径做粗排
        coarse_limit = max(50, entity_topk * 10)
        candidate_docs = []
        if len(candidate_records) > coarse_limit:
            for cand in candidate_records:
                walk_path_text = format_reasoning_path(cand["walk_path"], fallback_sep=", ")
                candidate_docs.append(f"Possible next entity: {cand['entity_name']} (id: {cand['entity_id']}), with its path: {walk_path_text}")
            pairs = [(query_text, doc) for doc in candidate_docs]

            try:
                scores = self.rerank_model.predict(pairs)
            except Exception:
                scores = [0.0] * len(candidate_records)

            scored_candidates = list(zip(candidate_records, scores))
            scored_candidates.sort(key=lambda x: float(x[1]), reverse=True)
            llm_candidates = [cand for cand, _ in scored_candidates[:coarse_limit]]
        else:
            llm_candidates = candidate_records

        if not llm_candidates:
            return []

        # 4. 构造LLM的候选列表
        indexed_candidates = {idx: cand for idx, cand in enumerate(llm_candidates)}
        candidate_lines = []
        for idx, cand in indexed_candidates.items():
            walk_path_text = format_reasoning_path(cand["walk_path"], fallback_sep=", ")
            candidate_lines.append(f"[{idx}] {cand['entity_name']} | local_path={walk_path_text}")

        system_prompt = entity_prune_system_prompt
        user_prompt = entity_prune_user_prompt.substitute(
            topk=entity_topk,
            question=self.question,
            path_history=walk_path_history_text,
            current_entities=current_entities_text,
            relation=relation_text,
            subgoal_context=subgoal_ctx,
            candidates="\n".join(candidate_lines),
        )

        llm_output = self.model(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0)[0]
        selected_indices = extract_candidate_indices(llm_output, set(indexed_candidates.keys()))

        if not selected_indices:
            logger.warning("LLM entity prune failed; using reranked fallback.")
            fallback_docs = []
            for candidate in indexed_candidates.values():
                candidate_path = format_reasoning_path(candidate["walk_path"], fallback_sep=", ")
                fallback_docs.append(
                    f"Possible next entity: {candidate['entity_name']} (id: {candidate['entity_id']}); path: {candidate_path}"
                )
            if self.rerank_model:
                try:
                    fallback_scores = self.rerank_model.predict(
                        [(query_text, document) for document in fallback_docs]
                    )
                except Exception:
                    logger.exception("Entity fallback rerank failed; using candidate order.")
                    fallback_scores = [0.0] * len(fallback_docs)
            else:
                fallback_scores = [0.0] * len(fallback_docs)
            selected_indices = [
                idx for idx, _ in sorted(
                    zip(indexed_candidates.keys(), fallback_scores),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )[:entity_topk]
            ]

        if len(selected_indices) > entity_topk:
            selected_indices = selected_indices[:entity_topk]

        # 5. 返回LLM选中的candidate，不再补齐到entity_topk
        return [
            indexed_candidates[idx]["candidate_index"]
            for idx in selected_indices[:entity_topk]
        ]


# --- mcts_core ---

def _backup_path(child, reward):
    """Back up one evaluated path with unit weight."""
    current = child
    while current:
        current.uct_visits += 1
        current.value_weight += 1.0
        current.total_reward += reward
        current = current.parent


def backup_evaluations(child_rewards):
    """Back up each evaluated child once, independent of its sibling batch."""
    for child, reward in child_rewards:
        _backup_path(child, reward)


def backpropagate(children):
    backup_evaluations([(child, float(child.path_value or 0.0))
                        for child in children or [] if child.parent is not None])


def evaluate_paths(question, reasoner_model, path_requests):
    if not path_requests:
        return []

    results = [None] * len(path_requests)
    pending = []
    for index, request in enumerate(path_requests):
        walk_path = ensure_triple_list(request.get("walk_path"))
        if not walk_path:
            results[index] = 0.0
            continue

        pending.append({
            "index": index,
            "walk_path": walk_path,
        })

    if not pending:
        return results

    candidate_lines = [
        "[{index}] {path}".format(
            index=item["index"],
            path=format_reasoning_path(
                item["walk_path"],
                fallback_sep=", ",
                relation_style="readable",
            ),
        )
        for item in pending
    ]
    user_prompt = evaluate_user_prompt.substitute(
        question=question,
        candidate_paths="\n".join(candidate_lines),
    )
    max_tokens = max(256, min(2048, 48 * len(pending) + 128))
    output = reasoner_model(
        system_prompt=evaluate_system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )[0]
    parsed = parse_batch_evaluation_output(
        output,
        [item["index"] for item in pending],
    )

    for item in pending:
        index = item["index"]
        if index in parsed:
            value = parsed[index]
        else:
            logger.warning(
                "Batch path evaluation missed candidate %s; using zero score.",
                index,
            )
            value = 0.0
        results[index] = value
    return results


def simulate_candidates(search, expansion_candidates):
    if not expansion_candidates:
        return []

    evaluated = evaluate_paths(
        question=search.question,
        reasoner_model=search.reasoner_model,
        path_requests=[
            {
                "walk_path": item["eval_walk_path"],
            }
            for item in expansion_candidates
        ],
    )

    children = []
    for index, item in enumerate(expansion_candidates):
        value = evaluated[index]
        logger.info(
            "  Score=%.1f | %s",
            float(value or 0.0),
            format_reasoning_path(item["eval_walk_path"], relation_style="raw"),
        )
        candidate = item["candidate"]
        child = item["parent"].add_child(
            new_entity_info=item["next_entity_info"],
            new_walk_path=item["next_walk_path"],
            relation=candidate["relation"],
            head=candidate["head"],
            new_evidence_path=item["next_evidence_path"],
            new_id_path=item["next_id_path"],
        )
        child.path_value = value
        if not is_freebase_entity_id(child.entity_id):
            child.is_fully_expanded = True
        children.append(child)
    return children
