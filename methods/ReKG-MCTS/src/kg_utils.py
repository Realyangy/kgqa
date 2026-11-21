import random
import logging

import freebase_func
from prompt_list import entity_prune_prompt
from utils import (
    build_relation_prune_prompt,
    clean_relations,
    parse_selected_ids,
    format_reasoning_path,
)

logger = logging.getLogger(__name__)


class KgSearchUtils:
    def __init__(self, question, model, relation_topk=5, entity_topk=5):
        self.question = question
        self.model = model
        self.relation_topk = relation_topk
        self.entity_topk = entity_topk


    # 根据实体查询其相连关系
    def get_relations(self, entity_id, blocked_reverse=None):
        if not freebase_func.is_freebase_entity_id(entity_id):
            return []

        head_res = freebase_func.execurte_sparql(freebase_func.sparql_search_out_rel % entity_id)
        head_relations = [freebase_func.normalize_relation(item["relation"]["value"]) for item in head_res]
        tail_res = freebase_func.execurte_sparql(freebase_func.sparql_search_in_rel % entity_id)
        tail_relations = [freebase_func.normalize_relation(item["relation"]["value"]) for item in tail_res]

        seen = set()
        total_relations = []
        for relation in head_relations:
            key = (relation, True)
            if blocked_reverse is not None and key == blocked_reverse:
                continue
            if key in seen:
                continue
            seen.add(key)
            total_relations.append({"relation": relation, "head": True})

        for relation in tail_relations:
            key = (relation, False)
            if blocked_reverse is not None and key == blocked_reverse:
                continue
            if key in seen:
                continue
            seen.add(key)
            total_relations.append({"relation": relation, "head": False})

        return sorted(freebase_func.filter_relations(total_relations), key=lambda item: (item["relation"], not item["head"]))


    # 关系扩展
    def relation_search_prune(self, entity_id, entity_name, path_history=None, relation_history=None):
        blocked_reverse = None
        if relation_history:
            last_relation, last_head = relation_history[-1]
            blocked_reverse = (last_relation, not last_head)

        # 关系搜索
        total_relations = self.get_relations(entity_id, blocked_reverse=blocked_reverse)
        logger.info("[Expansion]%s has %d candidates", entity_name, len(total_relations))
        if not total_relations:
            return []

        # 关系剪枝
        retain_topk = min(self.relation_topk, len(total_relations))
        prompt = build_relation_prune_prompt(self.question, entity_name, total_relations, retain_topk, path_history=path_history)
        llm_output = self.model(user_prompt=prompt)[0]
        pruned_relations = clean_relations(llm_output, entity_id, total_relations)
        if not pruned_relations:
            logger.warning("[Expansion]Relation pruning failed for %s.", entity_name)
            return []

        pruned_relations.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        logger.info("[Expansion]Selected relations:\n%s", pruned_relations[:retain_topk])
        return pruned_relations[:retain_topk]


    # 实体搜索
    def entity_search(self, entity_id, entity_name, relation, head=True):

        if not freebase_func.is_freebase_entity_id(entity_id):
            return []

        if head:
            entities = freebase_func.execurte_sparql(freebase_func.sparql_search_tail_entities % (entity_id, relation))
        else:
            entities = freebase_func.execurte_sparql(freebase_func.sparql_search_head_entities % (relation, entity_id))

        entity_ids = freebase_func.replace_entities_prefix(entities)
        valid_ids = [entity_id for entity_id in entity_ids if freebase_func.is_valid_entity(entity_id)]
        logger.info("[Expansion]%s -> %s has %d candidates.", entity_name, relation, len(valid_ids))
        return valid_ids


    # 实体剪枝
    def entity_prune(self, candidate_entities, node, relation):
        candidates = sorted(set(candidate_entities))
        if not candidates:
            return []
        
        # 候选实体过多时，随机采样保留50个
        if len(candidates) > 50:
            candidates = sorted(random.sample(candidates, 50))

        # 查询候选实体名称
        names = freebase_func.get_entity_names(candidates)
        named = [(entity_id, names[entity_id]) for entity_id in candidates if names[entity_id] != "Unknown"]
        if not named:
            chosen = random.sample(candidates, min(self.entity_topk, len(candidates)))
            logger.info("[Expansion]All %d candidates unnamed, retaining %d random IDs.", len(candidates), len(chosen))
            return chosen
        if len(named) <= self.entity_topk:
            return [entity_id for entity_id, _ in named]

        # 调用LLM进行实体剪枝
        current_entities = ", ".join(entity["entity_name"] for entity in node.entities_info)
        path_history = format_reasoning_path(node.path, fallback_sep=", ") or "None"
        prompt = entity_prune_prompt.substitute(
            top_k=self.entity_topk,
            question=self.question,
            current_entities=current_entities,
            current_relation=relation,
            path_history=path_history,
            candidates="\n".join(f"[{index}] {name}" for index, (_, name) in enumerate(named, start=1)),
        )
        output = self.model(user_prompt=prompt)[0]
        selected_ids = parse_selected_ids(output, len(named), self.entity_topk)
        return [named[index - 1][0] for index in selected_ids]
