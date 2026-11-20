import logging
import random

from prompt_list import entity_p_prompt
from freebase_func import (
    execurte_sparql,
    filter_relations,
    get_entity_name,
    is_freebase_entity_id,
    is_valid_entity,
    normalize_relation,
    replace_entities_prefix,
    sparql_search_head_entities,
    sparql_search_in_rel,
    sparql_search_out_rel,
    sparql_search_tail_entities,
)
from utils import (
    build_relation_prune_prompt,
    clean_relations,
    parse_selected_ids,
    format_reasoning_path,
)


logger = logging.getLogger(__name__)


class KgSearchUtils:
    def __init__(
        self,
        question,
        model,
        relation_topk=5,
        entity_topk=5,
        sample_seed=0,
    ):
        self.question = question
        self.model = model
        self.relation_topk = relation_topk
        self.entity_topk = entity_topk
        self.sample_seed = sample_seed

    def get_relations(self, entity_id, blocked_reverse=None):
        if not is_freebase_entity_id(entity_id):
            return []

        head_res = execurte_sparql(sparql_search_out_rel % entity_id)
        head_relations = [
            normalize_relation(item["relation"]["value"]) for item in head_res
        ]

        tail_res = execurte_sparql(sparql_search_in_rel % entity_id)
        tail_relations = [
            normalize_relation(item["relation"]["value"]) for item in tail_res
        ]

        total_relations = []
        seen = set()
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

        return sorted(
            filter_relations(total_relations),
            key=lambda item: (item["relation"], not item["head"]),
        )

    def relation_search_prune(
        self,
        entity_id,
        entity_name,
        path_history=None,
        relation_history=None,
        topk=None,
        include_path_history=False,
    ):
        blocked_reverse = None
        if relation_history:
            last_relation, last_head = relation_history[-1]
            blocked_reverse = (last_relation, not last_head)

        total_relations = self.get_relations(entity_id, blocked_reverse=blocked_reverse)
        logger.info("Relation search: %s (%s), %d candidates", entity_name, entity_id, len(total_relations))
        if not total_relations:
            return []

        requested_topk = topk or self.relation_topk
        retain_topk = min(requested_topk, len(total_relations))
        prompt = build_relation_prune_prompt(
            self.question,
            entity_name,
            total_relations,
            retain_topk,
            path_history=path_history if include_path_history else None,
        )
        llm_output = self.model(user_prompt=prompt)[0]
        success, pruned_relations = clean_relations(
            llm_output,
            entity_id,
            total_relations,
        )

        if not success or not pruned_relations:
            logger.warning(
                "Relation pruning failed for %s. raw_output=%r",
                entity_name,
                llm_output,
            )
            return []

        pruned_relations.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        logger.info("Selected relations: %s", pruned_relations[:retain_topk])
        return pruned_relations[:retain_topk]

    def entity_search(self, entity_id, relation, head=True):
        if not is_freebase_entity_id(entity_id):
            return []

        if head:
            entities = execurte_sparql(
                sparql_search_tail_entities % (entity_id, relation)
            )
        else:
            entities = execurte_sparql(
                sparql_search_head_entities % (relation, entity_id)
            )

        entity_ids = replace_entities_prefix(entities)
        valid_ids = [entity_id for entity_id in entity_ids if is_valid_entity(entity_id)]
        logger.info("Entity search: %s / %s, %d candidates", entity_id, relation, len(valid_ids))
        return valid_ids

    def entity_prune(self, candidate_entities, node, relation, head=True,
                     source_entity_info=None, topk=None):
        """Keep unnamed IDs when all candidates lack names, as in local ToG."""
        candidates = sorted(set(candidate_entities))
        if not candidates:
            return []
        limit = topk or self.entity_topk
        if len(candidates) >= 20:
            candidates = sorted(random.sample(candidates, min(limit, len(candidates))))
        pairs = [(entity_id, get_entity_name(entity_id)) for entity_id in candidates]
        named = [(entity_id, name) for entity_id, name in pairs if name != "Unknown"]
        if not named:
            return [entity_id for entity_id, _ in pairs]
        if len(named) <= limit:
            return [entity_id for entity_id, _ in named]

        current_entities = ", ".join(entity["entity_name"] for entity in node.entities_info)
        path_history = format_reasoning_path(node.path, fallback_sep=", ") or "None"
        prompt = entity_p_prompt.substitute(
            top_k=limit, question=self.question, current_entities=current_entities,
            current_relation=relation, path_history=path_history,
            candidates="\n".join(f"[{index}] {name}" for index, (_, name) in enumerate(named, start=1)),
        )
        output = self.model(user_prompt=prompt)[0]
        selected_ids = parse_selected_ids(output, len(named), limit)
        return [named[index - 1][0] for index in selected_ids]
