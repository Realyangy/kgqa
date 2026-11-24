"""Relation-centric MCTS: search never materializes endpoint entities."""
import logging
import math

from mcts_utils import (
    FREEBASE_NS,
    build_relation_query,
    format_relation_path,
    is_noise_relation,
    parse_batch_evaluation_output,
)
from sparql_utils import execute_sparql
from prompt_list import (
    extract_relation_system_prompt,
    extract_relation_user_prompt,
    evaluate_system_prompt,
    evaluate_user_prompt,
)
from utils import extract_candidate_indices, format_relation_name

logger = logging.getLogger(__name__)


class MCTSNode:
    """一个节点 = 起点实体 + 有向关系链；虚拟根仅负责多起点预算分配。"""
    def __init__(self, topic_entity=None, parent=None, relation_path=None, is_virtual_root=False):
        self.topic_entity = dict(topic_entity or {})
        self.relation_path = [dict(hop) for hop in (relation_path or [])]
        self.parent = parent
        self.children = []
        self.is_virtual_root = is_virtual_root
        self.is_fully_expanded = is_virtual_root or not self.topic_entity
        self.path_value = 0.0
        self.total_reward = 0.0
        self.value_weight = 0.0
        self.uct_visits = 0

    def add_child(self, relation):
        child = MCTSNode(topic_entity=self.topic_entity, parent=self, relation_path=self.relation_path + [relation])
        self.children.append(child)
        return child

    def get_uct_value(self, c_uct):
        if self.is_virtual_root or self.parent is None:
            return 0.0
        if self.uct_visits <= 0 or self.value_weight <= 0:
            return float("inf")
        return self.total_reward / self.value_weight + c_uct * math.sqrt(math.log(max(2, self.parent.uct_visits)) / self.uct_visits)

    def is_terminal(self, max_depth):
        return not self.is_virtual_root and len(self.relation_path) >= max_depth


class KgSearchUtils:
    def __init__(self, question, rerank_model, model, relation_topk):
        self.question = question
        self.rerank_model = rerank_model
        self.model = model
        self.relation_topk = relation_topk

    def relations_search(self, topic_entity_id, relation_path):
        """参考 CoR：在整个关系链匹配结果的末端查询所有可达出入边。"""
        blocked_reverse = None
        if relation_path:
            last = relation_path[-1]
            blocked_reverse = (last["relation"], not last["head"])
        seen = set()
        candidates = []
        for head in (True, False):
            rows = execute_sparql(build_relation_query(topic_entity_id, relation_path, head))
            for row in rows:
                uri = row.get("relation", {}).get("value", "")
                if not uri.startswith(FREEBASE_NS):
                    continue
                relation = uri[len(FREEBASE_NS):]
                key = (relation, head)
                if key == blocked_reverse or key in seen or is_noise_relation(relation):
                    continue
                seen.add(key)
                candidates.append({"relation": relation, "head": head})
        return sorted(candidates, key=lambda item: (item["relation"], not item["head"]))

    def _rerank_relations(self, candidates, topic_entity, relation_path, subgoal_context, topk):
        if not self.rerank_model:
            return candidates[:topk]
        query = (
            f"Question: {self.question}\n"
            f"Known relation path: {format_relation_path(topic_entity, relation_path)}\n"
            f"Static subgoals:\n{subgoal_context}"
        )
        docs = [
            f"Candidate relation: {format_relation_name(item['relation'])} "
            f"({'outgoing' if item['head'] else 'incoming'})"
            for item in candidates
        ]
        try:
            scores = self.rerank_model.predict([(query, doc) for doc in docs])
            if len(scores) != len(candidates):
                raise ValueError("Incomplete reranker scores")
            return [item for item, _ in sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)[:topk]]
        except Exception:
            logger.exception("Relation rerank failed; using candidate order.")
            return candidates[:topk]

    def relations_prune(self, topic_entity, relation_path, candidate_relations, subgoal_context=None):
        if len(candidate_relations) <= self.relation_topk:
            return list(candidate_relations)
        context = subgoal_context or "None"
        # 两个方向合计限制为 1000，超出时先按原有嵌入模型粗排，避免上下文爆炸。
        pool = candidate_relations
        if len(pool) > 1000:
            pool = self._rerank_relations(pool, topic_entity, relation_path, context, 1000)
        lines = [
            f"[{i}] {item['relation']} ({'outgoing' if item['head'] else 'incoming'})"
            for i, item in enumerate(pool)
        ]
        prompt = extract_relation_user_prompt.substitute(question=self.question, topic_entity=topic_entity.get("entity_name") or topic_entity["entity_id"],
                                                         relation_path=format_relation_path(topic_entity, relation_path), subgoal_context=context,
                                                         relations="\n".join(lines), topk=self.relation_topk)
        output = self.model(system_prompt=extract_relation_system_prompt, user_prompt=prompt, temperature=0.0)[0]
        selected = extract_candidate_indices(output, set(range(len(pool))))
        if not selected:
            logger.warning("Invalid relation selection; falling back to reranker.")
            return self._rerank_relations(pool, topic_entity, relation_path, context, self.relation_topk)
        return [pool[i] for i in selected[:self.relation_topk]]


def expand_node(search, node):
    """只扩展关系，不查询或剪枝实体，也不特殊处理 CVT。"""
    if node.is_virtual_root or node.is_fully_expanded or node.is_terminal(search.max_depth):
        return []
    candidates = search.kg_retrieval.relations_search(node.topic_entity["entity_id"], node.relation_path)
    selected = search.kg_retrieval.relations_prune(node.topic_entity, node.relation_path, candidates, search._get_subgoal_context())
    node.is_fully_expanded = True
    logger.info("Relation expansion: %d candidates -> %d selected", len(candidates), len(selected))
    return [
        {"parent": node, "topic_entity": node.topic_entity,
         "relation": hop, "relation_path": node.relation_path + [hop]}
        for hop in selected
    ]


def evaluate_paths(question, reasoner_model, path_requests):
    """一次 LLM 调用批量评价候选关系路径，仅针对原问题。"""
    if not path_requests:
        return []
    lines = [
        f"[{i}] {format_relation_path(item['topic_entity'], item['relation_path'])}"
        for i, item in enumerate(path_requests)
    ]
    output = reasoner_model(system_prompt=evaluate_system_prompt, user_prompt=evaluate_user_prompt.substitute(question=question, candidate_paths="\n".join(lines)),
                            max_tokens=max(256, min(2048, 48 * len(lines) + 128)))[0]
    parsed = parse_batch_evaluation_output(output, range(len(lines)))
    for i in range(len(lines)):
        if i not in parsed:
            logger.warning("Batch relation-path evaluation missed index %d; assigning zero.", i)
    return [parsed.get(i, 0.0) for i in range(len(lines))]


def evaluate_and_add_children(search, expansion_candidates):
    if not expansion_candidates:
        return []
    scores = evaluate_paths(search.question, search.reasoner_model, expansion_candidates)
    children = []
    for item, score in zip(expansion_candidates, scores):
        child = item["parent"].add_child(item["relation"])
        child.path_value = score
        logger.info("  Score=%.1f | %s", score, format_relation_path(child.topic_entity, child.relation_path))
        children.append(child)
    return children


def _backup_path(child, reward):
    """保留原来的逐子节点、单位权重回传。"""
    current = child
    while current:
        current.uct_visits += 1
        current.value_weight += 1.0
        current.total_reward += reward
        current = current.parent


def backpropagate(children):
    for child in children:
        _backup_path(child, float(child.path_value))
