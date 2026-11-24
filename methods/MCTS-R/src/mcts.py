"""Relation-centric MCTS: search never materializes endpoint entities."""
import logging
import math

from kg_utils import KgSearchUtils, get_entity_name, is_freebase_entity_id
from evidence import get_final_answer_nodes, make_path_record, try_early_stop
from prompt_list import evaluate_system_prompt, evaluate_user_prompt
from subgoal import SubgoalDecomposer, SubgoalState
from utils import format_relation_path, parse_batch_evaluation_output

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


class MCTSPathFinder:
    def __init__(self, search_model, planner_model, reasoner_model, rerank_model,
                 question, topic_entities, max_iterations=5, max_depth=5,
                 c_uct=0.15, relation_topk=5, answer_threshold=0.9,
                 good_threshold=0.7, path_topk=10, enable_decom=True,
                 enable_early_stop=True, step_history=None):
        self.question = question
        self.search_model = search_model
        self.planner_model = planner_model or search_model
        self.reasoner_model = reasoner_model or search_model
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.c_uct = c_uct
        self.path_topk = max(1, int(path_topk))
        self.answer_threshold = answer_threshold
        self.good_threshold = good_threshold
        self.enable_early_stop = enable_early_stop
        self.early_stop_verification_cache = {}
        self.actual_iterations = 0
        self.termination_reason = "not_started"
        self.final_answer_nodes = []
        self.step_history = step_history
        self.root = MCTSNode(is_virtual_root=True)
        # 多起点保持独立子树；只在初始化时解析给定起点名称。
        if isinstance(topic_entities, list):
            topics = [(eid, get_entity_name(eid)) for eid in topic_entities if is_freebase_entity_id(eid)]
        else:
            topics = [(eid, name or eid) for eid, name in topic_entities.items() if is_freebase_entity_id(eid)]
        for eid, name in dict(topics).items():
            self.root.children.append(MCTSNode(topic_entity={"entity_id": eid, "entity_name": name}, parent=self.root))
        self.kg_retrieval = KgSearchUtils(question, rerank_model, search_model, relation_topk)
        # 问题分解逻辑、输入及 Prompt 保持不变。
        if enable_decom:
            subgoals, self.is_complex = SubgoalDecomposer(self.planner_model).decompose(question=question, topic_entities=topic_entities)
            self.global_subgoal_state = SubgoalState(subgoals)
        else:
            self.is_complex = False
            self.global_subgoal_state = None

    def _get_subgoal_context(self):
        return self.global_subgoal_state.format_for_prompt() if self.global_subgoal_state else "None"

    def _record(self, event, **fields):
        if self.step_history is not None:
            self.step_history.append({"event": event, "iteration": self.actual_iterations, **fields})

    def _attempt_early_stop(self):
        if not self.enable_early_stop:
            return None
        result = try_early_stop(root=self.root, question=self.question, reasoner_model=self.reasoner_model,
                                good_threshold=self.good_threshold, answer_threshold=self.answer_threshold,
                                path_topk=self.path_topk, verification_cache=self.early_stop_verification_cache)
        self._record("early_stop", status=result["status"],
                     selected_paths=[make_path_record(node) for node in result.get("nodes", [])])
        if result["status"] == "accepted":
            self.final_answer_nodes = result["nodes"]
            self.termination_reason = "relation_early_stop"
            return [make_path_record(node) for node in self.final_answer_nodes]
        return None

    def search(self):
        self.actual_iterations = 0
        self.termination_reason = "max_iterations"
        for i in range(self.max_iterations):
            node = self._select(self.root)
            if node is None:
                self.termination_reason = "no_expandable_node"
                break
            self.actual_iterations = i + 1
            logger.info("========== Iteration %d/%d ==========", i + 1, self.max_iterations)
            logger.info("Selected path: %s", format_relation_path(node.topic_entity, node.relation_path))
            self._record("selection", path=make_path_record(node))
            candidates = expand_node(self, node)
            self._record("expansion", candidate_paths=[
                {"topic_entity": item["topic_entity"], "relation_path": item["relation_path"]}
                for item in candidates
            ])
            children = evaluate_and_add_children(self, candidates)
            backpropagate(children)
            self._record("evaluation_and_backup", children=[make_path_record(child) for child in children])
            paths = self._attempt_early_stop()
            if paths is not None:
                return paths
        # 无需重复验证未变化的树；迭代结束时返回高分关系链。
        self.final_answer_nodes = get_final_answer_nodes(self.root, self.good_threshold, self.path_topk)
        return [make_path_record(node) for node in self.final_answer_nodes]

    def _select(self, node):
        while not node.is_terminal(self.max_depth):
            if not node.is_fully_expanded:
                return node
            node = self._get_best_child(node)
            if node is None:
                return None
        return None

    def _has_expandable_node(self, node):
        stack = list(node.children) if node.is_virtual_root else [node]
        while stack:
            current = stack.pop()
            if not current.is_terminal(self.max_depth) and not current.is_fully_expanded:
                return True
            stack.extend(current.children)
        return False

    def _get_best_child(self, node):
        candidates = [child for child in node.children if self._has_expandable_node(child)]
        if not candidates:
            return None
        return max(candidates, key=lambda child: child.get_uct_value(self.c_uct))


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
