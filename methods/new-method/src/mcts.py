import logging

from mcts_core import KgSearchUtils
from mcts_core import MCTSNode
from mcts_core import backpropagate
from mcts_core import expand_node
from mcts_core import evaluate_and_add_children
from evidence import (
    get_final_answer_nodes,
)
from evidence import make_path_record
from evidence import try_early_stop
from sparql_utils import get_entity_name
from subgoal import SubgoalDecomposer, SubgoalState

logger = logging.getLogger(__name__)


class MCTSPathFinder:
    def __init__(self, search_model, planner_model, reasoner_model, rerank_model,
                 question, topic_entities,
                 max_iterations=5, max_depth=5, c_uct=0.15, relation_topk=5, entity_topk=5, cvt_relation_topk=2,
                 answer_threshold=0.9, good_threshold=0.7, path_topk=10,
                 enable_decom=True, step_history=None,
    ):
        ## LLM与功能开关
        self.search_model = search_model
        self.planner_model = planner_model if planner_model else search_model
        self.reasoner_model = reasoner_model if reasoner_model else search_model
        enable_decom = bool(enable_decom)
        ## 当前样本的问题和初始实体
        self.question = question
        self.top_entities = topic_entities

        ## MCTS超参数
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.c_uct = c_uct
        self.relation_topk = relation_topk
        self.entity_topk = entity_topk
        self.path_topk = max(1, int(path_topk or 1))

        # Answer verification and selection use the path-value scale.
        self.answer_threshold = answer_threshold
        self.good_threshold = good_threshold

        ## Early-stop verification state
        self.early_stop_verification_cache = {}
        self.early_stop_answer = ""
        self.actual_iterations = 0
        self.termination_reason = "not_started"
        self.final_answer_nodes = []
        self.step_history = step_history if step_history is not None else []
        ## 创建虚拟根
        self.root = MCTSNode(entity_info={}, is_virtual_root=True)

        ## 每个 topic entity 创建一个真实根
        if isinstance(topic_entities, list):

            entity_list = [{"entity_id": eid, "entity_name": get_entity_name(eid)} for eid in topic_entities]
        else:
            entity_list = [{"entity_id": entity_id, "entity_name": name} for entity_id, name in topic_entities.items()]
        self.initial_topic_entities = [dict(entity) for entity in entity_list]

        for ent_info in entity_list:
            real_root = MCTSNode(
                entity_info=ent_info,
                parent=self.root,
                visited_ids={ent_info["entity_id"]},
            )
            self.root.children.append(real_root)

        ## 初始化KG检索工具类
        self.kg_retrieval = KgSearchUtils(
            model=self.search_model,
            rerank_model=rerank_model,
            question=self.question,
            relation_topk=self.relation_topk,
            cvt_relation_topk=cvt_relation_topk,
            entity_topk=self.entity_topk,
        )

        ## 问题分解与全局子目标状态
        if enable_decom:
            subgoals, self.is_complex = SubgoalDecomposer(self.planner_model).decompose(
                question=self.question,
                topic_entities=topic_entities,
            )
            self.global_subgoal_state = SubgoalState(subgoals)
            logger.debug("  Complexity: %s", "complex" if self.is_complex else "simple")
        else:
            self.is_complex = False
            self.global_subgoal_state = None
            logger.debug("Question decomposition disabled.")

    def _get_subgoal_context(self):
        return self.global_subgoal_state.format_for_prompt() if self.global_subgoal_state else "None"

    def _attempt_early_stop(self, tree_nodes=None):
        result = try_early_stop(
            root=self.root,
            question=self.question,
            reasoner_model=self.reasoner_model,
            good_threshold=self.good_threshold,
            answer_threshold=self.answer_threshold,
            path_topk=self.path_topk,
            verification_cache=self.early_stop_verification_cache,
            tree_nodes=tree_nodes,
        )
        self.step_history.append({"event": "early_stop", "iteration": self.actual_iterations, "status": result["status"]})
        if result["status"] == "accepted":
            self.early_stop_answer = result["answer"]
            self.final_answer_nodes = list(result.get("nodes", []) or [])
            self.termination_reason = "early_stop"
        return result

    def _run_search_control(self):
        result = self._attempt_early_stop()
        return result["paths"] if result["status"] == "accepted" else None

    def search(self):
        self.actual_iterations = 0
        self.termination_reason = "max_iterations"
        self.final_answer_nodes = []
        for i in range(self.max_iterations):
            self.actual_iterations = i + 1
            logger.info("========== Iteration %s/%s ==========", i + 1, self.max_iterations)
            node = self._select(self.root)
            if node is None:
                self.termination_reason = "no_expandable_node"
                break
            logger.info("Selected entity: %s", node.entity_info.get("entity_name", ""))
            self.step_history.append({
                "event": "selection", "iteration": i + 1,
                "entity_id": node.entity_id, "entity_name": node.entity_name,
                "path": list(node.walk_path), "id_path": list(node.id_path),
            })
            candidates = expand_node(self, node)
            self.step_history.append({
                "event": "expansion", "iteration": i + 1,
                "candidate_count": len(candidates),
                "candidate_paths": [item["eval_walk_path"] for item in candidates],
            })
            children = evaluate_and_add_children(self, candidates) if candidates else []
            if children:
                backpropagate(children)
            self.step_history.append({
                "event": "evaluation_and_backup", "iteration": i + 1,
                "children": [
                    {"entity_id": child.entity_id, "path": list(child.walk_path),
                     "id_path": list(child.id_path), "score": child.path_value}
                    for child in children
                ],
            })
            paths = self._run_search_control()
            if paths is not None:
                return paths
        return self._finish_search()

    def _finish_search(self):
        early_result = self._attempt_early_stop()
        if early_result["status"] == "accepted":
            return early_result["paths"]

        # 搜索结束后收集高质量节点，作为最终回答模块的输入证据。
        best_nodes = get_final_answer_nodes(
            self.root,
            good_threshold=self.good_threshold,
            path_topk=self.path_topk,
        )
        self.final_answer_nodes = list(best_nodes)
        best_paths = [
            make_path_record(node)
            for node in best_nodes
        ]
        return best_paths if best_paths else []

    def _select(self, node):
        while not node.is_terminal(self.max_depth):
            if not node.is_fully_expanded:
                return node
            if not node.children:
                return None
            node = self._get_best_child(node)
            if node is None:
                return None
        return None

    def _has_expandable_node(self, node):
        stack = list(node.children) if node.is_virtual_root else [node]
        while stack:
            current = stack.pop()
            if (
                not current.is_terminal(self.max_depth)
                and not current.is_fully_expanded
            ):
                return True
            stack.extend(current.children)
        return False

    def _get_best_child(self, node):
        if not node.children:
            return None

        expandable_children = [child for child in node.children if self._has_expandable_node(child)]
        if not expandable_children:
            return None

        best_child = None
        best_uct_score = float("-inf")
        for child in expandable_children:
            curr_uct_value = child.get_uct_value(self.c_uct)
            if curr_uct_value > best_uct_score:
                best_uct_score = curr_uct_value
                best_child = child
        return best_child
