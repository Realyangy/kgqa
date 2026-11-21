import logging
import math
import random

from kg_utils import KgSearchUtils
from prompt_list import evaluate_prompt
from freebase_func import get_entity_name
from utils import construct_triple, format_reasoning_path, parse_json_object


logger = logging.getLogger(__name__)


class MCTSNode:
    def __init__(
        self,
        entities_info,
        triples=None,
        parent=None,
        visited_ids=None,
        entity_incoming=None,
    ):
        self.entities_info = [dict(entity) for entity in entities_info]
        self.parent = parent
        self.children = []

        self.path = triples or []
        self.v = 0.0
        self.explanation = None

        # The paper initializes every newly created node with one visit.
        self.visits = 1
        self.total_reward = 0.0
        self.is_fully_expanded = False

        self.expanded_relations = {
            entity["entity_id"]: set() for entity in self.entities_info
        }
        self.fully_expanded_entities = set()
        self.visited_ids = set(visited_ids or set())
        if entity_incoming is not None:
            self.entity_incoming = dict(entity_incoming)
        else:
            self.entity_incoming = {
                entity["entity_id"]: None for entity in self.entities_info
            }

    @property
    def y(self):
        return self.path

    @property
    def average_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else self.v

    def add_child(self, entity_id, new_entity_info, new_triple, relation, head):
        new_entities = []
        new_incoming = {}
        for entity in self.entities_info:
            if entity["entity_id"] == entity_id:
                continue
            copied = dict(entity)
            new_entities.append(copied)
            new_incoming[copied["entity_id"]] = self.entity_incoming.get(
                copied["entity_id"]
            )

        new_entities.append(dict(new_entity_info))
        new_incoming[new_entity_info["entity_id"]] = (relation, head)
        new_path = self.path + [new_triple]

        child = MCTSNode(
            entities_info=new_entities,
            triples=new_path,
            parent=self,
            visited_ids=self.visited_ids | {new_entity_info["entity_id"]},
            entity_incoming=new_incoming,
        )
        self.children.append(child)
        self.expanded_relations.setdefault(entity_id, set()).add((relation, head))
        return child

    def mark_entity_fully_expanded(self, entity_id):
        self.fully_expanded_entities.add(entity_id)
        entity_ids = {entity["entity_id"] for entity in self.entities_info}
        self.is_fully_expanded = entity_ids.issubset(self.fully_expanded_entities)

    def get_ucb_value(self, exploration_constant):
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration

    def is_terminal(self, max_depth):
        return len(self.path) >= max_depth or (self.is_fully_expanded and not self.children)

    def print_tree(self, limit=5):
        """Print the MCTS tree structure."""
        logger.info("  After Iteration, the Structure of MCTS is:")
        logger.info("  Node: visits=%s, value=%.3f, avg_reward=%.3f, total_reward=%.3f, path_len=%s",
                     self.visits, self.v, self.average_reward, self.total_reward, len(self.path))
        for i, child in enumerate(self.children):
            child._print_subtree(limit, prefix="", is_last=(i == len(self.children) - 1), depth=1)

    def _print_subtree(self, limit, prefix, is_last, depth):
        if depth > limit:
            return
        connector = "  └── " if is_last else "  ├── "
        logger.info("%s%sNode: visits=%s, value=%.3f, avg=%.3f, total_reward=%.3f, path_len=%s",
                     prefix, connector, self.visits, self.v,
                     self.average_reward, self.total_reward, len(self.path))
        next_prefix = prefix + ("      " if is_last else "  │   ")
        for i, child in enumerate(self.children):
            child._print_subtree(limit=limit, prefix=next_prefix,
                                 is_last=(i == len(self.children) - 1), depth=depth + 1)

    def __repr__(self):
        return (
            "MCTSNode("
            f"entities={len(self.entities_info)}, "
            f"path_len={len(self.path)}, "
            f"value={self.v:.3f}, "
            f"visits={self.visits}, "
            f"avg={self.average_reward:.3f}, "
            f"children={len(self.children)})"
        )


class MCTSPathFinder:
    def __init__(self, question, topic_entities, llm, evaluator=None,
                    max_depth=5, max_iterations=2, exploration_constant=0.5,relation_topk=5, entity_topk=5, 
                    rollout_steps=2, rollout_count=2, beam_size=2, score_threshold=0.8
                ):
        
        self.question = question
        self.llm = llm
        self.evaluator = evaluator if evaluator is not None else llm
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.rollout_steps = rollout_steps
        self.rollout_count = rollout_count
        self.beam_size = beam_size
        self.score_threshold = score_threshold

        # 提取起点实体的名称
        if isinstance(topic_entities, list):
            entities_info = [{"entity_id": entity_id, "entity_name": get_entity_name(entity_id)} for entity_id in topic_entities]
        else:
            entities_info = [{"entity_id": entity_id, "entity_name": name} for entity_id, name in topic_entities.items()]

        # 构造根节点
        self.root = MCTSNode(
            entities_info=entities_info,
            visited_ids={entity["entity_id"] for entity in entities_info},
        )

        # 初始化KG工具
        self.kg_utils = KgSearchUtils(question=question, model=llm, relation_topk=relation_topk, entity_topk=entity_topk)

        self.best_path = []
        self.best_path_value = 0.0
        self.best_path_source = None
        self.trace_events = []


    # 更新最佳路径
    def _update_best_path(self, path, value, source):
        if not path or (self.best_path and value <= self.best_path_value):
            return False
        self.best_path = list(path)
        self.best_path_value = value
        self.best_path_source = source
        return True


    # MCTS主流程
    def search(self):
        if not self.root.entities_info:
            logger.warning("No topic entities found for question: %s", self.question)
            return []

        for iteration in range(self.max_iterations):
            logger.info("===== Iteration %s/%s =====", iteration + 1, self.max_iterations)

            # 选择阶段
            logger.info("[Main]Selection")
            leaf = self._select(self.root)
            if leaf is None:
                logger.info("[Select]No expandable node remains.")
                break

            # 扩展阶段
            logger.info("[Main]Expansion")
            children = []
            if not leaf.is_terminal(self.max_depth):
                children = self._expand(leaf)

            # 成功扩展子节点
            if children:
                children.sort(key=lambda child: child.v, reverse=True)
                best_child = children[0]
                logger.info("[Expand]%s", format_reasoning_path(best_child.path, fallback_sep=", "))
                self._update_best_path(best_child.path, best_child.v, "expanded_child")
                
                # 子节点价值高于阈值，直接用于回答
                if self.best_path_value >= self.score_threshold:
                    logger.info("[Expand]Early stop triggered after expansion by %s | value=%.3f", self.best_path_source, self.best_path_value)
                    logger.info("[Expand]Final best path: %s", format_reasoning_path(self.best_path, fallback_sep=", "))
                    logger.info("[Expand]Path value: %.3f", self.best_path_value)
                    self.root.print_tree()
                    return self.best_path

            # 扩展失败
            else:
                logger.info("[Expand]No children expanded from this node.")
                if not leaf.children:
                    leaf.is_fully_expanded = True
                    logger.info("[Expand]Marked node as terminal: no child paths.")

            # 模拟阶段
            logger.info("[Main]Simulation and Backpropagation")
            if leaf.children and len(leaf.path) < self.max_depth:
                for index in range(self.rollout_count):
                    sim_path = self._simulate(leaf)
                    if not sim_path:
                        continue

                    # 评估并回传
                    reward, explanation = self.evaluate(sim_path)
                    self._backpropagate(leaf, reward)
                    self.trace_events.append({
                        "event": "rollout",
                        "iteration": iteration + 1,
                        "selected_path": list(leaf.path),
                        "simulated_path": list(sim_path),
                        "reward": reward,
                        "reason": explanation,
                    })
                    logger.info("[Simulate]Rollout %d/%d | reward=%.3f | path=%s", index + 1, self.rollout_count, reward, format_reasoning_path(sim_path, fallback_sep=", "))

        # 迭代终止时，根据节点的平均奖励选择最佳节点用于生成答案
        logger.info("[Main]MCTS iterations completed.")
        best_node = self._get_best_node()
        if best_node:
            self.best_path = list(best_node.path)
            self.best_path_value = best_node.average_reward
            self.best_path_source = "tree"

        logger.info("Final best path: %s", format_reasoning_path(self.best_path, fallback_sep=", "))
        logger.info("Path value: %.3f", self.best_path_value)
        self.root.print_tree()
        return self.best_path


    # 调用LLM评估KG路径
    def evaluate(self, triples):
        if not triples:
            return 0.0, None
        if (isinstance(triples, tuple) and len(triples) == 3 and not isinstance(triples[0], (list, tuple))):
            triples = [triples]

        formatted_triples = [f"{h}, {r}, {t}" for h, r, t in triples]
        triples_str = "\n".join(formatted_triples)
        prompt = evaluate_prompt.substitute(question=self.question, triple=triples_str)
        output = self.evaluator(user_prompt=prompt)[0]
        result = parse_json_object(output)
        score, reason = result.get("score"), result.get("reason")

        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("Path score must be a number between 0 and 1")
        if not isinstance(reason, str):
            raise ValueError("Path score reason must be a string")
        logger.info("[Evaluation]path=%s, score=%.3f", format_reasoning_path(triples), score)
        return float(score), reason


    # 选择阶段，逐层计算节点的UCB值，直到叶子节点
    def _select(self, node):
        if node.is_terminal(self.max_depth):
            return None
        if not node.is_fully_expanded:
            return node
        children = sorted(node.children, key=lambda child: child.get_ucb_value(self.exploration_constant), reverse=True)
        for child in children:
            selected = self._select(child)
            if selected is not None:
                return selected
        return None


    # 扩展阶段，为选择的叶子节点扩展新节点
    def _expand(self, node):
        if node.is_terminal(self.max_depth):
            return []

        existing_children = list(node.children)
        children = []

        # 遍历当前节点的实体
        for entity_info in list(node.entities_info):
            entity_id = entity_info["entity_id"]
            entity_name = entity_info["entity_name"]

            # 跳过已完全展开的实体
            if entity_id in node.fully_expanded_entities:
                continue

            # 关系扩展
            incoming = node.entity_incoming.get(entity_id)
            relation_history = [incoming] if incoming else []
            relations = self.kg_utils.relation_search_prune(
                entity_id=entity_id,
                entity_name=entity_name,
                relation_history=relation_history,
                path_history=node.path,
            )
            # 未扩展出关系时标记该实体完全展开
            if not relations:
                node.mark_entity_fully_expanded(entity_id)
                continue

            # 去除已经处理过的关系
            remaining_relations = [
                rel_info for rel_info in relations
                if (rel_info["relation"], rel_info["head"]) not in node.expanded_relations.setdefault(entity_id, set())
            ]
            logger.debug("[Relation Expansion]Entity=%s, num_remain_rels=%s", entity_name, len(remaining_relations))
            if not remaining_relations:
                node.mark_entity_fully_expanded(entity_id)
                continue

            # 遍历扩展的关系，进行实体扩展
            for relation_info in remaining_relations:
                rel_name = relation_info["relation"]
                logger.debug("[Entity Expansion]Current relation: %s, Direction: %s", rel_name, relation_info["head"])
                relation_key = (rel_name, relation_info["head"])
                node.expanded_relations[entity_id].add(relation_key)

                # 实体扩展
                target_entities = self.kg_utils.entity_search(entity_id, entity_name, rel_name, relation_info["head"])
                target_entities = [
                    target_id
                    for target_id in target_entities
                    if target_id != entity_id and target_id not in node.visited_ids
                ]
                if not target_entities:
                    continue
                candidate_ids = self.kg_utils.entity_prune(target_entities, node, rel_name)

                # 为扩展的实体构建新节点
                for target_id in candidate_ids:
                    target_name = get_entity_name(target_id)
                    new_triple = construct_triple(entity_name, rel_name, target_name, relation_info["head"])
                    children.append(node.add_child(
                        entity_id=entity_id,
                        new_entity_info={"entity_id": target_id, "entity_name": target_name},
                        new_triple=new_triple,
                        relation=rel_name,
                        head=relation_info["head"],
                    ))

            if len(node.expanded_relations[entity_id]) >= len(relations):
                node.mark_entity_fully_expanded(entity_id)

        # 当前节点所有实体都已完全展开后，标记当前节点已满
        if node.entities_info and all(entity["entity_id"] in node.fully_expanded_entities for entity in node.entities_info):
            node.is_fully_expanded = True

        # 遍历评估所有新节点，注意，这里不是模拟评估，只是对所有节点赋予一个初始的价值
        for child in children:
            score, explanation = self.evaluate(child.path)
            child.v = score
            child.explanation = explanation
            child.visits = 1
            child.total_reward = score

        # 保留beam_size个最优新节点
        children.sort(key=lambda child: child.v, reverse=True)
        retained = children[:max(0, self.beam_size - len(existing_children))]
        node.children = existing_children + retained
        self.trace_events.append({
            "event": "expand",
            "selected_path": list(node.path),
            "evaluated_children": [{"path": list(child.path), "score": child.v} for child in children],
            "retained_paths": [list(child.path) for child in retained],
        })
        logger.info("[Entity Expansion]Evaluated %s children; retained %s.", len(children), len(retained))
        return retained


    # 模拟阶段
    def _simulate(self, node):
        rollout_node = node

        # 先从node往下根据UCB选择扩展得到的新节点，作为模拟的起点
        while rollout_node.children and not rollout_node.is_terminal(self.max_depth):
            next_node = self._get_best_child(rollout_node)
            if next_node is None:
                break
            rollout_node = next_node

        sim_path = list(rollout_node.path)
        current_entity = rollout_node.entities_info[-1] if rollout_node.entities_info else None
        current_visited = set(rollout_node.visited_ids)
        incoming = rollout_node.entity_incoming.get(current_entity["entity_id"]) if current_entity else None
        relation_history = [incoming] if incoming else []

        # 计算剩余跳数
        remaining_depth = self.max_depth - len(sim_path)
        steps_left = (remaining_depth if self.rollout_steps <= 0 else min(self.rollout_steps, remaining_depth))

        for _ in range(max(0, steps_left)):
            if current_entity is None:
                break

            # 关系扩展
            relations = self.kg_utils.relation_search_prune(
                entity_id=current_entity["entity_id"],
                entity_name=current_entity["entity_name"],
                relation_history=relation_history,
                path_history=sim_path,
            )
            if not relations:
                break
            advanced = False
            remaining_relations = list(relations)

            # 从候选关系中将得分归一化为概率后采样
            while remaining_relations:
                max_score = max(relation["score"] for relation in remaining_relations)
                weights = [math.exp(relation["score"] - max_score) for relation in remaining_relations]
                relation_info = random.choices(remaining_relations, weights=weights, k=1)[0]
                remaining_relations.remove(relation_info)
                logger.info("[Rollout]Sampled relation: %s | score=%.3f", relation_info["relation"], relation_info["score"])

                # 实体扩展
                targets = self.kg_utils.entity_search(
                    current_entity["entity_id"],
                    current_entity["entity_name"],
                    relation_info["relation"],
                    relation_info["head"],
                )
                targets = [
                    target_id
                    for target_id in targets
                    if target_id != current_entity["entity_id"] and target_id not in current_visited
                ]
                if not targets:
                    continue

                temp_node = MCTSNode(
                    entities_info=[current_entity], triples=sim_path,
                    visited_ids=current_visited,
                )
                candidate_ids = self.kg_utils.entity_prune(targets, temp_node, relation_info["relation"])
                if not candidate_ids:
                    continue
                    
                # 取第一个候选实体作为下一步起点
                target_id = candidate_ids[0]
                target_name = get_entity_name(target_id)
                sim_path.append(construct_triple(
                    current_entity["entity_name"],
                    relation_info["relation"],
                    target_name,
                    relation_info["head"],
                ))
                current_visited.add(target_id)
                current_entity = {"entity_id": target_id, "entity_name": target_name}
                relation_history = [(relation_info["relation"], relation_info["head"])]
                advanced = True
                break

            if not advanced:
                break

        return sim_path


    # 回传节点模拟后的价值
    def _backpropagate(self, node, reward):
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent


    def _get_best_child(self, node):
        if not node.children:
            return None
        return max(
            node.children,
            key=lambda child: child.get_ucb_value(self.exploration_constant),
        )


    def _get_best_node(self):
        nodes = []
        stack = [self.root]
        while stack:
            current = stack.pop()
            if current.path:
                nodes.append(current)
            stack.extend(current.children)

        if not nodes:
            return None
        return max(
            nodes,
            key=lambda child: (child.average_reward, child.v, len(child.path)),
        )
