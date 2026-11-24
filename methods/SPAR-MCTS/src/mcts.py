import hashlib
import logging
import time

from mcts_core import KgSearchUtils
from mcts_core import MCTSNode
from mcts_core import backpropagate, rebuild_statistics
from mcts_core import expand_node, reopen_nodes
from mcts_core import simulate_candidates
from mcts_utils import node_path_key
from replan import Replanner
from evidence import (
    get_final_answer_nodes,
    real_tree_nodes,
)
from evidence import make_path_record
from evidence import try_early_stop
from sparql_utils import get_entity_name
from subgoal import SubgoalDecomposer
from subgoal import (
    attach_subgoal_answer_hits,
    check_full_tree_answers,
    refresh_root_activity,
)
from subgoal import SubgoalState
from subgoal import (
    SubgoalUpdater,
    build_new_answer_candidates,
)
from subgoal import assign_node_rewards, can_expand_after_progress
from subgoal import (
    format_answer_for_log,
    format_coverage_for_log,
    progress_signature,
)
from logging_utils import UsageTracker


logger = logging.getLogger(__name__)


def _execution_state_records(global_state):
    """Keep only state transitions needed to audit natural recovery events."""
    records = []
    for subgoal in getattr(global_state, "subgoals", []) or []:
        answers = []
        for answer in subgoal.get("answers", []) or []:
            answer_text = str(answer.get("answer", "") or "").strip()
            entity_id = str(answer.get("entity_id", "") or "").strip()
            if answer_text or entity_id:
                answers.append({
                    "answer": answer_text,
                    "entity_id": entity_id,
                })
        coverage = global_state.dependency_coverage(subgoal)
        records.append({
            "id": str(subgoal.get("id", "") or "").strip(),
            "status": str(
                subgoal.get("status", "unresolved") or "unresolved"
            ).strip(),
            "answers": answers,
            "coverage": [
                int(coverage.get("covered", 0) or 0),
                int(coverage.get("total", 0) or 0),
            ],
        })
    return records


class MCTSPathFinder:
    def __init__(self, search_model, planner_model, reasoner_model, rerank_model,
                 question, topic_entities,
                 max_iterations=5, max_depth=5, c_uct=0.15, relation_topk=5, entity_topk=5, cvt_relation_topk=2,
                 answer_threshold=0.9, good_threshold=0.7, path_topk=10,
                 enable_decom=True,
                 progress_alpha=0.3,
                 enable_progress_reward=True,
                 enable_dynamic_subgoal_update=True,
                 enable_subgoal_replan=True, replan_patience=2,
                 enable_replan_tree_reuse=True,
                 enable_full_tree_coordination=True,
    ):
        ## LLM与功能开关
        self.search_model = search_model
        self.planner_model = planner_model if planner_model else search_model
        self.reasoner_model = reasoner_model if reasoner_model else search_model
        enable_decom = bool(enable_decom)
        self.enable_dynamic_subgoal_update = bool(enable_dynamic_subgoal_update and enable_decom)
        self.progress_alpha = progress_alpha if enable_progress_reward and self.enable_dynamic_subgoal_update else 0.0
        self.enable_subgoal_replan = bool(enable_subgoal_replan and self.enable_dynamic_subgoal_update)
        self.enable_full_tree_coordination = bool(
            enable_full_tree_coordination and self.enable_dynamic_subgoal_update
        )
        self.enable_replan_tree_reuse = bool(
            enable_replan_tree_reuse and self.enable_subgoal_replan
        )
        self.replan_patience = (
            max(1, int(replan_patience or 1))
            if self.enable_full_tree_coordination or self.enable_subgoal_replan else 0
        )

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

        # Answer verification and final evidence selection stay on the
        # original path-value scale. Fused values are only used for MCTS control.
        self.answer_threshold = answer_threshold
        self.good_threshold = good_threshold

        ## Early-stop verification state
        self.early_stop_verification_cache = {}
        self.early_stop_answer = ""
        self.actual_iterations = 0
        self.termination_reason = "not_started"
        self.final_answer_nodes = []
        self.recovery_events = []
        self.no_subgoal_progress_rounds = 0
        self.subgoal_updater = SubgoalUpdater(self.reasoner_model) if enable_decom else None
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
            real_root.root_active = False
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

        self._refresh_root_activity()
        self.replanner = Replanner(self)

    @property
    def replan_count(self):
        return self.replanner.replan_count

    @property
    def replan_events(self):
        return self.replanner.replan_events

    def _recovery_checkpoint_id(self, iteration, tree_nodes):
        path_keys = tuple(
            node_path_key(node)
            for node in tree_nodes or []
        )
        payload = (
            getattr(self, "question", ""),
            int(iteration),
            (
                progress_signature(self.global_subgoal_state)
                if hasattr(self.global_subgoal_state, "subgoals")
                else ()
            ),
            path_keys,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def _recovery_stage_tracker(self):
        return (
            time.perf_counter(),
            UsageTracker(self.search_model),
        )

    @staticmethod
    def _finish_recovery_stage(started_at, tracker):
        return {
            "runtime_sec": round(time.perf_counter() - started_at, 4),
            "usage": tracker.delta(),
        }

    def _record_recovery_event(self, event):
        if not hasattr(self, "recovery_events"):
            self.recovery_events = []
        event.setdefault("event_id", len(self.recovery_events) + 1)
        self.recovery_events.append(event)

    def _refresh_root_activity(self):
        refresh_root_activity(
            self.root,
            self.global_subgoal_state,
            self.enable_dynamic_subgoal_update,
        )

    def _get_active_roots(self):
        return [
            root
            for root in self.root.children
            if getattr(root, "root_active", True)
        ]

    def _get_subgoal_context(self):
        if not self.global_subgoal_state:
            return "None"
        return self.global_subgoal_state.format_for_prompt(
            all_active=not self.enable_dynamic_subgoal_update,
        )

    def _all_real_tree_nodes(self, require_path=False):
        return real_tree_nodes(self.root, require_path=require_path)

    def _update_subgoals_after_simulation(self, answer_candidates):
        """Update only retrieval targets available before this call."""
        if (
            answer_candidates
            and self.global_subgoal_state
            and self.subgoal_updater
        ):
            self.global_subgoal_state = self.subgoal_updater.update(
                self.global_subgoal_state,
                answer_candidates,
            )
            self.global_subgoal_state = self.subgoal_updater.auto_resolve(
                self.global_subgoal_state,
            )

    def _sync_subgoal_answer_hits_to_tree(self, nodes=None):
        if not self.global_subgoal_state or not self.subgoal_updater:
            return
        attach_subgoal_answer_hits(
            self._all_real_tree_nodes() if nodes is None else nodes,
            self.global_subgoal_state,
        )

    def _assign_progress_credits_to_nodes(self, nodes):
        assign_node_rewards(
            nodes,
            self.global_subgoal_state,
            self.progress_alpha,
        )

    def _rebuild_tree_statistics_for_current_subgoals(
        self,
        sync_hits=True,
        nodes=None,
    ):
        nodes = (
            self._all_real_tree_nodes()
            if nodes is None
            else list(nodes)
        )
        if sync_hits:
            self._sync_subgoal_answer_hits_to_tree(nodes)
        self._assign_progress_credits_to_nodes(nodes)
        rebuild_statistics(
            self.root,
            nodes,
        )
        self._refresh_root_activity()

    def _check_subgoals_from_full_tree(
        self,
        rebuild_tree_statistics=True,
        reopen_answer_nodes=True,
        tree_nodes=None,
    ):
        nodes = (
            self._all_real_tree_nodes(require_path=True)
            if tree_nodes is None
            else tree_nodes
        )
        (
            self.global_subgoal_state,
            updated,
            answer_nodes,
        ) = check_full_tree_answers(
            nodes,
            self.global_subgoal_state,
            self.subgoal_updater,
        )
        if not updated:
            logger.info(
                "  Full-tree information check found no evidence-grounded subgoal progress."
            )
            return False
        if rebuild_tree_statistics:
            # check_full_tree_answers already synchronized every node's hits
            # after a successful full-tree state update.
            self._rebuild_tree_statistics_for_current_subgoals(
                sync_hits=False,
            )
        else:
            self._refresh_root_activity()
        if reopen_answer_nodes:
            reopened = reopen_nodes(self, answer_nodes)
            logger.info(
                "  Reopened %s answer node(s) found by the full-tree information check.",
                len(reopened),
            )
        logger.info("  Full-tree information check updated subgoal status:")
        self._log_current_subgoal_answers()
        return True

    def _log_current_subgoal_answers(self):
        if not self.global_subgoal_state:
            return
        for subgoal in self.global_subgoal_state.subgoals:
            answers = [
                format_answer_for_log(answer)
                for answer in subgoal.get("answers", [])
            ]
            logger.info(
                "  %s | %s | state=%s | answers=%s%s",
                subgoal["id"].upper(),
                subgoal["goal"],
                subgoal["status"],
                " | ".join(answers) or "None",
                format_coverage_for_log(
                    self.global_subgoal_state,
                    subgoal,
                ),
            )

    def _check_subgoal_progress_and_replan(
        self,
        iteration,
        made_progress,
        force_check=False,
        tree_nodes=None,
        trigger_reason="no_progress",
    ):
        if not self.global_subgoal_state or not (
            self.enable_full_tree_coordination or self.enable_subgoal_replan
        ):
            return None

        remaining_iterations = max(0, self.max_iterations - iteration)
        # A revision must leave one MCTS iteration in which to execute the
        # revised plan.  If only that final iteration remains, a single
        # stagnant round is enough to start recovery now; the final iteration
        # itself still cannot install a new plan.
        effective_patience = self.replan_patience
        if remaining_iterations == 1:
            effective_patience = min(effective_patience, 1)

        if force_check:
            logger.info(
                "  Force failure recovery because the completed plan did not pass early-stop verification."
            )
        elif made_progress:
            if self.no_subgoal_progress_rounds:
                logger.info(
                    "  Subgoal progress resumed, reset no-progress counter."
                )
            self.no_subgoal_progress_rounds = 0
            return None
        else:
            self.no_subgoal_progress_rounds += 1
            logger.debug(
                "  Subgoal progress stagnate round: (%s/%s; configured=%s, remaining=%s).",
                self.no_subgoal_progress_rounds,
                effective_patience,
                self.replan_patience,
                remaining_iterations,
            )
            if self.no_subgoal_progress_rounds < effective_patience:
                return None

        checkpoint_id = self._recovery_checkpoint_id(
            iteration,
            tree_nodes,
        )
        recovery_event = {
            "checkpoint_id": checkpoint_id,
            "iteration": iteration,
            "trigger": (
                "unverified_completed_plan"
                if force_check
                else trigger_reason
            ),
            "stagnation_rounds": self.no_subgoal_progress_rounds,
            "remaining_iterations": remaining_iterations,
            "effective_patience": effective_patience,
            "tree_path_node_count": len(
                tree_nodes or []
            ),
            "state_before": _execution_state_records(
                self.global_subgoal_state
            ),
        }
        coordination_started, coordination_tracker = (
            self._recovery_stage_tracker()
        )
        coordination_updated = (
            self._check_subgoals_from_full_tree(tree_nodes=tree_nodes)
            if self.enable_full_tree_coordination else False
        )
        recovery_event["coordination"] = {
            "executed": self.enable_full_tree_coordination,
            "state_updated": coordination_updated,
            **self._finish_recovery_stage(
                coordination_started,
                coordination_tracker,
            ),
        }
        if not self.enable_full_tree_coordination:
            recovery_event["coordination"]["runtime_sec"] = 0.0
            logger.info("  Full-tree evidence coordination is disabled; skip this recovery stage.")
        if coordination_updated:
            self.no_subgoal_progress_rounds = 0
            recovery_event.update({
                "outcome": "coordination_progress",
                "plan_changed": False,
                "state_after": _execution_state_records(
                    self.global_subgoal_state
                ),
            })
            self._record_recovery_event(recovery_event)
            return {
                "action": "check_full_tree_evidence",
                "success": True,
                "checked": True,
                "plan_changed": False,
                "state_updated": True,
                "tree_statistics_rebuilt": True,
            }

        if not self.enable_subgoal_replan:
            self.no_subgoal_progress_rounds = 0
            recovery_event.update({
                "outcome": "coordination_no_progress",
                "plan_changed": False,
                "state_after": _execution_state_records(self.global_subgoal_state),
            })
            self._record_recovery_event(recovery_event)
            return {
                "action": "check_full_tree_evidence",
                "success": False,
                "checked": True,
                "plan_changed": False,
                "state_updated": False,
            }

        if self.max_iterations - iteration <= 0:
            logger.debug(
                "  Skip plan repair: no search iteration remains after iteration %s.",
                iteration,
            )
            recovery_event.update({
                "outcome": "no_remaining_iterations",
                "plan_changed": False,
                "state_after": _execution_state_records(
                    self.global_subgoal_state
                ),
            })
            self._record_recovery_event(recovery_event)
            return {
                "action": "no_remaining_iterations",
                "success": False,
                "checked": True,
                "plan_changed": False,
                "state_updated": False,
            }

        if self.enable_full_tree_coordination:
            logger.info("  Replan triggered after the full-tree information check found no progress.")
        else:
            logger.info("  Replan triggered directly with full-tree evidence coordination disabled.")
        replan_events = getattr(self.replanner, "replan_events", [])
        replan_event_count_before = len(replan_events)
        revision_started, revision_tracker = self._recovery_stage_tracker()
        action = self.replanner.try_replan(
            iteration=iteration,
            tree_nodes=tree_nodes,
            checkpoint_id=checkpoint_id,
        )
        recovery_event["revision"] = {
            "action": (action or {}).get("action", "none"),
            **self._finish_recovery_stage(
                revision_started,
                revision_tracker,
            ),
        }
        replan_events = getattr(self.replanner, "replan_events", [])
        if len(replan_events) > replan_event_count_before:
            recovery_event["replan_event_id"] = (
                replan_events[-1].get("event_id")
            )
        recovery_event.update({
            "outcome": (action or {}).get("action", "none"),
            "plan_changed": bool(
                action and action.get("plan_changed")
            ),
            "state_after": _execution_state_records(
                self.global_subgoal_state
            ),
        })
        self._record_recovery_event(recovery_event)
        if action and action.get("checked"):
            self.no_subgoal_progress_rounds = 0
        return action

    def _attempt_early_stop(self, tree_nodes=None):
        result = try_early_stop(
            root=self.root,
            question=self.question,
            reasoner_model=self.reasoner_model,
            global_subgoal_state=self.global_subgoal_state,
            good_threshold=self.good_threshold,
            answer_threshold=self.answer_threshold,
            path_topk=self.path_topk,
            verification_cache=self.early_stop_verification_cache,
            tree_nodes=tree_nodes,
        )
        if result["status"] == "accepted":
            self.early_stop_answer = result["answer"]
            self.final_answer_nodes = list(result.get("nodes", []) or [])
            self.termination_reason = "early_stop"
        return result

    def _run_search_control(
        self,
        *,
        iteration,
        made_progress,
        trigger_reason="no_progress",
    ):
        """Try early stop, check full-tree evidence, then repair the plan."""
        # One tree traversal supplies distinct evidence views for early stop,
        # the full-tree information check, and (if needed) plan repair.
        tree_nodes = self._all_real_tree_nodes(require_path=True)

        early_result = self._attempt_early_stop(tree_nodes=tree_nodes)
        if early_result["status"] == "accepted":
            return early_result["paths"], None

        all_resolved_without_acceptance = bool(
            self.global_subgoal_state
            and self.global_subgoal_state.all_resolved()
        )
        action = self._check_subgoal_progress_and_replan(
            made_progress=made_progress,
            iteration=iteration,
            force_check=all_resolved_without_acceptance,
            tree_nodes=tree_nodes,
            trigger_reason=trigger_reason,
        )

        if action and action.get("early_stop_paths") is not None:
            return action["early_stop_paths"], action

        if action and action.get("state_updated"):
            logger.info("  Full-tree information check made progress; retry early-stop verification.")
            retry_result = self._attempt_early_stop(tree_nodes=tree_nodes)
            if retry_result["status"] == "accepted":
                return retry_result["paths"], action
        return None, action


    def search(self):
        self.actual_iterations = 0
        self.termination_reason = "max_iterations"
        self.final_answer_nodes = []
        for i in range(self.max_iterations):
            self.actual_iterations = i + 1
            logger.info("========== Iteration %s/%s ==========", i+1, self.max_iterations)

            # 1.选择：从虚拟根节点出发，每一层根据各节点的UCT值选择节点，逐步向下直至可扩展
            logger.info("Selecting")
            node = self._select(self.root)
            if node and node.entity_info:
                logger.info("  Selected entity: %s", node.entity_info["entity_name"])
            else:
                logger.info("  No expandable nodes, iteration end.")
                early_paths, replan_action = self._run_search_control(
                    made_progress=False,
                    iteration=i + 1,
                    trigger_reason="no_expandable_node",
                )
                if early_paths is not None:
                    return early_paths
                if replan_action and (
                    replan_action.get("plan_changed")
                    or replan_action.get("state_updated")
                ):
                    continue
                self.termination_reason = "no_expandable_node"
                break

            # 2.扩展：对选择的节点先后进行关系扩展和实体扩展，两次扩展中都会将当前子目标状态注入提示词引导，会扩展得到多条候选路径
            logger.info("Expanding")
            cand_paths = expand_node(self, node)
            if not cand_paths:
                logger.info("  No candidate paths expanded from this node.")
                children = []
            else:
                # 3.模拟：批量评估候选路径对原始问题的推进价值，这里不使用当前子目标状态进行引导
                logger.info("Simulating")
                children = simulate_candidates(self, cand_paths)
                if not children:
                    logger.info("  No children created after simulation.")

            if not children:
                early_paths, _ = self._run_search_control(
                    made_progress=False,
                    iteration=i + 1,
                    trigger_reason="no_new_children",
                )
                if early_paths is not None:
                    return early_paths
                continue

            before_subgoal_signature = progress_signature(
                self.global_subgoal_state
            )
            logger.info("Updating Subgoal Status")
            if self.enable_dynamic_subgoal_update:
                # 收集候选答案信息
                answer_candidates = build_new_answer_candidates(children)

                # 4-5. 每轮只更新轮初已可回答的检索子目标；本轮新激活的
                # 下游检索子目标等待下一轮新扩展证据。
                if answer_candidates:
                    logger.debug("  Grounding subgoal answer candidates in KG evidence.")
                    self._update_subgoals_after_simulation(
                        answer_candidates
                    )

                    # 6. auto更新完成后，按本轮最终状态回写命中并只计算新节点的进度奖励
                    self._sync_subgoal_answer_hits_to_tree(children)
                    self._assign_progress_credits_to_nodes(children)

                    # 根据当前active子目标的合法anchor和依赖命中刷新可搜索起点。
                    self._refresh_root_activity()

                else:
                    logger.debug("  No new subgoal answer candidates in this iteration.")

            # 记录更新后的状态，检查是否变化，变化才说明基于子目标的搜索计划被推进
            after_subgoal_signature = progress_signature(
                self.global_subgoal_state
            )
            made_subgoal_progress = before_subgoal_signature != after_subgoal_signature
            if made_subgoal_progress:
                logger.info("  Subgoal status updated:")
                self._log_current_subgoal_answers()
            elif self.enable_dynamic_subgoal_update:
                logger.info("  Subgoal status unchanged at iteration %s.", i+1)

            # 7. 满足条件时做早停验证；失败后维护停滞计数并按条件做全树信息检查，
            # 全树检查无法推进计划且仍有搜索预算时才调用计划修正LLM。
            early_paths, replan_action = self._run_search_control(made_progress=made_subgoal_progress, iteration=i+1)
            if early_paths is not None:
                return early_paths
            if replan_action and (replan_action.get("plan_changed") or replan_action.get("tree_statistics_rebuilt")):
                logger.info("  Tree statistics already rebuilt, skip duplicate backpropagation.")
                continue

            # 8. 回传阶段：按策略聚合融合奖励；每条已评估路径统一记一次UCT访问。
            logger.info("Backpropagating")
            backpropagate(
                children,
                self.progress_alpha,
            )

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
            if not node.is_fully_expanded and self._progress_allows_expansion(node):
                return node
            if not node.children:
                return None
            node = self._get_best_child(node)
            if node is None:
                return None
        return None

    def _progress_allows_expansion(self, node):
        if not self.enable_dynamic_subgoal_update or not self.global_subgoal_state:
            return True
        return can_expand_after_progress(node, self.global_subgoal_state)

    def _has_expandable_node(self, node):
        stack = self._get_active_roots() if node.is_virtual_root else [node]
        while stack:
            current = stack.pop()
            if (
                not current.is_terminal(self.max_depth)
                and not current.is_fully_expanded
                and self._progress_allows_expansion(current)
            ):
                return True
            stack.extend(current.children)
        return False

    def _get_best_child(self, node):
        if not node.children:
            return None

        candidate_children = node.children
        if node.is_virtual_root:
            candidate_children = [child for child in node.children if getattr(child, "root_active", True)]

        expandable_children = [child for child in candidate_children if self._has_expandable_node(child)]
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
