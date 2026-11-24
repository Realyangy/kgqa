# MCTS-R

Relation-centric Monte Carlo Tree Search for Knowledge Graph Question Answering.

当前为关系中心搜索的第一阶段实现，继承 new-method 的初始问题分解、多起点虚拟根、UCT 和逐子节点价值回传。

- 搜索状态：固定起点实体 + 有序、有方向的关系链。每次迭代联合查询当前关系链可达的下一跳出入边，LLM 关系剪枝，批量评价子路径并回传。
- 搜索阶段不检索具体中间/末端实体，不进行实体剪枝或 CVT 特殊处理；CVT 的每条实际关系各占一跳。
- 早停只判断关系结构是否足够；不生成答案。可使用 --no-enable_early_stop 固定预算搜索。
- 搜索结束后查询完整变量绑定（每条链 LIMIT 1000），默认每条路径最多向最终回答模型提供 30 条绑定，优先覆盖不同末端。保留字面量和未命名实体 ID。
- 最终 grounding 默认最多 path_topk 条关系路径；早停选中的路径完全无绑定时，尝试树中其他高分路径，仍无绑定则直接回答（不恢复搜索）。暂未实现跨路径符号 JOIN、完整约束验证、类型引导或缺口反馈。
- 保留原问题分解 Prompt 和解析器；关系剪枝沿用编号行格式，路径评分沿用 JSON 数组，早停使用 JSON 对象，最终回答沿用 answer tag。
- 输出在 output/<run_name>/，包含 JSONL、LOG 和 Details。JSONL 保留统一评估字段，另含 relation_paths；Details 保存逐轮搜索、grounding、异常及 LLM/SPARQL 历史。

运行（在 src 中）：

```bash
python main.py --dataset cwq --dataset_sample_num 50 --dataset_sample_seed 42 --no-enable_early_stop
```

运行前确认模型 API、Freebase endpoint 和本地 embedding_model_path；数据集从仓库根目录的 data/test 读取。旧 new-method 实验结果不自动迁移。
