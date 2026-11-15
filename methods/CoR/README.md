# CoR工程化重构

本目录基于`methods/old_CoR/src`进行工程化重构，在不修改CoR搜索算法和提示词内容的前提下，简化代码结构，并与仓库中ToG、PoG和CoG的工程实现保持一致。

## 一、代码结构

各文件功能如下：

- `main.py`：配置三个LLM、数据集采样、断点续跑、统一JSONL结果保存及可选的详细运行记录。
- `agent.py`：实现CoR的DFS搜索、关系选择、推理决策及答案处理。
- `freebase_func.py`：实现Freebase的SPARQL查询、关系路径构造和实体名称解析。
- `prompt_list.py`：保留`old_CoR`中的原始提示词，未修改任务描述及Few-shot示例。
- `schema.py`：定义实体和关系的数据结构。
- `utils.py`：实现LLM调用、Token统计、数据集读取和JSONL结果保存。
- `logging_utils.py`：实现日志记录，日志文件与结果文件使用相同名称。

重构后不再需要`dfs_stack.py`。直接使用Python列表实现DFS栈，并通过逆序压入候选关系，保持原来的后进先出（LIFO）搜索顺序。

## 二、运行方法

进入`methods/new_cor/src`目录：

### 1.语法检查与单元测试

```bash
python -m py_compile *.py
python -m unittest discover -s ../tests -v
```

### 2.运行KGQA实验

**Qwen3.5-27B-AWQ-4bit：5题测试**

```bash
python main.py --dataset cwq --model_name Qwen3.5-27b-AWQ-4bit --dataset_sample_num 5 --dataset_sample_seed 42
```

**DeepSeekV4.1-Flash：500题实验**

```bash
python main.py --dataset cwq --model_name DeepSeekV4.1-Flash --dataset_sample_num 500 --dataset_sample_seed 42
```

**GPT-4o-Mini：500题实验**

```bash
python main.py --dataset cwq --model_name GPT-4o-Mini --dataset_sample_num 500 --dataset_sample_seed 42
```

运行前需要安装`openai`、`tqdm`和`SPARQLWrapper`等依赖，并确保本地Freebase SPARQL服务正常运行。

默认SPARQL地址：

```text
http://localhost:8890/sparql
```

使用API模型时，还需要配置对应的API Key。

如有需要，可以通过环境变量`SPARQL_TIMEOUT`和`FREEBASE_LABEL_BATCH_SIZE`调整SPARQL查询超时时间及实体名称批量查询大小。

## 三、统一评估

CoR使用与ToG、PoG、CoG相同的公共评估程序。

进入仓库的`eval`目录：

```bash
python eval.py --dataset cwq --output_file ../methods/new_cor/output/<result_file>.jsonl
```

实验结果统一保存为JSONL格式，主要字段包括：

| 字段 | 含义 |
|---|---|
| `question` | 问题文本，具体字段名称由数据集决定 |
| `results` | 最终预测答案 |
| `reasoning_chains` | 推理关系路径 |
| `call_num` | LLM调用次数 |
| `input_token` | 输入Token数量 |
| `output_token` | 输出Token数量 |
| `total_token` | 总Token数量 |
| `time` | 单题执行时间 |

正常完成时，`results`保存答案字符串列表；执行失败时，保存为`"error"`。

**断点续跑与统计规则：**

- 已成功处理的样本直接跳过。
- 最后一次执行失败的样本重新运行。
- 同一问题存在多条记录时，以最后一条记录为准。
- 每次调用`LLMAPI.generate()`计为一次算法层面的LLM调用，内部API重试不额外计数。
- Token统计包含已成功返回Usage信息的请求。

## 四、保留的CoR算法逻辑

本次重构只调整工程结构，保留以下算法机制：

1. **起点实体选择**：与`old_CoR`一致，仅使用第一个已链接的起点实体进行搜索。
2. **关系级DFS搜索**：沿具有方向的关系路径进行深度优先搜索，通过LLM对候选关系进行剪枝。
3. **完整关系链实体检索**：根据当前完整关系路径查询候选实体，而不是在每一跳进行实体剪枝。
4. **推理决策**：保留`Stop`、`Constraint`（约束过滤）、`Forward`（继续探索）和`Backtrack`（回溯）四种决策。
5. **兜底回答**：DFS搜索结束后仍未得到答案时，调用LLM直接生成答案。
6. **搜索参数**：保留原来的搜索深度、宽度、关系候选阈值、过滤策略及模型生成温度。
7. **提示词**：完整保留原始提示词的任务描述、输出要求与Few-shot示例，不修改算法语义。

## 五、测试与注意事项

目前已针对代码结构、DFS栈顺序、SPARQL关系路径构造和算法层面的LLM调用统计编写模拟测试。

但重构阶段尚未在真实Freebase服务和LLM环境下完成端到端验证。

正式运行500题实验前，建议先在服务器上进行少量样本测试，重点检查DFS搜索顺序、关系方向、SPARQL查询结果及最终答案保存是否正常。

另外，断点续跑会复用已有实验结果。因此，修改模型生成参数或搜索参数后，应使用新的`--output_dir`，避免不同实验配置共用结果文件。