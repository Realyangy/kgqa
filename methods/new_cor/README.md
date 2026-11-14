# CoR (engineering refactor)

This directory refactors `methods/old_CoR/src` without intentionally changing CoR's search algorithm or any prompt text.

## Files

- `src/main.py`: three-model configurations, dataset sampling, resume, unified JSONL, optional detailed traces
- `src/agent.py`: CoR DFS search, relation selection, reasoning decisions and answer handling
- `src/freebase_func.py`: Freebase/SPARQL queries, path assembly, entity-name resolution
- `src/prompt_list.py`: **unchanged from old_CoR**
- `src/schema.py`: entity/relation data classes
- `src/utils.py`: LLM calls, token accounting, data loading and JSONL writing
- `src/logging_utils.py`: log name matches the JSONL result name
- `tests/test_smoke.py`: network-free structural and mocked-DFS smoke tests

`dfs_stack.py` is no longer required. The ordinary Python list stack preserves the previous LIFO order: children are pushed in reverse selected-relation order.

## Run

From `methods/new_cor/src`:

```bash
python -m py_compile *.py
python -m unittest discover -s ../tests -v

python main.py --dataset cwq --model_name Qwen3.5-27b-AWQ-4bit --dataset_sample_num 5 --dataset_sample_seed 42
python main.py --dataset cwq --model_name DeepSeekV4.1-Flash --dataset_sample_num 500 --dataset_sample_seed 42
python main.py --dataset cwq --model_name GPT-4o-Mini --dataset_sample_num 500 --dataset_sample_seed 42
```

Requirements include `openai`, `tqdm`, and `SPARQLWrapper`, a locally available Freebase SPARQL service (default `http://localhost:8890/sparql`), and credentials for API models. Set `SPARQL_TIMEOUT` and `FREEBASE_LABEL_BATCH_SIZE` if needed.

To assess with the same evaluator as ToG, PoG and CoG, run from the repository's `eval` directory:

```bash
python eval.py --dataset cwq --output_file ../methods/new_cor/output/<result_file>.jsonl
```

The common result fields are `question` (or the dataset question key), `results`, `reasoning_chains`, `call_num`, `input_token`, `output_token`, `total_token`, and `time`. `results` is a list of answer strings for completed samples and `"error"` for execution failures. The evaluator keeps only the last record per question. Calls count algorithm-level `LLMAPI.generate()` invocations, not internal retries. Token counts include requests for which usage is returned.

## Preserved CoR semantics

- Only the first linked topic entity is explored, as in `old_CoR`
- DFS over directed relation paths with relation-based pruning
- Entity candidates are retrieved for the **whole relation chain**, not incrementally pruned at each hop
- The decision branches remain `Stop`, `Constraint` (filter), `Forward`, `Backtrack`
- Direct-answer fallback after DFS exhaustion
- Same depth, width, relation threshold, filtering and model-generation temperatures
- Same prompt wording and few-shot examples (byte-for-byte)

## Limits of verification

Mocked tests cover stack ordering, query path construction and logical call accounting; no live Freebase or model requests were run in this refactoring session. Run a short server-side smoke test before a 500-question experiment. Use a separate `--output_dir` when changing generation/search parameters, because results are reused for resume.
