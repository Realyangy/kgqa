import os
import re
import json
import time
import random
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from model import LLM_Client
from mcts import MCTSPathFinder
from utils import prepare_dataset
from logging_utils import initialize_logger
from freebase_func import SPARQLPATH, disable_net_proxy, set_sparql_trace


MODELS_DICT = {
    "Qwen3.5-27b-AWQ-4bit": {
        "model_path": "/home/flyvideo/Sata2/Models/LLM/Qwen3.5-27B-AWQ-4bit",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
    },
    "DeepSeekV4.1-Flash": {
        "model_path": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("LYY_DEEPSEEK_API_KEY"),
    },
    "GPT-4o-Mini": {
        "model_path": "openai/gpt-4o-mini",
        "base_url": "https://router.shengsuanyun.com/api/v1",
        "api_key": os.getenv("LYY_SHENGSUANYUN_API_KEY"),
    },
}

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="ReKG-MCTS")
    parser.add_argument("--dataset", type=str.lower, choices=["cwq", "webqsp", "grailqa"], default="cwq")
    parser.add_argument("--dataset_sample_num", "--num_sample", dest="dataset_sample_num", type=int, default=500)
    parser.add_argument("--dataset_sample_seed", "--sample_seed", dest="dataset_sample_seed", type=int, default=42)
    parser.add_argument("--model_name", choices=MODELS_DICT, default="Qwen3.5-27b-AWQ-4bit")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--exploration_constant", type=float, default=None)
    parser.add_argument("--relation_topk", type=int, default=5)
    parser.add_argument("--entity_topk", type=int, default=5)
    parser.add_argument("--rollout_steps", type=int, default=2, help="Maximum new hops in each rollout")
    parser.add_argument("--rollout_count", type=int, default=2, help="Number of rollouts per MCTS iteration")
    parser.add_argument("--beam_size", type=int, default=2, help="Maximum child paths retained per expansion")
    parser.add_argument("--reward_threshold", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--save_detail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="/home/flyvideo/Sata2/lyy/kgqa/methods/ReKG-MCTS/output")
    args = parser.parse_args()

    model_config = MODELS_DICT[args.model_name]
    args.model_path = model_config["model_path"]
    args.base_url = model_config["base_url"]
    args.api_key = model_config["api_key"]

    # 根据数据集选择不同的探索系数
    if args.exploration_constant is None:
        args.exploration_constant = 0.7 if args.dataset == "cwq" else 0.5

    return args


# 加载已有的实验结果，避免重复实验
def load_previous_results(output_file, question_key):
    path = Path(output_file)
    if not path.exists():
        return {}
    results = {}
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not an object")
            except (ValueError, UnicodeDecodeError) as error:
                if not line.endswith(b"\n") and not stream.read(1):
                    logger.warning("Discarding incomplete final result at byte %d", start)
                    stream.truncate(start)
                    break
                raise ValueError(f"Invalid result record at byte {start} in {path}") from error
            question = record.get(question_key) or record.get("question")
            if question:
                results[question] = record
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return results



def append_result(output_file, record):
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    disable_net_proxy()
    random.seed(args.random_seed)
    # 构造实验运行名称。用于记录
    sample_label = "all" if args.dataset_sample_num is None else args.dataset_sample_num
    run_name = f"ReKG-MCTS_data_{args.dataset}_sampleNum_{sample_label}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(run_dir / f"{run_name}.jsonl")
    initialize_logger(output_file)
    
    # 记录本次运行的实际参数和服务配置
    safe_args = {key: value for key, value in vars(args).items() if key != "api_key"}
    logger.info("Arguments: %s", json.dumps(vars(safe_args), ensure_ascii=False, sort_keys=True))
    logger.info("Model path: %s", args.model_path)
    logger.info("Model API: %s", args.base_url)
    logger.info("Freebase endpoint: %s", SPARQLPATH)

    # 加载数据集，必要时进行采样
    data, question_key = prepare_dataset(args.dataset)
    total_size = len(data)
    if args.dataset_sample_num is not None:
        if not 0 <= args.dataset_sample_num <= len(data):
            raise ValueError("dataset_sample_num must be between 0 and the dataset size")
        indices = sorted(random.Random(args.dataset_sample_seed).sample(range(len(data)), args.dataset_sample_num))
        data = [data[index] for index in indices]

    # 加载已有的实验结果
    logger.info("\n===== Start ReKG-MCTS | dataset=%s | samples=%d/%d | seed=%d | model=%s =====", args.dataset, len(data), total_size, args.dataset_sample_seed, args.model_name)
    previous_results = load_previous_results(output_file, question_key)
    logger.info("Loaded %d previous result records.", len(previous_results))

    # 加载模型
    model = LLM_Client(args.model_path, args.base_url, args.api_key, temperature=args.temperature, max_new_tokens=args.max_length)

    # 开始遍历样本进行处理
    for item in tqdm(data, desc="Processing"):
        question = item[question_key]
        previous = previous_results.get(question)
        if (
            previous is not None and
            previous.get("results") != "error" and
            all(key in previous for key in ("results", "call_num", "total_token", "input_token", "output_token", "time"))
        ):
            continue
        if previous is not None:
            logger.info("Retry failed or incomplete question: %s", question)
        logger.info("\nCurrent Question: %s\nTopic Entity(s): %s", question, item.get("topic_entity") or {})

        # 记录时间和计算开销
        started = time.monotonic()
        before = model.get_usage_snapshot()

        # 记录提示词和SPARQL查询历史
        prompt_history, sparql_history = [], []
        model.set_prompt_history(prompt_history)
        set_sparql_trace(sparql_history)

        finder = None
        try:
            topics = item.get("topic_entity") or {}

            # 无起点实体时，直接回答
            if not topics:
                logger.info("No topic entity, answer directly without graph retrieval.")
                path, best_path_value = [], None
                raw_answer, answer = model.generate_direct_answer(question)

            # 有起点实体时，执行REKG-MCTS搜索
            else:
                finder = MCTSPathFinder(
                    question, topics, model,
                    max_iterations=args.iterations, max_depth=args.depth,
                    exploration_constant=args.exploration_constant,
                    relation_topk=args.relation_topk, entity_topk=args.entity_topk,
                    rollout_steps=args.rollout_steps, rollout_count=args.rollout_count,
                    beam_size=args.beam_size, score_threshold=args.reward_threshold
                )
                path = finder.search()

                # 根据搜索结果回答
                best_path_value = finder.best_path_value
                raw_answer, answer = model.generate_answer(question, path)

            # 记录该样本搜索结果
            record = {
                question_key: question,
                "results": answer,
                "reasoning_chains": path,
                "raw_answer": raw_answer,
                "best_path_value": best_path_value
            }
            logger.info("Answer: %s", answer)

        # 错误样本记录
        except Exception as error:
            logger.exception("Failed to process question: %s", question)
            record = {
                question_key: question,
                "results": "error",
                "reasoning_chains": None,
                "error": f"{type(error).__name__}: {error}"
            }

        # 记录时间和计算开销
        after = model.get_usage_snapshot()
        record.update({
            "call_num": after["num_calls"] - before["num_calls"],
            "input_token": after["prompt_tokens"] - before["prompt_tokens"],
            "output_token": after["completion_tokens"] - before["completion_tokens"],
            "total_token": after["total_tokens"] - before["total_tokens"],
            "time": round(time.monotonic() - started, 4)
        })
        append_result(output_file, record)
        previous_results[question] = record

        # 记录细节信息
        if args.save_detail:
            try:
                detail_dir = run_dir / "details"
                detail_dir.mkdir(parents=True, exist_ok=True)
                question_id = item.get("ID") or item.get("QuestionId") or item.get("qid") or question
                safe_id = re.sub(r"[^\w.\-]", "_", str(question_id))
                detail = {
                    "id": question_id,
                    "question": question,
                    "topic_entities": item.get("topic_entity") or {},
                    "result": record,
                    "prompt_history": prompt_history,
                    "sparql_history": sparql_history,
                    "mcts_history": finder.trace_events if finder is not None else [],
                }
                with (detail_dir / (safe_id + ".json")).open("w", encoding="utf-8") as stream:
                    json.dump(detail, stream, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed to save details for question: %s", question)
        model.set_prompt_history(None)
        set_sparql_trace(None)
    logger.info("Finished processing %d selected samples, results: %s", len(data), output_file)


if __name__ == "__main__":
    main()
