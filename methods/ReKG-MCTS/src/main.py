"""Run ReKG-MCTS and append one evaluation-compatible result per question."""
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

from tqdm import tqdm
from freebase_func import disable_net_proxy
from logging_utils import initialize_logger
from mcts import MCTSPathFinder
from model import LLM_Client
from prompt_list import answer_prompt

DATASETS = {
    "cwq": ("../../../data/test/cwq.json", "question"),
    "webqsp": ("../../../data/test/webqsp.json", "RawQuestion"),
}
MODELS_DICT = {
    "Qwen3.5-27b-AWQ-4bit": {
        "model_path": "/home/flyvideo/Sata2/Models/LLM/Qwen3.5-27B-AWQ-4bit",
        "base_url": "http://localhost:8001/v1", "api_key": "EMPTY",
    },
    "DeepSeekV4.1-Flash": {
        "model_path": "deepseek-flash", "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("LYY_DEEPSEEK_API_KEY"),
    },
    "GPT-4o-Mini": {
        "model_path": "openai/gpt-4o-mini", "base_url": "https://router.shengsuanyun.com/api/v1",
        "api_key": os.getenv("LYY_SHENGSUANYUN_API_KEY"),
    },
}
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ReKG-MCTS")
    parser.add_argument("--dataset", type=str.lower, choices=DATASETS, default="cwq")
    parser.add_argument("--data_path", default=None, help="Override the dataset JSON path")
    parser.add_argument("--model_name", choices=MODELS_DICT, default="Qwen3.5-27b-AWQ-4bit")
    parser.add_argument("--model", default=None, help="Override the model path or served model name")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--rollout_steps", type=int, default=2, help="Maximum new hops in each rollout")
    parser.add_argument("--rollout_count", type=int, default=2, help="Number of rollouts per MCTS iteration")
    parser.add_argument("--beam_size", type=int, default=2, help="Maximum child paths retained per expansion")
    parser.add_argument("--exploration_constant", type=float, default=None)
    parser.add_argument("--relation_topk", type=int, default=5)
    parser.add_argument("--entity_topk", type=int, default=5)
    parser.add_argument("--reward_threshold", type=float, default=0.8)
    parser.add_argument("--dataset_sample_num", "--num_sample", dest="dataset_sample_num", type=int, default=None)
    parser.add_argument("--dataset_sample_seed", "--sample_seed", dest="dataset_sample_seed", type=int, default=42)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_dir", default="../output")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()
    if args.beam_size < 1:
        parser.error("--beam_size must be at least 1")
    if args.rollout_count < 1:
        parser.error("--rollout_count must be at least 1")
    if args.exploration_constant is None:
        args.exploration_constant = 0.7 if args.dataset == "cwq" else 0.5
    config = MODELS_DICT[args.model_name]
    args.model = args.model or config["model_path"]
    args.base_url = args.base_url or config["base_url"]
    args.api_key = args.api_key or config["api_key"]
    if not args.api_key:
        parser.error("The selected model API key is missing; set its environment variable or use --api_key")
    name = f"ReKG-MCTS_data_{args.dataset}_sampleNum_{args.dataset_sample_num}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}"
    args.output_file = args.output_file or str(Path(args.output_dir) / (name + ".jsonl"))
    args.log_file = args.log_file or str(Path(args.output_dir) / (name + ".log"))
    return args


def load_previous_results(output_file, question_key):
    """Read the latest entry per question, recovering an incomplete final write."""
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
    initialize_logger(args.log_file)
    disable_net_proxy()
    dataset_path, question_key = DATASETS[args.dataset]
    with open(args.data_path or dataset_path, encoding="utf-8") as stream:
        data = json.load(stream)
    if args.dataset_sample_num is not None:
        if not 0 <= args.dataset_sample_num <= len(data):
            raise ValueError("dataset_sample_num must be between 0 and the dataset size")
        indices = sorted(random.Random(args.dataset_sample_seed).sample(range(len(data)), args.dataset_sample_num))
        data = [data[index] for index in indices]
    logger.info("Arguments: %s", {key: value for key, value in vars(args).items() if key != "api_key"})
    previous_results = load_previous_results(args.output_file, question_key)
    logger.info("Loaded %d previous result records.", len(previous_results))
    model = LLM_Client(args.model, base_url=args.base_url, api_key=args.api_key,
                       temperature=args.temperature, max_new_tokens=args.max_length)
    model.set_answer_prompt(answer_prompt)
    for item in tqdm(data, desc="Processing"):
        question = item[question_key]
        previous = previous_results.get(question)
        if previous is not None and previous.get("results") != "error" and all(
            key in previous for key in ("results", "call_num", "total_token", "input_token", "output_token", "time")
        ):
            continue
        if previous is not None:
            logger.info("Retry failed or incomplete question: %s", question)
        logger.info("\nCurrent Question: %s\nTopic Entity(s): %s", question, item.get("topic_entity") or {})
        random.seed(f"{args.random_seed}:{question}")
        started = time.monotonic()
        before = model.get_usage_snapshot()
        try:
            topics = item.get("topic_entity") or {}
            if not topics:
                logger.info("No topic entity; answer directly without graph retrieval.")
                path, best_path_value = [], None
                raw_answer, answer = model.generate_direct_answer(question)
            else:
                finder = MCTSPathFinder(question, topics, model,
                    relation_topk=args.relation_topk, entity_topk=args.entity_topk,
                    max_depth=args.depth, max_iterations=args.iterations,
                    rollout_steps=args.rollout_steps, rollout_count=args.rollout_count,
                    beam_size=args.beam_size, score_threshold=args.reward_threshold,
                    exploration_constant=args.exploration_constant, sample_seed=args.random_seed)
                path = finder.search()
                best_path_value = finder.best_path_value
                raw_answer, answer = model.generate_answer(question, path)
            record = {question_key: question, "results": answer, "reasoning_chains": path,
                      "raw_answer": raw_answer, "best_path_value": best_path_value}
            logger.info("Answer: %s", answer)
        except Exception as error:
            logger.exception("Failed to process question: %s", question)
            record = {question_key: question, "results": "error", "reasoning_chains": None,
                      "error": f"{type(error).__name__}: {error}"}
        after = model.get_usage_snapshot()
        record.update({"call_num": after["num_calls"] - before["num_calls"],
                       "input_token": after["prompt_tokens"] - before["prompt_tokens"],
                       "output_token": after["completion_tokens"] - before["completion_tokens"],
                       "total_token": after["total_tokens"] - before["total_tokens"],
                       "time": round(time.monotonic() - started, 4)})
        append_result(args.output_file, record)
        previous_results[question] = record
    logger.info("Finished processing %d selected samples; results: %s", len(data), args.output_file)


if __name__ == "__main__":
    main()
