import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from tqdm import tqdm

from agent import CoRAgent
from logging_utils import initialize_logger
from schema import Entity
from utils import load_previous_results, prepare_dataset, save_2_jsonl


MODELS_DICT = {
    "Qwen3.5-27b-AWQ-4bit": {
        "model_path": "/home/flyvideo/Sata2/Models/LLM/Qwen3.5-27B-AWQ-4bit",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY"
    },
    "DeepSeekV4.1-Flash": {
        "model_path": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("LYY_DEEPSEEK_API_KEY")
    },
    "GPT-4o-Mini": {
        "model_path": "openai/gpt-4o-mini",
        "base_url": "https://router.shengsuanyun.com/api/v1",
        "api_key": os.getenv("LYY_SHENGSUANYUN_API_KEY")
    }
}


def sample_count(value):
    if value.lower() == "none":
        return None
    try:
        count = int(value)
        if count < 0:
            raise ValueError
        return count
    except ValueError:
        raise argparse.ArgumentTypeError("expected a non-negative integer or None")


def normalize_topic_entities(raw_topic_entities):
    if isinstance(raw_topic_entities, dict):
        return [Entity(id=str(entity_id), name=str(name)) for entity_id, name in raw_topic_entities.items() if entity_id and name]
    if isinstance(raw_topic_entities, list):
        return [Entity(id=str(item["id"]), name=str(item["name"])) for item in raw_topic_entities if isinstance(item, dict) and item.get("id") and item.get("name")]
    return []


def save_detail(output_dir, run_name, question_id, data, question, result, prompt_history, sparql_history):
    detail_dir = Path(output_dir) / "details" / run_name
    detail_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w.\-]", "_", str(question_id))
    detail = {
        "id": question_id, "question": question, "topic_entities": data.get("topic_entity", {}),
        "action": result.get("action", ""), "results": result.get("results", []),
        "reasoning_chains": result.get("reasoning_chains", []),
        "step_history": result.get("step_history", []),
        "prompt_history": result.get("prompt_history", prompt_history),
        "sparql_history": result.get("sparql_history", sparql_history)
    }
    with open(detail_dir / f"{safe_id}.json", "w", encoding="utf-8") as outfile:
        json.dump(detail, outfile, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CoR Freebase KGQA baseline")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--data_dir", default=str(project_root / "data" / "test"))
    parser.add_argument("--dataset", choices=["cwq", "webqsp", "grailqa"], default="cwq")
    parser.add_argument("--dataset_sample_num", type=sample_count, default=500)
    parser.add_argument("--dataset_sample_seed", type=int, default=42)
    parser.add_argument("--model_name", choices=MODELS_DICT.keys(), default="DeepSeekV4.1-Flash")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature_exploration", type=float, default=0.3)
    parser.add_argument("--temperature_reasoning", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--sample_relation_threshold", type=int, default=1000)
    parser.add_argument("--remove_unnecessary_rel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_detail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parents[1] / "output"))
    args = parser.parse_args()

    model_config = MODELS_DICT[args.model_name]
    random.seed(args.random_seed)
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"):
        os.environ.pop(name, None)

    datas, question_string, id_field = prepare_dataset(args.dataset, args.data_dir)
    total_size = len(datas)
    if args.dataset_sample_num is not None:
        if not 1 <= args.dataset_sample_num <= total_size:
            parser.error("--dataset_sample_num must be between 1 and the dataset size, or None")
        indices = sorted(random.Random(args.dataset_sample_seed).sample(range(total_size), args.dataset_sample_num))
        datas = [datas[index] for index in indices]

    sample_label = "all" if args.dataset_sample_num is None else args.dataset_sample_num
    run_name = f"CoR_data_{args.dataset}_sampleNum_{sample_label}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}"
    output_file = os.path.join(args.output_dir, run_name + ".jsonl")
    logger = initialize_logger(output_file)
    logger.info("Start CoR | dataset=%s samples=%d/%d seed=%d model=%s", args.dataset, len(datas), total_size, args.dataset_sample_seed, args.model_name)

    previous_results = load_previous_results(output_file, question_string)
    logger.info("Loaded %d previous question results.", len(previous_results))
    agent = CoRAgent(model_name=model_config["model_path"], base_url=model_config["base_url"], api_key=model_config["api_key"], args=args)

    for data in tqdm(datas, desc="Processing"):
        question = data[question_string]
        last = previous_results.get(question)
        if last is not None and last.get("results") != "error":
            continue

        agent.llm_api.reset_usage()
        started = time.monotonic()
        prompt_history, sparql_history = [], []
        question_id = data.get(id_field, question)
        logger.info("Current Question: %s", question)

        try:
            topic_entities = normalize_topic_entities(data.get("topic_entity", {}))
            if len(topic_entities) > 1:
                logger.info("CoR uses the first of %d topic entities: %s", len(topic_entities), topic_entities[0].name)
            result = agent.answer(question=question, topic_entities=topic_entities, prompt_history=prompt_history, sparql_history=sparql_history)
            if result.get("action") == "ERROR":
                raise RuntimeError(str(result.get("reasoning_chains", "CoR failed")))
            if result.get("action") == "generate_directly" and prompt_history and prompt_history[-1].get("response") is None:
                raise RuntimeError("LLM failed to generate a direct answer")

            raw_results = result.get("results", [])
            if isinstance(raw_results, str):
                raw_results = [raw_results]
            predictions = [str(item.get("name", "")) if isinstance(item, dict) else str(item) for item in raw_results]
            predictions = [value for value in predictions if value]
            # Unified eval/eval.py accepts a list of answer strings in "results".
            save_2_jsonl(question, question_string, predictions, result.get("reasoning_chains", []),
                         agent.llm_api, started, output_file, id=question_id,
                         action=result.get("action", ""), predicted_entities=raw_results)

            if args.save_detail:
                try:
                    save_detail(args.output_dir, run_name, question_id, data, question, result, prompt_history, sparql_history)
                except Exception:
                    logger.exception("Failed to save detail for question=%s", question)
            logger.info("CoR complete | action=%s answers=%s calls=%d", result.get("action"), predictions, agent.llm_api.call_num)
        except Exception:
            logger.exception("Error on question=%s", question)
            save_2_jsonl(question, question_string, "error", None, agent.llm_api, started, output_file,
                         id=question_id, action="ERROR", predicted_entities=[])

    logger.info("CoR finished. Result file: %s", output_file)


if __name__ == "__main__":
    main()
