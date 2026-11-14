import os
import re
import json
import time
import random
import argparse
from tqdm import tqdm

from schema import Entity
from agent import CoRAgent
from logging_utils import initialize_logger
from utils import load_previous_results, prepare_dataset, save_2_jsonl


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


def build_parser():
    parser = argparse.ArgumentParser(description="Run CoR on a Freebase dataset")
    parser.add_argument("--data_dir", default="/home/flyvideo/Sata2/lyy/kgqa/data/test")    
    parser.add_argument("--dataset", choices=["cwq", "webqsp", "grailqa"], default="cwq")
    parser.add_argument("--dataset_sample_num", type=sample_count, default=500, help="Number of samples to evaluate, use None for all samples.")
    parser.add_argument("--dataset_sample_seed", type=int, default=42, help="Random seed used only for selecting the evaluation subset.")
    parser.add_argument("--model_name", choices=MODELS_DICT.keys(), default="DeepSeekV4.1-Flash")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature_exploration", type=float, default=0.3)
    parser.add_argument("--temperature_reasoning", type=float, default=0.1)    
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)    
    parser.add_argument("--sample_relation_threshold", type=int, default=1000, help="Maximum number of relation candidates sent to the LLM.")
    parser.add_argument("--remove_unnecessary_rel", action=argparse.BooleanOptionalAction, default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--save_detail", action=argparse.BooleanOptionalAction, default=True, help="Whether to save detailed results.")
    parser.add_argument("--output_dir", default="/home/flyvideo/Sata2/lyy/kgqa/methods/CoR/output")
    return parser


def disable_net_proxy():
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"):
        os.environ.pop(key, None)


# 规范化起点实体
def normalize_topic_entities(raw_topic_entities):
    if isinstance(raw_topic_entities, dict):
        return [Entity(id=str(entity_id), name=str(name)) for entity_id, name in raw_topic_entities.items() if entity_id and name]
    if isinstance(raw_topic_entities, list):
        return [Entity(id=str(item["id"]), name=str(item["name"])) for item in raw_topic_entities if isinstance(item, dict) and item.get("id") and item.get("name")]
    return []


# 为每题保存详细信息
def save_detail(output_dir, run_name, question_id, data, question, result, prompt_history, sparql_history):
    detail_dir = os.path.join(output_dir, "details", run_name)
    os.makedirs(detail_dir, exist_ok=True)
    safe_id = re.sub(r"[^\w.\-]", "_", str(question_id))
    detail = {
        "id": question_id,
        "question": question,
        "topic_entities": data.get("topic_entity", {}),
        "action": result.get("action", ""),
        "results": result.get("results", []),
        "reasoning_chains": result.get("reasoning_chains", []),
        "step_history": result.get("step_history", []),
        "prompt_history": result.get("prompt_history", prompt_history),
        "sparql_history": result.get("sparql_history", sparql_history),
    }
    with open(os.path.join(detail_dir, f"{safe_id}.json"), "w", encoding="utf-8") as outfile:
        json.dump(detail, outfile, ensure_ascii=False, indent=2)



def main():
    args = build_parser().parse_args()
    model_config = MODELS_DICT[args.model_name]
    model_path, base_url, api_key = model_config["model_path"], model_config["base_url"], model_config["api_key"]
    random.seed(args.random_seed)
    disable_net_proxy()

    # 加载数据集，必要时采样
    datas, question_string, id_field = prepare_dataset(args.dataset, args.data_dir)
    total_size = len(datas)
    if args.dataset_sample_num is not None:
        indices = sorted(random.Random(args.dataset_sample_seed).sample(range(total_size), args.dataset_sample_num))
        datas = [datas[index] for index in indices]

    # 构造运行文件名称
    sample_label = "all" if args.dataset_sample_num is None else args.dataset_sample_num
    run_name = f"CoR_data_{args.dataset}_sampleNum_{sample_label}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}"
    output_file = os.path.join(args.output_dir, run_name + ".jsonl")
    logger = initialize_logger(args.output_dir, args.dataset, args.model_name, sample_label, args.dataset_sample_seed)
    logger.info("Start CoR on dataset=%s(samples=%d/%d in seed=%d) using model=%s", args.dataset, len(datas), total_size, args.dataset_sample_seed, args.model_name)

    # 检查是否有以及处理过的样本
    previous = load_previous_results(output_file, question_string)
    logger.info("Loaded %d previous results", len(previous))

    # 实例化CoR Agent，开始CoR
    agent = CoRAgent(model_name=model_path, base_url=base_url, api_key=api_key, args=args)
    for data in tqdm(datas, desc="Processing"):

        # 跳过处理过的样本
        question = data[question_string]
        last = previous.get(question)
        if last is not None and last.get("results") != "error":
            continue

        # 记录处理的时间和计算开销
        agent.llm_api.reset_usage()
        started = time.monotonic()

        # 记录每次提示词和SPARQL查询结果
        prompt_history, sparql_history = [], []

        question_id = data.get(id_field, question)
        logger.info("Current Question: %s", question)
        try:
            # 仅使用第一个起点实体
            topic_entities = normalize_topic_entities(data.get("topic_entity", {}))
            if len(topic_entities) > 1:
                logger.info("CoR uses the first of %d topic entities: %s", len(topic_entities), topic_entities[0].name)
            result = agent.answer(question=question, topic_entities=topic_entities, prompt_history=prompt_history, sparql_history=sparql_history)

            # 检查CoR是否失败
            if result.get("action") == "ERROR":
                raise RuntimeError(str(result.get("reasoning_chains", "CoR failed")))
            if result.get("action") == "generate_directly" and prompt_history and prompt_history[-1].get("response") is None:
                raise RuntimeError("LLM failed to generate a direct answer")

            # 提取答案并保存
            raw_results = result.get("results", [])
            if isinstance(raw_results, str):
                raw_results = [raw_results]
            predictions = [str(item.get("name", "")) if isinstance(item, dict) else str(item) for item in raw_results]
            predictions = [name for name in predictions if name]
            answer = json.dumps({"A": {"Answer": predictions}}, ensure_ascii=False)
            save_2_jsonl(question, question_string, answer, result.get("reasoning_chains", []), agent.llm_api, started, output_file, id=question_id, action=result.get("action", ""), predicted_entities=raw_results)
            
            # 必要时记录详细信息
            if args.save_detail:
                try:
                    save_detail(args.output_dir, run_name, question_id, data, question, result, prompt_history, sparql_history)
                except Exception:
                    logger.exception("Failed to save detail for question=%s", question)
            logger.info("CoR complete | action=%s answers=%s calls=%d", result.get("action"), predictions, agent.llm_api.call_num)
        except Exception:
            logger.exception("Error on question=%s", question)
            save_2_jsonl(question, question_string, "error", None, agent.llm_api, started, output_file, id=question_id, action="ERROR", predicted_entities=[])

    logger.info("CoR finished. Result file: %s", output_file)


if __name__ == "__main__":
    main()
