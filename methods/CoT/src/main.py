import os
import re
import json
import time
import random
import argparse
from tqdm import tqdm
from collections import Counter

from utils import *
from prompt_list import *
from logging_utils import initialize_logger


def disable_net_proxy():
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"]:
        os.environ.pop(var, None)


# 解析LLM输出的最终答案
def parse_answer(response):
    data = json.loads(response)
    if not isinstance(data, dict) or "answer" not in data:
        raise ValueError("Invalid answer JSON")

    answer = data["answer"]
    if answer is not None and not isinstance(answer, str):
        raise ValueError("answer must be a string or null")

    return answer.strip() if answer is not None else None



# Self-Consistency答案多数投票
def majority_vote(answers):
    valid_answers = [answer.strip() for answer in answers if isinstance(answer, str) and answer.strip()]
    if not valid_answers:
        return None

    # 仅统一大小写和多余空格，保留数值及标点
    normalized_answers = [re.sub(r"\s+", " ", answer.casefold()) for answer in valid_answers]
    counter = Counter(normalized_answers)

    # 选择票数最多的答案，票数相同时选择最早出现的答案
    best_index = max(range(len(valid_answers)), key=lambda i: (counter[normalized_answers[i]], -i))
    return valid_answers[best_index]


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_methods", type=str, default="sc", choices=["io", "cot", "sc"], help="Prompting method.")
    parser.add_argument("--sc_samples", type=int, default=6, help="Number of sampled reasoning paths for SC.")
    parser.add_argument("--model_name", type=str, default="DeepSeekV4.1-Flash", choices=MODELS_DICT.keys(), help="Model configuration name.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum LLM output tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature.")
    parser.add_argument("--dataset", type=str, default="cwq", help="Dataset.")
    parser.add_argument("--dataset_sample_num", type=int, default=500, help="Number of dataset samples.")
    parser.add_argument("--dataset_sample_seed", type=int, default=42, help="Dataset sampling seed.")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory for saving logs and results.")
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = 0.7 if args.prompt_methods == "sc" else 0.0

    if args.prompt_methods == "sc" and args.sc_samples < 2:
        parser.error("--sc_samples must be at least 2 for SC")

    # 配置模型
    model_config = MODELS_DICT[args.model_name]
    args.LLM_type = model_config["model_path"]
    args.base_url = model_config["base_url"]
    args.api_key = model_config["api_key"]
    disable_net_proxy()

    # 获取数据集并固定随机采样
    datas, question_string = prepare_dataset(args.dataset)
    total_dataset_size = len(datas)

    if args.dataset_sample_num is not None:
        if not 1 <= args.dataset_sample_num <= total_dataset_size:
            parser.error("--dataset_sample_num must be between 1 and the dataset size")

        sample_rng = random.Random(args.dataset_sample_seed)
        sampled_indices = sorted(sample_rng.sample(range(total_dataset_size), args.dataset_sample_num))
        datas = [datas[index] for index in sampled_indices]

    # 输出文件与日志
    output_name = f"{args.prompt_methods.upper()}_data_{args.dataset}_sampleNum_{args.dataset_sample_num}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}"
    if args.prompt_methods == "sc":
        output_name += f"_scSamples_{args.sc_samples}"

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    logger = initialize_logger(output_file)

    logger.info("\n===== Start Running %s on %s dataset =====", args.prompt_methods.upper(), args.dataset)
    logger.info("Dataset sampling: %d/%d samples, seed=%d", len(datas), total_dataset_size, args.dataset_sample_seed)
    logger.info("Temperature: %s | Max length: %d", args.temperature, args.max_length)

    # 加载已有结果，用于断点续跑
    previous_results = load_previous_results(output_file, question_string)
    logger.info("Loaded %d previous result records.", len(previous_results))

    # 开始实验
    with open(output_file, "a", encoding="utf-8") as out:
        for data in tqdm(datas, total=len(datas)):
            question = data[question_string]

            # 已成功处理的样本直接跳过，错误样本重新运行
            previous_result = previous_results.get(question)
            if previous_result is not None:
                if previous_result.get("results") != "error":
                    continue
                logger.info("Retry failed question: %s", question)

            start_time = time.time()
            call_num = 0
            all_tokens = {"total": 0, "input": 0, "output": 0}
            sample_answers = []
            sample_reasons = []

            logger.info("\nCurrent Question: %s", question)

            try:
                # IO使用直接回答Prompt，CoT和SC使用CoT Prompt
                if args.prompt_methods == "io":
                    prompt = io_prompt.substitute(question=question)
                else:
                    prompt = cot_prompt.substitute(question=question)

                # SC多次采样，其余方法只调用一次
                num_samples = args.sc_samples if args.prompt_methods == "sc" else 1

                for index in range(num_samples):
                    call_num += 1
                    response, token_num = run_llm(prompt, args.temperature, args.max_length, args.base_url, args.api_key, args.LLM_type)

                    # 累计本次LLM的Token开销
                    for token_type in all_tokens:
                        all_tokens[token_type] += token_num[token_type]

                    # 解析JSON输出
                    answer, reason = parse_answer_json(response, args.prompt_methods)
                    sample_answers.append(answer)
                    sample_reasons.append(reason)

                    logger.info("LLM Response [%s - %d/%d]:\n%s", args.prompt_methods.upper(), index + 1, num_samples, response)

                # 获取最终答案
                if args.prompt_methods == "sc":
                    final_answer = majority_vote(sample_answers)
                    logger.info("[SC] Sample Answers: %s", sample_answers)
                    logger.info("[SC] Voted Answer: %s", final_answer)
                else:
                    final_answer = sample_answers[0]

                # 保存成功执行的样本
                result_data = {
                    question_string: question,
                    "results": final_answer,
                    "reasoning_chains": [],
                    "call_num": call_num,
                    "total_token": all_tokens["total"],
                    "input_token": all_tokens["input"],
                    "output_token": all_tokens["output"],
                    "time": time.time() - start_time
                }

                if args.prompt_methods == "cot":
                    result_data["reason"] = sample_reasons[0]
                elif args.prompt_methods == "sc":
                    result_data["sample_answers"] = sample_answers
                    result_data["sample_reasons"] = sample_reasons

                logger.info("Final Answer: %s", final_answer)

            except Exception:
                logger.exception("Error on question=%s", question)

                # 记录错误样本以及失败前产生的计算开销
                result_data = {
                    question_string: question,
                    "results": "error",
                    "reasoning_chains": [],
                    "call_num": call_num,
                    "total_token": all_tokens["total"],
                    "input_token": all_tokens["input"],
                    "output_token": all_tokens["output"],
                    "time": time.time() - start_time
                }

                if args.prompt_methods == "sc":
                    result_data["sample_answers"] = sample_answers
                    result_data["sample_reasons"] = sample_reasons

            # 立即写入，防止程序中断导致已经完成的结果丢失
            out.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            out.flush()

    logger.info("Experiment finished. Result file: %s", output_file)
