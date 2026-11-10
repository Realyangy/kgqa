import os
import re
import json
import time
import argparse
from tqdm import tqdm
from collections import Counter

from utils import *
from prompt_list import *
from logging_utils import initialize_logger


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
    valid_answers = [answer for answer in answers if isinstance(answer, str) and answer.strip()]
    if not valid_answers:
        return None

    normalized_answers = [re.sub(r"[\W_]+", "", answer.casefold()) for answer in valid_answers]
    counter = Counter(normalized_answers)

    # 票数相同时，选择最早出现的答案
    best_index = max(range(len(valid_answers)), key=lambda i: (counter[normalized_answers[i]], -i))
    return valid_answers[best_index]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_methods", type=str, default="cot", choices=["cot", "io"], help="Prompting method.")    
    parser.add_argument("--model_name", type=str, default="GPT-4o-Mini", choices=MODELS_DICT.keys(), help="Model configuration name defined in MODELS_DICT.")
    parser.add_argument("--max_length", type=int, default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature.")
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory for saving logs and results, relative to methods/CoT/src.")
    args = parser.parse_args()

    datas, question_string = prepare_dataset(args.dataset)
    model_config = MODELS_DICT[args.model_name]
    args.LLM_type = model_config["model_path"]
    args.base_url = model_config["base_url"]
    args.api_key = model_config["api_key"]
    disable_net_proxy()

    output_name = f"CoT_data_{args.dataset}_prompt_{args.prompt_methods}_modelName_{args.model_name.lower()}"
    output_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    logger = initialize_logger(args.output_dir, args.dataset, args.prompt_methods, args.model_name)
    logger.info("\n===== Start Running %s on %s dataset =====", args.prompt_methods.upper(), args.dataset)

    with open(output_file, 'a+', encoding="UTF-8") as out:
        for i in tqdm(datas, total=len(datas)):
            question = i[question_string]
            logger.info("\nCurrent Question: %s", question)
            if args.prompt_methods == "cot":
                prompt = cot_prompt.substitute(question=question)
            else:
                prompt = io_prompt.substitute(question=question)
            start_time = time.time()
            response, token_num = run_llm(prompt, args.temperature, args.max_length, args.base_url, args.api_key, args.LLM_type)
            answer, reason = parse_answer_json(response, args.prompt_methods)
            logger.info("LLM Response [%s]:\n%s", args.prompt_methods.upper(), response)
            logger.info("Answer: %s", answer)

            result_data = {
                question_string: question,
                "results": answer,
                "reasoning_chains": [],
                "call_num": 1,
                "total_token": token_num["total"],
                "input_token": token_num["input"],
                "output_token": token_num["output"],
                "time": time.time() - start_time
            }

            if args.prompt_methods == "cot":
                result_data["reason"] = reason

            out.write(json.dumps(result_data, ensure_ascii=False) + "\n")
            out.flush()
