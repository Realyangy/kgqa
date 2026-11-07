import os
import json
import argparse
from tqdm import tqdm

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--prompt_methods", type=str, default="cot", choices=["cot", "io"], help="Prompting method.")
    parser.add_argument("--max_length", type=int, default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature.")
    parser.add_argument("--model_name", type=str, default="GPT-4o-Mini", choices=MODELS_DICT.keys(), help="Model configuration name defined in MODELS_DICT.")
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
                prompt = cot_prompt + "\n\nQ: " + question + "\nA: "
            else:
                prompt = io_prompt + "\n\nQ: " + question + "\nA: "
            results = run_llm(prompt, args.temperature, args.max_length, args.base_url, args.api_key, args.LLM_type)
            logger.info("LLM Response [%s]:\n%s", args.prompt_methods.upper(), results)
            out.write(json.dumps({"question": question, "{}_result".format(args.prompt_methods): results}, ensure_ascii=False) + '\n')
            out.flush()
