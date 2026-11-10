import time
import json
from openai import OpenAI
from logging_utils import logger


def run_llm(prompt, temperature, max_tokens, base_url, api_key, engine="gpt-3.5-turbo"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information. Always respond in valid JSON format."},
        {"role": "user", "content": prompt}
    ]

    extra_body = None
    if "qwen" in engine.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif "deepseek" in engine.lower():
        extra_body = {"thinking": {"type": "disabled"}}

    completion = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
        extra_body=extra_body
    )
    content = completion.choices[0].message.content
    usage = completion.usage
    if usage is None:
        raise ValueError("LLM response is missing token usage")

    token_num = {
        "total": usage.total_tokens,
        "input": usage.prompt_tokens,
        "output": usage.completion_tokens
    }

    return content, token_num


# 解析LLM的JSON输出
def parse_answer_json(response, method):
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string")

    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError as error:
        raise ValueError("Invalid answer JSON") from error

    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")

    # 验证最终答案
    if "answer" not in data:
        raise ValueError("Missing answer field")

    answer = data["answer"]
    if answer is not None and not isinstance(answer, str):
        raise ValueError("answer must be a string or null")

    # CoT需要额外解析推理过程
    reason = None
    if method == "cot":
        reason = data.get("reason")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")

    return answer.strip() if answer is not None else None, reason




def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../../../data/test/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../../../data/test/webqsp.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../../../data/test/graliqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found")
        exit(-1)
    return datas, question_string
