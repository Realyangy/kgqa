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
