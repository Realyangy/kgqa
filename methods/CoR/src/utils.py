import json
import logging
import os
import time
from pathlib import Path

from openai import OpenAI


class LLMAPI:
    """OpenAI-compatible model client; one generate() is one algorithmic LLM call."""

    def __init__(self, model_name, base_url, api_key):
        if not api_key:
            raise ValueError("The selected model's API key is not configured")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.reset_usage()

    def reset_usage(self):
        self.call_num = 0
        self.input_token = 0
        self.output_token = 0
        self.total_token = 0

    def generate(self, user_prompt, temperature=0.1, max_tokens=512, system_prompt=None):
        self.call_num += 1
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        extra_body = None
        if "qwen" in self.model_name.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif "deepseek" in self.model_name.lower():
            extra_body = {"thinking": {"type": "disabled"}}

        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        # Preserve the old CoR's three application-level attempts. They count
        # as one logical agent action, but their returned tokens are accumulated.
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, frequency_penalty=0, presence_penalty=0,
                    response_format={"type": "json_object"}, extra_body=extra_body
                )
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens or 0,
                        "output_tokens": response.usage.completion_tokens or 0,
                        "total_tokens": response.usage.total_tokens or 0
                    }
                    self.input_token += usage["input_tokens"]
                    self.output_token += usage["output_tokens"]
                    self.total_token += usage["total_tokens"]

                if not response.choices:
                    raise ValueError("LLM returned no choices")
                choice = response.choices[0]
                if choice.finish_reason == "length":
                    raise ValueError("LLM JSON output was truncated")
                content = choice.message.content
                if not content or not content.strip():
                    raise ValueError("LLM returned an empty response")
                return content, usage
            except Exception as error:
                logging.warning("LLM request attempt %d/3 failed: %s", attempt + 1, error)
        return None, usage


def prepare_dataset(dataset, data_dir):
    datasets = {
        "cwq": ("cwq.json", "question", "ID"),
        "webqsp": ("webqsp.json", "RawQuestion", "QuestionId"),
        "grailqa": ("graliqa.json", "question", "qid")
    }
    filename, question_string, id_field = datasets[dataset]
    with open(Path(data_dir) / filename, encoding="utf-8") as infile:
        datas = json.load(infile)
    return datas, question_string, id_field


def load_previous_results(output_file, question_string):
    previous = {}
    if not os.path.exists(output_file):
        return previous
    with open(output_file, encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = record.get(question_string)
            if question is not None:
                previous[question] = record
    return previous


def save_2_jsonl(question, question_string, answer, reasoning_chains, llm, start_time, output_file, **extra):
    record = {
        question_string: question,
        "results": answer,
        "reasoning_chains": reasoning_chains,
        "call_num": llm.call_num,
        "total_token": llm.total_token,
        "input_token": llm.input_token,
        "output_token": llm.output_token,
        "time": time.monotonic() - start_time,
        **extra
    }
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
