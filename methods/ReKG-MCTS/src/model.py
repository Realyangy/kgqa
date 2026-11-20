"""OpenAI-compatible model calls used by ReKG-MCTS."""
import logging
import time

from openai import OpenAI
from prompt_list import answer_directly_prompt
from utils import extract_answer

logger = logging.getLogger(__name__)


class LLM_Client:
    def __init__(self, model_path, base_url, api_key, temperature=0.5, max_new_tokens=256):
        self.model_name = model_path
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.answer_prompt = None
        self.prompt_history = None
        self.usage_stats = {"num_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def set_prompt_history(self, history):
        self.prompt_history = history

    def get_usage_snapshot(self):
        return dict(self.usage_stats)

    def set_answer_prompt(self, prompt):
        self.answer_prompt = prompt

    def generate_answer(self, question, cluster_chain_of_entities, system_prompt=None):
        chain = "\n".join(f"{head}, {relation}, {tail}" for head, relation, tail in cluster_chain_of_entities)
        prompt = self.answer_prompt.substitute(question=question, knowledge_triplets=chain or "None")
        raw_answer = self(system_prompt=system_prompt, user_prompt=prompt)[0]
        return raw_answer, extract_answer(raw_answer)

    def generate_direct_answer(self, question):
        prompt = answer_directly_prompt.substitute(question=question)
        raw_answer = self(user_prompt=prompt)[0]
        return raw_answer, extract_answer(raw_answer)

    def __call__(self, system_prompt=None, user_prompt=None, n=1, max_tokens=None, temperature=None):
        if user_prompt is None:
            user_prompt, system_prompt = system_prompt or "", None
        messages = [
            {"role": "system", "content": system_prompt or "You are an AI assistant that helps people find information. Always respond in valid JSON format."},
            {"role": "user", "content": user_prompt},
        ]
        extra_body = None
        if "qwen" in self.model_name.lower():
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif "deepseek" in self.model_name.lower():
            extra_body = {"thinking": {"type": "disabled"}}
        results = []
        for _ in range(n):
            self.usage_stats["num_calls"] += 1
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name, messages=messages,
                        temperature=self.temperature if temperature is None else temperature,
                        max_tokens=max_tokens or self.max_new_tokens,
                        frequency_penalty=0, presence_penalty=0,
                        response_format={"type": "json_object"}, extra_body=extra_body,
                    )
                    if response.usage:
                        self.usage_stats["prompt_tokens"] += response.usage.prompt_tokens or 0
                        self.usage_stats["completion_tokens"] += response.usage.completion_tokens or 0
                        self.usage_stats["total_tokens"] += response.usage.total_tokens or 0
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise ValueError("LLM returned empty JSON")
                    if response.choices[0].finish_reason == "length":
                        raise ValueError("LLM JSON output was truncated")
                    if self.prompt_history is not None:
                        self.prompt_history.append({
                            "system_prompt": messages[0]["content"],
                            "prompt": user_prompt, "response": content,
                            "input_tokens": response.usage.prompt_tokens if response.usage else None,
                            "output_tokens": response.usage.completion_tokens if response.usage else None,
                        })
                    logger.info("LLM Response: %s", content, extra={"file_only": True})
                    results.append(content)
                    break
                except Exception as error:
                    logger.warning("OpenAI error, attempt %d/3: %s", attempt + 1, error)
                    if attempt == 2:
                        raise RuntimeError(f"Failed to call LLM after 3 retries: {error}") from error
                    time.sleep(2)
        return results
