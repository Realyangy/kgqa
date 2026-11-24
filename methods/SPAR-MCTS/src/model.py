"""OpenAI-compatible calls for search, planning and reasoning."""
import logging
import time
from openai import OpenAI

logger = logging.getLogger(__name__)


class ModelCallError(RuntimeError):
    """A model call failed after its retry attempts."""


class LLM_Client:
    def __init__(self, model_path, base_url="http://localhost:8001/v1", api_key="EMPTY", temperature=0.3, max_new_tokens=256, n=1):
        self.model_name = model_path
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", max_retries=0)
        self.temperature, self.max_new_tokens, self.n = temperature, max_new_tokens, n
        self.prompt_history = None
        self.usage_stats = {"num_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def set_prompt_history(self, history):
        self.prompt_history = history

    def get_usage_snapshot(self):
        return dict(self.usage_stats)

    def __call__(self, system_prompt=None, user_prompt=None, n=None, max_tokens=None, temperature=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt or ""})
        kwargs = {"model": self.model_name, "messages": messages, "temperature": self.temperature if temperature is None else temperature, "max_tokens": self.max_new_tokens if max_tokens is None else max_tokens}
        name = self.model_name.lower()
        if "qwen3" in name:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif "deepseek" in name:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        results = []
        for _ in range(self.n if n is None else n):
            for attempt in range(3):
                self.usage_stats["num_calls"] += 1
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    usage = getattr(response, "usage", None)
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or prompt_tokens + completion_tokens
                    for key, value in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens), ("total_tokens", total_tokens)):
                        self.usage_stats[key] += value
                    if usage is None:
                        logger.warning("Model response omitted usage; token counts for this call are unavailable.")
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        raise ValueError("Model returned empty content")
                    if getattr(response.choices[0], "finish_reason", None) == "length":
                        raise ValueError("Model response was truncated")
                    if self.prompt_history is not None:
                        self.prompt_history.append({"model": self.model_name, "system_prompt": system_prompt, "prompt": user_prompt, "response": content, "input_tokens": prompt_tokens if usage else None, "output_tokens": completion_tokens if usage else None})
                    results.append(content)
                    break
                except Exception as error:
                    logger.warning("Model call attempt %d/3 failed: %s", attempt + 1, error)
                    if attempt == 2:
                        raise ModelCallError(f"Model call failed after 3 attempts: {error}") from error
                    time.sleep(2)
        return results
