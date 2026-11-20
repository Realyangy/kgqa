import os
import json
import logging
from pathlib import Path

logger = logging.getLogger("ToG")


# 在终端输出中隐藏结构化搜索记录，日志文件仍然完整保留
class SearchTraceFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(("SEARCH_TRACE ", "LLM Response "))


# 初始化 INFO 级别日志，同时输出到终端和当前实验对应的日志文件
def initialize_logger(output_dir, dataset, model_name, dataset_sample_num, dataset_sample_seed):
    os.makedirs(output_dir, exist_ok=True)
    run_log_file = os.path.join(output_dir, f"ToG_data_{dataset}_sampleNum_{dataset_sample_num}_sampleSeed_{dataset_sample_seed}_modelName_{model_name.lower()}.log")

    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(
        run_log_file,
        mode="a",
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SearchTraceFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# 将结构化搜索记录写入统一日志文件
def append_search_trace(logger, dataset, question, depth, event, **kwargs):
    record = {
        "dataset": dataset,
        "question": question,
        "depth": depth,
        "event": event,
        **kwargs,
    }
    record_step(event, depth=depth, **kwargs)
    logger.info(
        "SEARCH_TRACE %s",
        json.dumps(record, ensure_ascii=False, default=str),
    )


# Per-question details; tracing is optional and never changes search decisions.
_active_detail = None


def begin_detail(data, question, output_file, enabled=True):
    global _active_detail
    if not enabled:
        _active_detail = None
        return
    _active_detail = {
        "question": question,
        "id": data.get("ID") or data.get("QuestionId") or data.get("id") or question,
        "topic_entities": data.get("topic_entity", {}),
        "prompt_history": [],
        "sparql_history": [],
        "step_history": [],
        "_output_file": str(output_file),
    }


def record_llm(prompt, response, usage=None, **metadata):
    if _active_detail is not None:
        _active_detail["prompt_history"].append({
            "prompt": prompt, "response": response,
            "token_usage": usage or {}, **metadata,
        })


def record_sparql(query, result_count=None, error=None):
    if _active_detail is not None:
        _active_detail["sparql_history"].append({
            "sparql": query, "result_count": result_count,
            "status": "error" if error is not None else "ok",
            **({"error": str(error)} if error is not None else {}),
        })


def record_step(event, **fields):
    if _active_detail is not None:
        _active_detail["step_history"].append({"event": event, **fields})


def finish_detail(record):
    global _active_detail
    active = _active_detail
    _active_detail = None
    if active is None:
        return
    try:
        import hashlib
        import re
        output = Path(active.pop("_output_file"))
        details_dir = output.parent / output.stem / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        raw_id = str(active["id"])
        safe_id = re.sub(r"[^\w.-]", "_", raw_id)[:120]
        unique_hash = hashlib.sha256(active["question"].encode("utf-8")).hexdigest()[:12]
        filename = f"{safe_id}_{unique_hash}.json"
        active["result"] = record
        with (details_dir / filename).open("w", encoding="utf-8") as stream:
            json.dump(active, stream, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.exception("Failed to save question details")
