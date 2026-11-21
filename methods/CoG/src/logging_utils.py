import logging
import json
from pathlib import Path


logger = logging.getLogger("CoG")


def initialize_logger(output_dir, dataset, model_name, sample_num, sample_seed):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_label = "all" if sample_num is None else sample_num
    log_file = output_path / (
        f"CoG_data_{dataset}_sampleNum_{sample_label}_sampleSeed_{sample_seed}_"
        f"modelName_{model_name.lower()}.log"
    )

    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


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
        details_dir = output.parent / "details"
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
