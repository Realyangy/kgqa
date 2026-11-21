import ast
import json
import logging
from pathlib import Path


logger = logging.getLogger("PoG")


class FileOnlyFilter(logging.Filter):
    """Hide records marked as file-only from the terminal handler."""

    def filter(self, record):
        return not getattr(record, "file_only", False)


def log_file_only(message, *args):
    """Write an INFO record to the log file without printing it in the terminal."""
    logger.info(message, *args, extra={"file_only": True})


def initialize_logger(output_dir, dataset, model_name, dataset_sample_num, dataset_sample_seed):
    """Initialize the per-run logger for both the terminal and output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    file_handler = logging.FileHandler(
        output_path / f"PoG_data_{dataset}_sampleNum_{dataset_sample_num}_sampleSeed_{dataset_sample_seed}_modelName_{model_name.lower()}.log",
        mode="a",
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(FileOnlyFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_subquestions(sub_questions):
    logger.info("[Subquestions]")

    if isinstance(sub_questions, list):
        displayed_subquestions = sub_questions
    else:
        try:
            displayed_subquestions = json.loads(sub_questions)
        except (json.JSONDecodeError, TypeError):
            try:
                displayed_subquestions = ast.literal_eval(sub_questions)
            except (ValueError, SyntaxError, TypeError):
                displayed_subquestions = [sub_questions]

    if not isinstance(displayed_subquestions, list):
        displayed_subquestions = [displayed_subquestions]

    for index, subquestion in enumerate(displayed_subquestions, start=1):
        logger.info("%d. %s", index, subquestion)


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
