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
