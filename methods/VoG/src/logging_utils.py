import logging
from pathlib import Path


logger = logging.getLogger("VoG")


def initialize_logger(output_dir, dataset, model_name):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / f"VoG_{dataset}_{model_name.lower()}.log"

    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
