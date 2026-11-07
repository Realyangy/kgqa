import logging
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
