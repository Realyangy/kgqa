import logging
import os


logger = logging.getLogger("CoT")


def initialize_logger(output_dir, dataset, prompt_method, model_name):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir,
        f"CoT_data_{dataset}_prompt_{prompt_method}_modelName_{model_name.lower()}.log",
    )

    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
