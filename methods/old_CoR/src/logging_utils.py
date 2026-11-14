import logging
import os


def initialize_logger(output_dir, dataset, model_name, sample_num, sample_seed):
    os.makedirs(output_dir, exist_ok=True)
    filename = (
        f"CoR_data_{dataset}_sampleNum_{sample_num}_sampleSeed_{sample_seed}_"
        f"modelName_{model_name.lower()}.log"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(os.path.join(output_dir, filename), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
