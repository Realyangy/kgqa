import logging
import os


def initialize_logger(output_file):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    log_file = os.path.splitext(output_file)[0] + ".log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter_file = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    formatter_console = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter_file)
    console_handler.setFormatter(formatter_console)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
