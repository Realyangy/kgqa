import logging
from pathlib import Path


class FileOnlyFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, "file_only", False)


def initialize_logger(output_file):
    path = Path(output_file).with_suffix(".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(FileOnlyFilter())
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)
