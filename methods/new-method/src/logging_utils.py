import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

USAGE_KEYS = ("num_calls", "prompt_tokens", "completion_tokens", "total_tokens")


def setup_logging(args):
    os.makedirs(args.run_dir, exist_ok=True)
    log_path = os.path.join(args.run_dir, args.run_name + ".log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Run started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    safe_args = {key: value for key, value in vars(args).items() if key != "api_key"}
    logger.info("Effective args: %s", json.dumps(safe_args, ensure_ascii=False, sort_keys=True))
    return log_path


class UsageTracker:
    def __init__(self, model):
        self.model = model
        self.before = model.get_usage_snapshot()

    def delta(self):
        after = self.model.get_usage_snapshot()
        return {key: max(0, int(after.get(key, 0) or 0) - int(self.before.get(key, 0) or 0)) for key in USAGE_KEYS}
