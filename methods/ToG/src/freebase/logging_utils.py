import os
import json
import logging

logger = logging.getLogger("ToG")


# 在终端输出中隐藏结构化搜索记录，日志文件仍然完整保留
class SearchTraceFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(("SEARCH_TRACE ", "LLM Response "))


# 初始化 INFO 级别日志，同时输出到终端和当前实验对应的日志文件
def initialize_logger(output_dir, dataset, model_name, dataset_sample_num, dataset_sample_seed):
    os.makedirs(output_dir, exist_ok=True)
    run_log_file = os.path.join(output_dir, f"ToG_data_{dataset}_sampleNum_{dataset_sample_num}_sampleSeed_{dataset_sample_seed}_modelName_{model_name.lower()}.log")

    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(
        run_log_file,
        mode="a",
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SearchTraceFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# 将结构化搜索记录写入统一日志文件
def append_search_trace(logger, dataset, question, depth, event, **kwargs):
    record = {
        "dataset": dataset,
        "question": question,
        "depth": depth,
        "event": event,
        **kwargs,
    }
    logger.info(
        "SEARCH_TRACE %s",
        json.dumps(record, ensure_ascii=False, default=str),
    )
