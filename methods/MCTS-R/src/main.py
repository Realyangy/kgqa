import os
import re
import json
import time
import random
import logging
import hashlib
import argparse
import traceback
from tqdm import tqdm
from pathlib import Path
from model import LLM_Client, BiEncoderPairScorer
from mcts import MCTSPathFinder
from evidence import ground_relation_paths
from evidence import collect_candidate_records
from kg_utils import SPARQLPATH, set_sparql_trace
from evidence import generate_direct_cot_answer, generate_final_answer
from utils import disable_net_proxy, prepare_dataset, topic_entity_names, UsageTracker, setup_logging


MODELS_DICT = {
    "Qwen3.5-27b-AWQ-4bit":{
        "model_path": "/home/flyvideo/Sata2/Models/LLM/Qwen3.5-27B-AWQ-4bit",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY"
    },
    "DeepSeekV4.1-Flash": {
        "model_path": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("LYY_DEEPSEEK_API_KEY")
    },
    "GPT-4o-Mini": {
        "model_path": "openai/gpt-4o-mini",
        "base_url": "https://router.shengsuanyun.com/api/v1",
        "api_key": os.getenv("LYY_SHENGSUANYUN_API_KEY")
    },
}


def optional_int(value):
    if value.strip().lower() in {"none", "null"}:
        return None
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer or None.")
    return number


# 规范化命令行参数
def _finalize_args(args):

    # 配置环境变量
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 配置模型参数
    settings = MODELS_DICT[args.model_name]
    args.model_path, args.base_url, args.api_key = settings["model_path"], settings["base_url"], settings["api_key"]
    
    # 检查数值参数的范围
    for option_name in ("answer_threshold", "good_threshold"):
        value = getattr(args, option_name)
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError(f"--{option_name} must be in [0, 1].")
    for option_name in ("path_topk", "relation_topk", "depth", "grounding_rows_per_path"):
        if getattr(args, option_name) < 1:
            raise argparse.ArgumentTypeError(f"--{option_name} must be >= 1.")
    if args.iterations < 0 or args.c_uct < 0:
        raise argparse.ArgumentTypeError("Iterations and UCT exploration coefficient must be non-negative.")
    if args.grounding_rows_per_path > 1000:
        raise argparse.ArgumentTypeError("--grounding_rows_per_path must be <= 1000.")

    return args


def parse_args():
    parser = argparse.ArgumentParser(description="MCTS-R: relation-centric Monte Carlo Tree Search for KGQA")

    # 硬件环境相关配置
    parser.add_argument("--gpu_id", type=int, default=0, help="Physical GPU index for the local embedding model, e.g. 0 or 1.")

    # 模型相关配置
    parser.add_argument("--model_name", choices=MODELS_DICT, default="Qwen3.5-27b-AWQ-4bit")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for all LLM clients.")
    parser.add_argument("--max_length", type=int, default=512, help="Default output token limit for model tasks without a task-specific limit.")
    parser.add_argument("--embedding_model_path", type=str, default="/home/liyuyang/lyy/Models/embedding_model/msmarco-distilbert-base-tas-b", help="Path to the SentenceTransformer bi-encoder used for candidate reranking.")

    # 数据集相关配置
    parser.add_argument("--dataset", type=str.lower, choices=["cwq", "webqsp", "grailqa"], default="cwq")
    parser.add_argument("--dataset_sample_num", "--num_sample", dest="dataset_sample_num", type=optional_int, default=500, help="Number of sampled questions; None uses the full dataset.")
    parser.add_argument("--dataset_sample_seed", "--sample_seed", dest="dataset_sample_seed", type=int, default=42)

    # MCTS参数相关配置
    parser.add_argument("--iterations", type=int, default=5, help="Number of MCTS search iterations per question.")
    parser.add_argument("--depth", type=int, default=5, help="Maximum KG path depth explored by MCTS.")
    parser.add_argument("--c_uct", type=float, default=0.2, help="UCT exploration coefficient.")
    parser.add_argument("--relation_topk", type=int, default=5, help="Maximum relations selected for expansion at each node.")
    parser.add_argument("--path_topk", type=int, default=5, help="Maximum relation paths used for structural verification and final grounding.")

    # 算法模块相关配置
    parser.add_argument("--enable_decom", action=argparse.BooleanOptionalAction, default=True, help="Enable initial subgoal decomposition before MCTS.")

    parser.add_argument("--enable_early_stop", action=argparse.BooleanOptionalAction, default=True, help="Enable structural relation-path sufficiency judgment; disable for fixed-budget search.")
    parser.add_argument("--grounding_rows_per_path", type=int, default=30, help="Maximum connected KG bindings supplied per selected relation path after search.")

    # 预设的阈值
    parser.add_argument("--answer_threshold", type=float, default=0.9, help="Relation-path score threshold for triggering structural early-stop verification.")
    parser.add_argument("--good_threshold", type=float, default=0.7, help="Minimum relation-path relevance score for selection.")

    # 输出相关配置
    parser.add_argument("--force_recompute", action="store_true", help="Ignore a matching existing result file and recompute the selected subset.")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory for experiment results.")
    parser.add_argument("--save_detail", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    # 参数规范化
    try:
        args = _finalize_args(args)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    return args


logger = logging.getLogger(__name__)


def run_relation_mcts(args, question, top_ent, model, rerank_model, step_history=None):
    logger.info("Current Question: %s", question)
    logger.info("Topic Entity(s): %s", top_ent or {})

    # 无起点实体时直接回答
    if not topic_entity_names(top_ent):
        logger.info("Dataset topic entities empty; skipping decomposition and KG search and using direct CoT answering.")
        answer, raw_answer = generate_direct_cot_answer(question, model)
        logger.info("========== Final Result ==========")
        logger.info("Question: %s", question)
        logger.info("Answer mode: direct_cot")
        logger.info("Answer: %s", answer)

        record = {
            "answer": answer,
            "raw_answer": raw_answer,
            "best_path": [],
            "best_evidence": [],
            "best_relation_paths": [],
            "grounding": [],
            "subgoal_mode": None,
            "num_subgoals": 0,
            "initial_subgoals": [],
            "answer_mode": "direct_cot",
            "answer_source": "direct_cot",
            "actual_iterations": 0,
            "termination_reason": "direct_cot",
            "tree_candidates": [],
        }
        return record

    # 初始化MCTS检索器
    finder = MCTSPathFinder(search_model=model, planner_model=model, reasoner_model=model, rerank_model=rerank_model,
                            question=question, topic_entities=top_ent, max_iterations=args.iterations, max_depth=args.depth,
                            c_uct=args.c_uct, relation_topk=args.relation_topk, path_topk=args.path_topk,
                            answer_threshold=args.answer_threshold, good_threshold=args.good_threshold,
                            enable_decom=args.enable_decom, enable_early_stop=args.enable_early_stop, step_history=step_history)

    # 子目标分解
    subgoals = finder.global_subgoal_state.subgoals if finder.global_subgoal_state else []
    if subgoals:
        logger.info("Decomposed Subgoals:")
        for sg in subgoals:
            logger.info("  %s: %s | depend: %s | answer_type: %s | resolve_mode: %s | anchor: %s", sg.get("id", ""), sg.get("goal", ""), sg.get("depend", []), sg.get("answer_type", ""), sg.get("resolve_mode", ""), sg.get("anchor_entity", None))
    else:
        logger.info("Decomposed Subgoals: None")

    # MCTS 只搜索关系链，结束后才进行实体落地。
    best_paths = finder.search()
    grounded = ground_relation_paths(best_paths, step_history=step_history, rows_per_path=args.grounding_rows_per_path)
    # 如果结构早停选中的路径未返回任何有效绑定，尝试树中其他高分路径。
    if not any(group["instances"] for group in grounded) and finder.termination_reason == "relation_early_stop":
        from evidence import get_final_answer_nodes, make_path_record
        from utils import node_path_key
        tried = {node_path_key(node) for node in finder.final_answer_nodes}
        alternatives = [
            node for node in get_final_answer_nodes(finder.root, 0.0, args.path_topk + len(tried))
            if node_path_key(node) not in tried
        ][:args.path_topk]
        if alternatives:
            extra_paths = [make_path_record(node) for node in alternatives]
            extra_grounded = ground_relation_paths(extra_paths, step_history=step_history, rows_per_path=args.grounding_rows_per_path)
            best_paths.extend(extra_paths)
            grounded.extend(extra_grounded)
            finder.final_answer_nodes.extend(alternatives)
        finder.termination_reason = "relation_early_stop_grounding_fallback"
    answer, answer_source = generate_final_answer(question, grounded, model)

    logger.info("========== Final Result ==========")
    logger.info("Question: %s", question)
    logger.info("Answer source: %s", answer_source)
    logger.info("Answer: %s", answer)

    # 保存结果
    record = {
        "answer": answer,
        "best_path": [instance["walk_path"] for group in grounded for instance in group["instances"]],
        "best_evidence": [instance["walk_path"] for group in grounded for instance in group["instances"]],
        "best_relation_paths": best_paths,
        "grounding": grounded,
        "subgoal_mode": finder.is_complex,
        "num_subgoals": len(finder.global_subgoal_state.subgoals) if finder.global_subgoal_state else 0,
        "initial_subgoals": subgoals,
        "answer_mode": "relation_mcts",
        "answer_source": answer_source,
        "actual_iterations": int(getattr(finder, "actual_iterations", 0) or 0),
        "termination_reason": str(getattr(finder, "termination_reason", "unknown") or "unknown"),
        "tree_candidates": collect_candidate_records(finder.root, selected_nodes=getattr(finder, "final_answer_nodes", [])),
    }
    return record


# 加载之前的实验结果，避免重新实验
def load_previous_results(path, question_key):
    if not path.exists():
        return {}
    records = {}
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("result is not an object")
            except (ValueError, UnicodeDecodeError) as error:
                if not line.endswith(b"\n") and not stream.read(1):
                    logger.warning("Discarding incomplete final result at byte %d", start)
                    stream.truncate(start)
                    break
                raise ValueError(f"Invalid result at byte {start}: {path}") from error
            question = record.get(question_key) or record.get("question")
            if question:
                records[question] = record
    return records


# 每完成一个样本就保存一个
def save_results(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records.values():
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


# 根据本次实验参数构造运行名称，用于记录实验结果和日志
def build_run_name(args):
    sample_label = "all" if args.dataset_sample_num is None else args.dataset_sample_num
    fingerprint_args = {key: value for key, value in vars(args).items() if not key.startswith("_") and key not in {"output_dir", "force_recompute", "save_detail", "run_name", "run_dir", "api_key"}}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_args, sort_keys=True).encode()).hexdigest()[:10]
    return f"MCTS-R_data_{args.dataset}_sampleNum_{sample_label}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}_{fingerprint}"


def run(args):
    # 构造实验记录文件夹
    args.run_name = build_run_name(args)
    args.run_dir = str(Path(args.output_dir) / args.run_name)
    setup_logging(args)
    output_path = Path(args.run_dir) / (args.run_name + ".jsonl")
    logger.info("Freebase endpoint: %s", SPARQLPATH)

    # 读取数据集，必要时采样，并避免重复实验
    data, question_key = prepare_dataset(args.dataset)
    if args.dataset_sample_num is not None:
        if not 0 <= args.dataset_sample_num <= len(data):
            raise ValueError("Sample count must be between 0 and the dataset size")
        indices = sorted(random.Random(args.dataset_sample_seed).sample(range(len(data)), args.dataset_sample_num))
        data = [data[index] for index in indices]
    logger.info("Loaded %d samples from dataset %s", len(data), args.dataset)
    previous = {} if args.force_recompute else load_previous_results(output_path, question_key)
    logger.info("Loaded %d previous result records", len(previous))

    # 配置模型
    model = LLM_Client(args.model_path, args.base_url, args.api_key, temperature=args.temperature, max_new_tokens=args.max_length)
    rerank_model = BiEncoderPairScorer(args.embedding_model_path)

    # 循环处理各样本
    for index, item in enumerate(tqdm(data, desc="Processing")):
        question = item[question_key]
        old = previous.get(question)
        if old and old.get("results") != "error" and all(key in old for key in ("results", "call_num", "input_token", "output_token", "total_token", "time")):
            continue
        if old:
            logger.info("Retrying failed or incomplete question: %s", question)

        # 执行MCTS并记录结果，无论正确还是错误
        started = time.monotonic()
        tracker = UsageTracker(model)
        error_traceback = None
        step_history = [] if args.save_detail else None
        if args.save_detail:
            prompt_history, sparql_history = [], []
            model.set_prompt_history(prompt_history)
            set_sparql_trace(sparql_history)
        try:
            topics = item.get("topic_entity") or {}
            algorithm_record = run_relation_mcts(args, question, topics, model, rerank_model, step_history)
            record = {
                question_key: question,
                "results": algorithm_record["answer"],
                "reasoning_chains": algorithm_record["best_path"],
                "relation_paths": algorithm_record["best_relation_paths"]
            }
        except Exception as error:
            logger.exception("Failed to process question: %s", question)
            error_traceback = traceback.format_exc()
            algorithm_record = None
            record = {
                question_key: question,
                "results": "error",
                "reasoning_chains": None,
                "error": f"{type(error).__name__}: {error}"
            }
        finally:
            if args.save_detail:
                model.set_prompt_history(None)
                set_sparql_trace(None)
        
        usage = tracker.delta()
        record.update({"call_num": usage["num_calls"], "input_token": usage["prompt_tokens"], "output_token": usage["completion_tokens"], "total_token": usage["total_tokens"], "time": round(time.monotonic() - started, 4)})
        previous[question] = record
        save_results(output_path, previous)
        if args.save_detail:
            try:
                detail_dir = Path(args.run_dir) / "details"
                detail_dir.mkdir(exist_ok=True)
                question_id = item.get("ID") or item.get("QuestionId") or item.get("qid") or index
                safe_id = re.sub(r"[^\w.\-]", "_", str(question_id))
                detail_path = detail_dir / f"{safe_id}_{hashlib.sha256(question.encode()).hexdigest()[:8]}.json"
                with detail_path.open("w", encoding="utf-8") as stream:
                    json.dump({
                        "id": question_id,
                        "question": question,
                        "topic_entities": item.get("topic_entity") or {},
                        "result": record,
                        "algorithm_record": algorithm_record,
                        "prompt_history": prompt_history,
                        "sparql_history": sparql_history,
                        "step_history": step_history,
                        "traceback": error_traceback
                        }, stream, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed to save details for question: %s", question)
    logger.info("Finished processing %d selected samples; results: %s", len(data), output_path)


if __name__ == "__main__":
    args = parse_args()
    disable_net_proxy()
    run(args)
