import os
import re
import json
import time
import random
import logging
import hashlib
import argparse
from tqdm import tqdm
from pathlib import Path
from embedding_model import BiEncoderPairScorer

from model import LLM_Client
from mcts import MCTSPathFinder
from evidence import unpack_path_record
from evidence import collect_candidate_records
from sparql_utils import SPARQLPATH, set_sparql_trace
from evidence import generate_direct_cot_answer, generate_final_answer
from utils import (
    disable_net_proxy,
    prepare_dataset,
    is_valid_answer,
    topic_entity_names,
)
from logging_utils import (
    UsageTracker,
    setup_logging,
)


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
    
    # 配置各模块的依赖关系
    # 1. 关闭问题分解 -> 关闭动态子目标更新
    if not args.enable_decom:
        args.enable_dynamic_subgoal_update = False
    
    # 2. 关闭动态子目标更新 -> 关闭进度奖励和后续所有计划修复操作
    if not args.enable_dynamic_subgoal_update:
        args.enable_progress_reward = False
        args.progress_alpha = 0.0
        args.enable_subgoal_replan = False
        args.enable_replan_tree_reuse = False
        args.replan_patience = 0
    else:
        # 3. 关闭进度奖励 -> 将进度奖励权重置零
        if not args.enable_progress_reward:
            args.progress_alpha = 0.0

        # 4. 关闭子目标修正 -> 关闭修正后的树复用
        if not args.enable_subgoal_replan:
            args.enable_replan_tree_reuse = False
    
    # 5. 关闭子目标动态更新 -> 关闭全树信息检查
    args.enable_full_tree_coordination = bool(args.enable_full_tree_coordination and args.enable_dynamic_subgoal_update)

    # 6. 关闭全树信息检查和子目标修正 -> 将停滞轮数置零
    if not (args.enable_full_tree_coordination or args.enable_subgoal_replan):
        args.replan_patience = 0

    # 检查数值参数的范围
    for option_name in ("answer_threshold", "good_threshold", "progress_alpha"):
        value = getattr(args, option_name)
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError(f"--{option_name} must be in [0, 1].")
    if args.path_topk < 1:
        raise argparse.ArgumentTypeError("--path_topk must >= 1.")
    if (args.enable_full_tree_coordination or args.enable_subgoal_replan) and args.replan_patience < 1:
        raise argparse.ArgumentTypeError("--replan_patience must >= 1.")

    return args


def parse_args():
    parser = argparse.ArgumentParser(description="SPAR-MCTS")

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
    parser.add_argument("--cvt_relation_topk", type=int, default=2, help="Maximum second-hop relations selected when expanding a CVT mediator.")
    parser.add_argument("--entity_topk", type=int, default=5, help="Maximum entities kept after entity pruning.")
    parser.add_argument("--path_topk", type=int, default=5, help="Maximum tree paths supplied to early-stop verification, replanning, and final-answer fallback when no path reaches --good_threshold; otherwise final answering uses every unique good path.")

    # 算法模块相关配置
    parser.add_argument("--enable_decom", action=argparse.BooleanOptionalAction, default=True, help="Enable initial subgoal decomposition before MCTS.")
    parser.add_argument("--enable_dynamic_subgoal_update", action=argparse.BooleanOptionalAction, default=True, help="Enable dynamic subgoal state updates during MCTS.")
    parser.add_argument("--enable_progress_reward", action=argparse.BooleanOptionalAction, default=True, help="Enable evidence-grounded subgoal-hit progress reward.")
    parser.add_argument("--progress_alpha", type=float, default=0.3, help="Weight of the evidence-grounded subgoal progress reward in fused node scores.")
    parser.add_argument("--enable_subgoal_replan", action=argparse.BooleanOptionalAction, default=True, help="Enable stagnation-triggered search plan repair.")
    parser.add_argument("--enable_full_tree_coordination", action=argparse.BooleanOptionalAction, default=True, help="Enable full-tree evidence coordination independently of plan repair.")
    parser.add_argument("--enable_replan_tree_reuse", action=argparse.BooleanOptionalAction, default=True, help="Reuse and reproject the old MCTS tree after plan repair; false restarts from fresh topic roots.")
    parser.add_argument("--replan_patience", type=int, default=2, help="Trigger enabled failure-recovery stages after this many rounds without subgoal progress.")

    # 预设的阈值
    parser.add_argument("--answer_threshold", type=float, default=0.9, help="Early-stop answer threshold on the original 0-1 path-value scale.")
    parser.add_argument("--good_threshold", type=float, default=0.7, help="Minimum evidence path threshold on the original 0-1 path-value scale.")

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


def run_spar_mcts(args, question, top_ent, model, rerank_model):
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
            "subgoal_mode": None,
            "num_subgoals": 0,
            "subgoal_status": [],
            "answer_mode": "direct_cot",
            "answer_source": "direct_cot",
            "actual_iterations": 0,
            "termination_reason": "direct_cot",
            "recovery_events": [],
            "tree_candidates": [],
        }
        if args.enable_subgoal_replan:
            record["replan_count"] = 0
            record["replan_events"] = []

        return record

    # 初始化MCTS检索器
    finder = MCTSPathFinder(
        search_model=model,
        planner_model=model,
        reasoner_model=model,
        rerank_model=rerank_model,
        question=question,
        topic_entities=top_ent,
        max_iterations=args.iterations,
        max_depth=args.depth,
        c_uct=args.c_uct,
        relation_topk=args.relation_topk,
        cvt_relation_topk=args.cvt_relation_topk,
        entity_topk=args.entity_topk,
        path_topk=args.path_topk,
        answer_threshold=args.answer_threshold,
        good_threshold=args.good_threshold,
        enable_decom=args.enable_decom,
        progress_alpha=args.progress_alpha,
        enable_progress_reward=args.enable_progress_reward,
        enable_dynamic_subgoal_update=args.enable_dynamic_subgoal_update,
        enable_subgoal_replan=args.enable_subgoal_replan,
        enable_full_tree_coordination=args.enable_full_tree_coordination,
        enable_replan_tree_reuse=args.enable_replan_tree_reuse,
        replan_patience=args.replan_patience,
    )

    # 子目标分解
    subgoals = finder.global_subgoal_state.subgoals if finder.global_subgoal_state else []
    if subgoals:
        logger.info("Decomposed Subgoals:")
        for sg in subgoals:
            logger.info("  %s: %s | depend: %s | answer_type: %s | resolve_mode: %s | anchor: %s", sg.get("id", ""), sg.get("goal", ""), sg.get("depend", []), sg.get("answer_type", ""), sg.get("resolve_mode", ""), sg.get("anchor_entity", None))
    else:
        logger.info("Decomposed Subgoals: None")

    # MCTS获取路径和答案
    best_paths = finder.search()

    # 验证答案准确性
    if is_valid_answer(finder.early_stop_answer):
        answer, answer_source = finder.early_stop_answer, "early_stop"
    else:
        subgoal_answers = finder.global_subgoal_state.format_answer_context() if finder.global_subgoal_state else "None"
        answer, answer_source = generate_final_answer(question, best_paths, model, subgoal_answers)

    logger.info("========== Final Result ==========")
    logger.info("Question: %s", question)
    if answer_source == "early_stop":
        logger.info("Final-answer generation skipped after accepted early stop.")
    elif answer_source == "paths":
        logger.info("Generated final answer directly from selected KG paths.")
    elif answer_source == "subgoal_answers":
        logger.info("No selected KG path; used direct answering with available subgoal answers.")
    logger.info("Answer: %s", answer)

    # 保存结果
    record = {
        "answer": answer,
        "best_path": [unpack_path_record(item)[0] for item in best_paths],
        "best_evidence": [unpack_path_record(item)[2] for item in best_paths],
        "subgoal_mode": finder.is_complex,
        "num_subgoals": len(finder.global_subgoal_state.subgoals) if finder.global_subgoal_state else 0,
        "subgoal_status": finder.global_subgoal_state.subgoals if finder.global_subgoal_state else [],
        "answer_mode": "spar_mcts",
        "answer_source": answer_source,
        "actual_iterations": finder.actual_iterations,
        "termination_reason": finder.termination_reason,
        "recovery_events": list(finder.recovery_events),
        "tree_candidates": collect_candidate_records(finder.root, selected_nodes=finder.final_answer_nodes),
    }
    if args.enable_subgoal_replan:
        record["replan_count"] = finder.replan_count
        record["replan_events"] = finder.replan_events

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
    fingerprint_args = {key: value for key, value in vars(args).items() if not key.startswith("_") and key not in {"output_dir", "force_recompute", "save_detail", "run_name", "run_dir"}}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_args, sort_keys=True).encode()).hexdigest()[:10]
    return f"SPAR-MCTS_data_{args.dataset}_sampleNum_{sample_label}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}_{fingerprint}"


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
        if args.save_detail:
            prompt_history, sparql_history = [], []
            model.set_prompt_history(prompt_history)
            set_sparql_trace(sparql_history)
        try:
            topics = item.get("topic_entity") or {}
            algorithm_record = run_spar_mcts(args, question, topics, model, rerank_model)
            record = {
                question_key: question,
                "results": algorithm_record["answer"],
                "reasoning_chains": algorithm_record["best_path"]
            }
        except Exception as error:
            logger.exception("Failed to process question: %s", question)
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
                        "sparql_history": sparql_history
                        }, stream, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed to save details for question: %s", question)
    logger.info("Finished processing %d selected samples; results: %s", len(data), output_path)


if __name__ == "__main__":
    args = parse_args()
    disable_net_proxy()
    run(args)
