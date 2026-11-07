import os
import random
import argparse
from tqdm import tqdm

from utils import *
from freebase_func import *
from logging_utils import append_search_trace, initialize_logger

MODELS_DICT = {
    "Qwen3.5-27b-AWQ-4bit": {
        "model_path": "/home/flyvideo/Sata2/Models/LLM/Qwen3.5-27B-AWQ-4bit",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
    },
    "DeepSeekV4.1-Flash": {
        "model_path": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("LYY_DEEPSEEK_API_KEY"),
    },
    "GPT-4o-Mini": {
        "model_path": "openai/gpt-4o-mini",
        "base_url": "https://router.shengsuanyun.com/api/v1",
        "api_key": os.getenv("LYY_SHENGSUANYUN_API_KEY"),
    },
}

def disable_net_proxy():
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"]:
        os.environ.pop(var, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--dataset_sample_num", type=int, default=100, help="Number of dataset samples to evaluate. Use None for all samples.")
    parser.add_argument("--dataset_sample_seed", type=int, default=42, help="Random seed used only for selecting the evaluation subset.")
    parser.add_argument("--model_name", type=str, default="GPT-4o-Mini", choices=MODELS_DICT.keys(), help="Model configuration name defined in MODELS_DICT.")
    parser.add_argument("--max_length", type=int, default=256, help="the max length of LLMs output.") # 注意，后面的实体打分时的max-length修改成1024了
    parser.add_argument("--temperature_exploration", type=float, default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float, default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int, default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int, default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool, default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--num_retain_entity", type=int, default=5, help="Number of candidate entities randomly retained for LLM-based entity scoring when more than 20 entities are retrieved.")
    parser.add_argument("--output_dir", type=str, default="../../output", help="Root directory for saving results, relative to methods/ToG/src/freebase.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for all candidate-entity sampling.")
    args = parser.parse_args()

    args.LLM_type = MODELS_DICT[args.model_name]["model_path"]
    args.base_url = MODELS_DICT[args.model_name]["base_url"]
    args.api_key = MODELS_DICT[args.model_name]["api_key"]

    random.seed(args.random_seed)
    disable_net_proxy()

    # 获取数据集，必要时进行采样
    datas, question_string = prepare_dataset(args.dataset)
    total_dataset_size = len(datas)
    if args.dataset_sample_num is not None:
        sample_rng = random.Random(args.dataset_sample_seed)
        sampled_indices = sorted(sample_rng.sample(range(total_dataset_size), args.dataset_sample_num))
        datas = [datas[index] for index in sampled_indices]

    args.output_file = os.path.join(args.output_dir, f"ToG_data_{args.dataset}_sampleNum_{args.dataset_sample_num}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}.jsonl")
    logger = initialize_logger(args.output_dir, args.dataset, args.model_name.lower(), args.dataset_sample_num, args.dataset_sample_seed)
    if args.dataset_sample_num is not None:
        logger.info("\nDataset sampling enabled: selected %d/%d samples with seed %d.", len(datas), total_dataset_size, args.dataset_sample_seed)
    logger.info("\n===== Start Running ToG on %s dataset =====", args.dataset)

    # 读取已有结果；每个问题仅保留结果文件中的最后一条记录
    previous_results = load_previous_results(args.output_file)
    logger.info("Loaded %d previous result records.", len(previous_results))

    # 开始遍历处理样本
    for data in tqdm(datas):
        question = data[question_string]

        # 重新处理错误样本
        previous_result = previous_results.get(question)
        if previous_result is not None:
            if previous_result.get("results") != "error":
                continue
            logger.info("Retry failed question: %s", question)

        # 起点实体信息：pre_relations记录上一跳关系，pre_head记录上一跳方向，用于避免立即回退
        try:
            topic_entities = [
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "pre_relations": [],
                    "pre_head": -1,
                }
                for entity_id, entity_name in data["topic_entity"].items()
            ]

            topic_entities_list = [f"{topic_entity['entity_name']}({topic_entity['entity_id']})" for topic_entity in topic_entities]
            logger.info("\nCurrent Question: %s\nTopic Entity(s): %s", question, ", ".join(topic_entities_list))

            # 用于存放所有检索到的三元组
            cluster_chain_of_entities = []

            # 起点实体为空，则调用LLM直接回答
            if len(topic_entities) == 0:
                results = generate_without_explored_paths(question, args)
                save_2_jsonl(question, results, [], output_file=args.output_file)
                continue

            # 用于标记当前问题是否已经生成并保存结果
            flag_printed = False

            for depth in range(1, args.depth+1):
                logger.info("\n----- Depth: %d -----", depth)
                current_entity_relations_list = []

                # 这里遍历本轮所有起点实体，为它们进行关系扩展
                for entity_info in topic_entities:
                    entity_id = entity_info["entity_id"]

                    # 关系剪枝
                    if entity_id!="[FINISH_ID]":
                        retrieve_relations_with_scores = relation_search_prune(entity_id, entity_info["entity_name"], entity_info["pre_relations"], entity_info["pre_head"], question, args)
                        current_entity_relations_list.extend(retrieve_relations_with_scores)
                        relation_trace = getattr(args, "_last_relation_search", {})
                        append_search_trace(
                            logger, args.dataset, question, depth, "relation_search",
                            entity_id=entity_id,
                            entity_name=entity_info["entity_name"],
                            all_candidates=relation_trace.get("all_candidates", []),
                            selected_candidates=retrieve_relations_with_scores,
                        )

                total_candidates = []
                total_scores = []
                total_relations = []
                total_entities_id = []
                total_topic_entities = []
                total_head = []

                # 下面对上面所有的两元组进行实体扩展
                logger.info("Searching:")
                for entity in current_entity_relations_list:
                    logger.info("Current pair: %s -> %s", entity['entity_name'], entity['relation']) # 这里需要提取实体名称
                    if entity['head']:
                        entity_candidates_id = entity_search(entity['entity_id'], entity['relation'], True)
                    else:
                        entity_candidates_id = entity_search(entity['entity_id'], entity['relation'], False)
                    retrieved_entity_ids = list(entity_candidates_id)

                    # 当前关系下的实体数量超过20时，随机选择num_retain_entity个
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                    append_search_trace(
                        logger, args.dataset, question, depth, "entity_search",
                        source_entity=f"{entity['entity_name']}({entity['entity_id']})",
                        relation=entity['relation'],
                        head=entity['head'],
                        retrieved_candidates=retrieved_entity_ids,
                        candidates_for_scoring=list(entity_candidates_id),
                    )

                    # 当前关系下没有实体时，直接跳过，继续处理下一个关系
                    if len(entity_candidates_id) ==0:
                        continue

                    # 实体选择，并汇总本轮所有候选实体
                    scores, entity_candidates, entity_candidates_id = entity_select(question, entity_candidates_id, entity['score'], entity['relation'], entity['entity_id'], args)
                    append_search_trace(
                        logger, args.dataset, question, depth, "entity_selection",
                        source_entity=f"{entity['entity_name']}({entity['entity_id']})",
                        relation=entity['relation'],
                        candidates=entity_candidates,
                        candidate_ids=entity_candidates_id,
                        scores=scores,
                    )
                    total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                    # logger.info("    New entity(s): %s", " | ".join(total_candidates))

                # 所有关系都没有后续实体时，中断搜索
                if len(total_candidates) ==0:
                    half_stop(question, cluster_chain_of_entities, args.output_file, depth, args)
                    flag_printed = True
                    break

                # 实体剪枝
                flag, chain_of_entities, selected_entity_ids, selected_relations, selected_heads, parent_entity_ids = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
                append_search_trace(
                    logger, args.dataset, question, depth, "entity_prune",
                    selected_paths=chain_of_entities,
                    selected_entity_ids=selected_entity_ids,
                    selected_relations=selected_relations,
                )
                cluster_chain_of_entities.append(chain_of_entities)

                # 本轮关系和实体处理完成后，根据新路径判断是否足以回答问题
                if flag:
                    stop, results = reasoning(question, cluster_chain_of_entities, args)

                    # 足以回答问题
                    if stop:
                        logger.info("Reason result:\nToG stoped at depth %d.", depth)
                        save_2_jsonl(question, results, cluster_chain_of_entities, output_file=args.output_file)
                        flag_printed = True
                        break

                    # 不足以回答问题
                    else:
                        # 可视化本轮保留的路径和推理结果
                        logger.info("\nNew triples:")
                        for source, relation, target in chain_of_entities[0]:
                            logger.info("%s --[%s]--> %s", source, relation, target)
                        logger.info("Reason result:\nDepth %d still not find the answer.", depth)

                        if all(entity_id == "[FINISH_ID]" for entity_id in selected_entity_ids):
                            half_stop(question, cluster_chain_of_entities, args.output_file, depth, args)
                            flag_printed = True
                            break

                        else:
                            next_topic_entities = []
                            for (entity_id, relation, head) in zip(selected_entity_ids, selected_relations, selected_heads):
                                if entity_id == "[FINISH_ID]":
                                    continue

                                # 构造新的实体记录
                                next_topic_entities.append(
                                    {
                                        "entity_id": entity_id,
                                        "entity_name": id2entity_name_or_type(entity_id),
                                        "pre_relations": [relation] if relation else [],
                                        "pre_head": head,
                                    }
                                )

                            # 根据实体 ID 去重，只保留第一次出现的状态
                            unique_topic_entities = {}
                            for entity_info in next_topic_entities:
                                unique_topic_entities.setdefault(entity_info["entity_id"], entity_info)
                            topic_entities = list(unique_topic_entities.values())
                
                else:
                    half_stop(question, cluster_chain_of_entities, args.output_file, depth, args)
                    flag_printed = True
                    break

            # LLM直接生成答案兜底
            if not flag_printed:
                # 有KG信息时结合回答
                if cluster_chain_of_entities:
                    results = generate_answer(question, cluster_chain_of_entities, args)
                    save_2_jsonl(question, results, cluster_chain_of_entities, output_file=args.output_file)
                else:
                    results = generate_without_explored_paths(question, args)
                    save_2_jsonl(question, results, [], output_file=args.output_file)

        # # 异常处理
        # except Exception:
        #     logger.exception("Failed to process question, save error result and continue: %s", question)
        #     save_error_result(args.output_file, question)
        #     continue

        # 异常处理
        except Exception as error:
            logger.exception("Failed to process question, save error result and continue: %s", question)
            reasoning_chains = None
            if isinstance(error, EntitySelectionParseError):
                reasoning_chains = {
                    "module": "entity_selection",
                    "candidates": error.candidates,
                    "raw_llm_output": error.raw_output,
                }

            save_error_result(args.output_file, question, reasoning_chains)
            continue
