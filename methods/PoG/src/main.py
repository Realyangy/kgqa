import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import argparse
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

from utils import *
from freebase_func import *
from logging_utils import initialize_logger, log_file_only, logger, log_subquestions


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
    }
}


# 根据问题找到特定样本
def get_one_data(datas, question_string, question):
    for data in datas:
        if data[question_string] == question:
            return [data]


# 将内部entity_relation_entity_dict转换为易读格式
def log_entity_graph(title, entity_graph, entity_id_to_name, file_only=False):
    log_info = log_file_only if file_only else logger.info
    log_info("%s", title)
    has_triple = False
    for topic_entity_id, directions in entity_graph.items():
        for direction, relations in directions.items():
            for relation, candidate_entity_ids in relations.items():
                for candidate_entity_id in candidate_entity_ids:
                    if direction == "head":
                        source_id, target_id = topic_entity_id, candidate_entity_id
                    else:
                        source_id, target_id = candidate_entity_id, topic_entity_id

                    source = entity_id_to_name.get(source_id, source_id)
                    target = entity_id_to_name.get(target_id, target_id)
                    log_info("%s --[%s]--> %s", source, relation, target)
                    has_triple = True

    if not has_triple:
        log_info("(no triples)")


def disable_net_proxy():
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"]:
        os.environ.pop(var, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3.5-27b-AWQ-4bit", choices=MODELS_DICT.keys(), help="Model configuration name defined in MODEL_DICT.")
    parser.add_argument("--max_length", type=int, default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float, default=0.3, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float, default=0.3, help="the temperature in reasoning stage.")
    parser.add_argument("--embedding_model", type=str, default="/home/flyvideo/Sata2/Models/Embedding-Model/msmarco-distilbert-base-tas-b")
    parser.add_argument("--output_dir", type=str, default="/home/flyvideo/Sata2/lyy/kgqa/methods/PoG/output", help="Root directory for logs, results, and memory files, relative to methods/PoG/src.")
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--dataset_sample_num",type=int, default=500, help="Number of dataset samples to evaluate. Use None for all samples.")
    parser.add_argument("--dataset_sample_seed", type=int, default=42, help="Random seed used only for selecting the evaluation subset.")
    parser.add_argument("--depth", type=int, default=4, help="choose the search depth of PoG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool, default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for all candidate-entity sampling.")
    parser.add_argument("--max_embedding_candidates", type=int, default=70, help="Maximum number of candidates retained by embedding-based filtering.")
    parser.add_argument("--max_reflection_times", type=int, default=5, help="Maximum number of reflection rounds for one question.")
    args = parser.parse_args()

    args.LLM_type = MODELS_DICT[args.model_name]["model_path"]
    args.base_url = MODELS_DICT[args.model_name]["base_url"]
    args.api_key = MODELS_DICT[args.model_name]["api_key"]

    # 确定随机数种子
    random.seed(args.random_seed)
    disable_net_proxy()

    # 加载数据集和已有结果：必要时采样，正常结果跳过，错误结果重跑
    datas, question_string = prepare_dataset(args.dataset)
    total_dataset_size = len(datas)
    if args.dataset_sample_num is not None:
        sample_rng = random.Random(args.dataset_sample_seed)
        sampled_indices = sorted(sample_rng.sample(range(total_dataset_size), args.dataset_sample_num))
        datas = [datas[index] for index in sampled_indices]
        
    # 加载模型
    embedding_model = SentenceTransformer(args.embedding_model)
    args.output_file = str(Path(args.output_dir) / f"PoG_data_{args.dataset}_sampleNum_{args.dataset_sample_num}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}.jsonl")
    logger = initialize_logger(args.output_dir, args.dataset, args.model_name.lower(), args.dataset_sample_num, args.dataset_sample_seed)
    logger.info("\n===== Start Running PoG on %s dataset =====", args.dataset)
    if args.dataset_sample_num is not None:
        logger.info("Dataset sampling enabled: selected %d/%d samples with seed %d.", len(datas), total_dataset_size, args.dataset_sample_seed)    

    previous_results = load_previous_results(args.output_file, question_string)
    logger.info("Loaded %d previous result records.", len(previous_results))

    # 开始完整PoG流程
    for data in tqdm(datas):
        question = data[question_string]

        # 重新处理错误样本
        previous_result = previous_results.get(question)
        if previous_result is not None:
            if previous_result.get("results") != "error":
                continue
            logger.info("Retry failed question: %s", question)

        try:
            # 记录当前样本的时间开销、LLM调用次数和token开销
            start_time = time.time()
            call_num = 0
            all_tokens = {'total': 0, 'input': 0, 'output': 0}

            # 创建当前样本的记忆文件，但是先不写入
            memory_path = Path(args.output_dir) / "memory" / args.dataset / args.model_name.lower() / question
            if not os.path.exists(memory_path):
                os.makedirs(memory_path)
            with open(memory_path / "mem", 'w', encoding='utf-8') as f:
                pass

            initial_topic_entity = data['topic_entity']
            topic_entities = [
                {
                    'entity_id': entity_id,
                    'entity_name': entity_name,
                    'pre_relations': [],
                    'pre_head': -1,
                }
                for entity_id, entity_name in initial_topic_entity.items()
            ]
            topic_entities_list = [f"{entity_info['entity_name']}({entity_info['entity_id']})" for entity_info in topic_entities]
            logger.info("\n[Current question]\n%s\n[Topic entity(s)]\n%s", question, ", ".join(topic_entities_list))

            # 问题分解，并记录相应开销
            call_num += 1
            sub_questions, token_num = get_subquestions(memory_path, question, args)
            for token_type in token_num.keys():
                all_tokens[token_type] += token_num[token_type]
            log_subquestions(sub_questions)

            cluster_chain_of_entities = []
            depth_entities_relations_entities_dict = {}

            # 用于记录反思模块补充实体的次数和具体的实体ID
            reflection_records = {'time': 0, 'ent': []}

            # 实体的名称和ID的相互对应关系
            entity_id_to_name, entity_name_to_id= {}, {}
            for entity_info in topic_entities:
                entity_id_to_name[entity_info['entity_id']] = entity_info['entity_name']
                entity_name_to_id[entity_info['entity_name']] = entity_info['entity_id']

            # 没有起点实体时，LLM直接回答
            if len(topic_entities) == 0:
                call_num += 1
                results, token_num = generate_without_explored_paths(question, sub_questions, args)
                for token_type in token_num.keys():
                    all_tokens[token_type] += token_num[token_type]
                save_2_jsonl(question, question_string, results, [], call_num, all_tokens, start_time, output_file=args.output_file)
                continue

            # 用于标记当前问题是否已经生成并保存结果
            flag_printed = False

            for depth in range(1, args.depth+1):
                logger.info("\n----- Depth: %d -----", depth)

                # 记录当前深度下所有起点实体的关系扩展结果
                current_entity_relations_list = [] 

                # 为所有起点实体进行关系扩展
                for entity_info in topic_entities:
                    entity_id = entity_info['entity_id']
                    if entity_id != "[FINISH_ID]":
                        retrieve_relations, token_num = relation_search_prune(entity_id, sub_questions, entity_info['entity_name'], entity_info['pre_relations'], entity_info['pre_head'], question, args)

                        # 空关系时不调用LLM
                        if token_num["total"] > 0:
                            call_num += 1

                        for token_type in token_num.keys():
                            all_tokens[token_type] += token_num[token_type]
                        current_entity_relations_list.extend(retrieve_relations)

                total_candidates = []
                total_relations = []
                total_entities_id = [] 
                total_topic_entities = [] 
                total_head = []

                # 记录探索得到的所有的实体和关系，按三个层次分组：起点实体ID -> 关系方向 -> 关系及其实体列表
                entity_relation_entity_dict = {}

                # 遍历关系剪枝后的二元组，进行实体扩展
                for entity_relation_pair in current_entity_relations_list:
                    # 在entity_relation_entity_dict中记录起点实体ID
                    if entity_relation_pair['entity'] not in entity_relation_entity_dict.keys():
                        entity_relation_entity_dict[entity_relation_pair['entity']] = {}

                    # 实体搜索
                    if entity_relation_pair['head']:
                        head_or_tail = 'head'
                        entity_candidates_id = entity_search(entity_relation_pair['entity'], entity_relation_pair['relation'], True)
                    else:
                        head_or_tail = 'tail'
                        entity_candidates_id = entity_search(entity_relation_pair['entity'], entity_relation_pair['relation'], False)

                    if len(entity_candidates_id) == 0:
                        direction = "outgoing" if entity_relation_pair['head'] else "incoming"
                        logger.info("No candidate entity for %s relation: %s --[%s]--", direction, entity_id_to_name.get(entity_relation_pair['entity'], entity_relation_pair['entity']), entity_relation_pair['relation'])
                        continue

                    # 将检索到的实体ID转化为可读的实体名称，并双向记录
                    entity_candidates, entity_candidates_id = get_entity_names(entity_candidates_id)
                    entity_name_to_id.update(dict(zip(entity_candidates, entity_candidates_id)))
                    entity_id_to_name.update(dict(zip(entity_candidates_id, entity_candidates)))

                    # 在entity_relation_entity_dict中记录关系方向和具体关系
                    if head_or_tail not in entity_relation_entity_dict[entity_relation_pair['entity']].keys():
                            entity_relation_entity_dict[entity_relation_pair['entity']][head_or_tail] = {}
                    if entity_relation_pair['relation'] not in entity_relation_entity_dict[entity_relation_pair['entity']][head_or_tail].keys():
                        entity_relation_entity_dict[entity_relation_pair['entity']][head_or_tail][entity_relation_pair['relation']] = []

                    # 在entity_relation_entity_dict中记录具体关系下本轮检索到的所有实体
                    for retrive_entities in entity_candidates_id:
                        if retrive_entities not in entity_relation_entity_dict[entity_relation_pair['entity']][head_or_tail][entity_relation_pair['relation']]:
                            entity_relation_entity_dict[entity_relation_pair['entity']][head_or_tail][entity_relation_pair['relation']].append(retrive_entities)
                    
                    total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity_relation_pair, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head)

                # 记录当前深度下的（即本轮）检索的关系和实体信息
                depth_entities_relations_entities_dict[depth] = entity_relation_entity_dict
                logger.info("[Searching]")
                log_entity_graph("Before entity pruning:",  entity_relation_entity_dict, entity_id_to_name, file_only=True)

                # 未成功扩展
                if len(total_candidates) == 0:
                    half_stop(question, question_string, sub_questions, cluster_chain_of_entities, depth, call_num, all_tokens, start_time, args)
                    flag_printed = True
                    break

                # new_entity_relation_entity_dict是实体自适应剪枝后的所有信息，和entity_relation_entity_dict不同
                flag, chain_of_entities, selected_entity_ids, selected_relations, selected_heads, selected_parent_entity_ids, new_entity_relation_entity_dict, cur_call_time, cur_token = entity_condition_prune(question, entity_relation_entity_dict, entity_id_to_name, entity_name_to_id, args, embedding_model)
                call_num += cur_call_time
                for token_type in cur_token.keys():
                    all_tokens[token_type] += cur_token[token_type]
                cluster_chain_of_entities.append(chain_of_entities)
                log_entity_graph("After entity pruning:", new_entity_relation_entity_dict, entity_id_to_name)

                # 实体剪枝成功
                if flag:

                    # 本轮实体剪枝保留的全部关系，作为下一轮共享的历史关系
                    previous_relations = list(selected_relations)

                    # 构造下一轮迭代的起点状态
                    next_topic_entities = []
                    for entity_id, relation, head in zip(selected_entity_ids, selected_relations, selected_heads):
                        next_topic_entities.append({
                            'entity_id': entity_id,
                            'entity_name': entity_id_to_name.get(entity_id, entity_id),
                            'pre_relations': [],
                            'pre_head': head,
                        })

                    # 根据本轮结果更新记忆
                    call_num += 1
                    token_num = update_memory(question, sub_questions, new_entity_relation_entity_dict, entity_id_to_name, memory_path, args)
                    for token_type in token_num.keys():
                        all_tokens[token_type] += token_num[token_type]

                    # 根据本轮结果推理，判断是否能够获得答案，判断是否需要反思回溯
                    call_num += 1
                    results, answer, sufficient, token_num = reasoning(question, new_entity_relation_entity_dict, entity_id_to_name, memory_path, args)
                    for token_type in token_num.keys():
                        all_tokens[token_type] += token_num[token_type]

                    # 判断是否能够停止搜索
                    answer_text = str(answer).strip()
                    stop = (
                        sufficient
                        and answer is not None
                        and answer_text.lower() not in {"", "null", "none", "[]"}
                        and not answer_text.startswith(("m.", '["m.', "['m."))
                    )

                    # LLM判断证据充分后，保存答案并停止搜索
                    if stop:
                        save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_tokens, start_time, output_file=args.output_file)
                        flag_printed = True
                        break
                    # 证据不足时进行反思，判断继续搜索还是换方向搜索
                    else:
                        # 记录反思选择的回溯实体
                        reflection_entity_list = []

                        # selected_entity_ids是本轮实体剪枝后保留的实体，depth_entities_relations_entities_dict中记录了到目前为止所有实体剪枝之前的关系和实体信息，便于回溯时选择回溯锚点
                        if reflection_records['time'] < args.max_reflection_times: 
                            remaining_entity_ids, reflection_entity_list, cur_call_time, cur_token = reflection(question, selected_entity_ids, depth_entities_relations_entities_dict, entity_id_to_name, entity_name_to_id, memory_path, results, cluster_chain_of_entities, args, embedding_model)
                            call_num += cur_call_time
                            for token_type in cur_token.keys():
                                all_tokens[token_type] += cur_token[token_type]

                            # 将反思模块选择的回溯实体加入记录，同时记录本次反思
                            reflection_entity_list = [entity for entity in reflection_entity_list if entity not in reflection_records['ent']]
                            if reflection_entity_list:
                                reflection_records['time'] += 1
                                reflection_records['ent'] += reflection_entity_list

                                # 将选择的实体添加到保留的子图中，便于后续作为搜索起点
                                add_entity_ids, add_relations, add_heads, add_parent_entity_ids, new_entity_relation_entity_dict = restore_backtrack_context(reflection_entity_list, depth_entities_relations_entities_dict, new_entity_relation_entity_dict)
                                
                                # 将 Reflection 恢复的关系加入本轮共享关系列表
                                previous_relations.extend(add_relations)
                                
                                log_entity_graph("\nTriples after reflection and backtrack:\n", new_entity_relation_entity_dict, entity_id_to_name)
                                for entity_id, relation, head in zip(add_entity_ids, add_relations, add_heads):
                                    next_topic_entities.append({
                                        'entity_id': entity_id,
                                        'entity_name': entity_id_to_name.get(entity_id, entity_id),
                                        'pre_relations': [],
                                        'pre_head': head,
                                    })

                            # 选择真正作为下一轮起点的实体，转化为集合去重：搜索阶段扩展得到的下一跳实体 + 反思模块补充的历史实体
                            remaining_entity_ids = set(remaining_entity_ids)
                            next_topic_entities = [
                                entity_info
                                for entity_info in next_topic_entities
                                if entity_info['entity_id'] in remaining_entity_ids or entity_info['entity_id'] in reflection_entity_list
                            ]

                        # 再次对下一轮的候选起点实体进行筛选，仅保留m.开头的，[FINISH_ID]和字面量实体都不能作为起点实体
                        next_topic_entities = [
                            entity_info
                            for entity_info in next_topic_entities
                            if entity_info['entity_id'].startswith('m.') and not if_topic_non_retrieve(entity_info['entity_id'])
                        ]

                        # 根据实体ID去重，仅保留第一次出现的状态
                        unique_topic_entities = {}
                        for entity_info in next_topic_entities:
                            unique_topic_entities.setdefault(entity_info['entity_id'], entity_info)
                        next_topic_entities = list(unique_topic_entities.values())

                        # 为下一轮所有搜索起点设置相同的上一轮关系列表
                        for entity_info in next_topic_entities:
                            entity_info['pre_relations'] = previous_relations.copy()

                        # 没有合适的起点实体时，结束迭代
                        if not next_topic_entities:
                            half_stop(question, question_string, sub_questions, cluster_chain_of_entities, depth, call_num, all_tokens, start_time, args)
                            flag_printed = True
                            break
                        else:
                            logger.info("\n[Topic entities in next iteration]\n%s", ", ".join(f"{entity_info['entity_name']} ({entity_info['entity_id']})" for entity_info in next_topic_entities))
                            topic_entities = next_topic_entities

                # 剪枝失败后，结束迭代
                else:
                    half_stop(question, question_string, sub_questions, cluster_chain_of_entities, depth, call_num, all_tokens, start_time, args)
                    flag_printed = True
                    break

            # 迭代结束仍然没有找到答案
            if not flag_printed:
                call_num += 1
                if cluster_chain_of_entities:
                    logger.info("Search reached the maximum depth, generate the final answer from retrieved triplets.")
                    results, token_num = generate_answer(question, sub_questions, cluster_chain_of_entities,args)
                    reasoning_chains = cluster_chain_of_entities
                else:
                    logger.info("No triplets were retrieved, generate the final answer without explored paths.")
                    results, token_num = generate_without_explored_paths(question, sub_questions, args)
                    reasoning_chains = []
                for token_type in token_num.keys():
                    all_tokens[token_type] += token_num[token_type]
                save_2_jsonl(question, question_string, results, reasoning_chains, call_num, all_tokens, start_time, output_file=args.output_file)

        # 运行过程中捕获异常
        except Exception:
            logger.exception("\n[Error]\nFailed to process question, save error result and continue: %s", question)
            save_error_result(question, question_string, args.output_file)
            continue
