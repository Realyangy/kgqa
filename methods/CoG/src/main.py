import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import time
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from collections import Counter
from sentence_transformers import SentenceTransformer

from utils import *
from freebase_func import *
from logging_utils import initialize_logger, logger


# 用于断点续跑，若已有结果文件，会跳过其中的样本
def repeat_unanswer(datas, question_string, output_file):
    answered_set = set()
    file_path = output_file
    if not os.path.exists(file_path):
        return datas
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                if question_string in data:
                    answered_set.add(data[question_string])
            except Exception:
                continue

    new_data = [x for x in datas if x[question_string] not in answered_set]
    return new_data


# 根据问题查找对应样本
def get_one_data(datas, question_string, question):
    for data in datas:
        if data[question_string] == question:
            return [data]


# CoG单题处理流程
def process_question(data, args, question_string, model, retriever):
    local_stats = Counter()
    try:
        start_time = time.time()
        call_num = 0
        all_tokens = {'total': 0, 'input': 0, 'output': 0}
        best_answer_so_far = "null" 
        
        final_json_to_save = None
        final_chain_to_save = []

        question = data[question_string]
        logger.info("\nCurrent Question：%s", question)
        safe_question_name = re.sub(r'[^\w\-. ]+', '_', question)[:200]
        memory_path = str(Path(args.output_dir) / "memory" / args.dataset / args.model_name.lower() / safe_question_name)
        if not os.path.exists(memory_path):
            os.makedirs(memory_path)
        with open(memory_path+'/mem', 'w', encoding='utf-8') as f:
            pass

        # 关系路径及其嵌入
        skeleton_chain = []
        global_skeleton_emb = None 

        # 起点实体
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
        topic_entity_names = list(initial_topic_entity.values())

        # 获取关系蓝图
        if retriever:
            try:
                skeleton_chain, skel_tokens, skeleton_str, spec_type = speculate_skeleton_chain(question, topic_entity_names, retriever, args, model)
                local_stats[spec_type] += 1
                call_num += skel_tokens.get('calls', 0)
                for token_type in skel_tokens.keys():
                    if token_type != 'calls': 
                        all_tokens[token_type] += skel_tokens.get(token_type, 0)
                
                if skeleton_str and model:
                    global_skeleton_emb = model.encode(skeleton_str)
                    
            except Exception as e:
                logger.info("Speculation Failed: %s", e)

        # 问题分解
        call_num += 1
        sub_questions, token_num = get_subquestions(memory_path, question, args)
        for token_type in token_num.keys():
            all_tokens[token_type] += token_num[token_type]

        cluster_chain_of_entities = []
        depth_entities_relations_entities_dict = {}
        reflection_records = {'time': 0, 'ent': []}

        entity_id_to_name = {}
        entity_name_to_id = {}
        for entity_info in topic_entities:
            entity_id_to_name[entity_info['entity_id']] = entity_info['entity_name']
            entity_name_to_id[entity_info['entity_name']] = entity_info['entity_id']
        
        # 没有起点实体的情况
        if len(topic_entities) == 0:
            logger.warning("No topic entity. Falling back to refinement.")
            call_num += 1
            refined_answer = generate_final_answer(question, [], args, all_tokens)
            save_2_jsonl(question, question_string, refined_answer, [], call_num, all_tokens, start_time, output_file=args.output_file)
            return local_stats

        flag_answered = False

        # 开始深度迭代
        for depth in range(1, args.depth+1):
            current_pred_rel = None

            # 取出本轮迭代对应的关系蓝图中的关系，这里是直接根据深度取得的蓝图位置，这里和论文中不一致
            if skeleton_chain and (depth - 1) < len(skeleton_chain):
                current_pred_rel = skeleton_chain[depth - 1]

            current_entity_relations_list = []
            layer_guard_budget = 1 

            # 多起点实体时，按照其与问题的语义相似度，从高到底排序
            if len(topic_entities) > 1:
                try:
                    entities_names = [str(entity_info['entity_name']) for entity_info in topic_entities]
                    question_emb = model.encode(question)
                    entity_embs = model.encode(entities_names)
                    scores = util.cos_sim(question_emb, entity_embs)[0].cpu().numpy()
                    sorted_indices = scores.argsort()[::-1]
                    topic_entities = [topic_entities[i] for i in sorted_indices]
                except Exception as e:
                    logger.info("Sort failed, keep original order: %s", e)

            # 为起点实体进行关系蓝图辅助的关系扩展
            for entity_info in topic_entities:
                entity = entity_info['entity_id']
                if entity != "[FINISH_ID]":

                    # 关系扩展
                    retrieve_relations, token_num, used_count, track_type = relation_search_prune(entity, sub_questions,  entity_info['entity_name'], entity_info['pre_relations'], entity_info['pre_head'], question,  args, model=model,  predicted_relation=current_pred_rel, global_skeleton_emb=global_skeleton_emb, speculation_list=skeleton_chain, guard_budget=layer_guard_budget)
                    local_stats[track_type] += 1
                    local_stats['total_steps'] += 1
                    layer_guard_budget -= used_count
                    call_num += token_num.get('calls', 0) 
                    for token_type in token_num.keys():
                        if token_type != 'calls':
                            all_tokens[token_type] += token_num.get(token_type, 0)
                    
                    current_entity_relations_list.extend(retrieve_relations)
            
            total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = [], [], [], [], []
            entity_relation_entity_dict = {}

            # 关系扩展失败后，使用当前已有信息生成答案
            if not current_entity_relations_list:
                logger.info("No approved relations. Fallback to Refinement.")
                call_num += 1
                final_json_to_save = generate_final_answer(question, cluster_chain_of_entities, args, all_tokens, best_answer_so_far)
                final_chain_to_save = cluster_chain_of_entities
                flag_answered = True
                break
            
            # 实体扩展
            for ent_rel_pair in current_entity_relations_list:
                if ent_rel_pair['entity'] not in entity_relation_entity_dict.keys():
                    entity_relation_entity_dict[ent_rel_pair['entity']] = {}
                
                if ent_rel_pair['head']:
                    head_or_tail = 'head'
                    entity_candidates_id = entity_search(ent_rel_pair['entity'], ent_rel_pair['relation'], True)
                else:
                    head_or_tail = 'tail'
                    entity_candidates_id = entity_search(ent_rel_pair['entity'], ent_rel_pair['relation'], False)
                if len(entity_candidates_id) == 0:
                    continue

                entity_candidates, entity_candidates_id = provide_triple(entity_candidates_id, ent_rel_pair['relation'])
                entity_name_to_id.update(dict(zip(entity_candidates, entity_candidates_id)))
                entity_id_to_name.update(dict(zip(entity_candidates_id, entity_candidates)))

                if head_or_tail not in entity_relation_entity_dict[ent_rel_pair['entity']].keys():
                        entity_relation_entity_dict[ent_rel_pair['entity']][head_or_tail] = {}
                if ent_rel_pair['relation'] not in entity_relation_entity_dict[ent_rel_pair['entity']][head_or_tail].keys():
                    entity_relation_entity_dict[ent_rel_pair['entity']][head_or_tail][ent_rel_pair['relation']] = []
                
                for retrive_entities in entity_candidates_id:
                    if retrive_entities not in entity_relation_entity_dict[ent_rel_pair['entity']][head_or_tail][ent_rel_pair['relation']]:
                        entity_relation_entity_dict[ent_rel_pair['entity']][head_or_tail][ent_rel_pair['relation']].append(retrive_entities)
                
                total_candidates, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, ent_rel_pair, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head)
            depth_entities_relations_entities_dict[depth] = entity_relation_entity_dict

            # 实体扩展失败后，也是根据已有信息回答
            if len(total_candidates) == 0:
                logger.info("Path pruned to empty (Candidates=0). Fallback to Refinement.")
                call_num += 1
                final_json_to_save = generate_final_answer(question, cluster_chain_of_entities, args, all_tokens, best_answer_so_far)
                final_chain_to_save = cluster_chain_of_entities
                flag_answered = True
                break
            
            # 实体剪枝
            flag, chain_of_entities, selected_entity_ids, selected_relations, selected_heads, new_entity_relation_entity_dict, cur_call_time, cur_token = entity_condition_prune(question, entity_relation_entity_dict, entity_id_to_name, entity_name_to_id, args, model)
            cluster_chain_of_entities.append(chain_of_entities)
            call_num += cur_call_time
            for token_type in cur_token.keys():
                all_tokens[token_type] += cur_token[token_type]

            # 实体剪枝成功时，记录本轮的扩展结果
            if flag:
                previous_relations = list(selected_relations)
                next_topic_entities = [
                    {
                        'entity_id': entity_id,
                        'entity_name': entity_id_to_name.get(entity_id, entity_id),
                        'pre_relations': [],
                        'pre_head': head,
                    }
                    for entity_id, head in zip(selected_entity_ids, selected_heads)
                ]

                # 更新记忆
                call_num += 1
                token_num = update_memory(question, sub_questions, new_entity_relation_entity_dict, entity_id_to_name, cluster_chain_of_entities, memory_path, args)
                for token_type in token_num.keys():
                    all_tokens[token_type] += token_num[token_type]

                # 推理，判断当前信息是否足够回答问题
                call_num += 1
                results, answer, sufficient, token_num = reasoning(question, sub_questions, new_entity_relation_entity_dict, entity_id_to_name, cluster_chain_of_entities, memory_path, args)
                for token_type in token_num.keys():
                    all_tokens[token_type] += token_num[token_type]

                # 判断LLM给出的答案是否有效，可能给出Freebase ID或者none这种无效的答案
                answer_is_id = str(answer).startswith('m.') or str(answer).startswith("['m.") or str(answer).startswith('["m.')
                answer_is_valid = str(answer).lower() not in ['null', 'none']

                # 答案有效则记录
                if answer_is_valid:
                    if best_answer_so_far in ['null', 'none', 'Null', None]:
                        best_answer_so_far = answer
                    elif (str(best_answer_so_far).startswith('m.') or str(best_answer_so_far).startswith('[')) and not answer_is_id:
                        best_answer_so_far = answer
                    elif not (str(best_answer_so_far).startswith('m.') or str(best_answer_so_far).startswith('[')) and answer_is_id:
                        pass 
                    else:
                        best_answer_so_far = answer

                # 充分性判断返回布尔值，保留原有无效答案过滤
                stop = (
                    sufficient
                    and answer is not None
                    and str(answer).strip().lower() not in {"", "null", "none", "[]"}
                    and not str(answer).startswith(("m.", '["m.', "['m."))
                )

                # 证据充足时直接回答
                if stop:
                    logger.info("Stop at depth %d. Answer found.", depth)
                    final_json_to_save = results
                    final_chain_to_save = cluster_chain_of_entities
                    flag_answered = True
                    break

                # 证据不充足时进行反思
                else:
                    reflection_entity_list = []
                    if reflection_records['time']<3:
                        selected_entity_ids, reflection_entity_list, cur_call_time, cur_token = reflection(question, selected_entity_ids, depth_entities_relations_entities_dict, entity_id_to_name, entity_name_to_id, memory_path, results, cluster_chain_of_entities, args, model)
                        call_num += cur_call_time
                        for token_type in cur_token.keys():
                            all_tokens[token_type] += cur_token[token_type]

                        # 提取反思选择的回溯实体
                        reflection_entity_list = [entity_id for entity_id in reflection_entity_list if entity_id not in reflection_records['ent']]
                        if reflection_entity_list:
                            reflection_records['time'] += 1
                            reflection_records['ent'] += reflection_entity_list

                            add_entity_ids, add_relations, add_heads, new_entity_relation_entity_dict = restore_backtrack_context(reflection_entity_list, depth_entities_relations_entities_dict, new_entity_relation_entity_dict, entity_id_to_name, entity_name_to_id, args)
                            previous_relations.extend(add_relations)
                            next_topic_entities.extend(
                                {
                                    'entity_id': entity_id,
                                    'entity_name': entity_id_to_name.get(entity_id, entity_id),
                                    'pre_relations': [],
                                    'pre_head': head,
                                }
                                for entity_id, head in zip(add_entity_ids, add_heads)
                            )

                    # 没有可继续搜索的起点，或已达到最大搜索深度时，根据已有信息回答
                    if not next_topic_entities or depth >= args.depth:
                        logger.info("Forced stop (Max Depth), fallback to Refinement.")
                        call_num += 1
                        final_json_to_save = generate_final_answer(question, cluster_chain_of_entities, args, all_tokens, best_answer_so_far)
                        final_chain_to_save = cluster_chain_of_entities
                        flag_answered = True
                        break

                    # 否则准备下一轮的起点实体
                    else:
                        # 将搜索阶段保留的和反思阶段选择的起点实体合并后去重
                        remaining_entity_ids = set(selected_entity_ids)
                        next_topic_entities = [
                            entity_info for entity_info in next_topic_entities
                            if entity_info['entity_id'] in remaining_entity_ids
                            or entity_info['entity_id'] in reflection_entity_list
                        ]

                        # 过滤去重后的起点实体，仅保留m.开头的
                        next_topic_entities = [
                            entity_info for entity_info in next_topic_entities
                            if entity_info['entity_id'].startswith('m.')
                            and not if_topic_non_retrieve(entity_info['entity_id'])
                        ]

                        # 再根据ID去重
                        unique_topic_entities = {}
                        for entity_info in next_topic_entities:
                            unique_topic_entities.setdefault(entity_info['entity_id'], entity_info)
                        topic_entities = list(unique_topic_entities.values())

                        # 没有合适的起点实体时，结束迭代，根据已有信息回答
                        if not topic_entities:
                            logger.info("No retrievable topic entities. Fallback to Refinement.")
                            call_num += 1
                            final_json_to_save = generate_final_answer(question, cluster_chain_of_entities, args, all_tokens, best_answer_so_far)
                            final_chain_to_save = cluster_chain_of_entities
                            flag_answered = True
                            break

                        # 为下一轮的起点实体设置上一跳关系，避免走回去
                        for entity_info in topic_entities:
                            entity_info['pre_relations'] = previous_relations.copy()

            # 实体剪枝失败时，根据已有信息回答
            else:
                logger.info("Path pruned to empty (Entity Pruning). Fallback to Refinement.")
                call_num += 1
                final_json_to_save = generate_final_answer(question, cluster_chain_of_entities, args, all_tokens, best_answer_so_far)
                final_chain_to_save = cluster_chain_of_entities
                flag_answered = True
                break
        
        is_negative_final = False
        current_answer_text = ""

        # 已经回答后，提取答案
        if final_json_to_save:
            try:
                if isinstance(final_json_to_save, str):
                    parsed = json.loads(final_json_to_save)
                    final_answer = parsed.get("answer", parsed.get("A", {}).get("Answer", ""))
                    current_answer_text = str(final_answer).lower()
                else:
                    current_answer_text = str(final_json_to_save).lower()
            except:
                current_answer_text = str(final_json_to_save).lower()
        else:
            is_negative_final = True

        # 判断答案是否有效，包含这些词可以视为无效
        negative_keywords = ["did not", "does not", "has not", "no record", "unknown", "unclear", "no information", "null", "none", "not directed", "no known movie", "insufficient"]
        if any(neg in current_answer_text for neg in negative_keywords):
            is_negative_final = True

        # LLM认为已经生成答案，但是是无效答案，那么重新生成一次
        if not flag_answered or is_negative_final:
            if is_negative_final:
                logger.info("[Safety] Answer is Negative ('%s...'). Force Refinement.", current_answer_text[:30])
            else:
                logger.info("[Safety] Loop finished without answer. Force Refinement.")

            call_num += 1
            final_json_to_save = generate_final_answer(question, cluster_chain_of_entities, args, all_tokens, best_answer_so_far)
            final_chain_to_save = cluster_chain_of_entities

        if final_json_to_save:
            save_2_jsonl(question, question_string, final_json_to_save, final_chain_to_save, call_num, all_tokens, start_time, output_file=args.output_file)
        else:
            logger.info("[Error] No result generated for: %s", question)

    except Exception:
        logger.exception("Error on question=%s", question)

    return local_stats


def disable_net_proxy():
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy", "NO_PROXY"]:
        os.environ.pop(var, None)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--dataset_sample_num", type=int, default=500, help="Number of examples; omit to run the full dataset.")
    parser.add_argument("--dataset_sample_seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen3.5-27b-AWQ-4bit", choices=MODELS_DICT.keys())
    parser.add_argument("--max_length", type=int, default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float, default=0.3, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float, default=0.3, help="the temperature in reasoning stage.")
    parser.add_argument("--embedding_model", type=str, default="/home/flyvideo/Sata2/Models/Embedding-Model/msmarco-distilbert-base-tas-b")
    parser.add_argument("--depth", type=int, default=4, help="choose the search depth of CoG.")
    parser.add_argument("--skeleton_similarity_threshold", type=float, default=0.92, help="Minimum top-1 similarity for directly copying a retrieved skeleton.")
    parser.add_argument("--remove_unnecessary_rel", type=bool, default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--relation_question_weight", type=float, default=0.60, help="Weight for question-to-relation similarity.")
    parser.add_argument("--relation_prediction_weight", type=float, default=0.25, help="Weight for predicted-relation-to-relation similarity.")
    parser.add_argument("--relation_skeleton_weight", type=float, default=0.15, help="Weight for global-skeleton-to-relation similarity.")
    parser.add_argument("--output_dir", type=str, default="../output")
    args = parser.parse_args()

    relation_weight_sum = args.relation_question_weight + args.relation_prediction_weight + args.relation_skeleton_weight
    if abs(relation_weight_sum - 1.0) > 1e-6:
        parser.error("--relation_question_weight, --relation_prediction_weight, and --relation_skeleton_weight must sum to 1.")

    # 配置模型
    disable_net_proxy()
    model_config = MODELS_DICT[args.model_name]
    args.LLM_type = model_config["model_path"]
    args.base_url = model_config["base_url"]
    args.api_key = model_config["api_key"]

    # 配置实验结果和日志保存路径
    sample_label = "all" if args.dataset_sample_num is None else args.dataset_sample_num
    args.output_file = str(Path(args.output_dir) / f"CoG_data_{args.dataset}_sampleNum_{sample_label}_sampleSeed_{args.dataset_sample_seed}_modelName_{args.model_name.lower()}.jsonl")
    logger = initialize_logger(args.output_dir, args.dataset, args.model_name, args.dataset_sample_num, args.dataset_sample_seed)
    logger.info("\n===== Start running CoG | dataset=%s | model=%s =====", args.dataset, args.model_name)

    # 配置数据集，必要时采样
    datas, question_string = prepare_dataset(args.dataset)
    total_dataset_size = len(datas)
    if args.dataset_sample_num is not None:
        random.seed(args.dataset_sample_seed)
        sampled_indices = sorted(random.Random(args.dataset_sample_seed).sample(range(total_dataset_size), args.dataset_sample_num))
        datas = [datas[index] for index in sampled_indices]
        logger.info("Dataset sampling enabled: selected %d/%d samples with seed %d.", len(datas), total_dataset_size, args.dataset_sample_seed)
    datas = repeat_unanswer(datas, question_string, args.output_file)

    # 配置嵌入模型
    model_path = args.embedding_model
    if os.path.isdir(model_path):
        logger.info("Loading local embedding model from %s", model_path)
        model = SentenceTransformer(model_path)
    else:
        logger.warning("Embedding model path not found: %s, loading it from Hugging Face Hub", model_path)
        model = SentenceTransformer('msmarco-distilbert-base-tas-b')

    # 配置关系蓝图数据库
    index_path = str(Path("../index") / f"{args.dataset}_index.pkl")
    if os.path.exists(index_path):
        logger.info("Loading skeleton index for %s from %s\n", args.dataset, index_path)
        retriever = SkeletonRetriever(index_path, model_name=model_path)
    else:
        logger.warning("Skeleton index not found at %s, running without skeleton retrieval\n", index_path)
        retriever = None

    global_stats = Counter()
    for data in tqdm(datas, total=len(datas), desc="Processing"):
        result = process_question(data, args, question_string, model, retriever)
        if result:
            global_stats += result

    logger.info("Results summary: %s", dict(global_stats))
    logger.info("CoG finished. Result file: %s", args.output_file)
