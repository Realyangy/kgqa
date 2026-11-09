import os
import re
import json
import time
import random
import pickle
import numpy as np
from pathlib import Path
from openai import OpenAI
from sentence_transformers import util, SentenceTransformer

from prompt_list import *
from logging_utils import logger


def retrieve_top_docs(query, docs, model, width=3):
    """用于 CoG 的候选实体粗筛。"""
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores


# 调用LLM
def run_llm(prompt, temperature, max_tokens, base_url, api_key, engine="gpt-3.5-turbo"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    extra_body = None
    if "qwen" in engine.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif "deepseek" in engine.lower():
        extra_body = {"thinking": {"type": "disabled"}}

    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                extra_body=extra_body
            )
            result = completion.choices[0].message.content or ""
            usage = completion.usage
            token_num = {
                "total": usage.total_tokens if usage else 0,
                "input": usage.prompt_tokens if usage else 0,
                "output": usage.completion_tokens if usage else 0
            }
            return result, token_num
        except Exception as e:
            logger.info("OpenAI error, retry: %s", e)
            time.sleep(2)

    raise RuntimeError("Failed to call LLM after 3 retries")

def convert_dict_name(ent_rel_ent_dict, entid_name):
    name_dict = {}
    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        if entid_name[topic_e] not in name_dict.keys():
            name_dict[entid_name[topic_e]] = {}

        for h_t, r_e_dict in h_t_dict.items():
            if h_t not in name_dict[entid_name[topic_e]].keys():
                name_dict[entid_name[topic_e]][h_t] = {}
            
            for rela, e_list in r_e_dict.items():
                if rela not in name_dict[entid_name[topic_e]][h_t].keys():
                    name_dict[entid_name[topic_e]][h_t][rela] = []
                for ent in e_list:
                    if entid_name[ent] not in name_dict[entid_name[topic_e]][h_t][rela]:
                        name_dict[entid_name[topic_e]][h_t][rela].append(entid_name[ent])
    return name_dict

    
def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, output_file):
    tt = time.time()-start_time
    dict = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "call_num": call_num, "total_token": all_t['total'], "input_token": all_t['input'], "output_token": all_t['output'], "time": tt}
    
    with open(output_file, "a", encoding='utf-8') as outfile:
        json_str = json.dumps(dict, ensure_ascii=False)
        outfile.write(json_str + "\n")


# 解析LLM输出的JSON对象
def parse_json_object(response):
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string")

    text = response.strip()

    # 兼容Markdown代码块
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON response") from error

    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")

    return data


# 解析LLM选择的候选编号
def parse_selected_ids(response, candidate_count):
    data = parse_json_object(response)
    selected_ids = data.get("selected_ids")
    if not isinstance(selected_ids, list):
        raise ValueError("selected_ids must be a list")

    selected_set = set()
    for candidate_id in selected_ids:
        if type(candidate_id) is not int:
            raise ValueError(f"Invalid candidate ID: {candidate_id}")
        if not 1 <= candidate_id <= candidate_count:
            raise ValueError(f"Candidate ID out of range: {candidate_id}")
        if candidate_id in selected_set:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")
        selected_set.add(candidate_id)

    return selected_ids


# 解析充分性判断结果
def extract_reason_and_anwer(response):
    data = parse_json_object(response)

    # 验证充分性判断
    sufficient = data.get("sufficient")
    if type(sufficient) is not bool:
        raise ValueError("sufficient must be a boolean")

    # 验证原因
    reason = data.get("reason")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    # 验证答案
    if "answer" not in data:
        raise ValueError("Missing answer field")

    answer = data["answer"]
    if not (
        answer is None
        or isinstance(answer, str)
        or (isinstance(answer, list) and all(isinstance(item, str) for item in answer))
    ):
        raise ValueError("answer must be a string, list of strings, or null")

    logger.info("Sufficient to answer: %s", sufficient)
    logger.info("Answer: %s", answer)
    logger.info("Reason: %s", reason)

    return answer, reason, sufficient


# 解析反思判断结果
def extract_add_and_reason(response):
    data = parse_json_object(response)
    add = data.get("add")
    reason = data.get("reason")

    if type(add) is not bool:
        raise ValueError("add must be a boolean")

    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    logger.info("Whether to backtrack: %s", add)
    logger.info("Reason: %s", reason)

    return add, reason


def generate_without_explored_paths(question, subquestions, args):
    prompt = cot_prompt + question 
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    return response, token_num



def break_question(question, args):
    prompt = subobjective_prompt.substitute(question=question)
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    data = parse_json_object(response)
    sub_questions = data.get("subobjectives")
    if not isinstance(sub_questions, list):
        raise ValueError("subobjectives must be a list")
    if not all(isinstance(item, str) for item in sub_questions):
        raise ValueError("Each subobjective must be a string")
    return sub_questions, token_num


def get_subquestions(q_mem_f_path, question, args):
    sub_questions, token_num = break_question(question, args)
    with open(q_mem_f_path+'/'+'subq', 'w', encoding='utf-8') as f:
        f.write(str(sub_questions))
    return sub_questions, token_num


def reflection(question, selected_entity_ids, depth_entities_relations_entities_dict, entity_id_to_name, entity_name_to_id, memory_path, results, cluster_chain_of_entities, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    with open(memory_path+'/mem', 'r', encoding='utf-8') as f:
        history_memory = f.read()

    if all(elem == "[FINISH_ID]" for elem in selected_entity_ids):
        remaining_entity_ids = []
    else:
        remaining_entity_ids = [elem for elem in selected_entity_ids if elem != "[FINISH_ID]"]
    

    all_entities_set = set()
    for depth, entity_relation_entity_dict in depth_entities_relations_entities_dict.items():
        for topic_entity, head_tail_dict in entity_relation_entity_dict.items():
            all_entities_set.add(topic_entity)
            for direction, relation_entities_dict in head_tail_dict.items():
                for relation, entities_list in relation_entities_dict.items():

                    if all(entity_id_to_name[item].startswith('m.') or entity_id_to_name[item].startswith('g.') for item in entities_list) and len(entities_list) > 10:
                        entities_list = random.sample(entities_list, 10)
                        

                    if len(entities_list) > 70:
                        #print('··········exceed 70 entities··········')
                        sorted_e_list = [entity_id_to_name[entity_id] for entity_id in entities_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        # print('sentence:', topn_entities)
                        entities_list = [entity_name_to_id[e_n] for e_n in topn_entities]
                        all_entities_set |= (set(entities_list))
                    else:
                        all_entities_set |= (set(entities_list))  
    
    clean_candidates_set = set()
    id_count = 0  
    
    for entity_id in all_entities_set:
        name = entity_id_to_name.get(entity_id, entity_id)
        is_id = str(name).startswith('m.') or str(name).startswith('g.') or str(name).isdigit()
        
        if not is_id:

            clean_candidates_set.add(entity_id)
        else:

            if id_count < 5:
                clean_candidates_set.add(entity_id)
                id_count += 1


    if not clean_candidates_set and all_entities_set:
        clean_candidates_set = set(list(all_entities_set)[:5])


    all_entities_set = clean_candidates_set


    # 构造反思提示词
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    current_entity_names = sorted({entity_id_to_name[entity_id] for entity_id in remaining_entity_ids})
    prompt = judge_reflection_prompt.substitute(question=question, entities=str(current_entity_names), memory=history_memory, knowledge_triplets=chain_prompt)

    cur_call_time += 1
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    for token_type in token_num:
        cur_token[token_type] += token_num[token_type]

    flag, reason = extract_add_and_reason(response)
    if flag:
        # 保留CoG原有的候选实体过滤策略，仅修改编号选择与ID映射
        other_entities = sorted(all_entities_set - set(remaining_entity_ids))
        if other_entities:
            numbered_entities = " ".join(f"[{i}] {entity_id_to_name[entity_id]}" for i, entity_id in enumerate(other_entities, start=1))
            logger.info("\n[Backtracking]\nCandidate entities: %s", numbered_entities)
            prompt = backtrack_entity_prompt.substitute(question=question, reason=reason, entity_candidates=numbered_entities, memory=history_memory)

            cur_call_time += 1
            response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
            for token_type in token_num:
                cur_token[token_type] += token_num[token_type]

            selected_ids = parse_selected_ids(response, len(other_entities))
            reflection_entity_list = sorted(other_entities[i - 1] for i in selected_ids)
            if reflection_entity_list:
                logger.info("add reverse ent: %d %s", len(reflection_entity_list), [entity_id_to_name[entity_id] for entity_id in reflection_entity_list])
                return remaining_entity_ids, reflection_entity_list, cur_call_time, cur_token

    return remaining_entity_ids, [], cur_call_time, cur_token


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open("../../../data/test/cwq.json", encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open("../../../data/test/webqsp.json", encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open("../../../data/test/graliqa.json", encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        logger.info("dataset not found, you should pick from {cwq, webqsp, grailqa}.")
        exit(-1)
    return datas, question_string


# 关系蓝图检索器
class SkeletonRetriever:
    def __init__(self, index_path, model_name):
        try:
            # 读取关系蓝图数据库
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
            self.data = data['data']
            self.corpus_emb = data['embeddings']

            # 加载嵌入模型
            if os.path.exists(model_name):
                self.model = SentenceTransformer(model_name)
            else:
                self.model = SentenceTransformer('msmarco-distilbert-base-tas-b')
            self.is_loaded = True

        except Exception as e:
            logger.info("Warning: Failed to load index (%s). Speculation will be disabled.", e)
            self.is_loaded = False


    def retrieve(self, query, topk=2):
        if not self.is_loaded:
            return [], []

        # 计算问题嵌入，并与数据库中的嵌入计算相似度，返回TopK
        query_emb = self.model.encode(query)
        scores = util.cos_sim(query_emb, self.corpus_emb)[0]
        scores_np = scores.cpu().numpy()
        top_k_indices = np.argsort(-scores_np)[:topk]
        return [self.data[i] for i in top_k_indices], scores_np[top_k_indices]


# 根据检索结果生成当前问题的关系蓝图
def speculate_skeleton_chain(question, topic_entity_names, retriever, args, model): 
    token_num = {'total': 0, 'input': 0, 'output': 0, 'calls': 0}
    if not retriever or not getattr(retriever, 'is_loaded', False):
        return [], token_num, None, "FAILED"
    
    try:
        masked_question = mask_entities(question, topic_entity_names)
        logger.info("Masked Question: %s", masked_question)

        # 检索结果
        exemplars, scores = retriever.retrieve(masked_question, topk=5) 

        # 相似度超过阈值时，直接使用
        if len(scores) > 0 and scores[0] > args.skeleton_similarity_threshold:
            logger.info("High similarity (%.4f) detected!\nQuestion: %s\nSkeleton: %s", scores[0], exemplars[0]['question'], exemplars[0]['skeleton'])
            skeleton = [s.strip() for s in exemplars[0]['skeleton'].split('->')]
            return skeleton, token_num, exemplars[0]['skeleton'], "DIRECT_COPY"

        if len(scores) > 0:
            logger.info("--- Retrieved Exemplars (Top-1 Score: %.4f) ---", scores[0])
            for i, ex in enumerate(exemplars):
                logger.info("[%d]\nQuestion: %s\nSkeleton: %s", i, ex['question'], ex['skeleton'])
            logger.info("---------------------------")
        
        if len(exemplars) < 2:
            return [], token_num, None, "FAILED"

        # 否则取TopK=2个，让LLM参考生成
        prompt = speculate_prompt.format(
            q1=exemplars[0]['question'], s1=exemplars[0]['skeleton'],
            q2=exemplars[1]['question'], s2=exemplars[1]['skeleton'],
            question=question
        )
        response, token = run_llm(prompt, temperature=0.0, max_tokens=150, base_url=args.base_url, api_key=args.api_key, engine=args.LLM_type)
        for token_type in token:
            if token_type != 'calls':
                token_num[token_type] += token.get(token_type, 0)
        token_num['calls'] += token.get('calls', 0)

        # 解析模型输出
        clean_resp = response.strip().split('\n')[0]
        if "Output:" in clean_resp:
            clean_resp = clean_resp.split("Output:")[-1].strip()

        skeleton = [s.strip() for s in clean_resp.split('->')]
        skeleton = [s for s in skeleton if '[' not in s and len(s) > 1]
        if len(skeleton) > 0:
            logger.info("Speculation Generated: %s", skeleton)
            logger.info("---------------------------")
            return skeleton, token_num, clean_resp, "GENERATED"
            
    except Exception as e:
        logger.info("[Speculation Warning] Failed: %s", e)

    return [], token_num, None, "FAILED"


# 将问题中的起点实体信息隐去，用mask代替
def mask_entities(question, entity_names, mask="[ENT]"):
    masked_q = question
    sorted_names = sorted(entity_names, key=len, reverse=True)
    for name in sorted_names:
        if name and name in masked_q:
            masked_q = re.sub(r'\b' + re.escape(name) + r'\b', mask, masked_q, flags=re.IGNORECASE)
    return masked_q
