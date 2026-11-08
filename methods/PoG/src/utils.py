import os
import re
import json
import time
import openai
from logging_utils import logger
from sentence_transformers import util

from prompt_list import *

def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores


# 调用LLM，计算调用的token开销
def run_llm(prompt, temperature, max_tokens, base_url, api_key, engine="qwen3.5-27b"):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
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

    completion = client.chat.completions.create(
            model=engine,
            messages = messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            extra_body=extra_body
            )
    result = completion.choices[0].message.content
    token_num = {
        "total": completion.usage.total_tokens,
        "input": completion.usage.prompt_tokens,
        "output": completion.usage.completion_tokens
    }
    return result, token_num



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

    
# 保存单样本结果
def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, output_file):
    tt = time.time()-start_time
    result = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "call_num": call_num, "total_token": all_t['total'], "input_token": all_t['input'], "output_token": all_t['output'], "time": tt}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as outfile:
        json_str = json.dumps(result, ensure_ascii=False)
        outfile.write(json_str + "\n")


def save_error_result(question, question_string, output_file):
    """Save a failed sample in the same schema as a normal PoG result."""
    error_result = {
        question_string: question,
        "results": "error",
        "reasoning_chains": None,
        "call_num": 0,
        "total_token": 0,
        "input_token": 0,
        "output_token": 0,
        "time": 0,
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(error_result, ensure_ascii=False) + "\n")


def load_previous_results(output_file, question_string):
    """Load the last saved record for each question from an existing jsonl file."""
    previous_results = {}
    if not os.path.exists(output_file):
        return previous_results

    with open(output_file, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = result.get(question_string)
            if question is not None:
                previous_results[question] = result
    return previous_results


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


# 解析LLM直接回答的JSON结果
def extract_direct_answer(response):
    data = parse_json_object(response)
    known = data.get("known")
    reason = data.get("reason")

    if type(known) is not bool:
        raise ValueError("known must be a boolean")

    if not isinstance(reason, str):
        raise ValueError("reason must be a string")

    if "answer" not in data:
        raise ValueError("Missing answer field")

    answer = data["answer"]
    if not (
        answer is None
        or isinstance(answer, str)
        or (isinstance(answer, list) and all(isinstance(item, str) for item in answer))
    ):
        raise ValueError("answer must be a string, list of strings, or null")

    logger.info("Known: %s", known)
    logger.info("Answer: %s", answer)
    logger.info("Reason: %s", reason)
    return answer


# 没有KG检索路径时直接回答
def generate_without_explored_paths(question, subquestions, args):
    prompt = answer_directly_prompt.substitute(question=question)
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    answer = extract_direct_answer(response)
    return answer, token_num


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
    with open(q_mem_f_path / 'subq', 'w', encoding='utf-8') as f:
        f.write(str(sub_questions))

    return sub_questions, token_num


# 反思
def reflection(question, enititiy_ids, depth_entity_relation_entity_dict, entity_id_name, entity_name_id, memory_path, results, cluster_chain_of_entities, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}
    with open(memory_path / 'mem', 'r', encoding='utf-8') as f:
        history_memory = f.read()

    # 提取所有未终止的实体ID
    if all(entity_id == "[FINISH_ID]" for entity_id in enititiy_ids):
        new_enititiy_ids = []
    else:
        new_enititiy_ids = [entity_id for entity_id in enititiy_ids if entity_id != "[FINISH_ID]"]

    logger.info("\n[Reflection]\nCurrent entities: %s", ", ".join(entity_id_name.get(entity_id, entity_id) for entity_id in new_enititiy_ids))

    # 收集可用于反思回溯的历史实体候选
    all_entities_set = set()
    for depth, entity_relation_entity_dict in depth_entity_relation_entity_dict.items():
        for topic_entities, head_tail_dict in entity_relation_entity_dict.items():
            # 先把起点实体加入all_entities_set
            all_entities_set.add(topic_entities)
            for direction, relation_entities_dict in head_tail_dict.items():
                for relation, entities_list in relation_entities_dict.items():
                    # 再把检索到的候选实体加入，候选实体数量过多时，使用嵌入模型筛选
                    if len(entities_list) > args.max_embedding_candidates:
                        logger.info("Too many entities, using embedding model to retain %d candidates.", args.max_embedding_candidates)
                        sorted_e_list = [entity_id_name[entity_id] for entity_id in entities_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, args.max_embedding_candidates)
                        entities_list = [entity_name_id[entity] for entity in topn_entities]

                    # 将检索到的候选实体也加入all_entities_set
                    all_entities_set |= (set(entities_list))
    
    # 构造反思提示词
    chain_prompt = '\n'.join(', '.join(str(x) for x in chain) for sublist in cluster_chain_of_entities for chain in sublist)
    current_entity_names = sorted({entity_id_name[entity_id] for entity_id in new_enititiy_ids})

    prompt = judge_reflection_prompt.substitute(
        question=question,
        entities=str(current_entity_names),
        memory=history_memory,
        knowledge_triplets=chain_prompt
    )
    cur_call_time += 1
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    for token_type in token_num:
        cur_token[token_type] += token_num[token_type]

    # 解析是否需要补充实体
    flag, reason = extract_add_and_reason(response)

    if flag:
        # 排除下一轮已经保留的实体，得到可回溯的历史候选实体
        other_entities = sorted(all_entities_set - set(new_enititiy_ids))

        if other_entities:
            # 构造编号候选，编号直接对应实体ID
            numbered_entities = " ".join(f"[{i}] {entity_id_name[entity_id]}" for i, entity_id in enumerate(other_entities, start=1))
            logger.info("\n[Backtracking]\nCandidate entities: %s", numbered_entities)

            prompt = backtrack_entity_prompt.substitute(
                question=question,
                reason=reason,
                entity_candidates=numbered_entities,
                memory=history_memory
            )
            cur_call_time += 1
            response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
            for token_type in token_num:
                cur_token[token_type] += token_num[token_type]

            # 解析选择的编号
            selected_ids = parse_selected_ids(response, len(other_entities))

            # 根据编号直接恢复历史实体ID
            add_ent_list = sorted(other_entities[i - 1] for i in selected_ids)
            if add_ent_list:
                logger.info("Selected history entities: %s", ", ".join(entity_id_name[entity_id] for entity_id in add_ent_list))
                return (new_enititiy_ids, add_ent_list, cur_call_time, cur_token)

    return new_enititiy_ids, [], cur_call_time, cur_token



def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('/home/flyvideo/Sata2/lyy/kgqa/data/test/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('/home/flyvideo/Sata2/lyy/kgqa/data/test/webqsp.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('/home/flyvideo/Sata2/lyy/kgqa/data/test/graliqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        logger.info("Dataset not found; choose from {cwq, webqsp, grailqa}.")
    return datas, question_string
