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

def extract_add_ent(string):
    first_brace_p = string.find('[')
    last_brace_p = string.rfind(']')
    string = string[first_brace_p:last_brace_p+1]
    try:
        new_string = eval(string)
    except:
        s_list = string.split('\', \'')
        if len(s_list) == 1:
            new_string = [s_list[0].strip('[\'').strip('\']')]
        else:
            new_string = [s.strip('[\'').strip('\']') for s in s_list]
    return new_string

def extract_memory(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    return string

def extract_reason_and_anwer(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    answer = re.search(r'"Answer":\s*"(.*?)"', string)
    if answer:
        answer = answer.group(1)
    else:
        answer = re.search(r'"Answer":\s*(\[[^\]]+\])', string).group(1)

    reason = re.search(r'"R":\s*"(.*?)"', string).group(1)
    sufficient = re.search(r'"Sufficient":\s*"(.*?)"', string).group(1)
    logger.info("Sufficient to answer: %s", sufficient)
    if sufficient.lower() == "yes":
        logger.info("Answer: %s", answer)
    logger.info("Reason: %s", reason)
    return answer, reason, sufficient

def extract_add_and_reason(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    flag = re.search(r'"Add":\s*"(.*?)"', string).group(1)
    reason = re.search(r'"Reason":\s*"(.*?)"', string).group(1)

    logger.info("Whether to backtrack: %s", flag)
    logger.info("Reason: %s", reason)
    if 'yes' in flag.lower():
        return True, reason
    else:
        return False, reason
    


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
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt = judge_reverse + question + '\nEntities set to be retrieved: ' + str(list(set(sorted([entity_id_name[ent_i] for ent_i in new_enititiy_ids])))) +'\nMemory: '+ history_memory + '\nKnowledge Triplets:' + chain_prompt
    cur_call_time += 1
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    for token_type in token_num.keys():
        cur_token[token_type] += token_num[token_type]
    flag, reason = extract_add_and_reason(response)

    if flag:
        # new_enititiy_ids是本轮实体剪枝后仍然保留，将会作为下一轮迭代的起点实体的实体，这里把它们去掉，剩下的就是LLM之前没有选择，现在可以重新选择的候选实体
        other_entities = sorted(list(all_entities_set - set(new_enititiy_ids)))
        other_entities_name = [entity_id_name[entity_id] for entity_id in other_entities]

        # 需要回溯时，调用LLM判断选择哪个实体作为回溯起点
        logger.info("\n[Backtracking]\nCandidate entities: %s", ", ".join(other_entities_name))
        prompt = add_ent_prompt+question+'\nReason: '+reason+'\nCandidate Entities: ' + str(sorted(other_entities_name))+'\nMemory: '+ history_memory
        cur_call_time += 1
        response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
        for token_type in token_num.keys():
            cur_token[token_type] += token_num[token_type]

        # 提取LLM选择的回溯实体，根据候选列表检查，避免幻觉输出
        add_ent_list = extract_add_ent(response)
        add_ent_list = [entity_name_id[ent_i] for ent_i in add_ent_list if ent_i in other_entities_name]
        add_ent_list = sorted(add_ent_list)
        if add_ent_list:
            logger.info("Selected history entities: %s", ", ".join(entity_id_name[ent_i] for ent_i in add_ent_list))
            return new_enititiy_ids, add_ent_list, cur_call_time, cur_token
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
