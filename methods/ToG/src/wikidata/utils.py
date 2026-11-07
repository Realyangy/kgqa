import re
import os
import json
import time
import logging
from openai import OpenAI
from prompt_list import *

logger = logging.getLogger("ToG")

# 判断得分是否都为零
def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


# 调用LLM
def run_llm(prompt, temperature, max_tokens, base_url, api_key, engine="gpt-3.5-turbo"):
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.info("OpenAI error, retry: %s", e)
            time.sleep(2)
    raise RuntimeError("Failed to call LLM after 3 retries")


# 判断检索得到的实体是否都为UnName_Entity
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


# 删除候选实体集合中的UnName_Entity
def del_unknown_entity(entity_candidates):
    # 若只有一个UnName_Entity，那么保留
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    # 否则删除所有的UnName_Entity
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


# 解析LLM返回的关系剪枝结果
def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue

        score = match.group("score")
        if not relation or not score:
            return False, "LLM relation prune output uncompleted"
        try:
            score = float(score)
        except ValueError:
            return False, "LLM relation prune output invalid score"
        
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    
    if not relations: 
        return False, "LLM relation prune failed, no relations found"
    return True, relations


# 解析LLM返回的候选实体打分结果
def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        logger.info("LLM entity evaluate failed, all entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)


# 保存样本结果，保存的内容包括问题、LLM的回答、检索到的三元组
def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, output_dir):
    logger.info("Answer: %s", answer)
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ToG_{}.jsonl".format(file_name)), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

# 初始化 INFO 级别日志，同时输出到终端和单个日志文件
def initialize_logger(output_dir, dataset):
    os.makedirs(output_dir, exist_ok=True)
    run_log_file = os.path.join(output_dir, f"ToG_{dataset}_run.log")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(run_log_file, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
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
    logger.info("SEARCH_TRACE %s", json.dumps(record, ensure_ascii=False, default=str))


# 提取LLM回答的答案
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

# 提取LLM在reasoning中的判断结果，yes表示三元组足以回答问题
def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


# 未检索到相关信息时，直接调用LLM回答问题
def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    return response


# 如果当前关系下未检索到实体，则关系下会记录[FINISH_ID]（具体可见freebase_func.update_history），所有分支都结束时，终止搜索
def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


# 加载数据集
def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('/home/liyuyang/lyy/ToG/data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('/home/liyuyang/lyy/ToG/data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        logger.info("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string
