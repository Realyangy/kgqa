import re
import ast
import os
import json
import time
from openai import OpenAI
from prompt_list import *
from logging_utils import logger

# 判断得分是否都为零
def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


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
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                extra_body=extra_body
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.info("OpenAI error, retry: %s", e)
            time.sleep(2)
    raise RuntimeError("Failed to call LLM after 3 retries")


# 判断检索得到的实体是否都为UnName_Entity
def all_unknown_entity(entity_pairs):
    return all(name == "UnName_Entity" for name, _ in entity_pairs)


# 删除候选实体集合中的UnName_Entity
def del_unknown_entity(entity_pairs):
    # 若只有一个UnName_Entity，那么保留
    if len(entity_pairs) == 1 and entity_pairs[0][0] == "UnName_Entity":
        return entity_pairs

    # 否则删除所有的UnName_Entity及其对应ID
    return [(name, eid) for name, eid in entity_pairs if name != "UnName_Entity"]


# 解析LLM输出的json
def parse_scored_candidates(response, candidate_count, require_all=False):
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string")

    text = response.strip()

    # 兼容模型偶尔输出的Markdown代码块
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()

    # 解析JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON response") from error

    if not isinstance(data, dict) or not isinstance(data.get("scores"), list):
        raise ValueError("JSON must contain a 'scores' list")

    selected_scores = {}
    for item in data["scores"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid score item")

        candidate_id = item.get("id")
        score = item.get("score")
        if type(candidate_id) is not int:
            raise ValueError(f"Invalid candidate ID: {candidate_id}")
        if not 1 <= candidate_id <= candidate_count:
            raise ValueError(f"Candidate ID out of range: {candidate_id}")
        
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score: {score}")
        if candidate_id in selected_scores:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")

        selected_scores[candidate_id] = float(score)

    # 实体评分需要覆盖全部候选
    if require_all and len(selected_scores) != candidate_count:
        raise ValueError(f"Expected {candidate_count} scores, got {len(selected_scores)}")

    return selected_scores


# 解析LLM返回的关系剪枝结果
def clean_relations(response, entity_id, entity_name, relation_candidates):
    selected_scores = parse_scored_candidates(response, len(relation_candidates), require_all=False)

    relations = []
    for candidate_id, score in selected_scores.items():
        if score == 0:
            continue

        # 根据编号取回真实关系及方向
        candidate = relation_candidates[candidate_id - 1]
        relations.append({
            "entity_id": entity_id,
            "entity_name": entity_name,
            "relation": candidate["relation"],
            "score": score,
            "head": candidate["head"]
        })
    return bool(relations), relations


# LLM实体选择结果无法解析时，保留原始回复供错误样本记录
class EntitySelectionParseError(ValueError):
    def __init__(self, raw_output, candidates):
        super().__init__("Could not parse entity scoring JSON")
        self.raw_output = raw_output
        self.candidates = candidates


# 解析LLM返回的实体名称列表，并仅保留候选实体中的名称
def parse_selected_entities(response, entity_candidates):
    try:
        return parse_scored_candidates(response, len(entity_candidates), require_all=True)
    except ValueError as error:
        raise EntitySelectionParseError(response, entity_candidates) from error


# 保存样本结果，保存的内容包括问题、LLM的回答、检索到的三元组
def save_2_jsonl(question, answer, cluster_chain_of_entities, output_file):
    logger.info("Answer: %s\n", answer)

    result = {
        "question": question,
        "results": answer,
        "reasoning_chains": cluster_chain_of_entities
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")


# 提取LLM回答的答案
def extract_answer(text):
    match = re.search(r"Final Answer:\s*\{([^{}]+)\}\.?", text)
    if match:
        return match.group(1).strip()
    return ""
    

# 提取LLM在reasoning中的判断结果，yes表示三元组足以回答问题
def if_true(prompt):
    return bool(re.match(r"^(?:A:\s*)?\{Yes\}", prompt.strip(), re.IGNORECASE))


# 未检索到相关信息时，直接调用LLM回答问题
def generate_without_explored_paths(question, args):
    prompt = cot_prompt.substitute(question=question)
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    logger.info("LLM Response [Direct Answer]:\n%s", response)
    result = extract_answer(response)
    return result if result else response


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
        with open('../../../../data/test/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../../../../data/test/webqsp.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../../../../data/test/graliqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        logger.info("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string


# 保存运行过程中出错的样本
# def save_error_result(output_file, question):
#     error_result = {
#         "question": question,
#         "results": "error",
#         "reasoning_chains": None
#     }

#     with open(output_file, "a", encoding="utf-8") as f:
#         f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

def save_error_result(output_file, question, reasoning_chains=None):
    error_result = {
        "question": question,
        "results": "error",
        "reasoning_chains": reasoning_chains,
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")


# 加载已经存在的实验结果文件，读取每个样本的最新实验结果
def load_previous_results(output_file):
    previous_results = {}

    if not os.path.exists(output_file):
        return previous_results

    # 读取实验结果文件
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 这里，新处理结果会覆盖旧结果
            question = result.get("question")
            if question:
                previous_results[question] = result

    return previous_results
