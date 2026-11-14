import re
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
def run_llm(prompt, temperature, max_tokens, base_url, api_key, engine="gpt-3.5-turbo", stats=None):
    # 每次调用run_llm计为一次算法层面的LLM调用，内部重试不重复计数
    if stats is not None:
        stats["call_num"] += 1
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information. Always respond in valid JSON format."
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
                response_format={"type": "json_object"},
                extra_body=extra_body
            )
            # 累计已经返回的Token，包括后续被判定为截断的输出
            if stats is not None and response.usage is not None:
                stats["input_token"] += response.usage.prompt_tokens or 0
                stats["output_token"] += response.usage.completion_tokens or 0
                stats["total_token"] += response.usage.total_tokens or 0

            content = response.choices[0].message.content

            # 检查输出是否为空或者被截断
            if not content or not content.strip():
                raise ValueError("LLM returned empty JSON")
            if response.choices[0].finish_reason == "length":
                raise ValueError("LLM JSON output was truncated")
            return content
        
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
        super().__init__("Could not parse entity selection JSON")
        self.raw_output = raw_output
        self.candidates = candidates


# 解析LLM返回的实体选择结果
def parse_selected_entities(response, entity_candidates):
    try:
        text = response.strip()
        match = re.fullmatch( r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()

        # 解析JSON
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Invalid JSON object")

        selected_ids = data.get("selected_ids")
        if not isinstance(selected_ids, list):
            raise ValueError("selected_ids must be a list")

        # 验证实体编号
        candidate_count = len(entity_candidates)
        selected_set = set()

        for candidate_id in selected_ids:
            if type(candidate_id) is not int:
                raise ValueError(f"Invalid entity ID: {candidate_id}")

            if not 1 <= candidate_id <= candidate_count:
                raise ValueError(f"Entity ID out of range: {candidate_id}")

            if candidate_id in selected_set:
                raise ValueError(f"Duplicate entity ID: {candidate_id}")

            selected_set.add(candidate_id)

        return selected_set

    except (ValueError, TypeError) as error:
        raise EntitySelectionParseError(response, entity_candidates) from error


# 整理单题的计算开销
def get_computational_cost(stats, start_time):
    stats = stats or {}
    return {
        "call_num": stats.get("call_num", 0),
        "total_token": stats.get("total_token", 0),
        "input_token": stats.get("input_token", 0),
        "output_token": stats.get("output_token", 0),
        "time": time.time() - start_time if start_time is not None else 0
    }


# 保存样本结果，保存的内容包括问题、LLM的回答、检索到的三元组及计算开销
def save_2_jsonl(question, answer, cluster_chain_of_entities, output_file, stats=None, start_time=None):
    logger.info("Answer: %s\n", answer)
    result = {"question": question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    result.update(get_computational_cost(stats, start_time))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")


# 解析回答类任务的JSON输出
def parse_answer_json(response, require_sufficient=False):
    if not isinstance(response, str):
        raise ValueError("LLM response is not a string")

    text = response.strip()

    # 兼容Markdown代码块
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()

    # 解析JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid answer JSON") from error

    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object")

    # 验证reason
    if not isinstance(data.get("reason"), str):
        raise ValueError("reason must be a string")

    # 验证answer
    if "answer" not in data:
        raise ValueError("Missing answer field")

    if data["answer"] is not None and not isinstance(data["answer"], str):
        raise ValueError("answer must be a string or null")

    # 充分性判断任务额外验证sufficient
    if require_sufficient:
        if type(data.get("sufficient")) is not bool:
            raise ValueError("sufficient must be a boolean")

    return data


# 未检索到相关信息时，直接调用LLM回答问题
def generate_without_explored_paths(question, args):
    prompt = answer_directly_prompt.substitute(question=question)
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type, stats=args._llm_stats)
    logger.info("LLM Response [Direct Answer]:\n%s", response)
    result = parse_answer_json(response)
    answer = result["answer"]
    return answer.strip() if answer is not None else None


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


def save_error_result(output_file, question, reasoning_chains=None, stats=None, start_time=None):
    error_result = {"question": question, "results": "error", "reasoning_chains": reasoning_chains}
    error_result.update(get_computational_cost(stats, start_time))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
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
