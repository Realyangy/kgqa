import ast
from utils import *
from logging_utils import logger
from SPARQLWrapper import SPARQLWrapper, JSON

# freebase 部署的端口路径
SPARQLPATH = "http://localhost:3001/sparql"

# 查询实体的出边关系
sparql_head_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
    ns:%s ?relation ?x .
}"""

# 查询实体的入边关系
sparql_tail_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
    ?x ?relation ns:%s .
}"""

# 查询出边关系连接的实体
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
    ns:%s ns:%s ?tailEntity .
}"""

# 查询入边关系连接的实体
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
    ?tailEntity ns:%s ns:%s .
}"""

# 根据实体ID查询其名称，用UNION联合了两种查询方式，第一种是使用type.object.name直接查询，第二种是通过owl:sameAs查询同一实体的其他标识
# (1) 单个实体查询
# sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>
# SELECT DISTINCT ?tailEntity
# WHERE {
#     {
#         ?entity ns:type.object.name ?tailEntity .
#         FILTER(?entity = ns:%s)
#     }
#     UNION
#     {
#         ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
#         FILTER(?entity = ns:%s)
#     }
# }"""

# (2) 多实体查询
sparql_ids = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity ?tailEntity
WHERE {
    VALUES ?entity { %s }

    {
        ?entity ns:type.object.name ?tailEntity .
    }
    UNION
    {
        ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    }
}"""


def abandon_rels(relation):
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
        or "http://" in relation
        or "https://" in relation
    ):
        return True


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]


def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


# 根据实体ID查询实体名称，有英语名称时返回英语的，没有英语名称时返回第一个，没有名称时将ID作为名称返回
# def id2entity_name_or_type(entity_id):
#     sparql_query = sparql_id % (entity_id, entity_id)
#     sparql = SPARQLWrapper(SPARQLPATH)
#     sparql.setQuery(sparql_query)
#     sparql.setReturnFormat(JSON)
#     results = sparql.query().convert()
#     if len(results["results"]["bindings"])==0:
#         return "UnName_Entity"
#     else:
#         for item in results["results"]["bindings"]:
#             entity = item["tailEntity"]
#             if entity.get("xml:lang") == "en":
#                 return entity["value"]
#         return results["results"]["bindings"][0]['tailEntity']['value']

# 批量查询版本
def batch_id2entity_name_or_type(entity_ids):
    entity_ids = list(dict.fromkeys(entity_ids))
    batch_size = 100
    entity_names = {entity_id: [] for entity_id in entity_ids}
    for start in range(0, len(entity_ids), batch_size):

        # 获取待查询batch
        batch_ids = entity_ids[start:start + batch_size]
        values = " ".join(f"ns:{entity_id}" for entity_id in batch_ids)

        # 批量查询
        sparql_query = sparql_ids % values
        results = execurte_sparql(sparql_query)
        for item in results:
            entity_id = item["entity"]["value"].replace("http://rdf.freebase.com/ns/", "")
            entity = item["tailEntity"]
            if entity_id in entity_names:
                entity_names[entity_id].append((entity["value"], entity.get("xml:lang")))

    # 优先提取英语名称
    resolved_names = {}
    for entity_id in entity_ids:
        candidates = entity_names[entity_id]
        english_name = next((value for value, language in candidates if language == "en"), None)
        if english_name is not None:
            resolved_names[entity_id] = english_name
        elif candidates:
            resolved_names[entity_id] = candidates[0][0]
        else:
            resolved_names[entity_id] = entity_id

    return resolved_names
    

import re
import json
import time
import openai
import random
from prompt_list import *
from freebase_func import *


# 解析LLM输出的关系名称列表，支持LLM输出的多种格式：Python/JSON列表、分号分隔、逗号分隔、单个关系
def parse_relation_names(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []
    
    text = raw_text.strip()
    bracket_lists = re.findall(r'\[[^\[\]]*\]', text, flags=re.S)
    if bracket_lists:
        candidate = bracket_lists[-1]

        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            text = candidate[1:-1]
        else:
            if not isinstance(parsed, (list, tuple)):
                return []
            return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]

    else:
        text = re.sub(r'^```(?:python|json|text)?\s*', '', text, flags=re.I)
        text = re.sub(r'\s*```$', '', text).strip()
    text = re.sub(r'^(?:the output is|selected relations?|relations?|output)\s*:\s*', '', text, flags=re.I)

    relation_names = []
    for part in re.split(r'[;,\n]+', text):
        relation = re.sub(r'^\s*(?:[-*]\s*|\d+[.)]\s*)', '', part)
        relation = relation.strip().strip('"\'`').strip()
        if relation:
            relation_names.append(relation)

    return relation_names


# 从LLM输出中解析选择的关系
def select_relations(response, entity_id, relation_candidates):
    selected_ids = parse_selected_ids(response, len(relation_candidates))

    relations = []
    for candidate_id in selected_ids:
        candidate = relation_candidates[candidate_id - 1]
        relations.append({
            "entity": entity_id,
            "relation": candidate["relation"],
            "head": candidate["head"]
        })

    if not relations:
        return False, "No relations found"

    return True, relations


# 构造关系剪枝提示词
def construct_relation_prune_prompt(question, sub_questions, entity_name, relation_candidates):
    numbered_relations = " ".join(
        f"[{i}] [{'outgoing' if item['head'] else 'incoming'}] {item['relation']}"
        for i, item in enumerate(relation_candidates, start=1)
    )

    return extract_relation_prompt.substitute(
        question=question,
        subobjectives=str(sub_questions),
        entity_name=entity_name,
        relations=numbered_relations
    )


# 关系搜索和剪枝
def relation_search_prune(entity_id, sub_questions, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))
    
    # 关系去重
    head_relations = sorted(set(head_relations))
    tail_relations = sorted(set(tail_relations))

    # 构造带方向的候选关系
    relation_candidates = (
        [{"relation": relation, "head": True} for relation in head_relations] +
        [{"relation": relation, "head": False} for relation in tail_relations]
    )

    # 保证候选编号顺序稳定
    relation_candidates.sort(key=lambda item: (item["relation"], not item["head"]))

    # 没有候选关系时直接返回
    if not relation_candidates:
        return [], {"total": 0, "input": 0, "output": 0}

    prompt = construct_relation_prune_prompt(question, sub_questions, entity_name, relation_candidates)
    result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.base_url, args.api_key, args.LLM_type)
    flag, retrieve_relations = select_relations(result, entity_id, relation_candidates)

    if flag:
        return retrieve_relations, token_num
    else:
        return [], token_num
    

# 实体检索
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)

    entity_ids = replace_entities_prefix(entities)
    return entity_ids


# 获取实体ID集合中的实体名称
# def get_entity_names(entity_candidates_id):
#     entity_candidates = []
#     for entity_id in entity_candidates_id:
#         if entity_id.startswith("m."):
#             entity_candidates.append(id2entity_name_or_type(entity_id))
#         else:
#             entity_candidates.append(entity_id)

#     if len(entity_candidates) <= 1:
#         return entity_candidates, entity_candidates_id

#     # 为了避免实体名称相同导致在字典中被替代，给同名实体名称后面加上ID标识
#     unique_names = []
#     name_count = Counter(entity_candidates)
#     for entity_name, entity_id in zip(entity_candidates, entity_candidates_id):
#         if name_count[entity_name] > 1:
#             unique_names.append(f"{entity_name}[{entity_id}]")
#         else:
#             unique_names.append(entity_name)

#     pairs = sorted(zip(unique_names, entity_candidates_id), key=lambda pair: pair[0])
#     entity_candidates = [entity_name for entity_name, _ in pairs]
#     entity_candidates_id = [entity_id for _, entity_id in pairs]
#     return entity_candidates, entity_candidates_id

# 批量处理版本
def get_entity_names(entity_candidates_id):
    # 批量获取m.实体的名称
    entities_ids = [
        entity_id
        for entity_id in entity_candidates_id
        if entity_id.startswith("m.")
    ]
    entities_id_name_dict = batch_id2entity_name_or_type(entities_ids)

    # 没有名称的实体使用原始ID
    entity_candidates = [
        entities_id_name_dict[entity_id]
        if entity_id.startswith("m.") else entity_id
        for entity_id in entity_candidates_id
    ]

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id

    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates = list(ent_id_dict.keys())
    entity_candidates_id = list(ent_id_dict.values())
    return entity_candidates, entity_candidates_id


def update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head):
    # 本轮没有扩展出实体时，在该样本的所有实体集合中加入[FINISH]，表示搜素结束
    if len(entity_candidates) == 0: 
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    # 为当前二元组下所有检索到的尾实体保留一个头实体、关系和关系方向
    candidates_relation = [ent_rel['relation']] * len(entity_candidates)
    topic_entities = [ent_rel['entity']] * len(entity_candidates)
    head_num = [ent_rel['head']] * len(entity_candidates)

    # 当前二元组检索到的所有信息，包括实体名称和ID、关系及其方向，可以理解为本轮扩展出的三元组
    total_candidates.extend(entity_candidates) 
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)

    return total_candidates, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, question_string, subquestions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args):
    logger.info("No new knowledge added during search depth %d, stop searching.", depth)
    call_num += 1
    answer, token_num = generate_answer(question, subquestions, cluster_chain_of_entities, args)

    for token_type in token_num.keys():
        all_t[token_type] += token_num[token_type]

    save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, output_file=args.output_file)


# 根据检索到的KG证据生成最终答案
def generate_answer(question, subquestions, cluster_chain_of_entities, args):
    # 构造检索到的KG路径
    chain_prompt = '\n'.join(', '.join(str(x) for x in chain) for sublist in cluster_chain_of_entities for chain in sublist)
    prompt = answer_prompt.substitute(question=question, knowledge_triplets=chain_prompt)
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    answer, reason, sufficient = extract_reason_and_anwer(response)
    return answer, token_num



def if_topic_non_retrieve(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_all_digits(lst):
    for s in lst:
        if not s.isdigit():
            return False
    return True


# 实体剪枝
def entity_condition_prune(question, entity_relation_entity_dict, entity_id_to_name, entity_name_to_id, args, embedding_model):
    call_count = 0
    token_usage = {'total': 0, 'input': 0, 'output': 0}

    pruned_entity_relation_entity_dict = {}
    non_prunable_relations = ['time', 'number', 'date']
    selected_entity_ids = []
    selected_topic_entity_names = []
    selected_relations = []
    selected_entity_names = []
    selected_heads = []
    selected_parent_entity_ids = []

    for topic_entity_id, head_tail_dict in sorted(entity_relation_entity_dict.items()):
        for direction, relation_entity_dict in sorted(head_tail_dict.items()):
            for relation, entity_id_list in sorted(relation_entity_dict.items()):
                # 不需要LLM剪枝的情况，保留全部实体
                if (is_all_digits(entity_id_list) or relation in non_prunable_relations or len(entity_id_list) <= 1):
                    selected_entity_ids_for_relation = sorted(entity_id_list)
                else:
                    if (all(entity_id_to_name[entity_id].startswith("m.") for entity_id in entity_id_list) and len(entity_id_list) > 10):
                        entity_id_list = random.sample(entity_id_list, 10)

                    # Embedding粗排
                    if len(entity_id_list) > args.max_embedding_candidates:
                        entity_names = [entity_id_to_name[entity_id] for entity_id in entity_id_list]
                        top_entity_names, _top_entity_scores = retrieve_top_docs(question, entity_names, embedding_model, args.max_embedding_candidates)
                        entity_id_list = [entity_name_to_id[entity_name] for entity_name in top_entity_names]

                    # 按实体ID排序，保证候选编号稳定
                    candidate_entity_ids = sorted(entity_id_list)

                    # 构造单行编号候选
                    numbered_entities = " ".join(f"[{i}] {entity_id_to_name[eid]}" for i, eid in enumerate(candidate_entity_ids, start=1))

                    # 构造Prompt
                    prompt = entity_prune_prompt.substitute(
                        question=question,
                        entity_name=entity_id_to_name[topic_entity_id],
                        relation=relation,
                        entity_candidates=numbered_entities
                    )

                    # 调用LLM
                    call_count += 1
                    llm_response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
                    for token_type in token_num.keys():
                        token_usage[token_type] += token_num[token_type]

                    # 解析选择的实体编号
                    selected_ids = parse_selected_ids(llm_response, len(candidate_entity_ids))

                    # 根据编号直接恢复Freebase实体ID
                    selected_entity_ids_for_relation = [candidate_entity_ids[i - 1] for i in selected_ids]
                
                if not selected_entity_ids_for_relation:
                    continue

                if topic_entity_id not in pruned_entity_relation_entity_dict.keys():
                    pruned_entity_relation_entity_dict[topic_entity_id] = {}
                if direction not in pruned_entity_relation_entity_dict[topic_entity_id].keys():
                    pruned_entity_relation_entity_dict[topic_entity_id][direction] = {}
                if relation not in pruned_entity_relation_entity_dict[topic_entity_id][direction].keys():
                    pruned_entity_relation_entity_dict[topic_entity_id][direction][relation] = []
                
                for selected_entity_id in selected_entity_ids_for_relation:
                    selected_entity_name = entity_id_to_name[selected_entity_id]
                    pruned_entity_relation_entity_dict[topic_entity_id][direction][relation].append(selected_entity_id)
                    selected_topic_entity_names.append(entity_id_to_name[topic_entity_id])
                    selected_relations.append(relation)
                    selected_entity_names.append(selected_entity_name)
                    selected_entity_ids.append(selected_entity_id)
                    selected_parent_entity_ids.append(topic_entity_id)
                    if direction == "head":
                        selected_heads.append(True)
                    else:
                        selected_heads.append(False)

    if len(selected_entity_ids) == 0:
        return False, [], [], [], [], [], pruned_entity_relation_entity_dict, call_count, token_usage

    cluster_chain_of_entities = [[
        (selected_topic_entity_names[index], selected_relations[index], selected_entity_names[index])
        for index in range(len(selected_entity_names))
    ]]
    return True, cluster_chain_of_entities, selected_entity_ids, selected_relations, selected_heads, selected_parent_entity_ids, pruned_entity_relation_entity_dict, call_count, token_usage


# 为反思模块选择的回溯实体添加相关的信息，使其称为下一轮的起点实体
def restore_backtrack_context(add_ent_list, depth_entity_relation_entity_dict, new_entity_relation_entity_dict):
    add_entities_id = sorted(add_ent_list)
    add_relations, add_head, add_parent_entity_ids = [], [], []

    # 遍历历史信息，查看是否记录过这个被选择的实体
    for current_entity in add_entities_id:
        flag = 0
        for depth, entity_relation_entity_dict in depth_entity_relation_entity_dict.items():
            for topic_entity, head_tail_dict in entity_relation_entity_dict.items():
                for direction, relation_entity_dict in head_tail_dict.items():
                    for relation, entities_list in relation_entity_dict.items():
                        if current_entity in entities_list:
                            if topic_entity not in new_entity_relation_entity_dict.keys():
                                new_entity_relation_entity_dict[topic_entity] = {}
                            if direction not in new_entity_relation_entity_dict[topic_entity].keys():
                                new_entity_relation_entity_dict[topic_entity][direction] = {}
                            if relation not in new_entity_relation_entity_dict[topic_entity][direction].keys():
                                new_entity_relation_entity_dict[topic_entity][direction][relation] = []
                            if current_entity not in new_entity_relation_entity_dict[topic_entity][direction][relation]:
                                new_entity_relation_entity_dict[topic_entity][direction][relation].append(current_entity)
                            
                            if not flag:
                                add_relations.append(relation)
                                add_parent_entity_ids.append(topic_entity)
                                if direction == 'head':
                                    add_head.append(True)
                                else:
                                    add_head.append(False)
                                flag = 1

        # 在历史搜索记录中找不到该实体的相关信息，则将其作为独立的起点实体
        if not flag:
            logger.info("Entity(%s) has no previous relations, treat it as an extra topic entity", current_entity)
            flag = 1
            add_head.append(-1)
            add_relations.append('')
            add_parent_entity_ids.append(current_entity)

            # 将回溯实体写回new_entity_relation_entity_dict，补全
            if current_entity not in new_entity_relation_entity_dict.keys():
                new_entity_relation_entity_dict[current_entity] = {}

    return add_entities_id, add_relations, add_head, add_parent_entity_ids, new_entity_relation_entity_dict


# 根据检索结果更新记忆模块
def update_memory(question, subquestions, entity_relation_entity_dict, entity_id_name, memory_path, args):
    # 读取历史记忆
    with open(memory_path / 'mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    # 构造检索到的KG路径
    chain_prompt = ''
    for topic_entities, head_tail_relations_dict in sorted(entity_relation_entity_dict.items()):
        for directions, relations_entities_dict in sorted(head_tail_relations_dict.items()):
            for relation, entities_list in sorted(relations_entities_dict.items()):
                sorted_e_list = [entity_id_name[e_id] for e_id in sorted(entities_list)]
                chain_prompt += (entity_id_name[topic_entities] + ' ' + relation + ' ' + str(sorted_e_list) + '\n')

    # 构造Prompt
    prompt = update_mem_prompt.substitute(
        question=question,
        subobjectives=str(subquestions),
        memory=his_mem,
        knowledge_triplets=chain_prompt
    )

    # 调用LLM
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    memory_items = parse_json_object(response)

    logger.info("\n[Memory Update]")
    for subobjective_id, statement in memory_items.items():
        logger.info("%s. %s", subobjective_id, statement)

    # 保存为标准JSON，供后续推理和反思读取
    with open(memory_path / 'mem', 'w', encoding='utf-8') as f:
        json.dump(memory_items, f, ensure_ascii=False, indent=4)

    return token_num



# 根据本轮检索结果，判断是否能够回答
def reasoning(question, entity_relation_entity_dict, entity_id_name, memory_path, args):
    with open(memory_path / 'mem', 'r', encoding='utf-8') as f:
        history_memory = f.read()

    # 构造本轮检索到的KG三元组
    chain_prompt = ''
    for topic_entities, head_tail_relations_dict in sorted(entity_relation_entity_dict.items()):
        for directions, relations_entities_dict in sorted(head_tail_relations_dict.items()):
            for relation, entities_list in sorted(relations_entities_dict.items()):
                sorted_e_list = [entity_id_name[e_id] for e_id in sorted(entities_list)]
                chain_prompt += (entity_id_name[topic_entities] + ', ' + relation + ', ' + str(sorted_e_list) + '\n')

    prompt = evaluate_prompt.substitute(
        question=question,
        memory=history_memory,
        knowledge_triplets=chain_prompt
    )
    logger.info("\n[Reasoning]")
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    answer, reason, sufficient = extract_reason_and_anwer(response)

    return response, answer, sufficient, token_num


    
