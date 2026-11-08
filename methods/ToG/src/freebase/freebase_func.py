from utils import *
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
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
    {
        ?entity ns:type.object.name ?tailEntity .
        FILTER(?entity = ns:%s)
    }
    UNION
    {
        ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
        FILTER(?entity = ns:%s)
    }
}"""

# 舍弃部分无用的关系
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


# 执行特定SAPRQL查询并获取结果
def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


# 去掉SPARQL查询结果中的关系前缀，仅保留关系名称
def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]


# 去掉SPARQL查询结果中的实体前缀，仅保留实体ID
def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


# 根据实体ID查询实体名称，没有名称时返回UnName_Entity
def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        for item in results["results"]["bindings"]:
            entity = item["tailEntity"]
            if entity.get("xml:lang") == "en":
                return entity["value"]
        return results["results"]["bindings"][0]['tailEntity']['value']


import re
import json
import time
from prompt_list import *


# 判断所有候选得分是否都为零
def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


# 构造关系剪枝提示词
def construct_relation_prune_prompt(question, entity_name, relation_candidates, args):
    numbered_relations = " ".join(f"[{i}] [{'outgoing' if item['head'] else 'incoming'}] {item['relation']}" for i, item in enumerate(relation_candidates, start=1))
    return extract_relation_prompt.substitute(
        relation_num=args.width,
        question=question,
        entity_name=entity_name,
        relations=numbered_relations,
    )


# 构造实体评分提示词
def construct_entity_selection_prompt(question, entity_name, relation, entity_pairs):
    numbered_entities = " ".join(f"[{i}] {name}" for i, (name, entity_id) in enumerate(entity_pairs, start=1))
    return select_entity_candidates_prompt.substitute(
        question=question,
        entity_name=entity_name,
        relation=relation,
        entity_candidates=numbered_entities,
    )


# 关系检索和剪枝
def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    # 开启remove_unnecessary_rel时舍弃部分无用关系
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

    # 构建带方向的候选关系列表
    relation_candidates = ([{"relation": relation, "head": True} for relation in head_relations] + [{"relation": relation, "head": False} for relation in tail_relations])

    # 保证候选顺序稳定
    relation_candidates.sort(key=lambda item: (item["relation"], not item["head"]))

    # 构造Prompt
    prompt = construct_relation_prune_prompt(question, entity_name, relation_candidates, args)
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.base_url, args.api_key, args.LLM_type)
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, entity_name, relation_candidates)

    args._last_relation_search = {"all_candidates": relation_candidates}

    if flag:
        return retrieve_relations_with_scores
    else:
        return []


# 实体检索
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (entity, relation)
        entities = execurte_sparql(head_entities_extract)

    # 去掉前缀，只保留以m.开头的实体
    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


# 选择候选实体，保留与后续路径剪枝兼容的得分列表
def entity_select(question, entity_candidates_id, relation_score, relation, entity_id, args):
    if not entity_candidates_id:
        return [], [], []

    entity_pairs = [(id2entity_name_or_type(eid), eid) for eid in entity_candidates_id]

    # 无名称实体无法按名称选择，沿用原来的平均分配策略
    if all_unknown_entity(entity_pairs):
        entity_candidates = [name for name, _ in entity_pairs]
        entity_candidates_id = [eid for _, eid in entity_pairs]
        return [relation_score / len(entity_pairs)] * len(entity_pairs), entity_candidates, entity_candidates_id

    entity_pairs = del_unknown_entity(entity_pairs)
    if len(entity_pairs) == 1:
        name, eid = entity_pairs[0]
        return [relation_score], [name], [eid]

    # 根据名称和ID排序，保证编号顺序稳定
    entity_pairs.sort(key=lambda item: (item[0], item[1]))

    # 分离实体名称和ID，保留原有返回结构
    entity_candidates = [name for name, eid in entity_pairs]
    entity_candidates_id = [eid for name, eid in entity_pairs]
    entity_name = id2entity_name_or_type(entity_id)

    prompt = construct_entity_selection_prompt(question, entity_name, relation, entity_pairs)
    response = run_llm(prompt, args.temperature_exploration, 1024, args.base_url, args.api_key, args.LLM_type)
    selected_ids = parse_selected_entities(response, entity_candidates)

    # 选中实体得分为1，未选中实体得分为0
    scores = [relation_score if i in selected_ids else 0.0 for i in range(1, len(entity_candidates) + 1)]
    return scores, entity_candidates, entity_candidates_id

# 收集整理当前关系下的所有候选实体的信息（存储在下方以total开头的六个变量中），用于后续的实体剪枝
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    # 未检索到实体
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity_id']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)

    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


# 检索不到额外信息时，中止搜索，调用下面的generate_answer回答
def half_stop(question, cluster_chain_of_entities, output_file, depth, args):
    logger.info("No new knowledge added during search depth %d, stop searching.", depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, output_file=output_file)


# 使用已经检索到的知识生成最终答案，并保存
def generate_answer(question, cluster_chain_of_entities, args): 
    chain_prompt = '\n'.join(
        f"{source}, {relation}, {target}"
        for round_data in cluster_chain_of_entities
        for chain in round_data
        for source, relation, target in chain
    )

    prompt = answer_prompt.substitute(
        question=question,
        knowledge_triplets=chain_prompt,
    )
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)

    # 记录LLM原始回答
    logger.info("LLM Response [Generate Answer]:\n%s", response)

    # 解析JSON
    result = parse_answer_json(response)
    answer = result["answer"]

    return answer.strip() if answer is not None else None


# 实体剪枝，这里不需要调用LLM，直接根据整理好的内容排序后剪枝即可
def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):

    # 将候选路径的各信息绑定，根据得分排序后剪枝
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_parent_entity_ids, sorted_heads, sorted_scores = zip(*sorted_zipped) if sorted_zipped else ([], [], [], [], [], [])
    
    sorted_entities_id = list(sorted_entities_id)
    sorted_relations = list(sorted_relations)
    sorted_candidates = list(sorted_candidates)
    sorted_parent_entity_ids = list(sorted_parent_entity_ids)
    sorted_heads = list(sorted_heads)
    sorted_scores = list(sorted_scores)

    entities_id = sorted_entities_id[:args.width]
    relations = sorted_relations[:args.width]
    candidates = sorted_candidates[:args.width]
    parent_entity_ids = sorted_parent_entity_ids[:args.width]
    heads = sorted_heads[:args.width]
    scores = sorted_scores[:args.width]

    # 删除得分为 0 的路径
    filtered_list = [
        (entity_id, relation, candidate, parent_entity_id, head, score)
        for (entity_id, relation, candidate, parent_entity_id, head, score) in zip(entities_id, relations, candidates, parent_entity_ids, heads, scores)
        if score != 0
    ]
    
    if not filtered_list:
        return False, [], [], [], [], []
    entities_id, relations, candidates, parent_entity_ids, heads, scores = map(list, zip(*filtered_list))

    # 构造三元组，保存下一轮需要扩展的实体、关系及其方向
    parent_entity_names = [id2entity_name_or_type(entity_id) for entity_id in parent_entity_ids]
    cluster_chain_of_entities = [[(parent_entity_names[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads, parent_entity_ids


# 判断当前路径是否足以回答问题
def reasoning(question, cluster_chain_of_entities, args):
    chain_prompt = '\n'.join(
        f"{source}, {relation}, {target}"
        for round_data in cluster_chain_of_entities
        for chain in round_data
        for source, relation, target in chain
    )

    # 构造Prompt
    prompt = evaluate_prompt.substitute(question=question, knowledge_triplets=chain_prompt)

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    logger.info("LLM Response [Reasoning]:\n%s", response)
    result = parse_answer_json(response, require_sufficient=True)
    sufficient, answer= result["sufficient"], result["answer"]

    # 证据充分且能够提取有效答案时停止
    if sufficient and answer is not None and answer.strip():
        return True, answer.strip()

    return False, ""
