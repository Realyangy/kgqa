import re
import json
import time
import random
import requests
import numpy as np
from sentence_transformers import util
from SPARQLWrapper import SPARQLWrapper, JSON

from utils import *
from prompt_list import *
from logging_utils import logger

# freebase端口
SPARQLPATH = "http://localhost:3001/sparql"

# 预设的查询命令
sparql_head_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ns:%s ?relation ?x .
}"""

sparql_tail_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ?x ?relation ns:%s .
}"""

sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
    ns:%s ns:%s ?tailEntity .
}"""

sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
    ?tailEntity ns:%s ns:%s .
}"""

sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  BIND(ns:%s AS ?entity)

  {
    # 先查实体的标准名称
    ?entity ns:type.object.name ?tailEntity .
    FILTER(LANG(?tailEntity) = "" || LANGMATCHES(LANG(?tailEntity), "en"))
    BIND(1 AS ?priority)
  }
  UNION
  {
    # 再查实体别名
    ?entity ns:common.topic.alias ?tailEntity .
    FILTER(LANG(?tailEntity) = "" || LANGMATCHES(LANG(?tailEntity), "en"))
    BIND(2 AS ?priority)
  }
  UNION
  {
    # 最后查询实体链接
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    BIND(3 AS ?priority)
  }
}
ORDER BY ASC(?priority)
LIMIT 1"""

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
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.info("SPARQL Error: %s", e)
        return []

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    # 1. 基础过滤：只处理 m.ID 和 http URL
    if not entity_id.startswith('m.') and not entity_id.startswith('http'):
        return entity_id

    final_name = entity_id 

    # 2. 本地 Freebase 查询 (针对 m.ID)
    if entity_id.startswith('m.'):
        try:
            try:
                sparql_query = sparql_id % (entity_id)
            except TypeError:
                sparql_query = sparql_id % (entity_id, entity_id, entity_id)

            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            
            
            results = sparql.query().convert()
            if "results" in results and "bindings" in results["results"]:
                bindings = results["results"]["bindings"]
                if len(bindings) > 0 and 'tailEntity' in bindings[0]:
                    final_name = bindings[0]['tailEntity']['value']
        except Exception:
            pass

    if "wikidata.org/entity/" in final_name:
        try:
            qid = final_name.split("/")[-1]
            # API URL
            url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&props=labels&languages=en&format=json"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json'
            }
            
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=headers, timeout=5)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        if "entities" in data and qid in data["entities"]:
                            entity_node = data["entities"][qid]
                            if "labels" in entity_node and "en" in entity_node["labels"]:
                                return entity_node["labels"]["en"]["value"] 
                        break 
                    elif resp.status_code == 429: # Too Many Requests
                        time.sleep(1) 
                        continue
                    else:
                        break 
                        
                except requests.exceptions.RequestException:

                    time.sleep(0.5)
                    continue
                    
        except Exception as e:
            pass


    return final_name


# 从LLM输出中解析选择的关系
def select_relations(response, entity_id, relation_candidates):
    selected_ids = parse_selected_ids(response, len(relation_candidates))

    relations = []
    for candidate_id in selected_ids:
        candidate = relation_candidates[candidate_id - 1]
        relations.append({"entity": entity_id, "relation": candidate["relation"], "head": candidate["head"]})

    if not relations:
        return False, "No relations found"
    return True, relations


# 构造关系剪枝提示词
def construct_relation_prune_prompt(question, sub_questions, entity_name, relation_candidates):
    numbered_relations = " ".join(
        f"[{i}] [{'outgoing' if item['head'] else 'incoming'}]{item['relation']}"
        for i, item in enumerate(relation_candidates, start=1)
    )
    return relation_extract_prompt.substitute(question=question, subobjectives=str(sub_questions), entity_name=entity_name, relations=numbered_relations)



def calculate_entropy(scores):
    try:
        exp_scores = np.exp(scores - np.max(scores)) 
        probs = exp_scores / exp_scores.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(scores)) if len(scores) > 1 else 1.0
        return entropy / max_entropy
    except:
        return 1.0 


def relation_search_prune(entity_id, sub_questions, entity_name, pre_relations, pre_head, question, args, model=None, predicted_relation=None, global_skeleton_emb=None, speculation_list=None, guard_budget=0):
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

    # 构造带方向的关系候选
    head_relations = sorted(set(head_relations))
    tail_relations = sorted(set(tail_relations))
    relation_candidates = (
        [{"relation": relation, "head": True} for relation in head_relations] +
        [{"relation": relation, "head": False} for relation in tail_relations]
    )
    relation_candidates.sort(key=lambda item: (item["relation"], not item["head"]))
    total_relations = [item["relation"] for item in relation_candidates]
    
    token_num = {'total': 0, 'input': 0, 'output': 0, 'calls': 0}
    if not total_relations:
        return [], token_num, 0, "EMPTY"


    def clean_rel_str(r):
        if not r:
            return ""
        return r.replace('.', ' ').replace('_', ' ').strip()


    # 计算各候选关系的嵌入
    cand_embs = None
    if model and total_relations:
        cleaned_candidates = [clean_rel_str(r) for r in total_relations]
        cand_embs = model.encode(cleaned_candidates)

    # 计算关系蓝图中当前选择的关系（下称"引导关系"）的嵌入
    if predicted_relation and model and (cand_embs is not None):
        cleaned_pred = clean_rel_str(predicted_relation)
        pred_emb = model.encode(cleaned_pred)

        # 计算各候选关系与引导关系的相似度
        scores = util.cos_sim(pred_emb, cand_embs)[0].cpu().numpy()
        best_idx = scores.argmax()
        max_score = scores[best_idx]
        entropy = calculate_entropy(scores)

        # 若有和引导关系相似度高于设定阈值的候选关系且是离群值（用entropy衡量），则直接返回，结束关系剪枝
        if max_score > 0.88 and entropy < 0.4:
            hit_candidate = relation_candidates[best_idx]
            logger.info("[Fast Track] Entropy=%.2f | Spec '%s' -> Hit '%s'", entropy, predicted_relation, hit_candidate["relation"])
            return [{"entity": entity_id, "relation": hit_candidate["relation"], "head": hit_candidate["head"]}], token_num, 0, "FAST_TRACK"


    # 计算候选关系和原问题的相似度，这里和论文中不一致，论文中说的是计算和子问题的相似度
    if model and total_relations and (cand_embs is not None):
        q_emb = model.encode(question)
        score_local = util.cos_sim(q_emb, cand_embs)[0].cpu().numpy()
        final_scores = score_local * args.relation_question_weight

        # 计算候选关系和引导关系的相似度
        if predicted_relation:
            if 'pred_emb' not in locals():
                pred_emb = model.encode(clean_rel_str(predicted_relation))
            score_spec = util.cos_sim(pred_emb, cand_embs)[0].cpu().numpy()
            final_scores += score_spec * args.relation_prediction_weight

        # 计算候选关系和关系蓝图中其他关系的相似度
        if global_skeleton_emb is not None:
            score_global = util.cos_sim(global_skeleton_emb, cand_embs)[0].cpu().numpy()
            final_scores += score_global * args.relation_skeleton_weight

        # 对候选关系集合重排序，同时保持关系与方向的映射
        if len(final_scores) > 15:
            logger.info("[Relation Pruning] Too many raltions(%s) remained, start sorting and pruning.", len(final_scores))
            top_k_indices = final_scores.argsort()[::-1][:15]
        else:
            top_k_indices = final_scores.argsort()[::-1]
        relation_candidates = [relation_candidates[i] for i in top_k_indices]
        total_relations = [item["relation"] for item in relation_candidates]

        
    # 调用LLM再进行一次关系剪枝
    prompt = construct_relation_prune_prompt(question, sub_questions, entity_name, relation_candidates)
    token_num['calls'] = 1
    result, llm_tokens = run_llm(prompt, args.temperature_exploration, args.max_length, args.base_url, args.api_key, args.LLM_type)
    for token_type in llm_tokens: 
        if token_type != 'calls':
            token_num[token_type] += llm_tokens.get(token_type, 0)

    # 解析LLM输出
    flag, retrieve_relations = select_relations(result, entity_id, relation_candidates)
    if not flag:
        retrieve_relations = []

    used_budget = 0 
    
    # LLM关系剪枝结果不足两条时，从蓝图重排序的关系候选中补充
    if relation_candidates and len(retrieve_relations) < 2 and guard_budget > 0:
        selected_relations = {(item["relation"], item["head"]) for item in retrieve_relations}

        for candidate in relation_candidates:
            if used_budget >= guard_budget or len(retrieve_relations) >= 2:
                break

            relation_key = (candidate["relation"], candidate["head"])
            if relation_key in selected_relations:
                continue

            retrieve_relations.append({"entity": entity_id, "relation": candidate["relation"], "head": candidate["head"]})
            selected_relations.add(relation_key)
            logger.info("Too few relations chosen by LLM, add relation: %s", candidate["relation"])
            used_budget += 1

    return retrieve_relations, token_num, used_budget, "SLOW_TRACK"

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


def provide_triple(entity_candidates_id, relation):
    entity_candidates = []
    for entity_id in entity_candidates_id:

        if entity_id.startswith("m.") or entity_id.startswith("http"): 
            entity_candidates.append(id2entity_name_or_type(entity_id))
        else:
            entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id


    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(ent_id_dict.values())
    return entity_candidates, entity_candidates_id


def update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    candidates_relation = [ent_rel['relation']] * len(entity_candidates)
    topic_entities = [ent_rel['entity']] * len(entity_candidates)
    head_num = [ent_rel['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_relations, total_entities_id, total_topic_entities, total_head


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
def entity_condition_prune(question, entity_relation_entity_dict, entity_id_to_name, entity_name_to_id, args, model):
    call_count = 0
    token_usage = {'total': 0, 'input': 0, 'output': 0}

    pruned_entity_relation_entity_dict = {}
    non_prunable_relations = ['time', 'number', 'date']
    selected_entity_ids, selected_topic_entity_names, selected_relations, selected_entity_names, selected_heads = [], [], [], [], []

    for topic_entity_id, head_tail_dict in sorted(entity_relation_entity_dict.items()):
        for direction, relation_entity_dict in sorted(head_tail_dict.items()):
            for relation, entity_id_list in sorted(relation_entity_dict.items()):
                if is_all_digits(entity_id_list) or relation in non_prunable_relations or len(entity_id_list) <= 1:
                    selected_entity_ids_for_relation = sorted(entity_id_list)
                else:
                    # 无名称实体过多时随机采样
                    if all(entity_id_to_name[item].startswith('m.') or entity_id_to_name[item].startswith('g.') for item in entity_id_list) and len(entity_id_list) > 10:
                        entity_id_list = random.sample(entity_id_list, 10)

                    # Embedding粗排
                    if len(entity_id_list) > 70:
                        sorted_entity_names = [entity_id_to_name[entity_id] for entity_id in entity_id_list]
                        top_entity_names, _ = retrieve_top_docs(question, sorted_entity_names, model, 70)
                        entity_id_list = [entity_name_to_id[entity_name] for entity_name in top_entity_names]

                    # 候选实体编号
                    candidate_entity_ids = sorted(entity_id_list)
                    numbered_entities = " ".join(f"[{i}] {entity_id_to_name[entity_id]}" for i, entity_id in enumerate(candidate_entity_ids, start=1))

                    # 构造提示词
                    prompt = entity_prune_prompt.substitute(question=question, entity_name=entity_id_to_name[topic_entity_id], relation=relation, entity_candidates=numbered_entities)

                    # 调用LLM
                    call_count += 1
                    llm_response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
                    for token_type in token_num:
                        token_usage[token_type] += token_num[token_type]

                    # JSON解析并通过编号恢复实体ID
                    selected_ids = parse_selected_ids(llm_response, len(candidate_entity_ids))
                    selected_entity_ids_for_relation = [candidate_entity_ids[i - 1] for i in selected_ids]

                    logger.info("Relations: %s | Entities: %s", relation, numbered_entities)
                    logger.info("Selected entities after pruning: %s", [entity_id_to_name[entity_id] for entity_id in selected_entity_ids_for_relation])

                if not selected_entity_ids_for_relation:
                    continue

                if topic_entity_id not in pruned_entity_relation_entity_dict:
                    pruned_entity_relation_entity_dict[topic_entity_id] = {}
                if direction not in pruned_entity_relation_entity_dict[topic_entity_id]:
                    pruned_entity_relation_entity_dict[topic_entity_id][direction] = {}
                if relation not in pruned_entity_relation_entity_dict[topic_entity_id][direction]:
                    pruned_entity_relation_entity_dict[topic_entity_id][direction][relation] = []

                for selected_entity_id in selected_entity_ids_for_relation:
                    selected_entity_name = entity_id_to_name[selected_entity_id]
                    pruned_entity_relation_entity_dict[topic_entity_id][direction][relation].append(selected_entity_id)
                    selected_topic_entity_names.append(entity_id_to_name[topic_entity_id])
                    selected_relations.append(relation)
                    selected_entity_names.append(selected_entity_name)
                    selected_entity_ids.append(selected_entity_id)
                    selected_heads.append(direction == 'head')

    if len(selected_entity_ids) == 0:
        return False, [], [], [], [], pruned_entity_relation_entity_dict, call_count, token_usage

    cluster_chain_of_entities = [[(selected_topic_entity_names[i], selected_relations[i], selected_entity_names[i]) for i in range(len(selected_entity_names))]]
    return True, cluster_chain_of_entities, selected_entity_ids, selected_relations, selected_heads, pruned_entity_relation_entity_dict, call_count, token_usage


def restore_backtrack_context(add_ent_list, depth_entities_relations_entities_dict, new_entity_relation_entity_dict, entity_id_to_name, entity_name_to_id, args):
    add_entities_id = sorted(add_ent_list)
    add_relations, add_head = [], []
    topic_ent = set()

    for current_entity in add_entities_id:
        flag = 0
        for depth, entity_relation_entity_dict in depth_entities_relations_entities_dict.items():
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
                                if direction == 'head':
                                    add_head.append(True)
                                else:
                                    add_head.append(False)
                                flag = 1


        if not flag:
            logger.info("No previous relation for entity %s", current_entity)
            flag = 1
            add_head.append(-1)
            add_relations.append('')
            if current_entity not in new_entity_relation_entity_dict.keys():
                new_entity_relation_entity_dict[current_entity] = {}

    return add_entities_id, add_relations, add_head, new_entity_relation_entity_dict

# 根据检索结果更新记忆
def update_memory(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, memory_path, args):
    with open(memory_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    chain_prompt = ''
    for topic_entities, head_tail_relations_dict in sorted(ent_rel_ent_dict.items()):
        for directions, relations_entities_dict in sorted(head_tail_relations_dict.items()):
            for relation, entities_list in sorted(relations_entities_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(entities_list)]
                chain_prompt += entid_name[topic_entities] + ' ' + relation + ' ' + str(sorted_e_list) + '\n'

    prompt = update_memory_prompt.substitute(question=question, subobjectives=str(subquestions), memory=his_mem, knowledge_triplets=chain_prompt)
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    memory_items = parse_json_object(response)

    logger.info("\n[Memory Update]")
    for subobjective_id, statement in memory_items.items():
        logger.info("%s. %s", subobjective_id, statement)

    with open(memory_path+'/mem', 'w', encoding='utf-8') as f:
        json.dump(memory_items, f, ensure_ascii=False, indent=4)
    return token_num


# 根据本轮检索结果判断证据是否充分
def reasoning(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, memory_path, args):
    with open(memory_path+'/mem', 'r', encoding='utf-8') as f:
        history_memory = f.read()

    chain_prompt = ''
    for topic_entities, head_tail_relations_dict in sorted(ent_rel_ent_dict.items()):
        for directions, relations_entities_dict in sorted(head_tail_relations_dict.items()):
            for relation, entities_list in sorted(relations_entities_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(entities_list)]
                chain_prompt += entid_name[topic_entities] + ', ' + relation + ', ' + str(sorted_e_list) + '\n'

    prompt = evaluate_prompt.substitute(question=question, memory=history_memory, knowledge_triplets=chain_prompt)
    logger.info("\n[Reasoning]")
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.base_url, args.api_key, args.LLM_type)
    answer, reason, sufficient = extract_reason_and_anwer(response)
    return response, answer, sufficient, token_num


def generate_final_answer(question, cluster_chain_of_entities, args, all_t, best_answer_so_far="Null"):
    logger.info("[Refinement] Graph search incomplete. Translating IDs and Bridging...")
    try:
        path_str = ""
        if cluster_chain_of_entities:
            for depth_chains in cluster_chain_of_entities:
                for group in depth_chains:
                    for chain in group:
                        if isinstance(chain, (list, tuple)) and len(chain) >= 3:
                            h_raw, r, t_raw = chain[0], chain[1], chain[2]
                            h_show = str(h_raw)
                            if h_show.startswith('m.'):
                                try:
                                    name = id2entity_name_or_type(h_raw)
                                    if name != h_raw: 
                                        h_show = f"{name} ({h_raw})" 
                                except: pass

                            t_show = str(t_raw)
                            if t_show.startswith('m.'):
                                try:
                                    name = id2entity_name_or_type(t_raw)
                                    if name != t_raw: 
                                        t_show = f"{name} ({t_raw})"
                                except: pass
                            
                            path_str += f"{h_show} -> {r} -> {t_show}\n"
                        else:
                            path_str += str(chain) + "\n"
        
        if not path_str.strip(): 
            path_str = "No valid paths retrieved."

        guess, t_tok = run_llm(answer_prompt, 0.0, 50, args.base_url, args.api_key, args.LLM_type)
        for k in t_tok: all_t[k] += t_tok[k]
        

        guess_ans = guess.replace("Answer:", "").strip().strip('"').strip("'").strip(".").split('\n')[0]
        logger.info("[Refinement] LLM Output: %s", guess_ans)

        final_ans = "Null"
        is_verified = False

        invalid_keywords = ["null", "unknown", "i don't know", "no answer"]
        if not guess_ans or any(k in guess_ans.lower() for k in invalid_keywords):
            if best_answer_so_far not in ["Null", None]:
                final_ans = best_answer_so_far
        else:
            final_ans = guess_ans
            if not (guess_ans.isdigit() or re.match(r'^\d{4}-\d{2}-\d{2}$', guess_ans)):
                 check_sparql = f"""PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?mid WHERE {{ {{ ?entity ns:type.object.name "{guess_ans}"@en . }} UNION {{ ?entity ns:common.topic.alias "{guess_ans}"@en . }} }} LIMIT 1"""
                 try:
                     if execurte_sparql(check_sparql): is_verified = True
                 except: pass


        is_sufficient = "Yes" if (final_ans and "null" not in final_ans.lower()) else "No"
        
        final_result_dict = {
            "A": {"Sufficient": is_sufficient, "Answer": final_ans},
            "R": f"Refined answer with ID decoding: {final_ans}"
        }


        return json.dumps(final_result_dict, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.info("[Refinement Critical Error] %s", e)

        err_dict = {"A": {"Sufficient": "No", "Answer": "Null"}, "R": f"Error: {str(e)}"}
        return json.dumps(err_dict, indent=2, ensure_ascii=False)
    
