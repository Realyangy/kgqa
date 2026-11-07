from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
from prompt_list import *
from logging_utils import logger
import random
import json
import time
import re
import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from collections import defaultdict

SPARQLPATH = "http://localhost:8890/sparql"

# 查询实体的出边关系
sparql_head_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ns:%s ?relation ?x .
}"""

# 查询实体的入边关系
sparql_tail_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
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

# 根据实体ID查询名称或同一实体的其他标识
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


class KG_Aware_UCBSelector:
    def __init__(
        self,
        strategies=["lookahead", "local", "global"],
        exploration_coef=1.4,
        lambda_global=0.2,
        lambda_local=0.05,
        lambda_lookahead=0.1,
        lambda_bonus=0.1,
    ):

        self.strategies = strategies
        self.exploration_coef = exploration_coef
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.lambda_lookahead = lambda_lookahead
        # self.lookahead_threshold = lookahead_threshold
        self.lambda_bonus = lambda_bonus
        self.counts = {s: 0 for s in strategies}
        self.values = {s: 0.0 for s in strategies}
        self.total_counts = 0

        # Record consecutive selection count
        self.consecutive_counts = {s: 0 for s in strategies}
        self.last_selected = None

    def select(self, depth=None, normalized_entropy=None, loop_flag=False, max_depth=3):
        total_counts = max(1, self.total_counts)
        ucb_scores = {}

        entropy_value = normalized_entropy if normalized_entropy is not None else 0.5
        depth_value = depth if depth is not None else 2
        depth_ratio = depth_value / max_depth  # depth ∈ [0,1]
        bonus_global = 0
        bonus_lookahead = 0
        penalty_global = 0
        penalty_local = 0
        penalty_lookahead = 0
        for s in self.strategies:
            if self.counts[s] == 0:
                return s
            # decaying exploration coefficient：
            # exploration_coef = self.exploration_coef / (1 + total_counts)
            exploration_coef = self.exploration_coef

            avg_value = self.values[s] / self.counts[s]
            ucb = avg_value + exploration_coef * np.sqrt(
                np.log(total_counts) / self.counts[s]
            )
            # Frequent global penalty
            penalty_global = (
                self.lambda_global * self.consecutive_counts["global"]
                if s == "global"
                else 0
            )
            # Consecutive local penalty
            penalty_local = (
                self.lambda_local * self.consecutive_counts["local"]
                if s == "local"
                else 0
            )
            # Long lookahead penalty
            penalty_lookahead = (
                self.lambda_lookahead * self.consecutive_counts["lookahead"]
                if s == "lookahead"
                else 0
            )

            # Dynamic adjustment: higher entropy → encourage global, deeper → penalize lookahead

            # KG-Aware priors
            if s == "global":
                # High entropy → encourage exploration → increase global bonus
                bonus_global = self.lambda_bonus * (
                    1 / (1 + np.exp(-6 * (entropy_value - 0.5)))
                )
                if loop_flag:
                    bonus_global += 0.2
            if s == "lookahead":
                # Large depth → penalize lookahead → decrease bonus
                bonus_lookahead = -self.lambda_bonus * np.tanh(4 * depth_ratio)

            if entropy_value > 0.8 and depth_ratio > 1.5 and s == "lookahead":
                ucb_scores[s] -= 3.0

            ucb_scores[s] = (
                ucb
                + bonus_global
                + bonus_lookahead
                - penalty_global
                - penalty_local
                - penalty_lookahead
            )

        return max(ucb_scores, key=ucb_scores.get)

    def update(self, strategy, reward):
        self.total_counts += 1
        self.counts[strategy] += 1
        self.values[strategy] += reward

        # Update consecutive selection count
        if self.last_selected == strategy:
            self.consecutive_counts[strategy] += 1
        else:
            # Reset consecutive count for non-current strategy
            for s in self.strategies:
                if s != strategy:
                    self.consecutive_counts[s] = 0
            self.consecutive_counts[strategy] = 1

        self.last_selected = strategy


def enhanced_evaluate_plan(
    steps,
    depth,
    question,
    cluster_chain_of_entities,
    embedding_model,
    verification_tokenizer,
    verification_model,
    weights=(0.3, 0.3, 0.2, 0.2),
    optimal_depth=3,
):
    penalty = 0.0
    # Return 0 if depth is invalid
    if depth > len(steps) or depth < 0:
        return 0.0
    logger.info("Depth=%s | plan_steps=%s", depth, len(steps))
    # Fact verification step
    verification_flag, _ = verifying(
        steps[depth - 1 : depth],
        cluster_chain_of_entities,
        verification_tokenizer,
        verification_model,
    )

    # If obvious factual error is found, apply significant penalty
    if verification_flag:
        logger.info("Fact verification failed, applying significant penalty.")
        penalty = 0.5

    # (semantic invalid penalty)
    try:
        final_answer = parse_finish_action(steps[-1]["answer"]).lower().strip()
        invalid_answers = ["no", "unknown", "none", "null"]

        if any(invalid_word in final_answer for invalid_word in invalid_answers):
            if depth <= optimal_depth:
                penalty_invalid = 0.1
            else:
                penalty_invalid = 0.1 + 0.1 * (depth - optimal_depth)

            penalty += penalty_invalid
            logger.info(
                f"semantic invalid penalty({final_answer}),{penalty_invalid:.2f}, current depth is {depth}"
            )
    except:
        pass

    current_step = steps[depth - 1]
    current_observation = ",".join(extract_triplets([current_step]))

    knowledge_graph_text = "\n".join(
        [
            ", ".join([str(x) for x in chain])
            for sublist in cluster_chain_of_entities
            for chain in sublist
        ]
    )

    # 1. Quality
    current_observation_embedding = embedding_model.encode(
        current_observation, convert_to_tensor=True
    )
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    progress_quality = util.pytorch_cos_sim(
        current_observation_embedding, question_embedding
    ).item()
    progress_quality = (progress_quality + 1) / 2  # [0,1]

    # 2. Thought Completion
    current_thought = steps[depth - 1]["thought"]
    knowledge_graph_embedding = embedding_model.encode(
        knowledge_graph_text, convert_to_tensor=True
    )
    thought_embedding = embedding_model.encode(current_thought, convert_to_tensor=True)
    objective_completion = util.pytorch_cos_sim(
        thought_embedding, knowledge_graph_embedding
    ).item()
    objective_completion = (objective_completion + 1) / 2  # [0,1]

    # 3. Efficiency
    if depth == 0:
        efficiency = 1.0
    else:
        prev_texts = [step["action"] + step["thought"] for step in steps[:-1]]
        previous_step_embeddings = embedding_model.encode(
            prev_texts, convert_to_tensor=True
        )
        similarities = util.pytorch_cos_sim(thought_embedding, previous_step_embeddings)
        efficiency = 1.0 - torch.max(similarities).item()
        efficiency = max(efficiency, 0.0)

    # 4. Question Alignment
    step_embedding = embedding_model.encode(str(current_step), convert_to_tensor=True)
    question_alignment = util.pytorch_cos_sim(step_embedding, question_embedding).item()
    question_alignment = (question_alignment + 1) / 2  # [0,1]

    #
    weighted_score = (
        objective_completion * weights[0]
        + progress_quality * weights[1]
        + efficiency * weights[2]
        + question_alignment * weights[3]
    )

    weighted_score = max(weighted_score - penalty, 0.0)
    return weighted_score


def process_entity_relations(relations_list):
    summary = []
    unique_entities = {item["entity"] for item in relations_list}

    if len(unique_entities) == 1:
        entity_id = unique_entities.pop()
        entity_name = id2entity_name_or_type(entity_id)
        summary.append(
            f"Please choose from below relations to form your next action and observations:"
        )
        summary += [
            f"{entity_name}{' ->' if rel['head'] else '<-'} {rel['relation']}"
            for rel in relations_list
        ]
    else:
        summary.append(
            f"Please use all these {len(unique_entities)} entities and their corresponding relations to plan your next action and observations:"
        )

        entity_group = defaultdict(list)
        for item in relations_list:
            entity_group[item["entity"]].append(item)

        for entity_id, relations in entity_group.items():
            entity_name = id2entity_name_or_type(entity_id)
            rel_count = len(relations)
            rel_names = ",".join([rel["relation"] for rel in relations])
            summary += [
                f"{entity_name}{' ->' if rel['head'] else '<-'} {rel['relation']}"
                for rel in relations
            ]
    result = "\n".join(summary)
    return result


def abandon_rels(relation):
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
    ):
        return True


def relation_filter(
    entity_id,
    steps,
    entity_name,
    pre_relations,
    pre_head,
    question,
    embedding_model,
    args,
):
    suggested = ",".join(extract_relations(steps))
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)

    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [
            relation for relation in head_relations if not abandon_rels(relation)
        ]
        tail_relations = [
            relation for relation in tail_relations if not abandon_rels(relation)
        ]

    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal

    combined_results = filtering(
        total_relations, embedding_model, question, suggested, 2
    )  # Remove duplicates
    # logger.info(combined_results)
    flag, retrieve_relations = select_relations(
        combined_results, entity_id, head_relations, tail_relations
    )
    return retrieve_relations  # format error or too small max_length


def retrieve_top_rels(total_relations, suggested, embedding_model, width=3):
    if isinstance(suggested, str):
        suggested_relations = suggested.split(",")
    else:
        suggested_relations = suggested

    scores = []
    for rel in total_relations:
        max_score = max(
            [
                util.dot_score(
                    embedding_model.encode(rel), embedding_model.encode(sug)
                )[0]
                .cpu()
                .item()
                for sug in suggested_relations
            ]
        )
        scores.append(max_score)
    doc_score_pairs = sorted(
        list(zip(total_relations, scores)), key=lambda x: x[1], reverse=True
    )
    corpus_embeddings = embedding_model.encode(
        total_relations, convert_to_tensor=True, device="cuda"
    )
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = embedding_model.encode(
        suggested_relations, convert_to_tensor=True, device="cuda"
    )
    query_embeddings = util.normalize_embeddings(query_embeddings)
    sim_matrix = util.dot_score(
        query_embeddings, corpus_embeddings
    )  # shape: (num_query, num_corpus)
    max_scores = torch.max(sim_matrix, dim=0).values  # Maximum score for each relation
    weights = torch.nn.functional.softmax(max_scores, dim=0)
    weights_np = weights.cpu().numpy()
    entropy = -np.sum(weights_np * np.log(weights_np + 1e-12))
    if len(total_relations) == 0 or np.log(len(total_relations)) == 0:
        entropy_norm = 0
    else:
        entropy_norm = entropy / np.log(len(total_relations))
    # adaptive width
    min_width, max_width = 3, 10
    soft_width = int(min_width + (max_width - min_width) * entropy_norm)
    soft_width = max(min(soft_width, len(total_relations)), min_width)

    # top soft_width
    top_indices = np.argsort(-weights_np)[:soft_width]
    top_docs = [total_relations[i] for i in top_indices]

    return top_docs


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # logger.info(results["results"]["bindings"])
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities):
    return [
        entity["tailEntity"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for entity in entities
    ]


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) == 0:
        return entity_id
    else:
        return results["results"]["bindings"][0]["tailEntity"]["value"]


def select_relations(string, entity_id, head_relations, tail_relations):
    if isinstance(string, str):
        last_brace_l = string.rfind("[")
        last_brace_r = string.rfind("]")

        if last_brace_l < last_brace_r:
            string = string[last_brace_l : last_brace_r + 1]

        relations = []
        rel_list = eval(string.strip())
    elif isinstance(string, list):
        rel_list = string
        relations = []
    for relation in rel_list:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": True})
        elif relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": False})

    if not relations:
        return False, "No relations found"
    return True, relations


def filtering(total_relations, embedding_model, question, suggested, breadth=5):
    result = []
    if len(total_relations) > 10:
        result += retrieve_top_rels(
            total_relations, question, embedding_model, width=breadth
        )
        if suggested:
            result += retrieve_top_rels(
                total_relations, suggested, embedding_model, width=2
            )
        combined_results = list(set(result))  # Remove duplicates
    else:
        result += retrieve_top_rels(
            total_relations, question, embedding_model, width=breadth
        )
        if suggested:
            result += retrieve_top_rels(
                total_relations, suggested, embedding_model, width=breadth
            )

        return total_relations
    return combined_results


def construct_relation_prune_prompt(
    question, steps, entity_name, total_relations, args
):
    return (
        EXTRACT_RELATION_PROMPT
        + question
        + "\nSuggested Relation: "
        + ",".join(extract_relations(steps))
        + "\nTopic Entity: "
        + entity_name
        + "\nRelations: "
        + "; ".join(total_relations)
    )


def construct_entity_score_prompt(
    question, entity, relation, observation, entity_candidates
):
    if observation:
        return (
            PRUNE_ENTITY_PROMPT.format(question)
            + "\nSuggested Entites:"
            + observation
            + "\nTriplets: "
            + entity
            + ", "
            + relation
            + ", ["
            + "; ".join(entity_candidates)
            + "]"
            + "\nOutput: "
        )
    else:
        return (
            PRUNE_ENTITY_PROMPT.format(question)
            + "\nTriplets: "
            + entity
            + ", "
            + relation
            + ", ["
            + "; ".join(entity_candidates)
            + "]"
            + "\nOutput: "
        )


def relation_search_prune(
    entity_id,
    steps,
    entity_name,
    pre_relations,
    pre_head,
    question,
    embedding_model,
    args,
):
    suggested = ",".join(extract_relations(steps))
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)

    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [
            relation for relation in head_relations if not abandon_rels(relation)
        ]
        tail_relations = [
            relation for relation in tail_relations if not abandon_rels(relation)
        ]

    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    if len(total_relations) > 10:
        total_relations = filtering(
            total_relations, embedding_model, question, suggested, breadth=3
        )

    total_relations.sort()  # make sure the order in prompt is always equal

    prompt = construct_relation_prune_prompt(
        question, steps, entity_name, total_relations, args
    )
    result, token_num = run_llm(
        prompt,
        args.temperature_exploration,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    flag, retrieve_relations = select_relations(
        result, entity_id, head_relations, tail_relations
    )

    if flag:
        return retrieve_relations, token_num
    else:
        result = filtering(
            total_relations, embedding_model, question, suggested, breadth=2
        )
        flag, retrieve_relations = select_relations(
            result, entity_id, head_relations, tail_relations
        )
        return retrieve_relations, token_num  # format error or too small max_length


def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract % (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract % (relation, entity)
        entities = execurte_sparql(head_entities_extract)
    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def provide_triple(entity_candidate_ids, relation):
    entity_candidates = []
    for entity_id in entity_candidate_ids:
        if entity_id.startswith("m."):
            entity_candidates.append(id2entity_name_or_type(entity_id))
        else:
            entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidate_ids

    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidate_ids)))
    entity_candidates, entity_candidate_ids = list(ent_id_dict.keys()), list(
        ent_id_dict.values()
    )
    return entity_candidates, entity_candidate_ids


import random


def entity_prune(
    total_entity_ids,
    total_relations,
    total_candidates,
    topic_entity_ids,
    head_directions,
    total_scores,
    args,
):
    zipped = list(
        zip(
            total_entity_ids,
            total_relations,
            total_candidates,
            topic_entity_ids,
            head_directions,
            total_scores,
        )
    )
    logger.info(zipped)
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_zipped = [x for x in sorted_zipped if x[5] != 0]
    (
        sorted_entity_ids,
        sorted_relations,
        sorted_candidates,
        sorted_topic_entities,
        sorted_head,
        sorted_scores,
    ) = (
        [x[0] for x in sorted_zipped],
        [x[1] for x in sorted_zipped],
        [x[2] for x in sorted_zipped],
        [x[3] for x in sorted_zipped],
        [x[4] for x in sorted_zipped],
        [x[5] for x in sorted_zipped],
    )

    entity_ids, relations, candidates, topics, heads, scores = (
        sorted_entity_ids,
        sorted_relations,
        sorted_candidates[:],
        sorted_topic_entities[:],
        sorted_head[:],
        sorted_scores[:],
    )
    merged_list = list(zip(entity_ids, relations, candidates, topics, heads, scores))
    filtered_list = [
        (id, rel, ent, top, hea, score)
        for id, rel, ent, top, hea, score in merged_list
        if score != 0
    ]
    if len(filtered_list) == 0:
        new_sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
        # logger.info(new_sorted_zipped)
        if len(new_sorted_zipped) > 3:
            random.shuffle(new_sorted_zipped)
            new_sorted_zipped = new_sorted_zipped[:3]
        (
            sorted_entity_ids,
            sorted_relations,
            sorted_candidates,
            sorted_topic_entities,
            sorted_head,
            sorted_scores,
        ) = (
            [x[0] for x in new_sorted_zipped],
            [x[1] for x in new_sorted_zipped],
            [x[2] for x in new_sorted_zipped],
            [x[3] for x in new_sorted_zipped],
            [x[4] for x in new_sorted_zipped],
            [x[5] for x in new_sorted_zipped],
        )
        entity_ids, relations, candidates, topics, heads, scores = (
            sorted_entity_ids,
            sorted_relations,
            sorted_candidates[:],
            sorted_topic_entities[:],
            sorted_head[:],
            sorted_scores[:],
        )
        merged_list = list(
            zip(entity_ids, relations, candidates, topics, heads, scores)
        )
        filtered_list = [
            (id, rel, ent, top, hea, score)
            for id, rel, ent, top, hea, score in merged_list
        ]
    if len(filtered_list) == 0:
        return False, [], [], [], []
    entity_ids, relations, candidates, tops, heads, scores = map(
        list, zip(*filtered_list)
    )
    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [
        [(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]
    ]
    return True, cluster_chain_of_entities, entity_ids, relations, heads


def update_history(
    entity_candidates,
    entity,
    scores,
    entity_candidate_ids,
    total_candidates,
    total_scores,
    total_relations,
    total_entity_ids,
    topic_entity_ids,
    head_directions,
):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidate_ids = ["[FINISH_ID]"]
    candidates_relation = [entity["relation"]] * len(entity_candidates)
    topic_entities = [entity["entity"]] * len(entity_candidates)
    head_num = [entity["head"]] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entity_ids.extend(entity_candidate_ids)
    topic_entity_ids.extend(topic_entities)
    head_directions.extend(head_num)
    return (
        total_candidates,
        total_scores,
        total_relations,
        total_entity_ids,
        topic_entity_ids,
        head_directions,
    )


def half_stop(
    question,
    question_string,
    subquestions,
    cluster_chain_of_entities,
    depth,
    call_num,
    all_tokens,
    start_time,
    args,
):
    logger.info(
        "No new knowledge added during search depth %d, stop searching." % depth
    )
    call_num += 1
    answer, token_num = generate_answer(
        question, subquestions, cluster_chain_of_entities, args
    )

    for token_type in token_num.keys():
        all_tokens[token_type] += token_num[token_type]

    save_2_jsonl(
        question,
        question_string,
        answer,
        cluster_chain_of_entities,
        subquestions,
        call_num,
        all_tokens,
        start_time,
        file_name=f"{args.dataset}_{args.model_name}",
        output_dir=args.output_dir,
    )


def generate_answer(question, subquestions, cluster_chain_of_entities, args):
    prompt = ANSWER_PROMPT + question
    chain_prompt = "\n".join(
        [
            ", ".join([str(x) for x in chain])
            for sublist in cluster_chain_of_entities
            for chain in sublist
        ]
    )
    prompt += "\nKnowledge Triplets: " + chain_prompt
    result, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    return result, token_num


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


def entity_score(
    question, steps, entity_candidate_ids, relation, entity, embedding_model, args
):
    entity_candidates = [
        id2entity_name_or_type(entity_id) for entity_id in entity_candidate_ids
    ]
    entity = id2entity_name_or_type(entity)
    observation = ",".join(extract_triplets(steps))
    # if all_unknown_entity(entity_candidates):
    #     return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidate_ids
    # entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [1.0], entity_candidates, entity_candidate_ids, ""
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidate_ids, ""
    if len(entity_candidates) > 10:
        if observation:
            entity_candidates = retrieve_top_rels(
                entity_candidates, observation, embedding_model, width=10
            )
        else:
            entity_candidates = retrieve_top_rels(
                entity_candidates, question, embedding_model, width=10
            )

    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidate_ids))
    entity_candidates, entity_candidate_ids = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidate_ids = list(entity_candidate_ids)
    prompt = construct_entity_score_prompt(
        question, entity, relation, observation, entity_candidates
    )
    result, token_num = run_llm(
        prompt,
        args.temperature_exploration,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    logger.info("Prompt: %s\nLLM response: %s", prompt, result)
    if all(float(x) == 0.0 for x in clean_results(result, entity_candidates)):
        logger.info("retry")
        prompt = construct_entity_score_prompt(
            question, entity, relation, "", entity_candidates
        )
        result, token_num = run_llm(
            prompt,
            args.temperature_exploration,
            args.max_length,
            args.api_key,
            args.LLM_type,
            base_url=args.base_url,
        )
        logger.info("Prompt: %s\nLLM response: %s", prompt, result)
    return (
        [float(x) * 1.0 for x in clean_results(result, entity_candidates)],
        entity_candidates,
        entity_candidate_ids,
        token_num,
    )


def entity_condition_prune(
    question,
    total_entity_ids,
    total_relations,
    total_candidates,
    topic_entity_ids,
    head_directions,
    ent_rel_ent_dict,
    entid_name,
    name_entid,
    args,
    embedding_model,
):
    cur_call_time = 0
    cur_token = {"total": 0, "input": 0, "output": 0}

    new_ent_rel_ent_dict = {}
    no_prune = ["time", "number", "date"]
    (
        filter_entities_id,
        filter_tops,
        filter_relations,
        filter_candidates,
        filter_head,
    ) = ([], [], [], [], [])
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                if is_all_digits(e_list) or rela in no_prune or len(e_list) <= 1:
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    select_ent = sorted_e_list
                else:
                    if (
                        all(entid_name[item].startswith("m.") for item in e_list)
                        and len(e_list) > 10
                    ):
                        e_list = random.sample(e_list, 10)

                    if len(e_list) > 20:
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(
                            question, sorted_e_list, embedding_model, 20
                        )
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        logger.info("Top sentence entities: %s", topn_entities)

                    prompt = PRUNE_ENTITY_PROMPT.format(question) + "\nTriples: "
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    prompt += (
                        entid_name[topic_e] + " " + rela + " " + str(sorted_e_list)
                    )

                    cur_call_time += 1
                    result, token_num = run_llm(
                        prompt,
                        args.temperature_reasoning,
                        args.max_length,
                        args.api_key,
                        args.LLM_type,
                        base_url=args.base_url,
                    )
                    for token_type in token_num.keys():
                        cur_token[token_type] += token_num[token_type]

                    last_brace_l = result.rfind("[")
                    last_brace_r = result.rfind("]")

                    if last_brace_l < last_brace_r:
                        result = result[last_brace_l : last_brace_r + 1]

                    try:
                        result = eval(result.strip())
                    except:
                        result = result.strip().strip("[").strip("]").split(", ")
                        result = [x.strip("'") for x in result]

                    select_ent = sorted(result)
                    select_ent = [x for x in select_ent if x in sorted_e_list]

                if len(select_ent) == 0 or all(x == "" for x in select_ent):
                    continue

                if topic_e not in new_ent_rel_ent_dict.keys():
                    new_ent_rel_ent_dict[topic_e] = {}
                if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                    new_ent_rel_ent_dict[topic_e][h_t] = {}
                if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                    new_ent_rel_ent_dict[topic_e][h_t][rela] = []

                for ent in select_ent:
                    if ent in sorted_e_list:
                        new_ent_rel_ent_dict[topic_e][h_t][rela].append(name_entid[ent])
                        filter_tops.append(entid_name[topic_e])
                        filter_relations.append(rela)
                        filter_candidates.append(ent)
                        filter_entities_id.append(name_entid[ent])
                        if h_t == "head":
                            filter_head.append(True)
                        else:
                            filter_head.append(False)

    if len(filter_entities_id) == 0:
        return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token

    cluster_chain_of_entities = [
        [
            (filter_tops[i], filter_relations[i], filter_candidates[i])
            for i in range(len(filter_candidates))
        ]
    ]
    return (
        True,
        cluster_chain_of_entities,
        filter_entities_id,
        filter_relations,
        filter_head,
        new_ent_rel_ent_dict,
        cur_call_time,
        cur_token,
    )


def verify(question, steps, cluster_chain_of_entities, args):
    prompt = VERIFY_PROMPT + "\n Current step: " + str(steps)

    chain_prompt = "\n".join(
        [
            ", ".join([str(x) for x in chain])
            for sublist in cluster_chain_of_entities
            for chain in sublist
        ]
    )
    prompt += "\nKnowledge Triplets:\n" + chain_prompt
    response, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )

    flag, reason, observation = extract_revise_and_observation(response)
    return flag, reason, observation, token_num


def verifying(
    steps, cluster_chain_of_entities, verification_tokenizer, verification_model
):
    observation_text = ",".join(extract_triplets(steps))
    logger.info(observation_text)

    kg_evidence_text = "".join(
        [", ".join([str(x) for x in chain]) for chain in cluster_chain_of_entities]
    )
    kg_evidence_text = str(kg_evidence_text)

    verification_text = observation_text + " [SEP] " + kg_evidence_text
    logger.info(verification_text)
    encoded_verification_text = verification_tokenizer.encode(
        verification_text, padding=True
    )
    prediction = verification_model(
        torch.tensor(torch.tensor([encoded_verification_text]), device="cuda")
    )["logits"]
    predicted_label = torch.argmax(prediction, dim=1)
    reverse_verification_text = kg_evidence_text + " [SEP] " + observation_text
    encoded_reverse_verification_text = verification_tokenizer.encode(
        reverse_verification_text, padding=True
    )
    reverse_prediction = verification_model(
        torch.tensor(torch.tensor([encoded_reverse_verification_text]), device="cuda")
    )["logits"]
    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
    if 0 in predicted_label or 0 in reverse_predicted_label:
        flag = True
        return flag, kg_evidence_text
    else:
        flag = False
        return flag, ""


def generate_plan(question, topic_entities, args):
    prompt = GENERATE_PLAN_PROMPT.format(question=question)
    if topic_entities:
        topic_entity = ",".join(list(topic_entities.values()))
        prompt += f"You should start with {topic_entity} in the first step."

    plan, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    steps = extract_all_thoughts_and_actions(plan)
    return plan, steps, token_num


def update_plan(question, steps, args):
    prompt = (
        UPDATE_PLAN_PROMPT
        + question
        + "\nSteps:"
        + format_steps(steps)
        + "Please revise given steps to complete the whole plan. \nOutput:"
    )
    response, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    new_steps = extract_all_thoughts_and_actions(response)
    logger.info("Updated plan: %s\nPrevious plan: %s", new_steps, steps)
    if "Thought 1" not in response:
        plan = steps + new_steps
    else:
        plan = new_steps
    return response, plan, token_num


def update_plan_lookahead(question, steps, current_entity_relations_list, args):
    prompt = UPDATE_PLAN_LOOKAHEAD_PROMPT.format(
        question=question,
        steps=format_steps(steps),
        info=process_entity_relations(current_entity_relations_list),
    )
    response, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    new_steps = extract_all_thoughts_and_actions(response)
    logger.info("Updated plan: %s\nPrevious plan: %s", new_steps, steps)
    if "Thought 1" not in response:
        plan = steps + new_steps
    else:
        plan = new_steps
    return response, plan, token_num


def update_plan_local(question, steps, observation, args):
    prompt = UPDATE_PLAN_LOCAL_PROMPT.format(
        question=question, steps=format_steps(steps), observation=observation
    )
    # prompt = UPDATE_PLAN_LOCAL_PROMPT+question+'\nSteps:'+format_steps(steps)+'\nCurrent observation:'+observation+'\nOutput:'
    response, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    new_steps = extract_all_thoughts_and_actions(response)
    logger.info("Updated plan: %s\nPrevious plan: %s", new_steps, steps)
    if "Thought 1" not in response:
        plan = steps + new_steps
    else:
        plan = new_steps
    return response, plan, token_num


def update_plan_global(question, steps, cluster_chain_of_entities, args):

    prompt = UPDATE_PLAN_GLOBAL_PROMPT.format(
        question=question,
        steps=format_steps(steps),
        KG="\n".join(
            [
                ", ".join([str(x) for x in chain])
                for sublist in cluster_chain_of_entities
                for chain in sublist
            ]
        ),
    )
    response, token_num = run_llm(
        prompt,
        args.temperature_reasoning,
        args.max_length,
        args.api_key,
        args.LLM_type,
        base_url=args.base_url,
    )
    plan = extract_all_thoughts_and_actions(response)

    return response, plan, token_num
