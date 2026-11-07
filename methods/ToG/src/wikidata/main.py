from tqdm import tqdm
import argparse
import random
from ToG.wikidata.wiki_func import *
from ToG.wikidata.client import *
from ToG.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the OpenAI-compatible model service.")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL of the OpenAI-compatible model service.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--addr_list", type=str,
                        default="server_urls.txt", help="The address of the Wikidata service.")
    parser.add_argument("--output_dir", type=str,
                        default="/home/liyuyang/lyy/ToG/output", help="Root directory for saving results.")
    args = parser.parse_args()

    logger = initialize_logger(args.output_dir, args.dataset)
        
    datas, question_string = prepare_dataset(args.dataset)
    logger.info("Start Running ToG on %s dataset.", args.dataset)
    for data in tqdm(datas):
        question = data[question_string]
        topic_entity = data['qid_topic_entity']
        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset, output_dir=args.output_dir)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        with open(args.addr_list, "r") as f:
            server_addrs = f.readlines()
            server_addrs = [addr.strip() for addr in server_addrs]
        logger.info("Server addresses: %s", server_addrs)
        wiki_client = MultiServerWikidataQueryClient(server_addrs)
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args, wiki_client)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                    relation_trace = getattr(args, "_last_relation_search", {})
                    append_search_trace(
                        logger, args.dataset, question, depth, "relation_search",
                        entity_id=entity,
                        entity_name=topic_entity[entity],
                        all_candidates=relation_trace.get("all_candidates", []),
                        selected_candidates=retrieve_relations_with_scores,
                    )
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                value_flag=False
                if entity['head']:
                    entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation'], wiki_client, True)
                else:
                    entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation'], wiki_client, False)
                retrieved_entity_ids = list(entity_candidates_id)
                retrieved_entity_names = list(entity_candidates_name)
                if len(entity_candidates_name)==0:
                    continue
                if len(entity_candidates_id) ==0: # values
                    value_flag=True
                    if len(entity_candidates_name) >=20:
                        entity_candidates_name = random.sample(entity_candidates_name, 10)
                        entity_candidates_id = ["[FINISH_ID]"] * len(entity_candidates_name)
                else: # ids
                    entity_candidates_id, entity_candidates_name = del_all_unknown_entity(entity_candidates_id, entity_candidates_name)
                    if len(entity_candidates_id) >=20:
                        indices = random.sample(range(len(entity_candidates_name)), 10)
                        entity_candidates_id = [entity_candidates_id[i] for i in indices]
                        entity_candidates_name = [entity_candidates_name[i] for i in indices]

                append_search_trace(
                    logger, args.dataset, question, depth, "entity_search",
                    source_entity=entity['entity'],
                    relation=entity['relation'],
                    head=entity['head'],
                    retrieved_candidates=(
                        [{"id": entity_id, "name": name}
                         for entity_id, name in zip(retrieved_entity_ids, retrieved_entity_names)]
                        if retrieved_entity_ids
                        else [{"id": None, "name": name} for name in retrieved_entity_names]
                    ),
                    candidates_for_scoring=[
                        {"id": entity_id, "name": name}
                        for entity_id, name in zip(entity_candidates_id, entity_candidates_name)
                    ],
                )

                if len(entity_candidates_id) ==0:
                    continue

                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity_candidates_name, entity['score'], entity['relation'], args)
                append_search_trace(
                    logger, args.dataset, question, depth, "entity_scoring",
                    source_entity=entity['entity'],
                    relation=entity['relation'],
                    candidates=entity_candidates,
                    candidate_ids=entity_candidates_id,
                    scores=scores,
                )
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag)
            
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client)
            append_search_trace(
                logger, args.dataset, question, depth, "entity_prune",
                selected_paths=chain_of_entities,
                selected_entity_ids=entities_id,
                selected_relations=pre_relations,
            )
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    logger.info("ToG stoped at depth %d.", depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset, output_dir=args.output_dir)
                    flag_printed = True
                    break
                else:
                    logger.info("depth %d still not find the answer.", depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.query_all("qid2label", entity).pop() for entity in entities_id])}
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset, output_dir=args.output_dir)
