import argparse
import os
import random
import copy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import *
from freebase_func import *
from logging_utils import initialize_logger, logger

MODELS_DICT = {
    "Qwen3.5-27b-AWQ-4bit": {
        "model_path": "/home/flyvideo/Sata2/Models/LLM/Qwen3.5-27B-AWQ-4bit",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
    },
    "DeepSeekV4.1-Flash": {
        "model_path": "deepseek-flash",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.getenv("LYY_DEEPSEEK_API_KEY"),
    },
    "GPT-4o-Mini": {
        "model_path": "openai/gpt-4o-mini",
        "base_url": "https://router.shengsuanyun.com/api/v1",
        "api_key": os.getenv("LYY_SHENGSUANYUN_API_KEY"),
    },
}


def repeat_unanswer(datas, question_string, output_file):
    answered_set = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as result_file:
            for line in result_file:
                try:
                    result = json.loads(line)
                    answered_set.add(result[question_string])
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed result line in %s", output_file)
    else:
        logger.info(
            "No previous result file at %s; processing all examples.", output_file
        )
    remaining = [item for item in datas if item[question_string] not in answered_set]
    logger.info("Resume check: %d/%d examples remain.", len(remaining), len(datas))
    return remaining


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cwq", choices=["cwq", "webqsp", "grailqa"]
    )
    parser.add_argument(
        "--model_name", type=str, default="GPT-4o-Mini", choices=MODELS_DICT.keys()
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum generated tokens."
    )
    parser.add_argument("--temperature_exploration", type=float, default=0.3)
    parser.add_argument("--temperature_reasoning", type=float, default=0.3)
    parser.add_argument("--remove_unnecessary_rel", type=bool, default=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Output directory relative to VoG/src.",
    )
    parser.add_argument(
        "--semantic_model_path",
        type=str,
        default="/home/liyuyang/lyy/Models/embedding_model/msmarco-bert-base-dot-v5",
    )
    parser.add_argument(
        "--verification_model_path",
        type=str,
        default="/home/liyuyang/lyy/Models/embedding_model/deberta-large-mnli",
    )

    args = parser.parse_args()
    model_config = MODELS_DICT[args.model_name]
    args.LLM_type = model_config["model_path"]
    args.base_url = model_config["base_url"]
    args.api_key = model_config["api_key"]
    if not args.api_key:
        parser.error(
            f"API key is missing for {args.model_name}; set the corresponding environment variable."
        )
    args.output_dir = str(Path(args.output_dir))
    args.output_file = str(
        Path(args.output_dir) / f"VoG_{args.dataset}_{args.model_name.lower()}.jsonl"
    )
    initialize_logger(args.output_dir, args.dataset, args.model_name.lower())
    logger.info(
        "Dataset: %s | model: %s | endpoint: %s",
        args.dataset,
        args.model_name,
        args.base_url,
    )
    datas, question_string = prepare_dataset(args.dataset)

    # Set the base reward threshold and beta.
    base_threshold = 0.73
    beta = 0.4

    logger.info("Dataset size: %d", len(datas))
    unanswered_datas = repeat_unanswer(datas, question_string, args.output_file)
    logger.info("Starting with %d unanswered examples.", len(unanswered_datas))
    datas = unanswered_datas
    embedding_model = SentenceTransformer(
        args.semantic_model_path, model_kwargs={"torch_dtype": "float16"}
    )
    verification_tokenizer = AutoTokenizer.from_pretrained(args.verification_model_path)
    verification_model = AutoModelForSequenceClassification.from_pretrained(
        args.verification_model_path
    ).cuda()

    logger.info("===== Start running VoG on %s =====", args.dataset)
    for index, data in enumerate(tqdm(datas[:]), start=0):
        if data not in unanswered_datas:
            continue
        try:
            start_time = time.time()
            call_num = 0
            all_tokens = {"total": 0, "input": 0, "output": 0}

            question = data[question_string]
            logger.info("New question start: %s", question)

            topic_entity = data["topic_entity"]
            cluster_chain_of_entities = []
            all_candidates = []
            candidate_counter = 0  # Candidate ID counter
            call_num += 1
            response, steps, token_num = generate_plan(question, topic_entity, args)
            results = parse_finish_action(response)
            save_2_jsonl(
                question,
                question_string,
                results,
                [],
                steps,
                call_num,
                all_tokens,
                start_time,
                file_name=f"{args.dataset}_initial_{args.model_name}",
                output_dir=args.output_dir,
            )
            candidate = Candidate(
                steps=steps,
                final_answer=results,
                trace_reward=1,
                candidate_id=candidate_counter,
            )
            candidate_counter += 1
            all_candidates.append(candidate)
            answer2candidates, answer2confidence, answer2cnt = (
                group_candidates_by_answer(all_candidates, {}, criteria="freq")
            )
            logger.info(
                "Candidate answers: %s | confidence: %s | counts: %s",
                answer2candidates,
                answer2confidence,
                answer2cnt,
            )
            for token_type in token_num.keys():
                all_tokens[token_type] += token_num[token_type]

            if len(topic_entity) == 0:
                results = parse_finish_action(response)
                save_2_jsonl(
                    question,
                    question_string,
                    results,
                    [],
                    steps,
                    call_num,
                    all_tokens,
                    start_time,
                    file_name=f"{args.dataset}_{args.model_name}",
                    output_dir=args.output_dir,
                )
                continue

            pre_relations = []
            pre_head_directions = [-1] * len(topic_entity)
            result_saved = False
            depth = 1
            mab_selector = KG_Aware_UCBSelector()
            while depth < len(steps):
                # Use length of steps to define depth
                current_entity_relations_list = []
                i = 0
                for entity in topic_entity:
                    try:
                        logger.info(
                            "Entity=%s (%s) | prior_relations=%s | prior_head=%s | question=%s",
                            entity,
                            topic_entity[entity],
                            pre_relations,
                            pre_head_directions[i],
                            question,
                        )
                        if len(topic_entity) > 1:
                            retrieve_relations, token_num = relation_search_prune(
                                entity,
                                steps[depth - 1 : depth + 1],
                                topic_entity[entity],
                                pre_relations,
                                pre_head_directions[i],
                                question,
                                embedding_model,
                                args,
                            )
                            logger.info(retrieve_relations)
                        else:
                            retrieve_relations, token_num = relation_search_prune(
                                entity,
                                steps[depth - 1 : depth],
                                topic_entity[entity],
                                pre_relations,
                                pre_head_directions[i],
                                question,
                                embedding_model,
                                args,
                            )
                            logger.info(retrieve_relations)
                        call_num += 1
                        for token_type in token_num.keys():
                            all_tokens[token_type] += token_num[token_type]
                    except IndexError as e:
                        logger.info(f"IndexError occurred: {e}")
                        logger.info(f"Detailed information:")
                        logger.info(
                            f"  topic_entity[entity]: {topic_entity.get(entity, 'Not found')}"
                        )
                        logger.info(
                            f"  pre_head_directions[i]: {pre_head_directions[i] if i < len(pre_head_directions) else 'Index out of range'}"
                        )
                    current_entity_relations_list.extend(retrieve_relations)
                    i += 1
                total_candidates = []
                total_scores = []
                total_relations = []
                total_entity_ids = []
                topic_entity_ids = []
                head_directions = []
                err = 0
                errflag = False
                for entity in current_entity_relations_list:
                    if entity["head"]:
                        entity_candidate_ids = entity_search(
                            entity["entity"], entity["relation"], True
                        )
                    else:
                        entity_candidate_ids = entity_search(
                            entity["entity"], entity["relation"], False
                        )

                    if len(entity_candidate_ids) == 0:
                        logger.info("Relation returned no entities: %s", entity)
                        continue
                    if (
                        all(item.startswith("m.") for item in entity_candidate_ids)
                        and len(entity_candidate_ids) > 8
                    ):
                        entity_candidate_ids = random.sample(entity_candidate_ids, 5)
                        err += 1
                    if err > 3:
                        errflag = True
                        logger.info("the relations without named entity.")
                        break
                    logger.info(
                        "Question=%s | candidates=%s | entity=%s | relation=%s",
                        question,
                        entity_candidate_ids,
                        entity["entity"],
                        entity["relation"],
                    )

                    scores, entity_candidates, entity_candidate_ids, token_num = (
                        entity_score(
                            question,
                            steps[depth - 1 : depth],
                            entity_candidate_ids,
                            entity["relation"],
                            entity["entity"],
                            embedding_model,
                            args,
                        )
                    )
                    call_num += 1
                    if token_num:
                        for token_type in token_num.keys():
                            all_tokens[token_type] += token_num[token_type]
                    (
                        total_candidates,
                        total_scores,
                        total_relations,
                        total_entity_ids,
                        topic_entity_ids,
                        head_directions,
                    ) = update_history(
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
                    )

                if len(total_candidates) == 0 or errflag:
                    logger.info("no candidate")
                    results = parse_finish_action(response)

                    candidates = {
                        "question": question,
                        "results": results,
                        "confidence": answer2confidence,
                        "final_candidates": [],
                    }
                    # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]

                    # save_jsonl(candidates, file_name=f"{args.dataset}_{args.model_name}", output_dir=args.output_dir+'_all_candidates')
                    save_2_jsonl(
                        question,
                        question_string,
                        results,
                        cluster_chain_of_entities,
                        steps,
                        call_num,
                        all_tokens,
                        start_time,
                        file_name=f"{args.dataset}_{args.model_name}",
                        output_dir=args.output_dir,
                    )
                    result_saved = True
                    break
                logger.info(
                    "Entities=%s | relations=%s | candidates=%s | topic_entities=%s | directions=%s",
                    total_entity_ids,
                    total_relations,
                    total_candidates,
                    topic_entity_ids,
                    head_directions,
                )
                (
                    flag,
                    chain_of_entities,
                    entity_ids,
                    pre_relations,
                    pre_head_directions,
                ) = entity_prune(
                    total_entity_ids,
                    total_relations,
                    total_candidates,
                    topic_entity_ids,
                    head_directions,
                    total_scores,
                    args,
                )
                cluster_chain_of_entities.append(chain_of_entities)
                logger.info(chain_of_entities)
                if flag:
                    call_num += 1
                    logger.info(f"Verifying:{question,steps[depth-1:depth]}")
                    revise_flag, reason, re_observation, token_num = verify(
                        question,
                        str(steps[depth - 1 : depth]),
                        cluster_chain_of_entities,
                        args,
                    )
                    revise_check, _ = verifying(
                        steps[depth - 1 : depth],
                        chain_of_entities,
                        verification_tokenizer,
                        verification_model,
                    )
                    observation = "".join(
                        [
                            ", ".join([str(x) for x in chain])
                            for chain in chain_of_entities
                        ]
                    )

                    if revise_flag or revise_check:
                        initial_steps = steps
                        steps[depth - 1]["observation"] = re_observation
                        # record = {
                        #     'question': question,
                        #     'iterations': []
                        # }
                        best_steps = None
                        best_score = -float("inf")
                        MAX_ITERATIONS = (
                            4  # Set a maximum number of attempts to avoid infinite loop
                        )
                        strategy_functions = [
                            (
                                "lookahead",
                                update_plan_lookahead,
                                (
                                    question,
                                    steps[:depth],
                                    current_entity_relations_list,
                                    args,
                                ),
                            ),
                            (
                                "local",
                                update_plan_local,
                                (question, initial_steps[:depth], observation, args),
                            ),
                            (
                                "global",
                                update_plan_global,
                                (
                                    question,
                                    steps[:depth],
                                    cluster_chain_of_entities,
                                    args,
                                ),
                            ),
                        ]

                        iteration = 0
                        while iteration < MAX_ITERATIONS:
                            normalized_entropy = confidence2entropy(answer2confidence)
                            loop_flag = has_cross_depth_entity_overlap(
                                cluster_chain_of_entities
                            )
                            selected_strategy_name = mab_selector.select(
                                depth, normalized_entropy, loop_flag
                            )
                            strategy = next(
                                s
                                for s in strategy_functions
                                if s[0] == selected_strategy_name
                            )
                            update_func, func_args = strategy[1], strategy[2]
                            if selected_strategy_name == "lookahead":
                                topic_entity_copy = {
                                    entity: id2entity_name_or_type(entity)
                                    for entity in entity_ids
                                }
                                pre_relations_copy = copy.deepcopy(pre_relations)
                                pre_head_directions_copy = copy.deepcopy(
                                    pre_head_directions
                                )
                                depth_copy = depth + 1
                                # Execute relation search and other operations for each step
                                current_entity_relations_list = []
                                i = 0
                                for entity in topic_entity_copy:
                                    try:
                                        retrieve_relations = relation_filter(
                                            entity,
                                            steps[depth_copy - 1 : depth_copy],
                                            topic_entity_copy[entity],
                                            pre_relations_copy,
                                            pre_head_directions_copy[i],
                                            question,
                                            embedding_model,
                                            args,
                                        )
                                        logger.info(retrieve_relations)
                                    except IndexError as e:
                                        logger.info(f"IndexError occurred: {e}")
                                        logger.info(f"Detailed information:")
                                        logger.info(
                                            f"  topic_entity[entity]: {topic_entity_copy.get(entity, 'Not found')}"
                                        )
                                        logger.info(
                                            f"  pre_head_directions[i]: {pre_head_directions_copy[i] if i < len(pre_head_directions_copy) else 'Index out of range'}"
                                        )
                                    i += 1
                                    current_entity_relations_list.extend(
                                        retrieve_relations
                                    )
                                response, new_steps, token_num = update_plan_lookahead(
                                    question,
                                    steps[:depth],
                                    current_entity_relations_list,
                                    args,
                                )

                            else:
                                response, new_steps, token_num = update_func(*func_args)
                            call_num += 1
                            for token_type in token_num.keys():
                                all_tokens[token_type] += token_num[token_type]

                            score = enhanced_evaluate_plan(
                                new_steps,
                                depth,
                                question,
                                cluster_chain_of_entities,
                                embedding_model,
                                verification_tokenizer,
                                verification_model,
                            )

                            candidate = Candidate(
                                steps=new_steps,
                                final_answer=parse_finish_action(
                                    new_steps[-1]["answer"]
                                ),
                                trace_reward=score,
                                candidate_id=candidate_counter,
                            )
                            candidate_counter += 1
                            all_candidates.append(candidate)
                            answer2candidates, answer2confidence, answer2cnt = (
                                group_candidates_by_answer(
                                    all_candidates, answer2candidates, criteria="freq"
                                )
                            )
                            normalized_entropy = confidence2entropy(answer2confidence)
                            weight_confidence = (
                                beta * np.exp(-1 * normalized_entropy)
                                if normalized_entropy > 0
                                else 0
                            )

                            for candidate in all_candidates:
                                for ans, conf in answer2confidence.items():
                                    if check_answers_equiv(candidate.final_answer, ans):
                                        candidate.confidence_reward = conf

                            candidate.final_reward = (
                                weight_confidence * candidate.confidence_reward
                                + (1 - weight_confidence) * candidate.trace_reward
                            )
                            logger.info(candidate.final_reward)
                            logger.info(
                                f"Iteration {iteration+1} | score:{score:.4f},reward: {candidate.final_reward:.4f}"
                            )

                            mab_selector.update(
                                selected_strategy_name, candidate.final_reward
                            )

                            logger.info(
                                f"Iteration {iteration+1} | {selected_strategy_name} selected, reward: {candidate.final_reward:.4f}"
                            )
                            if depth > 3:
                                reward_threshold = adaptive_reward_threshold(
                                    normalized_entropy,
                                    depth,
                                    max(answer2confidence.values()),
                                )
                            else:
                                reward_threshold = base_threshold
                            # record['iterations'].append({
                            #     'iteration': iteration + 1,
                            #     'initial_steps': initial_steps,
                            #     'selected_strategy': selected_strategy_name,
                            #     'steps': new_steps,
                            #     'score': score,
                            #     "final_reward":candidate.final_reward,
                            #     "threshold": reward_threshold,
                            #     'observation': observation,
                            #     "cluster_chain_of_entities":cluster_chain_of_entities,
                            #     'depth': depth,
                            #     'token_usage': token_num
                            # })
                            reward = candidate.final_reward
                            if reward > best_score:
                                best_score = reward
                                best_steps = new_steps
                            if best_score >= reward_threshold:
                                logger.info(
                                    f"Threshold reached ({best_score:.4f} >= {reward_threshold}), breaking loop."
                                )
                                break
                            iteration += 1
                        steps = best_steps if best_steps else steps
                        # save_jsonl(record, file_name=f"{args.dataset}_{args.model_name}", output_dir=args.output_dir)
                    if depth == len(steps) - 1 or call_num > 60 or len(steps) > 15:
                        results = parse_finish_action(steps[-1]["answer"])
                        logger.info("VoG stoped at depth %d." % depth)
                        logger.info("Answer: %s", results)
                        # if len(all_candidates) > 5 and normalized_entropy > 0.75:
                        # results,steps=select_answer_based_on_confidence(all_candidates)
                        candidates = {
                            "question": question,
                            "results": results,
                            "confidence": answer2confidence,
                            "final_candidates": [],
                        }
                        try:
                            if reward_threshold is not None:
                                candidates["threshold"] = reward_threshold
                        except NameError:
                            pass

                        # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                        # save_jsonl(candidates, file_name=f"{args.dataset}_{args.model_name}", output_dir=args.output_dir+'_all_candidates')

                        save_2_jsonl(
                            question,
                            question_string,
                            results,
                            cluster_chain_of_entities,
                            steps,
                            call_num,
                            all_tokens,
                            start_time,
                            file_name=f"{args.dataset}_{args.model_name}",
                            output_dir=args.output_dir,
                        )
                        result_saved = True
                        break
                    else:
                        logger.info("depth %d Verified." % depth)
                        depth += 1
                        topic_entity = {
                            entity: id2entity_name_or_type(entity)
                            for entity in entity_ids
                        }
                else:
                    results = parse_finish_action(response)

                    candidates = {
                        "question": question,
                        "results": results,
                        "confidence": answer2confidence,
                        "final_candidates": [],
                    }
                    try:
                        if reward_threshold is not None:
                            candidates["threshold"] = reward_threshold
                    except NameError:
                        pass
                    # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                    # save_jsonl(candidates, file_name=f"{args.dataset}_{args.model_name}", output_dir=args.output_dir+'_all_candidates')

                    save_2_jsonl(
                        question,
                        question_string,
                        results,
                        cluster_chain_of_entities,
                        steps,
                        call_num,
                        all_tokens,
                        start_time,
                        file_name=f"{args.dataset}_{args.model_name}",
                        output_dir=args.output_dir,
                    )
                    result_saved = True
                    break

            if not result_saved:
                results = parse_finish_action(response)
                candidates = {
                    "question": question,
                    "results": results,
                    "confidence": answer2confidence,
                    "final_candidates": [],
                }
                # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                # save_jsonl(candidates, file_name=f"{args.dataset}_{args.model_name}", output_dir=args.output_dir+'_all_candidates')

                save_2_jsonl(
                    question,
                    question_string,
                    results,
                    [],
                    steps,
                    call_num,
                    all_tokens,
                    start_time,
                    file_name=f"{args.dataset}_{args.model_name}",
                    output_dir=args.output_dir,
                )
        except KeyboardInterrupt:
            logger.info("Program interrupted by user.")
            break  # or use break to exit loop
        except Exception as e:
            logger.exception("Failed to process sample %d: %s", index, e)
            error_file = (
                Path(args.output_dir)
                / f"VoG_{args.dataset}_{args.model_name.lower()}_errors.jsonl"
            )
            with error_file.open("a", encoding="utf-8") as error_output:
                error_output.write(
                    json.dumps(
                        {"index": index, "error": str(e), "data": data},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            continue
