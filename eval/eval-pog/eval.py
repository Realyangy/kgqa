import re
import argparse
import numpy as np

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str, default="/home/flyvideo/Sata2/lyy/kgqa/methods/CoG/output/CoG_data_cwq_sampleNum_500_sampleSeed_42_modelName_gpt-4o-mini.jsonl", help="Result file.")
    args = parser.parse_args()

    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)
    count_q = {}
    right_q = {}
    re_list = []
    error_list = []

    num_right = 0
    num_error = 0
    error_question = []

    type_field = ''
    part_q = False
    aname_dict = {}
    alias_dict = {}
    add_ans_alias_dict = {}
    call_num_list = []
    time_list = []
    token_num_list = {
        "input": [],
        "output": [],
        "total": []
    }

    if args.dataset == 'cwq':
        type_field = 'compositionality_type'
        with open('../cope_alias/cwq_aname_dict.json', 'r', encoding='utf-8') as f:
            aname_dict = json.load(f)
        with open('../cope_alias/CWQ_aliase_data31158.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)
        with open('../cope_alias/ComplexWebQuestions_test_wans.json', 'r', encoding='utf-8') as f:
            q_all_list = json.load(f)
            for q_item in q_all_list:
                ans_list = []
                for ans_item in q_item['answers']:
                    if ans_item['answer']:
                        ans_list.append(ans_item['answer'])
                    else:
                        ans_list.append(ans_item['answer_id'])
                    if 'aliases' in ans_item.keys():
                        ans_list += ans_item['aliases']
                
                add_ans_alias_dict[q_item['question']] = ans_list

    elif args.dataset == 'webqsp':
        with open('../cope_alias/WQSP_aliase_data.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)
    elif args.dataset == 'grailqa':
        type_field = 'level'
            
    if part_q:
        q_set = []
        with open('../../pog/eval/analysis_question', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                q_set.append(line.strip())

    for data in output_datas:
        if part_q and data[question_string] not in q_set:
            continue

        answers, ori_data = align(args.dataset, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict)

        if 'time' in data.keys():
            call_num_list.append(data['call_num'])
            time_list.append(data['time'])
            token_num_list['input'].append(data['input_token'])
            token_num_list['output'].append(data['output_token'])
            token_num_list['total'].append(data['total_token'])

        if type_field:
            if ori_data[type_field] not in count_q.keys():
                count_q[ori_data[type_field]] = 0
            count_q[ori_data[type_field]] += 1
        start_i = data['results'].find('{')
        if start_i != -1:
            try:
                results = json.loads(data['results'][start_i:])
                if 'A' in results.keys():
                    response = results['A']['Answer']
                else:
                    response = results['Answer']
                
                if exact_match(str(response), answers):
                    num_right+=1
                    if type_field:
                        if ori_data[type_field] not in right_q.keys():
                            right_q[ori_data[type_field]] = 0
                        right_q[ori_data[type_field]] += 1
                else:
                    num_error+=1
                    error_question.append(data[question_string])
            except:
                pattern = r'"Answer":\s*["\']([^"\']+)["\']'
                match_ = list(re.finditer(pattern, data['results'][start_i:]))
                if match_:
                    response = match_[-1].group(1)
                    if exact_match(response, answers):
                        num_right+=1
                        if type_field:
                            if ori_data[type_field] not in right_q.keys():
                                right_q[ori_data[type_field]] = 0
                            right_q[ori_data[type_field]] += 1
                    else:
                        num_error+=1
                        error_question.append(data[question_string])
                else:
                    pattern = r'"Answer":\s*(\[[^\]]+\])'
                    match_ = re.search(pattern, data['results'][start_i:])
                    if match_:
                        list_string = match_.group(1)
                        list_obj = json.loads(list_string)
                        flag = 0
                        for response in list_obj:
                            if exact_match(str(response), answers):
                                if type_field:
                                    if ori_data[type_field] not in right_q.keys():
                                        right_q[ori_data[type_field]] = 0
                                    right_q[ori_data[type_field]] += 1
                                num_right+=1
                                flag = 1
                                break
                        if not flag:
                            num_error+=1
                            error_question.append(data[question_string])
                            
                    else:
                        if exact_match(str(response), answers):
                            if type_field:
                                if ori_data[type_field] not in right_q.keys():
                                    right_q[ori_data[type_field]] = 0
                                right_q[ori_data[type_field]] += 1
                            num_right+=1
                        else:
                            num_error+=1
                            error_question.append(data[question_string])
        else:
            response = data['results']
            if exact_match(response, answers):
                if type_field:
                    if ori_data[type_field] not in right_q.keys():
                        right_q[ori_data[type_field]] = 0
                    right_q[ori_data[type_field]] += 1
                num_right+=1
            else:
            
                num_error+=1

    print("Exact Match: {:.4f}".format(num_right / len(output_datas)))
    print("Right: {}, Error: {}".format(num_right, num_error))
    print('Average call num:', np.mean(np.array(call_num_list)))
    print('Average time:', np.mean(np.array(time_list)))
    for t_type, nu_l in token_num_list.items():
        print('Average {} token:'.format(t_type), np.mean(np.array(nu_l)))

