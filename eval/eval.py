import re
import json
import argparse
import numpy as np
from utils import *

# 兼容ToG的答案字符串和PoG/CoG的JSON答案
def extract_predictions(value):
    if value is None:
        return []

    if isinstance(value, dict):
        if "A" in value:
            return extract_predictions(value["A"])
        for key in ("Answer", "answer"):
            if key in value:
                return extract_predictions(value[key])
        return []

    if isinstance(value, list):
        return [answer for item in value for answer in extract_predictions(item)]

    if not isinstance(value, str):
        return [str(value)]

    text = value.strip()
    if not text or text.lower() in {"none", "null", "error"}:
        return []

    # 尝试解析PoG/CoG的JSON答案
    start = text.find("{")
    if start >= 0:
        try:
            return extract_predictions(json.loads(text[start:]))
        except json.JSONDecodeError:
            # 兼容旧模型输出的不完整JSON
            matches = re.findall(r'"Answer"\s*:\s*"([^"]+)"', text[start:])
            if matches:
                return [matches[-1]]

            match = re.search(r'"Answer"\s*:\s*(\[[^\]]*\])', text[start:])
            if match:
                try:
                    return extract_predictions(json.loads(match.group(1)))
                except json.JSONDecodeError:
                    pass

    # ToG直接保存答案字符串
    return [text]


# 输出评估结果
def print_evaluation_table(num_right, num_error, total, call_num_list, time_list, token_num_list):
    def average(values):
        return f"{np.mean(values):,.2f}" if values else "N/A"
    
    accuracy = f"{num_right / total:.2%}" if total else "N/A"
    metric_width = 20
    value_width = 10
    border = ("=" * metric_width + "=" * value_width)

    def section(title):
        print(f"{title:<{metric_width + value_width}}")
        print(border)

    def row(metric, value):
        print(f"{metric:<{metric_width}} | {str(value):<{len(str(value))}}")

    print(border)

    # 性能
    section("Performance")
    row("Evaluated Samples", total)
    row("Accuracy", accuracy)
    row("Correct", num_right)
    row("Incorrect", num_error)
    print(border)

    # 计算开销（每题平均）
    section("Computational Cost")
    row("Avg.LLM Calls", average(call_num_list))
    row("Avg.Input Tokens", average(token_num_list["input"]))
    row("Avg.Output Tokens", average(token_num_list["output"]))
    row("Avg.Total Tokens", average(token_num_list["total"]))
    row("Avg.Time (s)", average(time_list))

    print(border)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str, default="/home/flyvideo/Sata2/lyy/kgqa/methods/ToG/output/ToG_data_cwq_sampleNum_500_sampleSeed_42_modelName_deepseekv4.1-flash.jsonl", help="Result file.")
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
        with open('cope_alias/cwq_aname_dict.json', 'r', encoding='utf-8') as f:
            aname_dict = json.load(f)
        with open('cope_alias/CWQ_aliase_data31158.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)
        with open('cope_alias/ComplexWebQuestions_test_wans.json', 'r', encoding='utf-8') as f:
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
        with open('cope_alias/WQSP_aliase_data.json', 'r', encoding='utf-8') as f:
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

        # 提取不同算法的预测答案
        predictions = extract_predictions(data.get("results"))

        # 与答案及别名集合进行匹配
        is_correct = any(exact_match(str(prediction), answers) for prediction in predictions)

        if is_correct:
            num_right += 1

            if type_field:
                question_type = ori_data[type_field]
                right_q[question_type] = (right_q.get(question_type, 0) + 1)
        else:
            num_error += 1
            error_question.append(data[question_string])

    # 打印结果
    print_evaluation_table(
        num_right=num_right,
        num_error=num_error,
        total=len(output_datas),
        call_num_list=call_num_list,
        time_list=time_list,
        token_num_list=token_num_list,
    )

