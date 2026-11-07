import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# 训练集路径
DATASETS = {
    'cwq': '/home/flyvideo/Sata2/lyy/kgqa/data/train/cwq.json',
    'webqsp': '/home/flyvideo/Sata2/lyy/kgqa/data/train/webqsp.json',
    'grailqa': '/home/flyvideo/Sata2/lyy/kgqa/data/train/grailqa.json'
}

# 关系蓝图数据库输出路径
OUTPUT_DIR = '../index'

# 嵌入模型路径
MODEL_NAME = '/home/flyvideo/Sata2/Models/Embedding-Model/msmarco-distilbert-base-tas-b'

# 从SPARQL中提取关系链
def extract_skeleton(sparql):
    if not sparql:
        return None

    sparql_str = str(sparql)
    pattern = re.compile(
        r"""
        (?:
            [?$][A-Za-z_][A-Za-z0-9_]*      # 变量主语，例如?x
            | (?:ns:|kb:|:)[A-Za-z0-9_.]+   # 带前缀的主语，例如ns:m.01234
            | <[^<>\s]+>                    # 完整URI主语
        )
        \s+
        (?:
            (?:ns:|kb:|:)
            (?P<prefixed>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)
            (?![A-Za-z0-9_.])
            |
            <http://rdf\.freebase\.com/ns/
            (?P<uri>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)>
        )
        (?=\s)
        """,
        re.VERBOSE,
    )

    rels = [match.group('prefixed') or match.group('uri') for match in pattern.finditer(sparql_str)]

    # index-2
    # rels = [relation for relation in rels if not relation.startswith(('m.', 'g.'))]
    
    # index-3
    rels = [relation for relation in rels if not relation.startswith(("m.", "g.")) and "common." not in relation and "type.object" not in relation]
    if not rels:
        return None

    return " -> ".join(rels)


# 从webqsp数据集中提取SPARQL
def parse_webqsp_item(item):
    question = item.get('RawQuestion')
    sparql = None
    if 'Parses' in item and len(item['Parses']) > 0:
        sparql = item['Parses'][0].get('Sparql')
    return question, sparql


# 从grailqa数据集中提取SPARQL
def parse_grailqa_item(item):
    question = item.get('question')
    sparql = item.get('sparql_query')
    if not sparql and 'graph_query' in item:
        sparql = item['graph_query'].get('sparql')
    return question, sparql


# 从cwq数据集中提取SPARQL
def parse_cwq_item(item):
    question = item.get('question')
    sparql = item.get('sparql')
    return question, sparql


# 构建关系蓝图数据库
def build_index(dataset_name, input_path, output_path):
    print(f"\nProcessing {dataset_name} from {input_path}...")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 提取数据集中的样本：不同数据集结构不同，webqsp的单个样本是字典，cwq和grailqa则是列表
    data_list = []
    if dataset_name == 'webqsp':
        if isinstance(raw_data, dict) and "Questions" in raw_data:
            data_list = raw_data["Questions"]
            print(f"WebQSP structure detected. Loaded {len(data_list)} items.")
        else:
            print("Error: WebQSP format mismatch.")
            return
    else:
        if isinstance(raw_data, list):
            data_list = raw_data
            print(f"{dataset_name} structure detected. Loaded {len(data_list)} items.")
        else:
            print(f"Error: {dataset_name} format mismatch.")
            return

    # 遍历所有训练数据，提取其中SPARQL的关系路径
    unique_skeletons = {}
    for item in tqdm(data_list, desc="Extracting"):
        question, sparql = None, None
        
        # 路由到不同的解析器
        if dataset_name == 'webqsp':
            question, sparql = parse_webqsp_item(item)
        elif dataset_name == 'grailqa':
            question, sparql = parse_grailqa_item(item)
        elif dataset_name == 'cwq':
            question, sparql = parse_cwq_item(item)

        # 为每个问题提取关系路径，并去重，关系路径相同的情况下，保留文本最长的问题，因为这样信息量最大
        if question and sparql:
            skeleton = extract_skeleton(sparql)
            if skeleton:
                if skeleton not in unique_skeletons:
                    unique_skeletons[skeleton] = question
                elif len(question) > len(unique_skeletons[skeleton]):
                    unique_skeletons[skeleton] = question

    # 关系路径提取失败时警告
    if len(unique_skeletons) == 0:
        print(f"Warning: Extracted 0 skeletons for {dataset_name}. Check regex or field names.")
        return

    final_data = [{'question': question, 'skeleton': skeleton} for skeleton, question in unique_skeletons.items()]
    print(f"Extracted {len(final_data)} unique logic skeletons.")

    # 对所有"问题-关系路径"数据对进行编码
    print(f"Encoding vectors with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([pair['question'] for pair in final_data], show_progress_bar=True)

    # 保存编码后的关系路径数据库
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_obj = {'data': final_data, 'embeddings': embeddings}
    with open(output_path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"Index saved to {output_path}")


if __name__ == '__main__':
    for name in ['cwq', 'webqsp', 'grailqa']:
        path = DATASETS.get(name)
        if path:
            build_index(name, path, f'{OUTPUT_DIR}/{name}_index.pkl')
        else:
            print(f"Skipping {name}: Path not configured.")
