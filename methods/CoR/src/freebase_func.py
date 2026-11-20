import os
import time
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Dict, List, Optional, Tuple

from schema import Entity, Relation


SPARQLPATH = "http://localhost:3001/sparql"
SPARQL_LABEL_BATCH_SIZE = int(os.getenv("FREEBASE_LABEL_BATCH_SIZE", "100"))

_FREEBASE_URI_PREFIX = "http://rdf.freebase.com/ns/"
_ID2NAME_CACHE: Dict[str, str] = {}


# 查询关系
sparql_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
  %s
} LIMIT 1000
"""

# 查询实体
sparql_entities = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?targetEntity
WHERE {
  %s
} LIMIT 1000
"""

# 根据实体ID查询实体名称
sparql_id2name = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?targetEntity
WHERE {
  {
    ?entity ns:type.object.name ?targetEntity .
    FILTER(?entity = ns:%s)
  }
  UNION
  {
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?targetEntity .
    FILTER(?entity = ns:%s)
  }
} LIMIT 1
"""

# 批量根据实体ID查询实体名称
sparql_ids2names = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity ?targetEntity
WHERE {
    VALUES ?entity { %s }
    OPTIONAL {
        ?entity ns:type.object.name ?targetEntity .
        FILTER (lang(?targetEntity) = '' OR langMatches(lang(?targetEntity), 'en'))
    }
}
"""


# 构造关系查询的SPARQL
def build_sparql_relations(query):
    return sparql_relations % query

# 构造实体查询的SPARQL
def build_sparql_entities(query):
    return sparql_entities % query

# 构造根据实体ID查询实体名称的SPARQL
def _render_id2name_query(entity_id):
    return sparql_id2name % (entity_id, entity_id)

# 删除实体前缀
def _normalize_entity_uri(entity_uri: str) -> str:
    entity_uri_str = entity_uri.strip()
    if entity_uri_str.startswith(_FREEBASE_URI_PREFIX):
        return entity_uri_str[len(_FREEBASE_URI_PREFIX) :]
    return entity_uri_str


def _label_rank(lang: str) -> int:
    text = str(lang or "").strip().lower()
    if text == "en":
        return 0
    if text == "":
        return 1
    return 2


# 构造批量查询实体名称的SPARQL
def _render_ids2names_query(entity_ids: List[str]) -> str:
    values_clause = " ".join(f"ns:{entity_id}" for entity_id in entity_ids)
    return sparql_ids2names % values_clause


def execurte_sparql(sparql_txt):
    meta = execurte_sparql_with_meta(sparql_txt)
    return meta.get("rows", [])

"""
执行SPARQL查询，并记录查询状态
"rows": 查询结果,
"status": 查询状态,
"timed_out": 查询是否超时,
"attempts": 查询次数,
"error": 查询是否错误,
""" 
def execurte_sparql_with_meta(sparql_txt):
    last_error = ""
    timed_out = False
    for i in range(3):
        try:
            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return {
                "rows": results["results"]["bindings"],
                "status": "ok",
                "timed_out": False,
                "attempts": i + 1,
                "error": "",
            }
        except BaseException as e:
            last_error = str(e)
            if "timed out" in last_error.lower() or "timeout" in last_error.lower():
                timed_out = True
            logging.error(f"Error in executing SPARQL query (attempt {i+1}/3) on endpoint={SPARQLPATH}: {e}")
            if i < 2:
                time.sleep(2)
            continue
    logging.error(f"SPARQL query failed after 3 attempts on endpoint={SPARQLPATH}:\n{sparql_txt}")
    return {
        "rows": [],
        "status": "timeout" if timed_out else "error",
        "timed_out": timed_out,
        "attempts": 3,
        "error": last_error,
    }


def abandon_rels(relation):
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
    ):
        return True
    return False


def replace_relation_prefix(relations):
    relations = [relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "") for relation in relations]
    clean_relations = [relation for relation in relations if not relation.startswith(("http://", "https://"))]
    return clean_relations


# 区分字面量和实体ID，并删除实体ID的前缀
def replace_entities_prefix(entities):
    literal_entities, uri_entities = [], []
    for entity in entities:
        binding = entity.get("targetEntity", {})
        binding_type = binding.get("type", "")
        value = str(binding.get("value", "")).strip()
        if not value:
            continue
        if binding_type in ("literal", "typed-literal"):
            literal_entities.append(value)
        elif binding_type == "uri":
            uri_entities.append(value.replace("http://rdf.freebase.com/ns/", ""))
    return literal_entities, uri_entities


def id2entity_name_or_type(entity_id):
    entity_id = str(entity_id or "").strip()
    if not entity_id:
        return "UnName_Entity"
    if entity_id in _ID2NAME_CACHE:
        return _ID2NAME_CACHE[entity_id]

    sparql_str = _render_id2name_query(entity_id)
    for i in range(3):
        try:
            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setQuery(sparql_str)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            if len(bindings) == 0:
                _ID2NAME_CACHE[entity_id] = "UnName_Entity"
                return "UnName_Entity"

            best_label = ""
            best_rank = 10**9
            fallback_label = ""
            for binding in bindings:
                label_binding = binding.get("targetEntity", {})
                label = str(label_binding.get("value", "")).strip()
                if not label:
                    continue
                if not fallback_label:
                    fallback_label = label
                rank = _label_rank(str(label_binding.get("xml:lang", "")))
                if rank < best_rank:
                    best_rank = rank
                    best_label = label

            resolved = best_label or fallback_label or "UnName_Entity"
            _ID2NAME_CACHE[entity_id] = resolved
            return resolved
        except BaseException as e:
            logging.error(
                f"Error in id2entity_name_or_type (attempt {i+1}/3) entity={entity_id} "
                f"endpoint={SPARQLPATH}: {e}"
            )
            if i < 2:
                time.sleep(1)
    _ID2NAME_CACHE[entity_id] = "UnName_Entity"
    return "UnName_Entity"


# 根据实体ID查询名称
def id2entity_names(entity_ids: List[str]) -> Dict[str, str]:
    clean_ids: List[str] = []
    seen = set()
    for entity_id in entity_ids:
        text = str(entity_id or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean_ids.append(text)

    if not clean_ids:
        return {}

    result: Dict[str, str] = {}
    pending_ids: List[str] = []

    # 在_ID2NAME_CACH中检查是否已经查询过entity_id
    for entity_id in clean_ids:
        # 如果有，直接放进本次查询结果result
        if entity_id in _ID2NAME_CACHE:
            result[entity_id] = _ID2NAME_CACHE[entity_id]
        # 没有则放入pending_ids，随后查询
        else:
            pending_ids.append(entity_id)

    if pending_ids:
        batch_size = max(1, SPARQL_LABEL_BATCH_SIZE)
        total_batches = (len(pending_ids) + batch_size - 1) // batch_size
        logging.info("[Freebase][id2entity_names] start batch lookup | total_entities=%s | batch_size=%s | total_batches=%s", len(pending_ids), batch_size, total_batches)

        unresolved: List[str] = []
        # 批量查询
        for batch_index, start in enumerate(range(0, len(pending_ids), batch_size), start=1):
            chunk = pending_ids[start : start + batch_size]
            logging.info("[Freebase][id2entity_names] processing batch %s/%s | batch_entities=%s", batch_index, total_batches, len(chunk))

            # 构造当前批次的查询并执行
            query = _render_ids2names_query(chunk)
            meta = execurte_sparql_with_meta(query)
            rows = meta.get("rows", []) or []

            best: Dict[str, Tuple[int, str]] = {}
            for row in rows:
                # 取出实体ID
                entity_uri = str(row.get("entity", {}).get("value", "")).strip()
                entity_id = _normalize_entity_uri(entity_uri)
                if not entity_id:
                    continue

                # 取出实体名称
                label_binding = row.get("targetEntity", {})
                label = str(label_binding.get("value", "")).strip()
                if not label:
                    continue

                # 根据定义的语言顺序决定使用什么语言作为实体名称，最优选择是英语
                rank = _label_rank(str(label_binding.get("xml:lang", "")))
                previous = best.get(entity_id)
                if previous is None or rank < previous[0]:
                    best[entity_id] = (rank, label)

            # 记录当前批次中未查询到名称的实体
            for entity_id in chunk:
                if entity_id in best:
                    resolved = best[entity_id][1]
                    result[entity_id] = resolved
                    _ID2NAME_CACHE[entity_id] = resolved
                else:
                    unresolved.append(entity_id)

        # 对所有批次中查询失败的实体再次查询名称
        for entity_id in unresolved:
            resolved = id2entity_name_or_type(entity_id)
            result[entity_id] = resolved

    return result


# 将关系链转化为SPARQL路径，用于把实体作为隐变量的关系查询
def build_relation_extract(topic_entity_id: str, relation_chain: List[Relation]) -> str:

    # 还没有关系链时，说明只有起点实体，那么当前路径末端实体就是起点实体
    if not relation_chain:
        return "", f"ns:{topic_entity_id}"

    relation_extract = ""
    last_entity = f"ns:{topic_entity_id}"
    for index, hop in enumerate(relation_chain, start=1):
        relation_node = f"ns:{hop.id}"
        if hop.left:
            relation_extract += f"{last_entity} {relation_node} ?e_{index} .\n"
        else:
            relation_extract += f"?e_{index} {relation_node} {last_entity} .\n"
        last_entity = f"?e_{index}"
    return relation_extract, last_entity


# 检查实体查询结果中是否有自带实体名称的结果，从targetEntityLabel中将其名称提取出来
def extract_inline_entity_labels(rows: List[Dict]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for row in rows:
        entity_binding = row.get("targetEntity", {})
        if entity_binding.get("type", "") != "uri":
            continue
        entity_uri = str(entity_binding.get("value", "")).strip()
        entity_id = entity_uri.replace("http://rdf.freebase.com/ns/", "").strip()
        if not entity_id or entity_id in label_map:
            continue
        label = str(row.get("targetEntityLabel", {}).get("value", "")).strip()
        if label:
            label_map[entity_id] = label
    return label_map


# 根据起点实体和关系链生成用于查询末端实体的SPARQL查询片段
def build_entities_extract(topic_entity: Entity, relation_chain: List[Relation]) -> Tuple[str, str]:
    if not relation_chain:
        return "", "relation_chain is empty"

    sparql_entities_extract = ""
    last_entity = f"ns:{topic_entity.id}"
    for index, relation in enumerate(relation_chain, start=1):
        is_last = index == len(relation_chain)
        if is_last:
            next_entity = "?targetEntity"
        else:
            next_entity = f"?e_{index}"

        relation_node = f"ns:{relation.id}"
        if relation.left:
            sparql_entities_extract += f"{last_entity} {relation_node} {next_entity} .\n"
        else:
            sparql_entities_extract += f"{next_entity} {relation_node} {last_entity} .\n"

        if not is_last:
            last_entity = next_entity

    return sparql_entities_extract.strip(), ""


def build_relation_snippets(
    topic_entity: Entity,
    relation_chain: List[Relation],
    relation_display_names: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    def _display_name(relation_id: str) -> str:
        if relation_display_names is None:
            return relation_id
        name = str(relation_display_names.get(relation_id, "")).strip()
        return name or relation_id

    base = f"TopicEntity({topic_entity.name})"
    if not relation_chain:
        return f"{base} ?relation ?x", f"?x ?relation {base}"

    triple_parts: List[str] = []
    last_entity = base
    step = 0
    for relation in relation_chain:
        step += 1
        next_entity = f"e{step}"
        relation_name = _display_name(relation.id)
        if relation.left:
            triple_parts.append(f"{last_entity} {relation_name} {next_entity} .")
        else:
            triple_parts.append(f"{next_entity} {relation_name} {last_entity} .")
        last_entity = next_entity

    prefix = " ".join(triple_parts)
    forward_snippet = f"{prefix} {last_entity} ?relation ?x"
    backward_snippet = f"{prefix} ?x ?relation {last_entity}"
    return forward_snippet, backward_snippet
