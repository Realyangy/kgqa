import os
import time

from SPARQLWrapper import JSON, SPARQLWrapper


SPARQLPATH = "http://localhost:3001/sparql"

_SPARQL_TRACE = None


def set_sparql_trace(trace):
    """Collect SPARQL queries for the current question (single-threaded runner)."""
    global _SPARQL_TRACE
    _SPARQL_TRACE = trace

sparql_search_out_rel = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ns:%s ?relation ?x .
}"""

sparql_search_in_rel = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ?x ?relation ns:%s .
}"""

sparql_search_tail_entities = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?Entity
WHERE {
    ns:%s ns:%s ?Entity .
}
LIMIT 1000
"""

sparql_search_head_entities = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?Entity
WHERE {
    ?Entity ns:%s ns:%s .
}
LIMIT 1000
"""

# Batch name resolution: prefer English names and aliases; follow sameAs to a
# *named* linked entity instead of treating the linked URI as a name.
sparql_ids2names = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity ?targetEntity ?priority
WHERE {
    VALUES ?entity { %s }
    {
        ?entity ns:type.object.name ?targetEntity .
        BIND(1 AS ?priority)
    }
    UNION
    {
        ?entity ns:common.topic.alias ?targetEntity .
        BIND(2 AS ?priority)
    }
    UNION
    {
        ?entity <http://www.w3.org/2002/07/owl#sameAs> ?sameAs .
        ?sameAs ns:type.object.name ?targetEntity .
        BIND(3 AS ?priority)
    }
}
"""

def disable_net_proxy():
    for var in [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    ]:
        os.environ.pop(var, None)


def execurte_sparql(sparql_query):
    started = time.monotonic()
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        rows = sparql.query().convert()["results"]["bindings"]
        if _SPARQL_TRACE is not None:
            _SPARQL_TRACE.append({
                "sparql": sparql_query, "result_count": len(rows),
                "elapsed_s": round(time.monotonic() - started, 4), "status": "ok",
            })
        return rows
    except Exception as error:
        if _SPARQL_TRACE is not None:
            _SPARQL_TRACE.append({
                "sparql": sparql_query, "result_count": 0,
                "elapsed_s": round(time.monotonic() - started, 4),
                "status": "error", "error": str(error),
            })
        raise


def normalize_relation(rel: str) -> str:
    prefix_map = (
        ("http://rdf.freebase.com/ns/", ""),
        ("http://rdf.freebase.com/key/", ""),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
        ("http://www.w3.org/2002/07/owl#", "owl:"),
        ("ns:", ""),
    )
    for prefix, replacement in prefix_map:
        if rel.startswith(prefix):
            return replacement + rel[len(prefix) :]
    return rel


def replace_entities_prefix(entities):
    entity_ids = []
    for entity in entities:
        value = None
        if "Entity" in entity:
            value = entity["Entity"]["value"]
        elif "tailEntity" in entity:
            value = entity["tailEntity"]["value"]
        if value is not None:
            entity_ids.append(value.replace("http://rdf.freebase.com/ns/", ""))
    return entity_ids


_FREEBASE_ENTITY_PREFIXES = ("m.", "g.", "en.")
_NOISE_URI_PREFIXES = (
    "http://",
    "https://",
    "/",
    "rdf:",
    "rdfs:",
    "owl:",
    "xsd:",
    "ns:",
)


def is_freebase_entity_id(entity_id: str) -> bool:
    return isinstance(entity_id, str) and entity_id.startswith(_FREEBASE_ENTITY_PREFIXES)


def is_valid_entity(entity: str) -> bool:
    if not entity or not isinstance(entity, str):
        return False
    if is_freebase_entity_id(entity):
        return True
    if entity.startswith(_NOISE_URI_PREFIXES):
        return False
    return True


ENTITY_NAME_CACHE = {}


def get_entity_names(entity_ids):
    """Resolve up to a batch of Freebase IDs per SPARQL call, retaining every ID."""
    unique_ids = list(dict.fromkeys(str(entity_id) for entity_id in entity_ids))
    resolved = {}
    pending = []
    for entity_id in unique_ids:
        if not is_freebase_entity_id(entity_id):
            resolved[entity_id] = entity_id
        elif entity_id in ENTITY_NAME_CACHE:
            resolved[entity_id] = ENTITY_NAME_CACHE[entity_id]
        else:
            pending.append(entity_id)

    for offset in range(0, len(pending), 100):
        batch = pending[offset:offset + 100]
        query = sparql_ids2names % " ".join("ns:" + entity_id for entity_id in batch)
        rows = execurte_sparql(query)
        best = {}
        for item in rows:
            entity_uri = str(item.get("entity", {}).get("value", ""))
            entity_id = entity_uri.removeprefix("http://rdf.freebase.com/ns/")
            label_field = item.get("targetEntity", {})
            label = str(label_field.get("value", "")).strip()
            if entity_id not in batch or not label or label_field.get("type") == "uri":
                continue
            lang = str(label_field.get("xml:lang", "")).lower()
            english = lang == "en" or lang.startswith("en-")
            try:
                priority = int(item.get("priority", {}).get("value", "3"))
            except (TypeError, ValueError):
                priority = 3
            priority = min(3, max(1, priority))
            # English name > English alias > English sameAs-name > other labels.
            rank = (priority - 1) if english else (priority + 2)
            choice = (rank, label)
            if entity_id not in best or choice < best[entity_id]:
                best[entity_id] = choice
        for entity_id in batch:
            name = best[entity_id][1] if entity_id in best else "Unknown"
            ENTITY_NAME_CACHE[entity_id] = name
            resolved[entity_id] = name
    return resolved


def get_entity_name(entity_id):
    return get_entity_names([entity_id])[str(entity_id)]

def is_noise_relation(relation: str) -> bool:
    """Apply the relation exclusions used by the official implementation."""
    if not relation or not isinstance(relation, str):
        return True
    return (
        relation.startswith(_NOISE_URI_PREFIXES)
        or relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameas" in relation.lower()
    )


def filter_relations(relations):
    if not relations:
        return []
    if isinstance(relations[0], str):
        return [rel for rel in relations if not is_noise_relation(rel)]
    return [
        rel for rel in relations
        if not is_noise_relation(rel.get("relation"))
    ]
