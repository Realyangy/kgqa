import os

from SPARQLWrapper import JSON, SPARQLWrapper


SPARQLPATH = "http://localhost:3001/sparql"

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
"""

sparql_search_head_entities = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?Entity
WHERE {
    ?Entity ns:%s ns:%s .
}
"""

sparql_name_en = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  ?entity ns:type.object.name ?tailEntity .
  FILTER(?entity = ns:%s)
  FILTER(LANG(?tailEntity) = "en")
}
ORDER BY STR(?tailEntity)
LIMIT 1"""
sparql_name_any = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  ?entity ns:type.object.name ?tailEntity .
  FILTER(?entity = ns:%s)
}
ORDER BY STR(?tailEntity)
LIMIT 1"""
sparql_same_as = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
  FILTER(?entity = ns:%s)
}
ORDER BY STR(?tailEntity)
LIMIT 1"""


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
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


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


def get_entity_name(entity_id):
    if entity_id in ENTITY_NAME_CACHE:
        return ENTITY_NAME_CACHE[entity_id]
    if not is_freebase_entity_id(entity_id):
        return entity_id

    for template in (sparql_name_en, sparql_name_any, sparql_same_as):
        results = execurte_sparql(template % entity_id)
        if results:
            name = results[0].get("tailEntity", {}).get("value", "")
            if name:
                ENTITY_NAME_CACHE[entity_id] = name
                return name
    ENTITY_NAME_CACHE[entity_id] = "Unknown"
    return "Unknown"


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
