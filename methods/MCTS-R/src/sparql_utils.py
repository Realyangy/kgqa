"""Shared Freebase/SPARQL infrastructure."""

from functools import lru_cache


SPARQLPATH = "http://localhost:3001/sparql"
_SPARQL_TRACE = None


def set_sparql_trace(trace):
    global _SPARQL_TRACE
    _SPARQL_TRACE = trace

_ENTITY_NAME_QUERY = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?name
WHERE {
    {
        ns:%s ns:type.object.name ?name .
        FILTER (lang(?name) = "en")
    }
    UNION
    {
        ns:%s <http://www.w3.org/2002/07/owl#sameAs> ?alias .
        ?alias ns:type.object.name ?name .
        FILTER (lang(?name) = "en")
    }
}
"""

_FREEBASE_ENTITY_PREFIXES = ("m.", "g.", "en.")


def execute_sparql(query):
    from SPARQLWrapper import JSON, SPARQLWrapper

    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()["results"]["bindings"]
        if _SPARQL_TRACE is not None:
            _SPARQL_TRACE.append({"query": query, "result_count": len(results)})
        return results
    except Exception as error:
        if _SPARQL_TRACE is not None:
            _SPARQL_TRACE.append({"query": query, "error": f"{type(error).__name__}: {error}"})
        raise


def is_freebase_entity_id(entity_id):
    return isinstance(entity_id, str) and entity_id.startswith(_FREEBASE_ENTITY_PREFIXES)


@lru_cache(maxsize=100_000)
def _get_freebase_entity_name(entity_id):
    results = execute_sparql(_ENTITY_NAME_QUERY % (entity_id, entity_id))
    return results[0]["name"]["value"] if results else "Unknown"


def get_entity_name(entity_id):
    if not is_freebase_entity_id(entity_id):
        return entity_id
    return _get_freebase_entity_name(entity_id)



def get_entity_names(entity_ids):
    """Batch English label lookup for grounded bindings (not used in MCTS)."""
    from mcts_utils import FREEBASE_NS, freebase_term

    ids = list(dict.fromkeys(entity_ids))
    names = {}
    for offset in range(0, len(ids), 100):
        batch = ids[offset:offset + 100]
        values = " ".join(freebase_term(eid) for eid in batch)
        query = f"""PREFIX ns: <{FREEBASE_NS}>
SELECT DISTINCT ?entity ?name ?priority WHERE {{
  VALUES ?entity {{ {values} }}
  {{
    ?entity ns:type.object.name ?name .
    BIND(1 AS ?priority)
  }} UNION {{
    ?entity ns:common.topic.alias ?name .
    BIND(2 AS ?priority)
  }} UNION {{
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?alias .
    ?alias ns:type.object.name ?name .
    BIND(3 AS ?priority)
  }}
  FILTER(isLiteral(?name) && (lang(?name) = "" || langMatches(lang(?name), "en")))
}}"""
        best = {}
        for row in execute_sparql(query):
            entity = row.get("entity", {}).get("value", "")
            if not entity.startswith(FREEBASE_NS):
                continue
            eid = entity[len(FREEBASE_NS):]
            name = row.get("name", {}).get("value", "").strip()
            if eid not in batch or not name:
                continue
            rank = (int(row.get("priority", {}).get("value", 3)), name)
            if eid not in best or rank < best[eid]:
                best[eid] = rank
        for eid in batch:
            names[eid] = best[eid][1] if eid in best else eid
    return names
