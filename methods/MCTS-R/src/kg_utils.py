"""Freebase querying and relation-centric KG retrieval."""
import logging
import re
from functools import lru_cache

from prompt_list import extract_relation_system_prompt, extract_relation_user_prompt
from utils import extract_candidate_indices, format_relation_path

logger = logging.getLogger(__name__)


FREEBASE_NS = "http://rdf.freebase.com/ns/"
PREFIX = "PREFIX ns: <http://rdf.freebase.com/ns/>\n"


def freebase_term(identifier):
    """Accept only Freebase local identifiers, never arbitrary SPARQL text."""
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", identifier):
        raise ValueError(f"Invalid Freebase identifier: {identifier!r}")
    return f"<{FREEBASE_NS}{identifier}>"


def build_relation_extract(topic_entity_id, relation_path):
    """Build one connected basic graph pattern; each hop has a fresh variable."""
    last = freebase_term(topic_entity_id)
    triples = []
    for index, hop in enumerate(relation_path, start=1):
        relation = freebase_term(hop["relation"])
        target = f"?e_{index}"
        if hop["head"]:
            triples.append(f"{last} {relation} {target} .")
        else:
            triples.append(f"{target} {relation} {last} .")
        last = target
    return "\n".join(triples), last


def build_relation_query(topic_entity_id, relation_path, limit=1000):
    """同时查询可达末端的入边和出边，有向关系合计最多返回 1000 个。"""
    extract, last = build_relation_extract(topic_entity_id, relation_path)
    return (
        PREFIX + "SELECT DISTINCT ?relation ?direction WHERE {\n"
        + extract + "\n"
        + f'{{ {last} ?relation ?x . BIND("outgoing" AS ?direction) }}\n'
        + f'UNION {{ ?x ?relation {last} . BIND("incoming" AS ?direction) }}\n'
        + f'FILTER(STRSTARTS(STR(?relation), "{FREEBASE_NS}"))\n'
        + "}\nORDER BY ?relation DESC(?direction)\n" + f"LIMIT {min(1000, int(limit))}"
    )


def build_grounding_query(topic_entity_id, relation_path, limit=1000):
    """Project the entire matching binding, not independent entity sets."""
    if not relation_path:
        raise ValueError("Cannot ground an empty relation path")
    extract, _ = build_relation_extract(topic_entity_id, relation_path)
    variables = " ".join(f"?e_{i}" for i in range(1, len(relation_path) + 1))
    return (
        PREFIX + f"SELECT DISTINCT {variables} WHERE {{\n"
        + extract + "\n}\nORDER BY " + variables + f"\nLIMIT {int(limit)}"
    )


_EXACT_RELATION_BLACKLIST = {
    "en",
    "rdf:type",
    "type.object.name",
    "type.object.type",
    "type.object.key",
    "type.type.instance",
    "type.object.permission",
    "common.topic.description",
    "common.topic.webpage",
    "common.topic.topic_equivalent_webpage",
    "common.document.text",
}

_RELATION_PREFIX_BLACKLIST = (
    "freebase.",
    "dataworld.",
    "pipeline.",
    "wikipedia.",
    "wikidata.",
    "imdb.",
    "musicbrainz.",
    "opencyc.",
    "yago.",
    "nytimes.",
    "cia.",
    "geonames.",
    "un.",
    "kg.",
    "base.ontologies.",
    "base.skosbase.",
    "base.type.",
    "base.yago.",
    "base.kwebbase.",
    "base.wikibase.",
    "base.triples.",
    "base.sameas.",
)

_RELATION_KEYWORD_BLACKLIST = (
    "guid",
    "sameas",
    "permission",
    "annotation",
    "provenance",
    "photo",
    "thumbnail",
    "official_website",
    "webpage",
    "social_media",
)

_RELATION_REGEX_BLACKLIST = (
    re.compile(r"^/"),
    re.compile(r"^(rdf:|rdfs:|owl:|xsd:)"),
    re.compile(r"^https?://"),
    re.compile(r"\.mid$"),
)


def is_noise_relation(relation):
    if not relation or not isinstance(relation, str):
        return True
    if relation in _EXACT_RELATION_BLACKLIST:
        return True
    if relation.startswith(_RELATION_PREFIX_BLACKLIST):
        return True
    relation_lower = relation.lower()
    if any(keyword in relation_lower for keyword in _RELATION_KEYWORD_BLACKLIST):
        return True
    return any(pattern.search(relation) for pattern in _RELATION_REGEX_BLACKLIST)


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


class KgSearchUtils:
    def __init__(self, question, model, relation_topk):
        self.question = question
        self.model = model
        self.relation_topk = relation_topk

    def relations_search(self, topic_entity_id, relation_path):
        """参考 CoR：在整个关系链匹配结果的末端查询所有可达出入边。"""
        blocked_reverse = None
        if relation_path:
            last = relation_path[-1]
            blocked_reverse = (last["relation"], not last["head"])
        seen = set()
        candidates = []
        rows = execute_sparql(build_relation_query(topic_entity_id, relation_path))
        for row in rows:
            uri = row.get("relation", {}).get("value", "")
            direction = row.get("direction", {}).get("value", "")
            if not uri.startswith(FREEBASE_NS) or direction not in ("outgoing", "incoming"):
                continue
            head = direction == "outgoing"
            relation = uri[len(FREEBASE_NS):]
            key = (relation, head)
            if key == blocked_reverse or key in seen or is_noise_relation(relation):
                continue
            seen.add(key)
            candidates.append({"relation": relation, "head": head})
        return sorted(candidates, key=lambda item: (item["relation"], not item["head"]))

    def relations_prune(self, topic_entity, relation_path, candidate_relations, subgoal_context=None):
        if len(candidate_relations) <= self.relation_topk:
            return list(candidate_relations)
        context = subgoal_context or "None"
        lines = [
            f"[{i}] {item['relation']} ({'outgoing' if item['head'] else 'incoming'})"
            for i, item in enumerate(candidate_relations)
        ]
        prompt = extract_relation_user_prompt.substitute(question=self.question, topic_entity=topic_entity.get("entity_name") or topic_entity["entity_id"],
                                                         relation_path=format_relation_path(topic_entity, relation_path), subgoal_context=context,
                                                         relations="\n".join(lines), topk=self.relation_topk)
        output = self.model(system_prompt=extract_relation_system_prompt, user_prompt=prompt, temperature=0.0)[0]
        selected = extract_candidate_indices(output, set(range(len(candidate_relations))))
        if not selected:
            logger.warning("Invalid relation selection; no relations selected for expansion.")
        return [candidate_relations[i] for i in selected[:self.relation_topk]]
