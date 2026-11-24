"""Directed relation paths, SPARQL joins and score parsing for MCTS-R."""
import re
from utils import extract_json

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


def build_relation_query(topic_entity_id, relation_path, head=True, limit=1000):
    extract, last = build_relation_extract(topic_entity_id, relation_path)
    edge = f"{last} ?relation ?x ." if head else f"?x ?relation {last} ."
    return (
        PREFIX + "SELECT DISTINCT ?relation WHERE {\n"
        + extract + "\n" + edge + "\n"
        + f'FILTER(STRSTARTS(STR(?relation), "{FREEBASE_NS}"))\n'
        + "}\nORDER BY ?relation\n" + f"LIMIT {int(limit)}"
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


def format_relation_path(topic_entity, relation_path):
    text = str(topic_entity.get("entity_name") or topic_entity["entity_id"])
    for index, hop in enumerate(relation_path, start=1):
        arrow = " -> " if hop["head"] else " <- "
        text += arrow + hop["relation"] + arrow + f"?e{index}"
    return text


def node_path_key(node):
    return (
        node.topic_entity.get("entity_id", ""),
        tuple((hop["relation"], hop["head"]) for hop in node.relation_path),
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



def _normalize_path_score(score):
    try:
        score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    return round(score, 1)


def parse_batch_evaluation_output(output, expected_indices):
    """Parse LLM path-evaluation scores from JSON or a loose text fallback."""
    expected = set(expected_indices)
    parsed = extract_json(output)
    if isinstance(parsed, dict):
        for key in ("results", "scores", "evaluations", "candidates"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    results = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index not in expected:
                continue
            results[index] = _normalize_path_score(item.get("score"))

    if results:
        return results

    pattern = re.compile(
        r"(?is)(?:candidate|path|index)\s*\[?#?\s*(\d+)\s*\]?\s*[:\n].*?"
        r"score\s*:\s*((?:0(?:\.\d+)?|1(?:\.0+)?))\b"
        r"(?=(?:candidate|path|index)\s*\[?#?\s*\d+\s*\]?\s*[:\n]|\Z)"
    )
    for match in pattern.finditer(output or ""):
        index = int(match.group(1))
        if index not in expected:
            continue
        results[index] = _normalize_path_score(match.group(2))
    return results
