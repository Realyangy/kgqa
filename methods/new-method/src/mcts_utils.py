"""MCTS expansion, retrieval, and simulation helpers."""

import re

from sparql_utils import is_freebase_entity_id
from utils import (
    ensure_triple_list,
    extract_candidate_indices,
    extract_json,
    format_entity_name,
)


SPARQL_SEARCH_OUT_REL = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ns:%s ?relation ?x.
}"""

SPARQL_SEARCH_IN_REL = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ?x ?relation ns:%s.
}"""

SPARQL_SEARCH_TAIL_ENTITIES = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?Entity
WHERE {
    ns:%s ns:%s ?Entity .
}
LIMIT 1000
"""

SPARQL_SEARCH_HEAD_ENTITIES = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?Entity
WHERE {
    ?Entity ns:%s ns:%s .
}
LIMIT 1000
"""


def normalize_relation(relation):
    prefix_map = (
        ("http://rdf.freebase.com/ns/", ""),
        ("http://rdf.freebase.com/key/", ""),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
        ("http://www.w3.org/2002/07/owl#", "owl:"),
        ("ns:", ""),
    )
    for prefix, replacement in prefix_map:
        if relation.startswith(prefix):
            return replacement + relation[len(prefix):]
    return relation


def extract_entity_ids(bindings):
    return [
        binding["Entity"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for binding in bindings
    ]


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


def is_valid_candidate_entity(entity):
    if not entity or not isinstance(entity, str):
        return False
    if is_freebase_entity_id(entity):
        return True
    return not entity.startswith(_NOISE_URI_PREFIXES)


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


def path_key(walk_path):
    return tuple(tuple(triple) for triple in ensure_triple_list(walk_path))


def node_path_key(node):
    """Return the internal path identity without exposing IDs to prompts."""
    id_path = getattr(node, "id_path", None) or []
    if id_path:
        return path_key(id_path)
    return (
        ("legacy_display_path", path_key(getattr(node, "walk_path", None) or [])),
        ("terminal_entity_id", str(getattr(node, "entity_id", "") or "")),
    )


def cvt_normalized_path_key(walk_path):
    return tuple(
        (format_entity_name(head), str(relation), format_entity_name(tail))
        for head, relation, tail in ensure_triple_list(walk_path)
    )


def deduplicate_cvt_candidates(candidates):
    deduplicated = []
    seen = set()
    for candidate in candidates or []:
        id_path = candidate.get("eval_id_path") or []
        key = (
            ("id_path", path_key(id_path))
            if id_path
            else (
                "legacy_display_path",
                cvt_normalized_path_key(candidate.get("eval_walk_path", [])),
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return deduplicated


def construct_evidence_triple(source_name, relation, target_name, is_head):
    if is_head:
        return [source_name, relation, target_name]
    return [target_name, relation, source_name]


def construct_walk_triple(source_name, relation, target_name, is_head=True):
    if is_head:
        return [source_name, relation, target_name]
    return [source_name, f"reverse {relation}", target_name]


def construct_id_triple(source_id, relation, target_id, is_head=True):
    """Build the non-display path step used for internal deduplication."""
    return construct_walk_triple(source_id, relation, target_id, is_head)


def build_candidate(
    source_name,
    relation,
    relation_direction,
    entity_id,
    entity_name,
    source_id=None,
    walk_path=None,
    evidence_path=None,
    id_path=None,
):
    if walk_path is None:
        walk_path = construct_walk_triple(source_name, relation, entity_name, relation_direction)
    if evidence_path is None:
        evidence_path = construct_evidence_triple(source_name, relation, entity_name, relation_direction)
    if id_path is None and source_id:
        id_path = construct_id_triple(
            source_id,
            relation,
            entity_id,
            relation_direction,
        )
    return {
        "entity_id": entity_id,
        "entity_name": entity_name,
        "relation": relation,
        "head": relation_direction,
        "walk_path": ensure_triple_list(walk_path),
        "evidence_path": ensure_triple_list(evidence_path),
        "id_path": ensure_triple_list(id_path),
    }


def parse_pruned_relations(output, entity_id, indexed_relations):
    selected = [
        indexed_relations[index]
        for index in extract_candidate_indices(output, set(indexed_relations))
    ]
    return bool(selected), [
        {
            "entity": entity_id,
            "relation": relation["relation"],
            "head": relation["head"],
        }
        for relation in selected
    ]


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
