"""Focused, offline tests for the ReKG-MCTS KG/retrieval and rollout alignment.

From methods/ReKG-MCTS/src or repo root:
  python -m unittest discover -s methods/ReKG-MCTS/tests -v
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import freebase_func
import kg_utils
from mcts import MCTSNode, MCTSPathFinder


class FakeModel:
    def __init__(self):
        self.prompts = []

    def __call__(self, user_prompt=None, **kwargs):
        self.prompts.append(user_prompt)
        return ['{"selected_ids": [1, 2, 3, 4, 5]}']


class CandidatePruningTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.search = kg_utils.KgSearchUtils("sample question", self.model, entity_topk=5)
        self.node = MCTSNode([{"entity_id": "m.start", "entity_name": "Start"}])

    def test_sample_before_name_lookup_then_llm_top5(self):
        ids = [f"m.id{i}" for i in range(80)]
        observed = []

        def name_lookup(candidate_ids):
            observed.extend(candidate_ids)
            return {eid: f"Entity {eid}" for eid in candidate_ids}

        with patch.object(kg_utils, "get_entity_names", side_effect=name_lookup):
            results = self.search.entity_prune(ids, self.node, "relation")
        self.assertEqual(len(observed), 50)
        self.assertEqual(len(results), 5)
        self.assertEqual(len(self.model.prompts), 1)
        self.assertTrue(set(results).issubset(set(ids)))

    def test_all_unnamed_retains_only_topk_ids(self):
        ids = [f"m.id{i}" for i in range(80)]
        with patch.object(kg_utils, "get_entity_names", side_effect=lambda candidate_ids: {eid: "Unknown" for eid in candidate_ids}):
            results = self.search.entity_prune(ids, self.node, "relation")
        self.assertEqual(len(results), 5)
        self.assertTrue(set(results).issubset(set(ids)))
        self.assertFalse(self.model.prompts)

    def test_mixed_names_prefer_named(self):
        ids = [f"m.id{i}" for i in range(8)]
        with patch.object(kg_utils, "get_entity_names", side_effect=lambda candidate_ids: {
            eid: ("Named " + eid if i < 3 else "Unknown")
            for i, eid in enumerate(candidate_ids)
        }):
            results = self.search.entity_prune(ids, self.node, "relation")
        self.assertEqual(len(results), 3)
        self.assertFalse(self.model.prompts)


class NameResolutionTests(unittest.TestCase):
    def test_prioritizes_english_label_and_never_uses_sameas_uri(self):
        def row(eid, label, priority, lang="", field_type="literal"):
            return {
                "entity": {"value": "http://rdf.freebase.com/ns/" + eid},
                "targetEntity": {"type": field_type, "value": label, "xml:lang": lang},
                "priority": {"value": str(priority)},
            }
        rows = [
            row("m.one", "Deutscher Name", 1, "de"),
            row("m.one", "English Alias", 2, "en"),
            row("m.one", "English Name", 1, "en"),
            row("m.two", "http://rdf.freebase.com/ns/m.other", 3, field_type="uri"),
            row("m.three", "Linked English Name", 3, "en"),
        ]
        with patch.object(freebase_func, "execurte_sparql", return_value=rows):
            freebase_func.ENTITY_NAME_CACHE.clear()
            result = freebase_func.get_entity_names(["m.one", "m.two", "m.three"])
        self.assertEqual(result["m.one"], "English Name")
        self.assertEqual(result["m.two"], "Unknown")
        self.assertEqual(result["m.three"], "Linked English Name")


class RolloutTests(unittest.TestCase):
    def test_rollout_starts_from_selected_node_and_backs_up_to_it(self):
        root = MCTSNode([{"entity_id": "m.start", "entity_name": "Start"}])
        child = MCTSNode([{"entity_id": "m.next", "entity_name": "Next"}],
                         triples=[("Start", "rel", "Next")], parent=root)
        child.v = child.total_reward = 0.3
        root.is_fully_expanded = True
        finder = MCTSPathFinder.__new__(MCTSPathFinder)
        finder.root = root
        finder.max_iterations = 1
        finder.max_depth = 5
        finder.rollout_count = 2
        finder.score_threshold = 1.1
        finder.best_path = []
        finder.best_path_value = 0.0
        finder.best_path_source = None
        finder.trace_events = []
        seen = []
        finder._select = lambda node: root

        def expand(node):
            self.assertIs(node, root)
            root.children = [child]
            return [child]

        def simulate(node):
            seen.append(node)
            return list(child.path)

        finder._expand = expand
        finder._simulate = simulate
        finder.evaluate = lambda path: (0.6, "test")
        finder._get_best_node = lambda: child
        answer_path = finder.search()
        self.assertEqual(seen, [root, root])
        self.assertEqual(root.visits, 3)
        self.assertEqual(child.visits, 1)
        self.assertEqual(answer_path, child.path)
        self.assertEqual(len(finder.trace_events), 2)


if __name__ == "__main__":
    unittest.main()
