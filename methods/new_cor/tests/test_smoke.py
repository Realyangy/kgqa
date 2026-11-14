import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import CoRAgent
from freebase_func import build_entities_extract, build_relation_extract
from schema import Entity, Relation
from utils import LLMAPI


class CoRRefactorSmokeTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(
            temperature_exploration=0.3, temperature_reasoning=0.1,
            max_length=512, depth=3, width=3,
            sample_relation_threshold=1000, remove_unnecessary_rel=True
        )
        self.agent = CoRAgent("mock-model", "http://127.0.0.1:8001/v1", "EMPTY", args)
        self.root = Entity("m.root", "Root")

    def test_query_building_respects_relation_directions(self):
        chain = [Relation("r.one", left=True), Relation("r.two", left=False)]
        rel_query, last = build_relation_extract(self.root.id, chain)
        self.assertEqual(last, "?e_2")
        self.assertIn("ns:m.root ns:r.one ?e_1 .", rel_query)
        self.assertIn("?e_2 ns:r.two ?e_1 .", rel_query)
        entity_query, err = build_entities_extract(self.root, chain)
        self.assertEqual(err, "")
        self.assertIn("?targetEntity ns:r.two ?e_1 .", entity_query)

    def test_dfs_backtracks_then_reaches_next_branch(self):
        visited = []
        relations = [Relation("r.first", left=True), Relation("r.second", left=True)]

        def entity_search(*args, **kwargs):
            return ["m.answer"], ["Alice"], []

        def reasoning(*args, **kwargs):
            relation_chain = kwargs["relation_chain"]
            name = relation_chain[-1].id
            visited.append(name)
            if name == "r.first":
                return "Backtrack", "{}", []
            return "Stop", "{}", ["Alice"]

        with patch.object(self.agent, "relation_search", return_value=(["r.first", "r.second"], [])), \\
             patch.object(self.agent, "relation_prune", return_value=relations), \\
             patch.object(self.agent, "entity_search", side_effect=entity_search), \\
             patch.object(self.agent, "reasoning", side_effect=reasoning):
            result = self.agent.answer("Who is Alice?", [self.root], prompt_history=[], sparql_history=[])

        self.assertEqual(visited, ["r.first", "r.second"])
        self.assertEqual(result["action"], "Stop")

    def test_llm_retries_count_as_one_logical_call(self):
        api = object.__new__(LLMAPI)
        api.model_name = "mock-model"
        api.reset_usage()
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"answer":[] }'))]
        )
        create = MagicMock(side_effect=[RuntimeError("temporary failure"), response])
        api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        answer, usage = api.generate("test", temperature=0.1)
        self.assertEqual(api.call_num, 1)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(api.total_token, 10)
        self.assertEqual(usage["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
