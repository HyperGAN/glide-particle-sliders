"""Contract and leakage checks for the synthetic concision experiment data."""

from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dataset_builder", PROJECT / "scripts" / "build_dataset.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

IN_DOMAIN = {
    "coding_explanations",
    "customer_support",
    "household_planning",
    "educational_explanations",
}
HELD_OUT = {"travel_logistics", "workplace_communication"}
ROW_KEYS = {
    "id", "group", "split", "domain", "concise_slot", "preferred_slot",
    "control", "neutral", "positive",
}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = BUILDER.build_dataset()
        cls.rows = cls.dataset["rows"]
        cls.groups = defaultdict(list)
        for row in cls.rows:
            cls.groups[row["group"]].append(row)

    def test_serialized_deliverable_matches_builder(self):
        saved = json.loads(BUILDER.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(saved, self.dataset)
        self.assertEqual(set(saved), {"schema", "rows"})
        self.assertEqual(saved["schema"], "glide-concise-pairs-v1")

    def test_exact_group_and_split_counts(self):
        self.assertEqual(len(self.rows), 108)
        self.assertEqual(len(self.groups), 54)
        self.assertEqual(len({row["id"] for row in self.rows}), 108)
        representatives = [rows[0] for rows in self.groups.values()]
        main = [row for row in representatives if not row["control"]]
        controls = [row for row in representatives if row["control"]]
        self.assertEqual(Counter(row["split"] for row in main), {"train": 24, "dev": 8, "test": 16})
        self.assertEqual(Counter(row["split"] for row in controls), {"train": 2, "dev": 2, "test": 2})
        self.assertEqual(Counter(row["split"] for row in self.rows), {"train": 52, "dev": 20, "test": 36})
        self.assertEqual(Counter(row["domain"] for row in main if row["split"] == "train"), {domain: 6 for domain in IN_DOMAIN})
        self.assertEqual(Counter(row["domain"] for row in main if row["split"] == "dev"), {domain: 2 for domain in IN_DOMAIN})
        self.assertEqual(Counter(row["domain"] for row in main if row["split"] == "test"), {domain: 8 for domain in HELD_OUT})

    def test_native_request_contract_and_instruction_only_difference(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(set(row), ROW_KEYS)
                self.assertIs(type(row["control"]), bool)
                self.assertIs(type(row["concise_slot"]), int)
                self.assertIs(type(row["preferred_slot"]), int)
                self.assertIn(row["concise_slot"], (0, 1))
                self.assertIn(row["preferred_slot"], (0, 1))
                self.assertIn(row["split"], ("train", "dev", "test"))
                for name, instruction in (("neutral", BUILDER.NEUTRAL_INSTRUCTIONS), ("positive", BUILDER.POSITIVE_INSTRUCTIONS)):
                    req = row[name]
                    self.assertEqual(set(req), {"state", "questions"})
                    self.assertIsInstance(req["state"], str)
                    self.assertTrue(req["state"].strip())
                    self.assertEqual(set(req["questions"]), {"preference"})
                    question = req["questions"]["preference"]
                    self.assertEqual(set(question), {"type", "instructions", "criteria"})
                    self.assertEqual(question["type"], "choice")
                    self.assertEqual(question["instructions"], instruction)
                    self.assertEqual(list(question["criteria"]), ["first", "second"])
                normalized = deepcopy(row["positive"])
                normalized["questions"]["preference"]["instructions"] = BUILDER.NEUTRAL_INSTRUCTIONS
                self.assertEqual(normalized, row["neutral"])

    def test_group_pairs_reverse_contents_and_semantic_labels(self):
        for group, rows in self.groups.items():
            with self.subTest(group=group):
                self.assertEqual(len(rows), 2)
                by_slot = {row["concise_slot"]: row for row in rows}
                self.assertEqual(set(by_slot), {0, 1})
                first, second = by_slot[0], by_slot[1]
                for field in ("group", "split", "domain", "control"):
                    self.assertEqual(first[field], second[field])
                self.assertEqual(first["preferred_slot"], 1 - second["preferred_slot"])
                for name in ("neutral", "positive"):
                    self.assertEqual(first[name]["state"], second[name]["state"])
                    a = list(first[name]["questions"]["preference"]["criteria"].values())
                    b = list(second[name]["questions"]["preference"]["criteria"].values())
                    self.assertEqual(a, b[::-1])
                    self.assertNotEqual(a[0], a[1])

    def test_semantic_labels_and_lengths_include_quality_controls(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                candidates = list(row["neutral"]["questions"]["preference"]["criteria"].values())
                short_length = len(candidates[row["concise_slot"]].split())
                long_length = len(candidates[1 - row["concise_slot"]].split())
                self.assertLess(short_length, long_length)
                self.assertGreaterEqual(short_length, 20)
                self.assertLessEqual(long_length, 60)
                expected = 1 - row["concise_slot"] if row["control"] else row["concise_slot"]
                self.assertEqual(row["preferred_slot"], expected)

    def test_no_group_or_domain_crosses_the_holdout_boundary(self):
        contexts = {}
        answers = {}
        for group, rows in self.groups.items():
            with self.subTest(group=group):
                self.assertEqual(len({row["split"] for row in rows}), 1)
                representative = rows[0]
                domain = representative["domain"]
                if representative["split"] == "test":
                    self.assertIn(domain, HELD_OUT)
                else:
                    self.assertIn(domain, IN_DOMAIN)
                # Detect exact scenario/answer reuse independently of group names.
                # Semantic task-family separation additionally needs human review.
                state = representative["neutral"]["state"]
                self.assertNotIn(state, contexts)
                contexts[state] = group
                for answer in representative["neutral"]["questions"]["preference"]["criteria"].values():
                    self.assertNotIn(answer, answers)
                    answers[answer] = group

    def test_order_and_preference_labels_are_balanced_within_splits(self):
        for split in ("train", "dev", "test"):
            rows = [row for row in self.rows if row["split"] == split]
            self.assertEqual(Counter(row["concise_slot"] for row in rows), {0: len(rows) // 2, 1: len(rows) // 2})
            self.assertEqual(Counter(row["preferred_slot"] for row in rows), {0: len(rows) // 2, 1: len(rows) // 2})


if __name__ == "__main__":
    unittest.main()
