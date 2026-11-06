"""Verify the frozen fresh dataset without evaluating any model."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
BUILDER_PATH = PROJECT / "scripts" / "build_fresh_dataset_v2.py"
SPEC = importlib.util.spec_from_file_location("fresh_dataset_builder_v2", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
FROZEN_SHA256 = "7f3a247377d59faf537a3329bf2908183d9962b05805dd65afdfcd29aef2bdfe"
DOMAINS = {"arts_crafts", "recreation_game_rules"}
ROW_KEYS = {
    "id", "group", "split", "domain", "concise_slot", "preferred_slot",
    "control", "neutral", "positive",
}


class FreshDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = BUILDER.DEFAULT_OUTPUT.read_bytes()
        cls.dataset = json.loads(cls.payload)
        cls.rows = cls.dataset["rows"]
        cls.groups = defaultdict(list)
        for row in cls.rows:
            cls.groups[row["group"]].append(row)

    def test_frozen_bytes_and_deterministic_builder_agree(self):
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), FROZEN_SHA256)
        self.assertEqual(self.dataset, BUILDER.build_dataset())
        regenerated = (json.dumps(BUILDER.build_dataset(), ensure_ascii=False, indent=2)+"\n").encode("utf-8")
        self.assertEqual(self.payload, regenerated)
        self.assertEqual(set(self.dataset), {"schema", "rows"})
        self.assertEqual(self.dataset["schema"], "glide-concise-pairs-v1")

    def test_exact_new_domain_and_quality_counts(self):
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(len(self.groups), 40)
        self.assertEqual(len({row["id"] for row in self.rows}), 80)
        self.assertEqual({row["split"] for row in self.rows}, {"fresh"})
        representatives = [rows[0] for rows in self.groups.values()]
        self.assertEqual(Counter(row["domain"] for row in representatives), {domain: 20 for domain in DOMAINS})
        self.assertEqual(Counter(row["domain"] for row in representatives if not row["control"]), {domain: 16 for domain in DOMAINS})
        self.assertEqual(Counter(row["domain"] for row in representatives if row["control"]), {domain: 4 for domain in DOMAINS})

    def test_requests_differ_only_in_question_instructions(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(set(row), ROW_KEYS)
                self.assertIs(type(row["control"]), bool)
                self.assertIs(type(row["concise_slot"]), int)
                self.assertIs(type(row["preferred_slot"]), int)
                self.assertIn(row["concise_slot"], (0, 1))
                self.assertIn(row["preferred_slot"], (0, 1))
                for arm, instruction in (("neutral", BUILDER.NEUTRAL_INSTRUCTIONS), ("positive", BUILDER.POSITIVE_INSTRUCTIONS)):
                    request = row[arm]
                    self.assertEqual(set(request), {"state", "questions"})
                    self.assertIsInstance(request["state"], str)
                    self.assertTrue(request["state"].strip())
                    self.assertEqual(set(request["questions"]), {"preference"})
                    question = request["questions"]["preference"]
                    self.assertEqual(set(question), {"type", "instructions", "criteria"})
                    self.assertEqual(question["type"], "choice")
                    self.assertEqual(question["instructions"], instruction)
                    self.assertEqual(list(question["criteria"]), ["first", "second"])
                normalized = deepcopy(row["positive"])
                normalized["questions"]["preference"]["instructions"] = BUILDER.NEUTRAL_INSTRUCTIONS
                self.assertEqual(normalized, row["neutral"])

    def test_each_group_has_exactly_reversed_options_and_labels(self):
        for group, rows in self.groups.items():
            with self.subTest(group=group):
                self.assertEqual(len(rows), 2)
                by_slot = {row["concise_slot"]: row for row in rows}
                self.assertEqual(set(by_slot), {0, 1})
                a, b = by_slot[0], by_slot[1]
                for field in ("group", "split", "domain", "control"):
                    self.assertEqual(a[field], b[field])
                self.assertEqual(a["preferred_slot"], 1-b["preferred_slot"])
                for arm in ("neutral", "positive"):
                    self.assertEqual(a[arm]["state"], b[arm]["state"])
                    left = list(a[arm]["questions"]["preference"]["criteria"].values())
                    right = list(b[arm]["questions"]["preference"]["criteria"].values())
                    self.assertEqual(left, right[::-1])
                    self.assertNotEqual(left[0], left[1])

    def test_length_and_control_label_contract(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                answers = list(row["neutral"]["questions"]["preference"]["criteria"].values())
                concise = len(answers[row["concise_slot"]].split())
                verbose = len(answers[1-row["concise_slot"]].split())
                self.assertGreaterEqual(concise, 20)
                self.assertLess(concise, verbose)
                self.assertLessEqual(verbose, 60)
                self.assertEqual(row["preferred_slot"], 1-row["concise_slot"] if row["control"] else row["concise_slot"])
        self.assertEqual(Counter(row["preferred_slot"] for row in self.rows), {0: 40, 1: 40})
        self.assertEqual(Counter(row["concise_slot"] for row in self.rows), {0: 40, 1: 40})

    def test_new_domains_and_no_exact_source_reuse(self):
        # This checks exact reuse; semantic task-family separation is also a
        # manual authoring/review requirement, not proven by string uniqueness.
        old = json.loads((PROJECT / "data" / "concise-preference-v1.json").read_text())["rows"]
        self.assertTrue(DOMAINS.isdisjoint({row["domain"] for row in old}))
        self.assertTrue(set(self.groups).isdisjoint({row["group"] for row in old}))
        old_states = {row["neutral"]["state"] for row in old}
        old_answers = {answer for row in old for answer in row["neutral"]["questions"]["preference"]["criteria"].values()}
        states, answers = set(), set()
        for rows in self.groups.values():
            representative = rows[0]
            state = representative["neutral"]["state"]
            self.assertNotIn(state, old_states)
            self.assertNotIn(state, states)
            states.add(state)
            for answer in representative["neutral"]["questions"]["preference"]["criteria"].values():
                self.assertNotIn(answer, old_answers)
                self.assertNotIn(answer, answers)
                answers.add(answer)

    def test_builder_never_overwrites_different_frozen_bytes(self):
        with tempfile.TemporaryDirectory(prefix="glide-fresh-contract-") as directory:
            output = Path(directory) / "fresh.json"
            command = [sys.executable, str(BUILDER_PATH), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(output.read_bytes(), self.payload)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            output.write_bytes(b"frozen-different-bytes\n")
            denied = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("Fresh dataset is frozen", denied.stderr)
            self.assertEqual(output.read_bytes(), b"frozen-different-bytes\n")


if __name__ == "__main__":
    unittest.main()
