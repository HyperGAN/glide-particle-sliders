"""Fresh-v3 contracts; these tests never load or evaluate a model."""

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
BUILDER_PATH = PROJECT / "scripts" / "build_fresh_dataset_v3.py"
SPEC = importlib.util.spec_from_file_location("fresh_dataset_builder_v3", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
FROZEN_SHA256 = "995871d69e3c1b7e0b8ee052d5e2fa68c9a472f8455408d39ba74cf2bf1b9059"
PREFERENCE_DOMAINS = {"natural_history_records", "audio_editing_briefs"}
UNRELATED_DOMAINS = {"ticket_intent", "binary_routing", "entailed_facts"}
ROW_KEYS = {
    "id", "group", "split", "domain", "concise_slot", "preferred_slot",
    "control", "neutral", "positive",
}


class FreshDatasetV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = BUILDER.DEFAULT_OUTPUT.read_bytes()
        cls.dataset = json.loads(cls.payload)
        cls.rows = cls.dataset["rows"]
        cls.groups = defaultdict(list)
        for row in cls.rows:
            cls.groups[row["group"]].append(row)

    def test_frozen_bytes_and_builder_agree(self):
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), FROZEN_SHA256)
        self.assertEqual(self.dataset, BUILDER.build_dataset())
        regenerated = (json.dumps(BUILDER.build_dataset(), ensure_ascii=False, indent=2)+"\n").encode("utf-8")
        self.assertEqual(self.payload, regenerated)
        self.assertEqual(set(self.dataset), {"schema", "rows"})
        self.assertEqual(self.dataset["schema"], "glide-concise-pairs-v1")

    def test_exact_preference_and_unrelated_counts(self):
        self.assertEqual(len(self.rows), 104)
        self.assertEqual(len(self.groups), 52)
        self.assertEqual(len({row["id"] for row in self.rows}), 104)
        self.assertEqual(Counter(row["split"] for row in self.rows), {"fresh": 80, "unrelated": 24})
        representatives = [rows[0] for rows in self.groups.values()]
        fresh = [row for row in representatives if row["split"] == "fresh"]
        unrelated = [row for row in representatives if row["split"] == "unrelated"]
        self.assertEqual(Counter(row["domain"] for row in fresh), {domain: 20 for domain in PREFERENCE_DOMAINS})
        self.assertEqual(Counter(row["domain"] for row in fresh if not row["control"]), {domain: 16 for domain in PREFERENCE_DOMAINS})
        self.assertEqual(Counter(row["domain"] for row in fresh if row["control"]), {domain: 4 for domain in PREFERENCE_DOMAINS})
        self.assertEqual(Counter(row["domain"] for row in unrelated), {domain: 4 for domain in UNRELATED_DOMAINS})

    def test_native_request_and_row_contract(self):
        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(set(row), ROW_KEYS)
                self.assertIn(row["split"], {"fresh", "unrelated"})
                self.assertIs(type(row["control"]), bool)
                for key in ("preferred_slot", "concise_slot"):
                    self.assertIs(type(row[key]), int)
                    self.assertIn(row[key], (0, 1))
                for arm in ("neutral", "positive"):
                    request = row[arm]
                    self.assertEqual(set(request), {"state", "questions"})
                    self.assertIsInstance(request["state"], str)
                    self.assertTrue(request["state"].strip())
                    self.assertEqual(set(request["questions"]), {"preference"})
                    question = request["questions"]["preference"]
                    self.assertEqual(set(question), {"type", "instructions", "criteria"})
                    self.assertEqual(question["type"], "choice")
                    self.assertEqual(list(question["criteria"]), ["first", "second"])
                    self.assertTrue(all(isinstance(value, str) and value.strip() for value in question["criteria"].values()))

    def test_preference_instruction_only_difference_and_labels(self):
        for row in self.rows:
            if row["split"] != "fresh":
                continue
            with self.subTest(row=row["id"]):
                normalized = deepcopy(row["positive"])
                normalized["questions"]["preference"]["instructions"] = BUILDER.NEUTRAL_INSTRUCTIONS
                self.assertEqual(normalized, row["neutral"])
                self.assertEqual(row["neutral"]["questions"]["preference"]["instructions"], BUILDER.NEUTRAL_INSTRUCTIONS)
                self.assertEqual(row["positive"]["questions"]["preference"]["instructions"], BUILDER.POSITIVE_INSTRUCTIONS)
                answers = list(row["neutral"]["questions"]["preference"]["criteria"].values())
                concise = len(answers[row["concise_slot"]].split())
                verbose = len(answers[1-row["concise_slot"]].split())
                self.assertGreaterEqual(concise, 20)
                self.assertLess(concise, verbose)
                self.assertLessEqual(verbose, 60)
                expected = 1-row["concise_slot"] if row["control"] else row["concise_slot"]
                self.assertEqual(row["preferred_slot"], expected)

    def test_unrelated_requests_have_no_teacher_edit(self):
        source = {item["group"]: item for item in BUILDER.UNRELATED_CASES}
        for row in self.rows:
            if row["split"] != "unrelated":
                continue
            with self.subTest(row=row["id"]):
                self.assertEqual(row["neutral"], row["positive"])
                self.assertTrue(row["control"])
                self.assertNotEqual(row["neutral"]["questions"]["preference"]["instructions"], BUILDER.POSITIVE_INSTRUCTIONS)
                answers = list(row["neutral"]["questions"]["preference"]["criteria"].values())
                item = source[row["group"]]
                self.assertEqual(answers[row["preferred_slot"]], item["options"][item["correct"]])
                # Description length is metadata, never the factual truth label.
                self.assertLessEqual(len(answers[row["concise_slot"]].split()), len(answers[1-row["concise_slot"]].split()))

    def test_both_orders_reverse_options_and_truth_labels(self):
        for group, rows in self.groups.items():
            with self.subTest(group=group):
                self.assertEqual(len(rows), 2)
                a, b = rows
                self.assertEqual(a["split"], b["split"])
                self.assertEqual(a["domain"], b["domain"])
                self.assertEqual(a["control"], b["control"])
                self.assertEqual(a["concise_slot"], 1-b["concise_slot"])
                self.assertEqual(a["preferred_slot"], 1-b["preferred_slot"])
                for arm in ("neutral", "positive"):
                    self.assertEqual(a[arm]["state"], b[arm]["state"])
                    left = list(a[arm]["questions"]["preference"]["criteria"].values())
                    right = list(b[arm]["questions"]["preference"]["criteria"].values())
                    self.assertEqual(left, right[::-1])
                    self.assertNotEqual(left[0], left[1])
        self.assertEqual(Counter(row["preferred_slot"] for row in self.rows), {0: 52, 1: 52})

    def test_preference_domains_and_sources_are_new(self):
        # Exact reuse can be detected automatically; task-family novelty also
        # requires the manual, model-blind authoring review described by builder.
        prior = []
        for name in ("concise-preference-v1.json", "concise-fresh-v2.json"):
            prior.extend(json.loads((PROJECT / "data" / name).read_text())["rows"])
        self.assertTrue(PREFERENCE_DOMAINS.isdisjoint({row["domain"] for row in prior}))
        self.assertTrue(set(self.groups).isdisjoint({row["group"] for row in prior}))
        old_states = {row["neutral"]["state"] for row in prior}
        old_answers = {answer for row in prior for answer in row["neutral"]["questions"]["preference"]["criteria"].values()}
        seen_states, seen_fresh_answers = set(), set()
        for rows in self.groups.values():
            representative = rows[0]
            state = representative["neutral"]["state"]
            self.assertNotIn(state, old_states)
            self.assertNotIn(state, seen_states)
            seen_states.add(state)
            if representative["split"] == "fresh":
                for answer in representative["neutral"]["questions"]["preference"]["criteria"].values():
                    self.assertNotIn(answer, old_answers)
                    self.assertNotIn(answer, seen_fresh_answers)
                    seen_fresh_answers.add(answer)

    def test_builder_preserves_different_existing_bytes(self):
        with tempfile.TemporaryDirectory(prefix="glide-fresh3-contract-") as directory:
            output = Path(directory) / "fresh.json"
            command = [sys.executable, str(BUILDER_PATH), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(output.read_bytes(), self.payload)
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            output.write_bytes(b"different-frozen-data\n")
            denied = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("Fresh dataset is frozen", denied.stderr)
            self.assertEqual(output.read_bytes(), b"different-frozen-data\n")


if __name__ == "__main__":
    unittest.main()
