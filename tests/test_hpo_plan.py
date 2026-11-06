"""Search declaration/provenance tests; never launch or run an HPO arm."""
from copy import deepcopy
import importlib.util
from itertools import product
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hpo_controller", ROOT / "scripts/run_hpo_v5.py")
hpo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hpo)
WORKER_SPEC = importlib.util.spec_from_file_location("hpo_worker", ROOT / "scripts/hpo_worker_v5.py")
# The executable worker imports its sibling in the normal script execution path.
import sys
sys.modules["run_hpo_v5"] = hpo
worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker)


def fixture_plan():
    plan = dict(schema="glide-fixed-hpo-v5", project=str(ROOT),
        candidates=hpo.candidate_specs(ROOT), resource_policy=hpo.POLICY.copy(),
        selection=hpo.CRITERION.copy(), budget=dict(arms=13, reused_arms=1,
            new_arms=12, new_unique_updates=6144, reused_unique_updates=512, horizon=512))
    plan["plan_sha256"] = hpo.digest(plan)
    return plan


def test_independent_absolute_rate_cartesian_product():
    candidates = hpo.candidate_specs(ROOT)
    actual = {(c["base_rates"]["adapter"], c["base_rates"]["critic"], c["base_rates"]["bank"])
              for c in candidates if c["config"]["options"]["batch"] == 8}
    assert actual == set(product((.0003, .001, .003), (.003, .018), (.003, .03)))
    assert len(actual) == 12
    for candidate in candidates:
        options = candidate["config"]["options"]
        assert math.isclose(options["lr"] * options["d_lr_mult"], candidate["base_rates"]["critic"], rel_tol=1e-14)
        assert math.isclose(options["lr"] * options["prior_lr_mult"], candidate["base_rates"]["bank"], rel_tol=1e-14)


def test_exact_single_anchor_reuse_and_unique_budget():
    candidates = hpo.candidate_specs(ROOT)
    reuse = [c for c in candidates if c["reuse"]]
    assert len(reuse) == 1
    assert reuse[0]["base_rates"] == dict(adapter=.001, critic=.018, bank=.03)
    assert reuse[0]["config"]["options"]["batch"] == 8
    assert reuse[0]["config"]["options"]["d_lr_mult"] == 18.
    assert reuse[0]["config"]["options"]["prior_lr_mult"] == 30.
    assert len(candidates) == 13
    assert sum(not c["reuse"] for c in candidates) == 12
    assert sum(c["new_unique_updates"] for c in candidates) == 6144
    batch32 = [c for c in candidates if c["config"]["options"]["batch"] == 32]
    assert len(batch32) == 1 and not batch32[0]["reuse"]
    assert batch32[0]["base_rates"] == reuse[0]["base_rates"]
    for candidate in candidates:
        options = candidate["config"]["options"]
        assert (options["seed"], options["steps"], options["device"], options["ema_decay"], options["critic"]) == (7, 512, "cpu", None, "residual64")
        assert options["cache_file"] == str((ROOT / hpo.CACHE).resolve())


@pytest.mark.parametrize("field,value", [("seed", 11), ("steps", 128), ("batch", 32),
    ("critic", "conditional16"), ("cache_file", "/tmp/other.pt")])
def test_configuration_drift_cannot_claim_exact_reuse(field, value):
    plan = fixture_plan()
    next(c for c in plan["candidates"] if c["reuse"])["config"]["options"][field] = value
    plan["plan_sha256"] = hpo.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="candidate grid"):
        hpo.validate_plan(plan)


def test_plan_budget_and_cpu_policy_are_binding():
    plan = fixture_plan()
    hpo.validate_plan(plan)
    for key, value in (("max_workers", 3), ("early_stopping", True), ("torch_threads_per_worker", 8),
                       ("automatic_retries", 1), ("time_limit_seconds", 60)):
        altered = deepcopy(plan)
        altered["resource_policy"][key] = value
        altered["plan_sha256"] = hpo.digest({k: v for k, v in altered.items() if k != "plan_sha256"})
        with pytest.raises(ValueError, match="resource"):
            hpo.validate_plan(altered)


def test_immutable_plan_can_only_be_recreated_identically(tmp_path):
    path = tmp_path / "plan.json"
    plan = fixture_plan()
    hpo.write_json(path, plan, immutable=True)
    original = path.read_bytes()
    hpo.write_json(path, plan, immutable=True)
    assert path.read_bytes() == original
    plan["budget"]["horizon"] = 256
    with pytest.raises(ValueError, match="immutable plan"):
        hpo.write_json(path, plan, immutable=True)
    assert path.read_bytes() == original


def test_sole_selection_metric_ignores_diagnostics_and_partial_runs():
    results = [dict(index=0, status="completed", completed_steps=512, dev=dict(mean=.8),
                    train=dict(mean=99), dev_preference=dict(main_accuracy=0)),
               dict(index=1, status="completed", completed_steps=512, dev=dict(mean=.9),
                    train=dict(mean=0), dev_preference=dict(main_accuracy=1)),
               dict(index=2, status="completed", completed_steps=256, dev=dict(mean=0)),
               dict(index=3, status="failed", completed_steps=512, dev=dict(mean=0)),
               dict(index=4, status="completed", completed_steps=512, dev=dict(mean=float("nan")))]
    assert hpo.choose_winner(results)["index"] == 0
    # Equal SSE is resolved only by the predeclared index.
    results.append(dict(index=5, status="reused", completed_steps=512, dev=dict(mean=.8)))
    assert hpo.choose_winner(results)["index"] == 0


def test_optimizer_logging_uses_actual_group_base_and_scheduled_rates():
    bank, adapter, critic = object(), object(), object()
    game = SimpleNamespace(backend=SimpleNamespace(particles=bank),
        opt_g=SimpleNamespace(param_groups=[dict(params=[adapter], lr=.0005), dict(params=[bank], lr=.015)]),
        opt_d=SimpleNamespace(param_groups=[dict(params=[critic], lr=.009)]),
        base_lrs=[[.001, .03], [.018]])
    result = worker.optimizer_rates(game)
    assert [(r["role"], r["base_lr"], r["scheduled_lr"]) for r in result] == [
        ("adapter", .001, .0005), ("bank", .03, .015), ("critic", .018, .009)]


def test_registry_derivative_allows_only_four_exact_metadata_entries(tmp_path):
    # Exact historical metadata, kept as text so a weight-free checkout can test
    # the source-derivative rule without private/local experiment artifacts.
    original = ROOT / "tests/fixtures/historical-pyproject-v1.toml"
    archived = tmp_path / "artifacts/source-v2-label-probe/pyproject.toml"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(original.read_bytes())
    current = tmp_path / "pyproject.toml"
    current.write_bytes((ROOT / "pyproject.toml").read_bytes())
    receipt = hpo.registry_derivative(tmp_path, hpo.file_sha(archived), hpo.file_sha(current))
    assert len(receipt["exact_toml_equality_after_removing"]) == 4
    current.write_text(current.read_text().replace('version = "0.1.0"', 'version = "99.0.0"'))
    # Assert the test actually introduced a separate metadata change.
    if current.read_bytes() == (ROOT / "pyproject.toml").read_bytes():
        current.write_text(current.read_text() + '\n[unapproved]\nchanged = true\n')
    with pytest.raises(ValueError, match="beyond four"):
        hpo.registry_derivative(tmp_path, hpo.file_sha(archived), hpo.file_sha(current))
