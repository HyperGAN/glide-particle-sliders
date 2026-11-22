from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from glide_recurrent_sliders import RecurrentChoice
from glide_recurrent_sliders.identity import decode_json, load_replay, model_identity, verify_runtime
from particle_sliders.prompt_steering import PromptSteering


class FixtureModel:
    """Exercise hook order on CPU without importing publisher code or weights."""

    def __init__(self, passes=2):
        self.passes = passes
        self.outputs = []
        self.scoring_calls = 0
        self.network = SimpleNamespace(backbone=SimpleNamespace(
            recurrence=dict(blocks=((12, 17), (18, 23)), passes=2, strategy="euler",
                            conv="A", gamma=1, n_gate=0, n_lora=0),
            config=SimpleNamespace(hidden_size=4096, num_hidden_layers=32),
            layers=[nn.Identity() for _ in range(32)],
        ))

    def _prepare(self, request):
        name, question = next(iter(request["questions"].items()))
        return [(name, question["type"], list(question["criteria"]), "prepared")], None

    def decide(self, _state, questions):
        self.scoring_calls += 1
        self.outputs = []
        residual = torch.ones(1, 2, 4096, dtype=torch.bfloat16)
        pristine = residual[:, 0].clone()
        for _ in range(self.passes):
            residual = self.network.backbone.layers[23](residual)
            self.outputs.append(residual[0, -1, 0].item())
            assert torch.equal(residual[:, 0], pristine)
        name, question = next(iter(questions.items()))
        a, b = question["criteria"]
        return dict(answers={name: dict(choice=a, probabilities={a: 0.75, b: 0.25})}, usage=dict(output_tokens=0))


def fixture_request():
    return dict(state="state", questions={"decision": dict(type="choice", criteria={"a": "First", "b": "Second"})})


def test_resolved_two_passes_and_matched_replay():
    model = FixtureModel()
    steering = PromptSteering(4096, device="cpu")
    with torch.no_grad():
        steering.probe.weight.zero_()
        steering.probe.bias.zero_()
        steering.direction.weight.zero_()
        steering.probe.weight[0, 0] = steering.direction.weight[0, 0] = 1
    scorer = RecurrentChoice(model, steering, dict(raw_sha256="r"), "p", 1.0)
    request = fixture_request()
    adaptive = scorer.score(request, mode="adaptive")
    assert model.outputs == [2, 4]
    assert adaptive["trace"]["resolved_coefficients"] == [1.0, 2.0]
    matched = scorer.score(request, mode="matched", reference=adaptive["trace"])
    assert model.outputs == [2.5, 4]
    assert matched["trace"]["resolved_coefficients"] == [1.5, 1.5]
    off = scorer.score(request, mode="off")
    assert model.outputs == [1, 1] and off["trace"]["resolved_coefficients"] == [0.0, 0.0]
    assert not model.network.backbone.layers[23]._forward_hooks
    with pytest.raises(ValueError, match="another request"):
        scorer.score(dict(request, state="different"), mode="matched", reference=adaptive["trace"])
    reference = dict(adaptive["trace"], resolved_coefficients=[1., 1.0000000000000002])
    with pytest.raises(ValueError, match="no exact binary64"):
        scorer.score(request, mode="matched", reference=reference)
    assert model.scoring_calls == 3
    fractional = dict(adaptive["trace"], resolved_coefficients=[1.0, 1.0 + 2**-23])
    exact = scorer.score(request, mode="matched", reference=fractional)
    assert exact["trace"]["resolved_coefficients"] == [1.0 + 2**-24] * 2


def test_rejects_request_or_native_shape_before_reusing_hook():
    model = FixtureModel(passes=1)
    steering = PromptSteering(4096, device="cpu")
    scorer = RecurrentChoice(model, steering, dict(raw_sha256="r"), "p", 1.0)
    actual = fixture_request()
    with pytest.raises(ValueError, match="text-only"):
        scorer.score(dict(actual, images=["x"]), mode="adaptive")
    with pytest.raises(ValueError, match="two distinct"):
        scorer.score(dict(actual, questions={"q": dict(type="choice", criteria={"a": "only"})}), mode="adaptive")
    assert model.scoring_calls == 0
    with pytest.raises(RuntimeError, match="both recurrent passes"):
        scorer.score(actual, mode="off")
    assert not model.network.backbone.layers[23]._forward_hooks
    with pytest.raises(ValueError, match="Repeated JSON"):
        decode_json(b'{"questions":{},"questions":{}}')


def _write_json(path, value):
    data = json.dumps(value, allow_nan=False).encode()
    path.write_bytes(data)
    return sha256(data).hexdigest()


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _replay_fixture(root):
    sample = PromptSteering(4096, device="cpu").state_dict()
    experiment = {"sha256": "1" * 64}
    full_file = root / "full.pt"
    cursor = dict(completed=80, epoch=5, next_order_index=0, epoch_means=[1.] * 5, stop_reason="declared_budget")
    moments = {index: dict(step=torch.tensor(80.), exp_avg=torch.zeros_like(value), exp_avg_sq=torch.zeros_like(value))
               for index, value in enumerate(sample[key] for key in ("direction.weight", "probe.weight", "probe.bias"))}
    optimizer = dict(param_groups=[dict(params=[0, 1, 2], lr=0.)], state=moments)
    state = dict(
        experiment=experiment, fitted_probe=sample, optimizer=optimizer, scheduler=dict(last_epoch=80, _step_count=81, _last_lr=[0.]),
        cpu_rng=torch.zeros(1, dtype=torch.uint8), cuda_rng=torch.zeros(1, dtype=torch.uint8),
        shuffle_state=torch.zeros(1, dtype=torch.uint8), orders_sha256=experiment["sha256"],
        losses=[1.] * 80, cursor=cursor, raw_weights=True, ema_decay=None, mutation_marker=dict(sha256="2" * 64),
    )
    torch.save(state, full_file)
    receipt_file = root / "receipt.json"
    receipt = dict(state=dict(sha256=_digest(full_file)), experiment=experiment, cursor=cursor, update=80,
                   mutation_marker=state["mutation_marker"], ema_decay=None)
    receipt_sha = _write_json(receipt_file, receipt)
    raw_file = root / "raw.pt"
    raw = dict(experiment=experiment, full=dict(sha256=receipt_sha), completed=80, probe=sample,
               raw_weights=True, ema_decay=None)
    torch.save(raw, raw_file)
    calibration_file = root / "calibration.json"
    calibration = dict(schema="glide-recurrent-calibration-v1", raw_sha256=_digest(raw_file),
                       selected_strength=1.0, selection_metric="mean_matched_intended_effect_on_minus_off")
    calibration_sha = _write_json(calibration_file, calibration)
    plan_file = root / "plan.json"
    plan = dict(schema="glide-recurrent-replay-v1", identity=model_identity(), raw_sha256=_digest(raw_file),
                full_sha256=_digest(full_file), fit_recipe_sha256=experiment["sha256"], completed_updates=80,
                calibration_sha256=calibration_sha, ema_decay=None)
    return plan_file, plan, raw_file, full_file, receipt_file, calibration_file


def test_only_digest_bound_raw_and_full_replay(tmp_path):
    plan_file, plan, raw, full, receipt, calibration = _replay_fixture(tmp_path)
    digest = _write_json(plan_file, plan)
    loaded, strength, probe = load_replay(plan_file, digest, raw, full, receipt, calibration)
    assert loaded == plan and strength == 1.0 and len(probe) == 3
    with pytest.raises(ValueError, match="plan digest"):
        load_replay(plan_file, "3" * 64, raw, full, receipt, calibration)

    incomplete = _write_json(plan_file, dict(plan, completed_updates=20))
    with pytest.raises(ValueError, match="completed author fit"):
        load_replay(plan_file, incomplete, raw, full, receipt, calibration)
    ema = _write_json(plan_file, dict(plan, ema_decay=.9))
    with pytest.raises(ValueError, match="weight policy"):
        load_replay(plan_file, ema, raw, full, receipt, calibration)

    calibration.write_bytes(calibration.read_bytes() + b" ")
    digest = _write_json(plan_file, plan)
    with pytest.raises(ValueError, match="calibration digest"):
        load_replay(plan_file, digest, raw, full, receipt, calibration)


def test_pinned_runtime_and_packaged_manifest():
    verify_runtime()
    identity = model_identity()
    assert identity["model_id"] == "fastino/GLiDE-9B"
    assert identity["asset_manifest_sha256"] == "010bbd6644fe012721aa8d48d757acf58a4cc5b044a241aa6db3c2810a8d2048"
    assert not torch.cuda.is_initialized()
