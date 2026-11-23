import pytest
import torch

from glide_recurrent_sliders import RecurrentChoice, write_block17_development_plan
from glide_recurrent_sliders import identity
from particle_sliders.prompt_steering import PromptSteering

from test_recurrent_replay import FixtureModel, _digest, _replay_fixture, _write_json, fixture_request


class Block17Model(FixtureModel):
    route = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    )

    def decide(self, _state, questions):
        self.scoring_calls += 1
        self.outputs = []
        residual = torch.ones(1, 2, 4096, dtype=torch.bfloat16)
        residual[:, 0, 0] = -0.0
        pristine = residual[:, 0].view(torch.int16).clone()
        for layer in self.route:
            residual = self.network.backbone.layers[layer](residual)
            if layer == 17:
                self.outputs.append(residual[0, -1, 0].item())
            assert torch.equal(residual[:, 0].view(torch.int16), pristine)
        name, question = next(iter(questions.items()))
        a, b = question["criteria"]
        return dict(answers={name: dict(choice=a, probabilities={a: 0.75, b: 0.25})}, usage=dict(output_tokens=0))


def block17_scorer():
    model = Block17Model()
    steering = PromptSteering(4096, device="cpu")
    with torch.no_grad():
        steering.probe.weight.zero_()
        steering.probe.bias.zero_()
        steering.direction.weight.zero_()
        steering.probe.weight[0, 0] = steering.direction.weight[0, 0] = 1
    scorer = RecurrentChoice(model, steering, dict(schema=identity.BLOCK17_SCHEMA, raw_sha256="r"), "p", 1.0)
    return model, scorer


def test_block17_two_causal_fp32_edits_and_stock_off(monkeypatch):
    model, scorer = block17_scorer()
    request = fixture_request()
    einsum = torch.einsum
    calls = []

    def traced_outer(equation, amount, direction):
        assert equation == "bs,d->bsd"
        assert amount.dtype == direction.dtype == torch.float32 and amount.shape == (1, 2)
        calls.append(amount[0].tolist())
        return einsum(equation, amount, direction)

    monkeypatch.setattr(torch, "einsum", traced_outer)
    adaptive = scorer.score(request, mode="adaptive")["trace"]
    assert model.outputs == [2, 4]
    assert calls == [[0.0, 1.0], [0.0, 2.0]]
    assert adaptive["visit_clocks"] == [18, 24] and adaptive["resolved_coefficients"] == [1.0, 2.0]
    matched = scorer.score(request, mode="matched", reference=adaptive)["trace"]
    assert model.outputs == [2.5, 4]
    assert matched["resolved_coefficients"] == [1.5, 1.5]
    assert matched["binary32_dose"] == dict(adaptive_dose="3", constant_dose="3", constant_minus_adaptive="0",
                                            exact=True, side="exact", requested_control="nearest")
    assert scorer.score(request, mode="off")["trace"]["resolved_coefficients"] == [0.0, 0.0]
    assert model.outputs == [1, 1]
    assert calls == [[0.0, 1.0], [0.0, 2.0], [0.0, 1.5], [0.0, 1.5]]
    with pytest.raises(ValueError, match="another request or replay"):
        scorer.score(request, mode="matched", reference=dict(adaptive, visit_clocks=[30, 36]))
    assert all(not layer._forward_hooks for layer in model.network.backbone.layers)
    assert not torch.cuda.is_initialized()


def test_block17_refuses_missing_or_reordered_route_and_removes_hooks():
    model, scorer = block17_scorer()
    route = model.route
    model.route = route[:-1]
    with pytest.raises(RuntimeError, match="did not complete"):
        scorer.score(fixture_request(), mode="off")
    assert all(not layer._forward_hooks for layer in model.network.backbone.layers)
    model.route = (*route[:18], 13, *route[19:])
    with pytest.raises(RuntimeError, match="left the registered"):
        scorer.score(fixture_request(), mode="adaptive")
    assert all(not layer._forward_hooks for layer in model.network.backbone.layers)


def test_block17_both_inexact_binary32_bracket_sides_have_exact_residuals():
    _, scorer = block17_scorer()
    request = fixture_request()
    adaptive = scorer.score(request, mode="adaptive")["trace"]
    reference = dict(adaptive, resolved_coefficients=[1.0, 1.0 + 2**-23])
    nearest = scorer.score(request, mode="matched", reference=reference)["trace"]
    lower = scorer.score(request, mode="matched", reference=reference, match_side="lower")["trace"]
    upper = scorer.score(request, mode="matched", reference=reference, match_side="upper")["trace"]
    assert nearest["resolved_coefficients"] == lower["resolved_coefficients"] == [1.0, 1.0]
    assert upper["resolved_coefficients"] == [1.0 + 2**-23] * 2
    assert nearest["binary32_dose"] == dict(adaptive_dose="16777217/8388608", constant_dose="2",
                                            constant_minus_adaptive="-1/8388608", exact=False,
                                            side="lower", requested_control="nearest")
    assert lower["binary32_dose"]["constant_minus_adaptive"] == "-1/8388608"
    assert upper["binary32_dose"] == dict(adaptive_dose="16777217/8388608", constant_dose="8388609/4194304",
                                          constant_minus_adaptive="1/8388608", exact=False, side="upper",
                                          requested_control="upper")
    broken = dict(adaptive, resolved_coefficients=[1.0, 1.0 + 2**-24])
    with pytest.raises(ValueError, match="resolved binary32"):
        scorer.score(request, mode="matched", reference=broken)


def test_block17_serializer_binds_tied_evidence_without_cross_accepting_other_site(tmp_path, monkeypatch):
    plan_file, block23, raw, full, receipt, calibration = _replay_fixture(tmp_path)
    evidence = tmp_path / "development.json"
    strengths = [7.5, 8.75]
    evidence_sha = _write_json(evidence, dict(
        schema="glide-block17-development-evidence-v1", held_out_cases=0,
        fit=dict(raw_checkpoint_sha256=_digest(raw), full_checkpoint_sha256=_digest(full),
                 full_checkpoint_receipt_sha256=_digest(receipt), ema_decay=None, completed_updates=80),
        development_ranking=dict(selected_strength=None, identical_adaptive_choices_across_strengths=True,
                                 co_leading_strengths=strengths, verification_sha256=identity.BLOCK17_RANK_SHA256),
    ))
    for attribute, digest in (
        ("BLOCK17_RAW_SHA256", _digest(raw)), ("BLOCK17_FULL_SHA256", _digest(full)),
        ("BLOCK17_RECEIPT_SHA256", _digest(receipt)), ("BLOCK17_FIT_SHA256", block23["fit_recipe_sha256"]),
        ("BLOCK17_DEVELOPMENT_SHA256", evidence_sha),
    ):
        monkeypatch.setattr(identity, attribute, digest)
    paths = dict(raw_path=raw, full_path=full, full_receipt_path=receipt, development_evidence_path=evidence)
    with pytest.raises(ValueError, match="verified tied development trial"):
        write_block17_development_plan(plan_file, trial_strength=0.5, **paths)
    assert not plan_file.exists()
    digest = write_block17_development_plan(plan_file, trial_strength=7.5, **paths)
    plan, trial, _ = identity.load_replay(plan_file, digest, **paths)
    assert plan["identity"] == identity.model_identity(block=17)
    assert plan["selected_strength"] is None and trial == 7.5
    with pytest.raises(FileExistsError):
        write_block17_development_plan(plan_file, trial_strength=7.5, **paths)
    with pytest.raises(ValueError, match="without a selected calibration"):
        identity.load_replay(plan_file, digest, raw, full, receipt, calibration)
    for modified in (dict(plan, selected_strength=7.5), dict(plan, identity=identity.model_identity())):
        modified_sha = _write_json(plan_file, modified)
        with pytest.raises(ValueError, match="reviewed fit, identity and no selected strength"):
            identity.load_replay(plan_file, modified_sha, **paths)
    block23_digest = _write_json(plan_file, block23)
    with pytest.raises(ValueError, match="without development evidence"):
        identity.load_replay(plan_file, block23_digest, **paths)
    with pytest.raises(ValueError, match="Block 17 fit requires a development replay plan"):
        identity.load_replay(plan_file, block23_digest, raw, full, receipt, calibration)
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(ValueError, match="development evidence digest"):
        write_block17_development_plan(tmp_path / "changed.json", trial_strength=7.5, **paths)
