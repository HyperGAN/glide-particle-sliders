"""Synthetic publisher-shaped CPU fixtures for the installed Site23 opt-in."""

import json
import os
import unittest
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import torch
from glide_recurrent_sliders import (
    FullSuffixChoice,
    site23_alignment_commitment,
    write_site23_full_suffix_replay,
)
from glide_recurrent_sliders import full_suffix as site23
from glide_recurrent_sliders.full_suffix import (
    ROUTE,
    VISITS,
    UnrepresentableConstant,
    _bits,
    receipt_digest,
)
from glide_recurrent_sliders.identity import verify_block17_dose_source
from particle_sliders.prompt_steering import PromptSteering
from torch import nn

DOSE_WHEEL = os.environ.get("GLIDE_SITE23_DOSE_WHEEL")


class TinyPublisher:
    """Expose only publisher metadata/scoring ABI backed by fake BF16 residuals."""

    def __init__(self, *, inputs=None, suffix=2):
        self.network = nn.Module()
        self.network.visual = None
        self.network.backbone = nn.Module()
        self.network.backbone.layers = nn.ModuleList(nn.Identity() for _ in range(32))
        self.network.backbone.recurrence = {
            "blocks": ((12, 17), (18, 23)),
            "passes": 2,
            "strategy": "euler",
            "conv": "A",
            "gamma": 1,
            "n_gate": 0,
            "n_lora": 0,
        }
        self.network.backbone.config = SimpleNamespace(
            hidden_size=4096, num_hidden_layers=32
        )
        self.network.readout = nn.Identity()
        self.network.eval()
        self.temperature = 1
        self.codes = tuple(range(255))
        self.max_len = 64
        self.inputs = inputs or {30: (0.0, 1.0), 36: (0.0, 0.0)}
        self.suffix = suffix
        self.decisions = 0
        self.route = ROUTE
        self.snapshots = {}
        self.last_readout = None
        self.corrupt_zero = False

    def _prepare(self, request):
        ((key, question),) = request["questions"].items()
        return [
            (key, question["type"], list(question["criteria"]), request["state"])
        ], None

    def tokenizer(self, prompts, *, add_special_tokens):
        assert add_special_tokens is False
        (prompt,) = prompts
        prefix, count = prompt.split(":")
        return {"input_ids": [[int(prefix), *range(20, 20 + int(count))]]}

    def decide(self, state, questions):
        self.decisions += 1
        assert torch.is_inference_mode_enabled()
        assert state == f"8:{self.suffix}"
        residual = torch.zeros((1, self.suffix + 1, 4096), dtype=torch.bfloat16)
        residual[0, :, 1] = -1
        residual[0, 0, 0] = -0.0
        residual[0, 0, 2] = 41.0
        prefix = residual[:, :1].clone()
        for call, block in enumerate(self.route, 1):
            if call in VISITS and self.inputs[call] is not None:
                residual[:, 1:, 0] = torch.tensor(self.inputs[call])
            residual = self.network.backbone.layers[block](residual)
            assert torch.equal(
                residual[:, :1].view(torch.int16), prefix.view(torch.int16)
            )
            if call in VISITS:
                self.snapshots[call] = residual.clone()
        readout = torch.zeros((1, 255), dtype=torch.bfloat16)
        readout[0, : self.suffix] = residual[0, 1:, 1]
        if self.corrupt_zero and self.decisions % 2 == 0:
            readout[0, -1] = -0.0
        self.last_readout = self.network.readout(readout).clone()
        (name,) = questions
        return {
            "answers": {
                name: {
                    "choice": "first",
                    "probabilities": {"first": 0.5, "second": 0.5},
                }
            },
            "usage": {"output_tokens": 0},
        }


def sealed_pair(positions):
    questions = {
        "choice": {
            "type": "choice",
            "criteria": {"first": "Option A", "second": "Option B"},
        }
    }
    return {"state": f"8:{positions}", "questions": questions}, {
        "state": f"9:{positions}",
        "questions": questions,
    }


def controller(weight=2.0**-23, bias=1.0):
    steering = PromptSteering(4096)
    with torch.no_grad():
        steering.probe.weight.zero_()
        steering.probe.weight[0, 0] = weight
        steering.probe.bias.fill_(bias)
        steering.direction.weight.zero_()
        steering.direction.weight[0, 1] = 1
    return steering.eval().requires_grad_(False)


@unittest.skipUnless(
    DOSE_WHEEL, "set GLIDE_SITE23_DOSE_WHEEL to the exact shared PR #338 or #339 wheel"
)
class FullSuffixBridge(unittest.TestCase):
    def bind(self, model=None, steering=None, strength=1.0):
        model = TinyPublisher() if model is None else model
        steering = controller() if steering is None else steering
        pair = sealed_pair(model.suffix)
        plan = site23._replay(site23_alignment_commitment(model, *pair), strength)
        adapter = FullSuffixChoice._from_verified_components(
            model,
            steering,
            plan,
            "2" * 64,
            site23._dose_source(DOSE_WHEEL),
        )
        self.assertEqual(model.decisions, 0)
        return model, adapter, pair

    def receipts(self, adapter, pair):
        parity = adapter.zero_parity(*pair)
        adaptive = adapter.adaptive(*pair)
        return {
            "adaptive": adaptive,
            "adaptive_sha256": receipt_digest(adaptive),
            "zero_parity": parity,
            "zero_parity_sha256": receipt_digest(parity),
        }

    def assert_hooks_removed(self, model):
        self.assertTrue(
            all(not block._forward_hooks for block in model.network.backbone.layers)
        )
        self.assertFalse(model.network.readout._forward_hooks)

    def test_exact_binary64_seam_and_both_visit_echoes(self):
        model, adapter, pair = self.bind()
        evidence = self.receipts(adapter, pair)
        self.assertEqual(
            evidence["adaptive"]["visits"][0]["post_strength_binary64_bits"],
            [_bits(1.0), _bits(1.0 + 2**-23)],
        )
        self.assertEqual(model.snapshots[30][0, -2:, 1].tolist(), [0, 2**-23])
        self.assertEqual(
            evidence["zero_parity"]["stock_readout_bf16_sha256"],
            evidence["zero_parity"]["zero_readout_bf16_sha256"],
        )
        result = adapter.exact_constant(*pair, **evidence)
        self.assertEqual([row["visit"] for row in result["visits"]], [30, 36])
        self.assertEqual(
            result["visits"][0]["post_strength_binary64_bits"], [_bits(1 + 2**-24)] * 2
        )
        self.assertEqual(
            result["visits"][1]["post_strength_binary64_bits"], [_bits(1)] * 2
        )
        self.assertEqual(model.snapshots[30][0, -2:, 1].tolist(), [2**-24] * 2)
        self.assertEqual([row["residual"] for row in result["doses"]], ["0", "0"])
        self.assertEqual(evidence["adaptive"]["changed_bf16_positions"], [2, 2])
        self.assertEqual(evidence["zero_parity"]["changed_bf16_positions"], [0, 0])
        self.assertEqual(result["changed_bf16_positions"], [2, 2])
        self.assertEqual(model.decisions, 4)
        self.assert_hooks_removed(model)

    def test_second_probe_observes_the_actual_first_adaptive_edit(self):
        model = TinyPublisher(inputs={30: (0.5, 1.0), 36: None})
        steering = controller(1, 0)
        with torch.no_grad():
            steering.direction.weight.zero_()
            steering.direction.weight[0, 0] = 1
        _, adapter, pair = self.bind(model, steering)
        evidence = self.receipts(adapter, pair)
        before, after = evidence["adaptive"]["visits"]
        self.assertEqual(
            before["post_strength_binary64_bits"], [_bits(0.5), _bits(1.0)]
        )
        self.assertEqual(after["post_strength_binary64_bits"], [_bits(1.0), _bits(2.0)])
        fixed = adapter.exact_constant(*pair, **evidence)
        self.assertEqual(
            [row["constant_binary64_bits"] for row in fixed["doses"]],
            [_bits(0.75), _bits(1.5)],
        )
        self.assertEqual(model.decisions, 4)

    def test_small_nonzero_coefficient_is_not_a_bf16_change(self):
        model, adapter, pair = self.bind(steering=controller(0.0, 2**-50))
        result = adapter.adaptive(*pair)
        self.assertEqual(result["changed_bf16_positions"], [0, 0])
        self.assertEqual(
            [row["post_strength_binary64_bits"] for row in result["visits"]],
            [[_bits(2**-50)] * 2] * 2,
        )
        self.assertEqual(model.decisions, 1)

    def test_second_visit_unrepresentable_stops_before_fixed_decide(self):
        model = TinyPublisher(inputs={30: (0.5, 1.0), 36: (1.0, 2.0**-60)})
        _, adapter, pair = self.bind(model, controller(1, 0))
        evidence = self.receipts(adapter, pair)
        with self.assertRaisesRegex(UnrepresentableConstant, r"visit\(s\) 36") as error:
            adapter.exact_constant(*pair, **evidence)
        self.assertEqual(
            [row["residual"] for row in error.exception.doses],
            ["0", "-1/1152921504606846976"],
        )
        self.assertEqual(model.decisions, 3)
        self.assert_hooks_removed(model)

    def test_odd_denominator_stops_before_fixed_decide(self):
        model = TinyPublisher(
            inputs={30: (1.0, 0.0, 0.0), 36: (1.0, 1.0, 1.0)}, suffix=3
        )
        _, adapter, pair = self.bind(model, controller(1, 0))
        evidence = self.receipts(adapter, pair)
        with self.assertRaisesRegex(UnrepresentableConstant, r"visit\(s\) 30"):
            adapter.exact_constant(*pair, **evidence)
        self.assertEqual(model.decisions, 3)
        self.assert_hooks_removed(model)

    def test_negative_strength_keeps_signed_dose(self):
        _, adapter, pair = self.bind(strength=-1.0)
        evidence = self.receipts(adapter, pair)
        result = adapter.exact_constant(*pair, **evidence)
        self.assertEqual(result["doses"][0]["adaptive_dose"], "-16777217/8388608")
        self.assertEqual(
            result["visits"][0]["post_strength_binary64_bits"], [_bits(-1 - 2**-24)] * 2
        )

    def test_scope_and_digest_mismatch_never_enters_fixed_forward(self):
        model, adapter, pair = self.bind()
        evidence = self.receipts(adapter, pair)
        modified = deepcopy(evidence["adaptive"])
        modified["visits"][0]["post_strength_binary64_bits"][0] = _bits(4)
        with self.assertRaisesRegex(ValueError, "digest-bound"):
            adapter.exact_constant(*pair, **dict(evidence, adaptive=modified))
        fixed = deepcopy(evidence["zero_parity"])
        fixed["binding"]["raw_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "digest-bound"):
            adapter.exact_constant(
                *pair,
                **dict(
                    evidence,
                    zero_parity=fixed,
                    zero_parity_sha256=receipt_digest(fixed),
                ),
            )
        changed = dict(pair[1], state="7:2")
        with self.assertRaisesRegex(ValueError, "registered role"):
            adapter.exact_constant(pair[0], changed, **evidence)
        self.assertEqual(model.decisions, 3)

    def test_device_preflight_detects_silent_fp32_narrowing_before_fixed_forward(self):
        model, adapter, pair = self.bind()
        evidence = self.receipts(adapter, pair)
        full = torch.full

        def accidentally_narrow(shape, value, *, dtype, device):
            rounded = torch.tensor(value, dtype=torch.float32).item()
            return full(shape, rounded, dtype=dtype, device=device)

        with (
            patch.object(torch, "full", accidentally_narrow),
            self.assertRaisesRegex(
                RuntimeError,
                "Device FP64 constant preflight",
            ),
        ):
            adapter.exact_constant(*pair, **evidence)
        self.assertEqual(model.decisions, 3)
        self.assert_hooks_removed(model)

    def test_fp64_only_bits_rejected_as_adaptive_reference(self):
        model, adapter, pair = self.bind()
        evidence = self.receipts(adapter, pair)
        evidence["adaptive"]["visits"][0]["post_strength_binary64_bits"][0] = _bits(
            1 + 2**-24
        )
        evidence["adaptive_sha256"] = receipt_digest(evidence["adaptive"])
        with self.assertRaisesRegex(ValueError, "requires resolved binary32"):
            adapter.exact_constant(*pair, **evidence)
        self.assertEqual(model.decisions, 3)

    def test_readout_parity_is_bitwise_including_signed_zero(self):
        model, adapter, pair = self.bind()
        model.corrupt_zero = True
        with self.assertRaisesRegex(RuntimeError, "differs from stock"):
            adapter.zero_parity(*pair)
        self.assertEqual(model.decisions, 2)
        self.assert_hooks_removed(model)

    def test_original_route_and_cleanup_are_required(self):
        model, adapter, pair = self.bind()
        model.route = ROUTE[:-1]
        with self.assertRaisesRegex(RuntimeError, "did not close"):
            adapter.adaptive(*pair)
        model.route = (1, *ROUTE[1:])
        with self.assertRaisesRegex(RuntimeError, "44-call"):
            adapter.adaptive(*pair)
        self.assert_hooks_removed(model)

    def test_shared_wheel_pin_is_verified_without_changing_block17(self):
        verify_block17_dose_source()
        site23._dose_source(DOSE_WHEEL)
        verify_block17_dose_source()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "modified.whl"
            with ZipFile(DOSE_WHEEL) as source, ZipFile(path, "w") as output:
                output.writestr(
                    site23.DOSE_MEMBER, b" " + source.read(site23.DOSE_MEMBER)[1:]
                )
            with self.assertRaisesRegex(ValueError, "reviewed shared dose module"):
                site23._dose_source(path)

    def test_public_open_and_registration_require_exact_trial_and_no_ema(self):
        model = TinyPublisher()
        alignment = site23_alignment_commitment(model, *sealed_pair(model.suffix))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "replay.json"
            with self.assertRaisesRegex(TypeError, "FullSuffixChoice.open"):
                FullSuffixChoice()
            with self.assertRaisesRegex(ValueError, "explicit finite"):
                write_site23_full_suffix_replay(
                    path, alignment_sha256=alignment, trial_strength=0.0
                )
            digest = write_site23_full_suffix_replay(
                path, alignment_sha256=alignment, trial_strength=0.75
            )
            with self.assertRaises(FileExistsError):
                write_site23_full_suffix_replay(
                    path, alignment_sha256=alignment, trial_strength=1.0
                )
            with (
                patch.object(
                    site23, "_raw_state", return_value=controller().state_dict()
                ),
                patch.object(
                    site23.identity, "verify_snapshot", return_value=Path(folder)
                ),
                patch.object(
                    site23.runtime, "_load_publisher", return_value=model
                ) as loader,
                patch.object(torch.cuda, "device_count", return_value=1),
                patch.object(torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(PromptSteering, "to", lambda probe, *_: probe),
            ):
                kwargs = {
                    "replay_file": path,
                    "raw_file": "unused-fixture",
                    "dose_core_wheel": DOSE_WHEEL,
                }
                with self.assertRaisesRegex(ValueError, "replay digest differs"):
                    FullSuffixChoice.open(folder, replay_sha256="3" * 64, **kwargs)
                adapter = FullSuffixChoice.open(folder, replay_sha256=digest, **kwargs)
                result = adapter.adaptive(*sealed_pair(model.suffix))
                self.assertEqual(
                    result["binding"]["inference_strength_binary64_bits"], _bits(0.75)
                )
                self.assertEqual(result["binding"]["raw_sha256"], site23.RAW_SHA256)
                self.assertEqual(result["binding"]["replay_sha256"], digest)
                wrong = dict(json.loads(path.read_text()), ema_decay=0.9)
                payload = json.dumps(wrong).encode()
                path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, "registered scope"):
                    FullSuffixChoice.open(
                        folder, replay_sha256=sha256(payload).hexdigest(), **kwargs
                    )
                loader.assert_called_once()

    def test_raw_materializer_checks_archival_fit_lineage_and_weight_policy(self):
        parameters = controller().state_dict()
        artifact = {
            "probe": parameters,
            "completed": 80,
            "raw_weights": True,
            "ema_decay": None,
            "experiment": {"path": "fit.json", "bytes": 1, "sha256": site23.FIT_SHA256},
            "full": {
                "path": "full.json",
                "bytes": 1,
                "sha256": site23.FULL_RECEIPT_SHA256,
            },
        }
        with TemporaryDirectory() as folder:
            path = Path(folder) / "raw.pt"
            for payload, invalid in (
                (artifact, False),
                (dict(artifact, ema_decay=0.9), True),
                (dict(artifact, raw_weights=False), True),
                (dict(artifact, completed=79), True),
                (dict(artifact, full=dict(artifact["full"], sha256="0" * 64)), True),
            ):
                torch.save(payload, path)
                data = path.read_bytes()
                with (
                    patch.object(site23, "RAW_SHA256", sha256(data).hexdigest()),
                    patch.object(site23, "RAW_BYTES", len(data)),
                ):
                    if invalid:
                        with self.assertRaisesRegex(
                            ValueError, "reviewed completed Site23 RAW80"
                        ):
                            site23._raw_state(path)
                    else:
                        loaded = site23._raw_state(path)
                        self.assertTrue(
                            all(
                                torch.equal(loaded[key], value)
                                for key, value in parameters.items()
                            )
                        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
