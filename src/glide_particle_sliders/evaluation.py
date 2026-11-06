"""One RAW FP64 equal-example metric, plus separate semantic diagnostics."""
from __future__ import annotations

import random
from collections import defaultdict

import torch


def relative_sse(student, teacher, neutral):
    """Cast operands before subtracting; never hide a nonfinite student row."""
    s, t, n = [x.detach().to(dtype=torch.float64, device="cpu") for x in (student, teacher, neutral)]
    if s.shape != t.shape or t.shape != n.shape or s.ndim < 2:
        raise ValueError("matching [examples, ...coordinates] tensors required")
    if not all(torch.isfinite(x).all() for x in (s, t, n)):
        raise ValueError("all student, teacher and neutral fields must be finite")
    energy = (t - n).square().flatten(1).sum(1)
    valid = energy != 0
    errors = (s - t).square().flatten(1).sum(1)
    values = [float(errors[i] / energy[i]) if bool(valid[i]) else None for i in range(len(s))]
    if not valid.any():
        raise ValueError("all examples have exact-zero teacher edit energy")
    return dict(mean=float((errors[valid] / energy[valid]).mean()), per_example=values,
                valid=[bool(x) for x in valid], excluded=int((~valid).sum()))


def position_bias(teacher, neutral):
    """Fit the optimal constant slot-logit edit on TRAIN for the primary metric."""
    delta = teacher.double() - neutral.double()
    energy = delta.square().sum(1)
    valid = energy != 0
    if not valid.any():
        raise ValueError("no teacher edit energy for bias baseline")
    weights = 1 / energy[valid]
    return (delta[valid] * weights[:, None]).sum(0) / weights.sum()


def diagnostics(student, teacher, neutral, rows):
    result = relative_sse(student, teacher, neutral)
    probs = torch.softmax(student.double(), -1)
    target_probs = torch.softmax(teacher.double(), -1)
    idx = torch.tensor([r["concise_slot"] for r in rows])
    preferred = torch.tensor([r["preferred_slot"] for r in rows])
    main = torch.tensor([not r["control"] for r in rows])
    controls = ~main
    by_group = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[row["group"]].append(i)
    swap_errors, swap_agreement = [], []
    for indices in by_group.values():
        if len(indices) != 2:
            raise ValueError("exactly two option orders per group required")
        a, b = indices
        pa, pb = probs[a, idx[a]], probs[b, idx[b]]
        swap_errors.append(abs(float(pa - pb)))
        ca = int(probs[a].argmax()) == int(idx[a])
        cb = int(probs[b].argmax()) == int(idx[b])
        swap_agreement.append(ca == cb)
    result.update(examples=len(rows), groups=len(by_group),
                  mean_p_concise_main=float(probs[main, idx[main]].mean()) if main.any() else None,
                  concise_choice_rate_main=float((probs[main].argmax(1) == idx[main]).double().mean()) if main.any() else None,
                  teacher_p_concise_main=float(target_probs[main, idx[main]].mean()) if main.any() else None,
                  control_accuracy=float((probs[controls].argmax(1) == preferred[controls]).double().mean()) if controls.any() else None,
                  swap_mean_abs_probability_gap=sum(swap_errors) / len(swap_errors),
                  swap_semantic_choice_agreement=sum(swap_agreement) / len(swap_agreement))
    # Resample whole scenario groups, keeping both orders together.
    group_values = []
    for indices in by_group.values():
        values = [result["per_example"][i] for i in indices if result["per_example"][i] is not None]
        if values:
            group_values.append(values)
    rng = random.Random(19)
    means = []
    for _ in range(2000):
        draw = [v for group in rng.choices(group_values, k=len(group_values)) for v in group]
        means.append(sum(draw) / len(draw))
    means.sort()
    result["group_bootstrap_95pct"] = [means[49], means[1949]]
    return result
