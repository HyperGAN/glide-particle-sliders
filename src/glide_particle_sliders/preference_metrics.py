"""Order-aware, scenario-grouped preference diagnostics, separate from ranking."""
from collections import defaultdict
import random

import torch


def preference_metrics(student, neutral, rows):
    s, n = student.detach().double().cpu(), neutral.detach().double().cpu()
    if s.shape != n.shape or s.shape != (len(rows), 2) or not torch.isfinite(s).all() or not torch.isfinite(n).all():
        raise ValueError("finite matched two-choice logits required")
    indices = torch.tensor([r["preferred_slot"] for r in rows])
    p, p0 = torch.softmax(s, -1), torch.softmax(n, -1)
    desired = p[torch.arange(len(rows)), indices]
    baseline = p0[torch.arange(len(rows)), indices]
    correct, correct0 = s.argmax(1) == indices, n.argmax(1) == indices
    nll = -torch.log_softmax(s, -1)[torch.arange(len(rows)), indices]
    result = {}
    for name, select in (("all", lambda row: True), ("main", lambda row: not row["control"]),
                         ("control", lambda row: row["control"])):
        by_group = defaultdict(list)
        for index, row in enumerate(rows):
            if select(row):
                by_group[row["group"]].append(index)
        if not by_group:
            continue
        if any(len(ids) != 2 for ids in by_group.values()):
            raise ValueError("each scenario requires both option orders")
        ids = [i for group in by_group.values() for i in group]
        worst = [float(desired[g].min()) for g in by_group.values()]
        worst0 = [float(baseline[g].min()) for g in by_group.values()]
        gains = [float((desired[g] - baseline[g]).mean()) for g in by_group.values()]
        worst_gains = [a - b for a, b in zip(worst, worst0)]
        rng = random.Random(91)
        draws = [rng.choices(range(len(gains)), k=len(gains)) for _ in range(2000)]
        boot = sorted(sum(gains[i] for i in draw) / len(draw) for draw in draws)
        worst_boot = sorted(sum(worst_gains[i] for i in draw) / len(draw) for draw in draws)
        result[name] = dict(groups=len(by_group), ordered_judgments=len(ids),
            mean_p_preferred=float(desired[ids].mean()), baseline_mean_p_preferred=float(baseline[ids].mean()),
            mean_probability_gain=sum(gains) / len(gains), group_bootstrap_95pct_gain=[boot[49], boot[1949]],
            choice_accuracy=float(correct[ids].double().mean()), baseline_choice_accuracy=float(correct0[ids].double().mean()),
            both_orders_correct_rate=sum(bool(correct[g].all()) for g in by_group.values()) / len(by_group),
            baseline_both_orders_correct_rate=sum(bool(correct0[g].all()) for g in by_group.values()) / len(by_group),
            mean_worst_order_p_preferred=sum(worst) / len(worst), baseline_mean_worst_order_p_preferred=sum(worst0) / len(worst0),
            mean_worst_order_gain=sum(worst_gains) / len(worst_gains), group_bootstrap_95pct_worst_order_gain=[worst_boot[49], worst_boot[1949]],
            both_orders_probability_improved_rate=sum(bool(((desired[g] - baseline[g]) > 0).all()) for g in by_group.values()) / len(by_group),
            mean_order_probability_gap=sum(abs(float(desired[g[0]] - desired[g[1]])) for g in by_group.values()) / len(by_group),
            mean_preferred_nll=float(nll[ids].mean()), mean_preferred_brier=float((1-desired[ids]).square().mean()))
    return result
