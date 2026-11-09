import pytest
import torch

from glide_particle_sliders.preference_metrics import preference_metrics


def test_order_aware_control_metrics():
    rows = [dict(group="main", preferred_slot=0, control=False), dict(group="main", preferred_slot=1, control=False),
            dict(group="control", preferred_slot=1, control=True), dict(group="control", preferred_slot=0, control=True)]
    neutral = torch.zeros(4,2)
    student = torch.tensor([[2.,0.], [0.,2.], [2.,0.], [0.,2.]])
    m = preference_metrics(student, neutral, rows)
    assert m["main"]["choice_accuracy"] == 1
    assert m["main"]["both_orders_correct_rate"] == 1
    assert m["main"]["mean_probability_gain"] > 0
    assert m["control"]["choice_accuracy"] == 0
    assert m["control"]["mean_probability_gain"] < 0
    assert m["all"]["groups"] == 2
    assert m["main"]["mean_order_probability_gap"] == 0


def test_one_good_order_cannot_fake_robust_preference():
    rows = [dict(group="g", preferred_slot=0, control=False), dict(group="g", preferred_slot=1, control=False)]
    m = preference_metrics(torch.tensor([[4.,0.],[1.,0.]]), torch.zeros(2,2), rows)["main"]
    assert m["mean_probability_gain"] > 0
    assert m["mean_worst_order_gain"] < 0
    assert m["both_orders_probability_improved_rate"] == 0


def test_missing_reverse_order_refused():
    with pytest.raises(ValueError, match="both option orders"):
        preference_metrics(torch.zeros(1,2),torch.zeros(1,2),[dict(group="g",preferred_slot=0,control=False)])
