import pytest
import torch

from glide_particle_sliders.evaluation import relative_sse, position_bias


def test_metric_reference_points_and_equal_examples():
    n = torch.tensor([[0., 0.], [0., 0.]])
    t = torch.tensor([[1., 0.], [10., 0.]])
    assert relative_sse(n, t, n)["mean"] == 1
    assert relative_sse(t, t, n)["mean"] == 0
    assert relative_sse(-t, t, n)["mean"] == 4
    assert relative_sse(torch.tensor([[0., 0.], [10., 0.]]), t, n)["mean"] == .5


def test_zero_energy_cannot_exclude_failed_student():
    t = torch.tensor([[0., 0.], [1., 0.]])
    assert relative_sse(t, t, torch.zeros_like(t))["excluded"] == 1
    s = t.clone(); s[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        relative_sse(s, t, torch.zeros_like(t))


def test_fp64_operands_before_subtraction():
    # Two finite FP32 operands whose FP32 subtraction would overflow.
    t = torch.tensor([[3e38, 0.]])
    n = torch.tensor([[-3e38, 0.]])
    assert relative_sse(n, t, n)["mean"] == 1


def test_bias_is_metric_weighted_train_optimum():
    n = torch.zeros(2, 2)
    t = torch.tensor([[1., 0.], [2., 0.]])
    assert torch.equal(position_bias(t, n), torch.tensor([1.2, 0.], dtype=torch.float64))
