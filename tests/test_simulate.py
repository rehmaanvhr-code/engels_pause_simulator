import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate


def test_steady_state_is_flat_without_shock():
    df = simulate.run(years=30, shock=0.0)
    assert abs(df["wage_index"].iloc[-1] - 100) < 1.0
    assert abs(df["output_index"].iloc[-1] - 100) < 1.0


def test_shock_raises_output_before_wages():
    df = simulate.run(years=40, shock=0.5, diffusion_years=20)
    assert df["output_index"].iloc[20] > df["wage_index"].iloc[20]


def test_broad_ownership_shortens_pause():
    slow = simulate.run(years=60, shock=0.5, diffusion_years=10, ownership_spread=0.0005)
    fast = simulate.run(years=60, shock=0.5, diffusion_years=10, ownership_spread=0.03)
    assert simulate.pause_length(fast) < simulate.pause_length(slow)


def test_gini_bounds():
    x = np.array([1, 1, 1, 1])
    assert abs(simulate.gini(x)) < 1e-9
    y = np.array([0, 0, 0, 10])
    assert simulate.gini(y) > 0.7
