"""
test_mobility.py — Unit tests for mobility.py.
Run with:  pytest tests/test_mobility.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from mobility import MobilityModel, UEState, _reflect
import config as cfg


@pytest.fixture
def ped_model():
    """Pedestrian mobility model with fixed seed."""
    rng = np.random.default_rng(seed=42)
    return MobilityModel(num_ue=5, scenario='pedestrian', rng=rng)


@pytest.fixture
def fast_model():
    """Fast vehicle mobility model with fixed seed."""
    rng = np.random.default_rng(seed=42)
    return MobilityModel(num_ue=5, scenario='fast_vehicle', rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialisation:

    def test_correct_ue_count(self, ped_model):
        assert len(ped_model.ues) == 5

    def test_positions_within_bounds(self, ped_model):
        for ue in ped_model.ues:
            assert 0.0 <= ue.x <= cfg.SIM_AREA_M, f"UE {ue.ue_id} x={ue.x} out of bounds"
            assert 0.0 <= ue.y <= cfg.SIM_AREA_M, f"UE {ue.ue_id} y={ue.y} out of bounds"

    def test_pedestrian_speeds_in_range(self, ped_model):
        v_min, v_max = cfg.SCENARIO_VELOCITIES['pedestrian']
        for ue in ped_model.ues:
            if not ue.is_paused:
                kmh = ue.velocity_kmh
                assert v_min <= kmh <= v_max, (
                    f"Pedestrian speed {kmh:.2f} km/h outside [{v_min}, {v_max}]"
                )

    def test_fast_vehicle_speeds_in_range(self, fast_model):
        v_min, v_max = cfg.SCENARIO_VELOCITIES['fast_vehicle']
        for ue in fast_model.ues:
            if not ue.is_paused:
                kmh = ue.velocity_kmh
                assert v_min <= kmh <= v_max, (
                    f"Fast vehicle speed {kmh:.2f} km/h outside [{v_min}, {v_max}]"
                )

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            MobilityModel(num_ue=2, scenario='hyperloop',
                          rng=np.random.default_rng(0))


# ─────────────────────────────────────────────────────────────────────────────
# Movement
# ─────────────────────────────────────────────────────────────────────────────

class TestMovement:

    def test_fast_ue_moves_further_than_slow(self):
        """Fast vehicle must cover more distance per step than pedestrian."""
        rng_ped  = np.random.default_rng(seed=1)
        rng_fast = np.random.default_rng(seed=1)
        ped  = MobilityModel(num_ue=1, scenario='pedestrian',   rng=rng_ped)
        fast = MobilityModel(num_ue=1, scenario='fast_vehicle', rng=rng_fast)

        x0_ped,  y0_ped  = ped.ues[0].x,  ped.ues[0].y
        x0_fast, y0_fast = fast.ues[0].x, fast.ues[0].y

        for _ in range(10):
            ped.step()
            fast.step()

        dist_ped  = np.sqrt((ped.ues[0].x  - x0_ped)**2  + (ped.ues[0].y  - y0_ped)**2)
        dist_fast = np.sqrt((fast.ues[0].x - x0_fast)**2 + (fast.ues[0].y - y0_fast)**2)

        assert dist_fast > dist_ped, (
            f"Fast vehicle ({dist_fast:.1f}m) should move further than pedestrian ({dist_ped:.1f}m)"
        )

    def test_positions_stay_within_bounds_after_many_steps(self, fast_model):
        """After 3000 steps, all UEs must remain inside the simulation area."""
        for _ in range(3000):
            fast_model.step()
        for ue in fast_model.ues:
            assert 0.0 <= ue.x <= cfg.SIM_AREA_M, f"UE {ue.ue_id} x={ue.x:.1f} out of bounds"
            assert 0.0 <= ue.y <= cfg.SIM_AREA_M, f"UE {ue.ue_id} y={ue.y:.1f} out of bounds"

    def test_paused_ue_does_not_move(self):
        """A paused UE must remain at the same position."""
        rng = np.random.default_rng(seed=99)
        model = MobilityModel(num_ue=1, scenario='pedestrian', rng=rng)
        ue = model.ues[0]
        # Force a pause
        ue.is_paused = True
        ue.pause_remaining_s = 1.0
        x_before, y_before = ue.x, ue.y
        model.step(dt=0.1)
        assert ue.x == x_before and ue.y == y_before, "Paused UE should not move"

    def test_deterministic_with_same_seed(self):
        """Same seed must produce identical trajectories."""
        def run(seed):
            rng = np.random.default_rng(seed=seed)
            model = MobilityModel(num_ue=3, scenario='slow_vehicle', rng=rng)
            for _ in range(100):
                model.step()
            return [(ue.x, ue.y) for ue in model.ues]

        positions_a = run(seed=7)
        positions_b = run(seed=7)
        assert positions_a == positions_b, "Same seed must yield same trajectory"

    def test_different_seeds_yield_different_trajectories(self):
        """Different seeds should (almost certainly) produce different trajectories."""
        def run(seed):
            rng = np.random.default_rng(seed=seed)
            model = MobilityModel(num_ue=1, scenario='fast_vehicle', rng=rng)
            for _ in range(50):
                model.step()
            return (model.ues[0].x, model.ues[0].y)

        pos_a = run(seed=1)
        pos_b = run(seed=2)
        assert pos_a != pos_b, "Different seeds should give different positions"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary reflection helper
# ─────────────────────────────────────────────────────────────────────────────

class TestReflect:

    def test_reflect_above(self):
        assert _reflect(510.0, 0.0, 500.0) == pytest.approx(490.0)

    def test_reflect_below(self):
        assert _reflect(-10.0, 0.0, 500.0) == pytest.approx(10.0)

    def test_reflect_within(self):
        assert _reflect(250.0, 0.0, 500.0) == pytest.approx(250.0)

    def test_reflect_exact_boundary(self):
        assert _reflect(500.0, 0.0, 500.0) == pytest.approx(500.0)
        assert _reflect(0.0,   0.0, 500.0) == pytest.approx(0.0)
