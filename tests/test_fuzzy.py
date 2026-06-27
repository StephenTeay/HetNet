"""
test_fuzzy.py — Sanity checks for the HandoverFIS.

Each test encodes a physically meaningful scenario where the expected
handover decision is unambiguous. If any test fails, the membership
functions or rule base contain a logical error.

Run with:  pytest tests/test_fuzzy.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from fuzzy_engine import HandoverFIS


@pytest.fixture(scope="module")
def fis():
    """Single FIS instance shared across all tests in this module."""
    return HandoverFIS()


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Critical signal scenarios (expected: HIGH score → trigger HO)
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticalSignal:

    def test_worst_case_triggers_ho(self, fis):
        """R01: Terrible RSRP, terrible SINR → must trigger handover."""
        score, trigger = fis.compute_and_decide(
            rsrp=-115, sinr=-8, velocity=30, load=40, battery=80
        )
        assert trigger, f"Expected HO trigger (score={score:.3f}), got stay"
        assert score > 0.6, f"Score too low for worst case: {score:.3f}"

    def test_poor_rsrp_medium_sinr_triggers_ho(self, fis):
        """R02: Poor RSRP even with medium SINR → high urgency."""
        score, trigger = fis.compute_and_decide(
            rsrp=-110, sinr=10, velocity=30, load=40, battery=80
        )
        assert trigger, f"Poor RSRP should trigger HO (score={score:.3f})"

    def test_low_sinr_medium_rsrp_triggers_ho(self, fis):
        """R04: Interference problem even with decent RSRP → HO."""
        score, trigger = fis.compute_and_decide(
            rsrp=-82, sinr=-5, velocity=30, load=40, battery=80
        )
        assert trigger, f"Low SINR should trigger HO (score={score:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Mobility scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestMobilityDriven:

    def test_fast_ue_medium_signal_triggers_ho(self, fis):
        """R06: Fast-moving UE at borderline signal → HO before overshoot."""
        score, trigger = fis.compute_and_decide(
            rsrp=-87, sinr=9, velocity=95, load=35, battery=80
        )
        assert trigger, f"Fast UE at medium signal should trigger HO (score={score:.3f})"

    def test_slow_ue_medium_signal_stays(self, fis):
        """R08: Slow UE at borderline signal → stay (ping-pong risk)."""
        score, trigger = fis.compute_and_decide(
            rsrp=-86, sinr=9, velocity=3, load=35, battery=80
        )
        assert not trigger, f"Slow UE should stay (score={score:.3f})"

    def test_fast_ue_good_signal_stays(self, fis):
        """R12: Fast UE but good signal → no urgency to handover."""
        score, trigger = fis.compute_and_decide(
            rsrp=-65, sinr=22, velocity=100, load=30, battery=80
        )
        assert not trigger, f"Good signal UE should stay even at speed (score={score:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Load and battery context
# ─────────────────────────────────────────────────────────────────────────────

class TestContextualFactors:

    def test_overloaded_target_suppresses_ho(self, fis):
        """R15: Medium signal + highly loaded target → stay."""
        score, trigger = fis.compute_and_decide(
            rsrp=-87, sinr=9, velocity=30, load=90, battery=80
        )
        assert not trigger, f"Congested target should suppress HO (score={score:.3f})"

    def test_critical_battery_suppresses_unnecessary_ho(self, fis):
        """R19: Critical battery + medium signal → conserve power, stay."""
        score, trigger = fis.compute_and_decide(
            rsrp=-86, sinr=10, velocity=20, load=40, battery=5
        )
        assert not trigger, f"Critical battery should suppress HO (score={score:.3f})"

    def test_critical_battery_poor_signal_still_triggers(self, fis):
        """R20: Critical battery but terrible signal → must HO regardless."""
        score, trigger = fis.compute_and_decide(
            rsrp=-112, sinr=-6, velocity=20, load=30, battery=5
        )
        assert trigger, f"Poor signal overrides battery concern (score={score:.3f})"

    def test_low_load_fast_ue_triggers_ho(self, fis):
        """R23: Fast UE + uncongested target → opportunistic HO encouraged."""
        score, trigger = fis.compute_and_decide(
            rsrp=-87, sinr=9, velocity=90, load=15, battery=80
        )
        assert trigger, f"Fast UE + low load should trigger HO (score={score:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary and edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_output_within_bounds(self, fis):
        """Score must always be within [0, 1]."""
        test_cases = [
            dict(rsrp=-120, sinr=-10, velocity=0,   load=0,   battery=0),
            dict(rsrp=-44,  sinr=30,  velocity=120, load=100, battery=100),
            dict(rsrp=-82,  sinr=10,  velocity=50,  load=50,  battery=50),
        ]
        for case in test_cases:
            score = fis.compute(**case)
            assert 0.0 <= score <= 1.0, (
                f"Score {score:.4f} out of [0,1] for inputs {case}"
            )

    def test_scores_ordered_by_signal_quality(self, fis):
        """Better signal quality should produce lower HO urgency (all else equal)."""
        base = dict(sinr=10, velocity=30, load=40, battery=80)
        score_poor   = fis.compute(rsrp=-110, **base)
        score_medium = fis.compute(rsrp=-84,  **base)
        score_good   = fis.compute(rsrp=-63,  **base)
        assert score_poor > score_medium, (
            f"Poor RSRP ({score_poor:.3f}) should score higher than medium ({score_medium:.3f})"
        )
        assert score_medium >= score_good, (
            f"Medium RSRP ({score_medium:.3f}) should score >= good ({score_good:.3f})"
        )

    def test_scores_ordered_by_velocity(self, fis):
        """Higher velocity should produce higher HO urgency (medium signal, all else equal)."""
        base = dict(rsrp=-86, sinr=9, load=35, battery=80)
        score_slow = fis.compute(velocity=3,  **base)
        score_fast = fis.compute(velocity=95, **base)
        assert score_fast > score_slow, (
            f"Fast UE ({score_fast:.3f}) should score higher than slow ({score_slow:.3f})"
        )

    def test_compute_is_deterministic(self, fis):
        """Same inputs must always produce the same output."""
        inputs = dict(rsrp=-87, sinr=9, velocity=60, load=50, battery=60)
        score1 = fis.compute(**inputs)
        score2 = fis.compute(**inputs)
        assert score1 == score2, f"FIS is non-deterministic: {score1} vs {score2}"
