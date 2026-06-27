"""
test_channel.py — Unit tests for channel_model.py.

Each test encodes a physically meaningful expectation.
Run with:  pytest tests/test_channel.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from channel_model import (
    BaseStation, BSType, build_network,
    compute_path_loss, compute_rsrp, compute_sinr,
    measure_all_links,
)
import config as cfg


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="module")
def network():
    return build_network()


@pytest.fixture(scope="module")
def macro(network):
    return next(bs for bs in network if bs.bs_type == BSType.MACRO)


@pytest.fixture(scope="module")
def pico(network):
    return next(bs for bs in network if bs.bs_type == BSType.PICO)


# ─────────────────────────────────────────────────────────────────────────────
# Network topology
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkTopology:

    def test_correct_bs_count(self, network):
        """Build topology must return 1 macro + 3 pico + 2 femto = 6 total."""
        assert len(network) == 6

    def test_bs_types(self, network):
        types = [bs.bs_type for bs in network]
        assert types.count(BSType.MACRO)  == 1
        assert types.count(BSType.PICO)   == 3
        assert types.count(BSType.FEMTO)  == 2

    def test_unique_ids(self, network):
        ids = [bs.bs_id for bs in network]
        assert len(ids) == len(set(ids)), "Duplicate BS IDs found"

    def test_tx_powers(self, network):
        for bs in network:
            if bs.bs_type == BSType.MACRO:
                assert bs.tx_power_dbm == cfg.MACRO_TX_POWER_DBM
            elif bs.bs_type == BSType.PICO:
                assert bs.tx_power_dbm == cfg.PICO_TX_POWER_DBM
            else:
                assert bs.tx_power_dbm == cfg.FEMTO_TX_POWER_DBM


# ─────────────────────────────────────────────────────────────────────────────
# Path loss — sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestPathLoss:

    def test_path_loss_increases_with_distance(self, macro, rng):
        """Path loss must monotonically increase as UE moves away from BS."""
        distances = [50, 100, 200, 350, 500]
        path_losses = []
        for d in distances:
            pl, _ = compute_path_loss(
                macro, macro.x + d, macro.y,
                h_ue=1.5, fc_ghz=cfg.CARRIER_FREQ_GHZ,
                rng=np.random.default_rng(seed=99),   # fixed seed for determinism
                ue_id=999
            )
            path_losses.append(pl)

        for i in range(len(path_losses) - 1):
            assert path_losses[i] < path_losses[i + 1], (
                f"Path loss did not increase: d={distances[i]}m "
                f"pl={path_losses[i]:.1f} → d={distances[i+1]}m pl={path_losses[i+1]:.1f}"
            )

    def test_path_loss_positive(self, macro, rng):
        """Path loss must always be positive (signal always attenuates)."""
        pl, _ = compute_path_loss(
            macro, macro.x + 100, macro.y,
            h_ue=1.5, fc_ghz=cfg.CARRIER_FREQ_GHZ,
            rng=rng, ue_id=1
        )
        assert pl > 0, f"Path loss should be positive, got {pl:.2f} dB"

    def test_pico_path_loss_less_at_short_range(self, network, rng):
        """
        A pico cell very close to a UE should have lower path loss than
        a distant macro cell to the same UE.
        """
        pico = next(bs for bs in network if bs.bs_type == BSType.PICO)
        macro = next(bs for bs in network if bs.bs_type == BSType.MACRO)

        # Place UE 20m from pico, far from macro
        ue_x = pico.x + 20.0
        ue_y = pico.y

        fixed_rng = np.random.default_rng(seed=7)
        pl_pico,  _ = compute_path_loss(pico, ue_x, ue_y, 1.5,
                                         cfg.CARRIER_FREQ_GHZ, fixed_rng, 1)
        pl_macro, _ = compute_path_loss(macro, ue_x, ue_y, 1.5,
                                         cfg.CARRIER_FREQ_GHZ, fixed_rng, 1)
        assert pl_pico < pl_macro, (
            f"Pico PL ({pl_pico:.1f}) should be less than macro PL ({pl_macro:.1f})"
            " when UE is 20m from pico"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RSRP
# ─────────────────────────────────────────────────────────────────────────────

class TestRSRP:

    def test_rsrp_within_bounds(self, network, rng):
        """RSRP must always fall within the configured measurement range."""
        for bs in network:
            pl, _ = compute_path_loss(
                bs, 250.0, 250.0, 1.5, cfg.CARRIER_FREQ_GHZ,
                np.random.default_rng(seed=0), ue_id=0
            )
            rsrp = compute_rsrp(bs.tx_power_dbm, pl)
            assert cfg.RSRP_MIN <= rsrp <= cfg.RSRP_MAX, (
                f"RSRP {rsrp:.1f} out of bounds for BS {bs.bs_id}"
            )

    def test_closer_ue_has_stronger_rsrp(self, macro):
        """A UE 50m from macro must have stronger RSRP than one 400m away."""
        fixed_rng_close = np.random.default_rng(seed=1)
        fixed_rng_far   = np.random.default_rng(seed=1)

        pl_close, _ = compute_path_loss(macro, macro.x + 50,  macro.y,
                                         1.5, cfg.CARRIER_FREQ_GHZ,
                                         fixed_rng_close, ue_id=10)
        pl_far,   _ = compute_path_loss(macro, macro.x + 400, macro.y,
                                         1.5, cfg.CARRIER_FREQ_GHZ,
                                         fixed_rng_far, ue_id=11)

        rsrp_close = compute_rsrp(macro.tx_power_dbm, pl_close)
        rsrp_far   = compute_rsrp(macro.tx_power_dbm, pl_far)
        assert rsrp_close > rsrp_far, (
            f"Close RSRP ({rsrp_close:.1f}) should exceed far RSRP ({rsrp_far:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SINR
# ─────────────────────────────────────────────────────────────────────────────

class TestSINR:

    def test_sinr_within_bounds(self):
        """SINR output must always be clipped to config range."""
        # Best case: very strong serving signal, no interference
        sinr_best = compute_sinr(serving_rsrp_dbm=-44.0, interferer_rsrps_dbm=[])
        assert cfg.SINR_MIN <= sinr_best <= cfg.SINR_MAX

        # Worst case: serving signal weak, many strong interferers
        sinr_worst = compute_sinr(
            serving_rsrp_dbm=-120.0,
            interferer_rsrps_dbm=[-44.0, -44.0, -44.0]
        )
        assert cfg.SINR_MIN <= sinr_worst <= cfg.SINR_MAX

    def test_more_interference_lowers_sinr(self):
        """Adding more interferers must decrease (or not increase) SINR."""
        sinr_no_intf = compute_sinr(-80.0, [])
        sinr_one_intf = compute_sinr(-80.0, [-85.0])
        sinr_many_intf = compute_sinr(-80.0, [-85.0, -82.0, -78.0])
        assert sinr_no_intf >= sinr_one_intf >= sinr_many_intf, (
            f"SINR should decrease with more interference: "
            f"{sinr_no_intf:.1f} / {sinr_one_intf:.1f} / {sinr_many_intf:.1f}"
        )

    def test_stronger_serving_improves_sinr(self):
        """Better serving signal must improve SINR with same interference."""
        interferers = [-90.0, -88.0]
        sinr_weak   = compute_sinr(-95.0, interferers)
        sinr_strong = compute_sinr(-70.0, interferers)
        assert sinr_strong > sinr_weak


# ─────────────────────────────────────────────────────────────────────────────
# Full measurement pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasurementPipeline:

    def test_measure_all_links_returns_all_bs(self, network):
        """measure_all_links must return one link entry per BS."""
        rng = np.random.default_rng(seed=5)
        meas = measure_all_links(
            ue_id=0, ue_x=250.0, ue_y=250.0,
            serving_bs_id=0,
            all_bs=network, rng=rng
        )
        assert len(meas.links) == len(network)

    def test_links_sorted_by_rsrp_descending(self, network):
        """Links must be sorted strongest-first."""
        rng = np.random.default_rng(seed=6)
        meas = measure_all_links(
            ue_id=1, ue_x=100.0, ue_y=150.0,
            serving_bs_id=1,
            all_bs=network, rng=rng
        )
        rsrps = [lk.rsrp_dbm for lk in meas.links]
        assert rsrps == sorted(rsrps, reverse=True), (
            f"Links not sorted by RSRP: {[f'{r:.1f}' for r in rsrps]}"
        )

    def test_rsrq_within_bounds(self, network):
        rng = np.random.default_rng(seed=7)
        meas = measure_all_links(
            ue_id=2, ue_x=250.0, ue_y=250.0,
            serving_bs_id=0,
            all_bs=network, rng=rng
        )
        assert cfg.RSRQ_MIN <= meas.rsrq_db <= cfg.RSRQ_MAX, (
            f"RSRQ {meas.rsrq_db:.2f} out of bounds"
        )

    def test_sinr_within_bounds_pipeline(self, network):
        rng = np.random.default_rng(seed=8)
        meas = measure_all_links(
            ue_id=3, ue_x=250.0, ue_y=250.0,
            serving_bs_id=0,
            all_bs=network, rng=rng
        )
        assert cfg.SINR_MIN <= meas.sinr_db <= cfg.SINR_MAX
