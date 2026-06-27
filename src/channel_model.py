"""
channel_model.py — 3GPP-compliant radio channel model for HetNet simulation.

Implements:
  - UMa  (Urban Macro)  path loss — for macro BS links  [3GPP TR 36.873 Table B.1.2.2-1]
  - UMi  (Urban Micro)  path loss — for pico BS links   [3GPP TR 36.873 Table B.1.2.2-2]
  - Indoor path loss              — for femto BS links   [3GPP TR 36.873 Table B.1.2.2-4]
  - Log-normal shadow fading
  - RSRP, RSRQ, SINR computation

Decision D-3.1 : Model selection (UMa/UMi/indoor per BS type)
Decision D-3.2 : Probabilistic LOS/NLOS assignment per link
Decision D-3.3 : Shadow fading — log-normal, σ from 3GPP TR 36.873
Decision D-3.4 : Fast fading excluded (averages out at 100ms TTI)
Decision D-3.5 : Thermal noise = −174 dBm/Hz + 9 dB NF + 10 MHz BW
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

class BSType(Enum):
    MACRO  = "macro"
    PICO   = "pico"
    FEMTO  = "femto"


@dataclass
class BaseStation:
    """A single base station in the simulation."""
    bs_id:    int
    bs_type:  BSType
    x:        float          # metres
    y:        float          # metres
    tx_power_dbm: float      # transmit power
    height_m: float          # antenna height

    # Per-link shadow fading cache: {ue_id: fading_db}
    # Shadow fading is correlated in space — we use a per-link fixed
    # realisation drawn once per simulation run (decision D-3.3b).
    _shadow_cache: dict = field(default_factory=dict, repr=False)

    def distance_to(self, ue_x: float, ue_y: float) -> float:
        """2D horizontal distance in metres."""
        return float(np.sqrt((self.x - ue_x)**2 + (self.y - ue_y)**2))

    def shadow_fading_db(self, ue_id: int, rng: np.random.Generator,
                          los: bool) -> float:
        """
        Return the shadow fading realisation for a specific UE link.
        Drawn once and cached — shadow fading changes slowly (correlated
        over tens of metres), so we treat it as fixed per run.

        Decision D-3.3b: per-link fixed shadow realisation.
        """
        key = (ue_id, los)
        if key not in self._shadow_cache:
            if self.bs_type == BSType.MACRO:
                std = (cfg.SHADOW_STD_UMA_LOS_DB if los
                       else cfg.SHADOW_STD_UMA_NLOS_DB)
            elif self.bs_type == BSType.PICO:
                std = (cfg.SHADOW_STD_UMI_LOS_DB if los
                       else cfg.SHADOW_STD_UMI_NLOS_DB)
            else:
                std = cfg.SHADOW_STD_INDOOR_DB
            self._shadow_cache[key] = float(rng.normal(0.0, std))
        return self._shadow_cache[key]


@dataclass
class LinkMeasurement:
    """Signal measurements from one UE to one BS."""
    bs_id:        int
    bs_type:      BSType
    rsrp_dbm:     float    # Reference Signal Received Power
    path_loss_db: float    # for diagnostics
    los:          bool     # line-of-sight flag


@dataclass
class UEMeasurement:
    """Full set of measurements for one UE at one time step."""
    ue_id:          int
    serving_bs_id:  int
    links:          List[LinkMeasurement]  # one entry per BS (sorted by RSRP desc)
    rsrq_db:        float   # of the serving link
    sinr_db:        float   # of the serving link


# ─────────────────────────────────────────────────────────────────────────────
# LOS probability functions  (3GPP TR 36.873 Table B.1.2.1-1)
# ─────────────────────────────────────────────────────────────────────────────

def _los_prob_uma(d_2d: float) -> float:
    """
    LOS probability for Urban Macro scenario.
    3GPP TR 36.873 Table B.1.2.1-1.
    d_2d: 2D distance in metres.
    """
    if d_2d <= 18.0:
        return 1.0
    return (18.0 / d_2d + np.exp(-d_2d / 63.0) * (1.0 - 18.0 / d_2d))


def _los_prob_umi(d_2d: float) -> float:
    """
    LOS probability for Urban Micro scenario.
    3GPP TR 36.873 Table B.1.2.1-1.
    """
    if d_2d <= 18.0:
        return 1.0
    return (18.0 / d_2d + np.exp(-d_2d / 36.0) * (1.0 - 18.0 / d_2d))


def _determine_los(bs_type: BSType, d_2d: float,
                   rng: np.random.Generator) -> bool:
    """
    Decision D-3.2: probabilistic LOS assignment.
    Draw a Bernoulli sample against the 3GPP LOS probability.
    Femto cells are assumed indoor — always NLOS from the outdoor channel
    perspective (the indoor path loss model implicitly captures LOS within
    the building).
    """
    if bs_type == BSType.FEMTO:
        return False   # indoor model; LOS within building not separately modelled
    prob = (_los_prob_uma(d_2d) if bs_type == BSType.MACRO
            else _los_prob_umi(d_2d))
    return bool(rng.random() < prob)


# ─────────────────────────────────────────────────────────────────────────────
# Path loss models
# ─────────────────────────────────────────────────────────────────────────────

def _path_loss_uma(d_2d: float, d_3d: float, fc_ghz: float,
                   h_bs: float, h_ue: float, los: bool) -> float:
    """
    Urban Macro path loss (dB).
    3GPP TR 36.873 Table B.1.2.2-1.

    Parameters
    ----------
    d_2d   : 2D distance (m)
    d_3d   : 3D distance (m)  √(d_2d² + (h_bs - h_ue)²)
    fc_ghz : carrier frequency (GHz)
    h_bs   : BS antenna height (m)
    h_ue   : UE height (m), default 1.5m
    los    : line-of-sight flag
    """
    # Effective environment height (h_E) — 3GPP TR 36.873 eq B-3
    # For UMa, h_E is typically 1.0m for outdoor UEs
    h_E = 1.0
    h_bs_prime = h_bs - h_E
    h_ue_prime = h_ue  - h_E   # h_ue = 1.5m → h_ue_prime = 0.5m

    # Breakpoint distance d_BP' (m)
    d_BP = 4.0 * h_bs_prime * h_ue_prime * fc_ghz * 1e9 / 3e8

    if los:
        if d_2d <= d_BP:
            pl = (28.0 + 22.0 * np.log10(d_3d)
                  + 20.0 * np.log10(fc_ghz))
        else:
            pl = (28.0 + 40.0 * np.log10(d_3d)
                  + 20.0 * np.log10(fc_ghz)
                  - 9.0  * np.log10(d_BP**2 + (h_bs - h_ue)**2))
    else:  # NLOS
        pl_los = _path_loss_uma(d_2d, d_3d, fc_ghz, h_bs, h_ue, los=True)
        pl_nlos = (13.54 + 39.08 * np.log10(d_3d)
                   + 20.0  * np.log10(fc_ghz)
                   - 0.6   * (h_ue - 1.5))
        pl = max(pl_los, pl_nlos)

    return float(pl)


def _path_loss_umi(d_2d: float, d_3d: float, fc_ghz: float,
                   h_bs: float, h_ue: float, los: bool) -> float:
    """
    Urban Micro (street canyon) path loss (dB).
    3GPP TR 36.873 Table B.1.2.2-2.
    """
    h_E = 1.0
    h_bs_prime = h_bs - h_E
    h_ue_prime = h_ue  - h_E
    d_BP = 4.0 * h_bs_prime * h_ue_prime * fc_ghz * 1e9 / 3e8

    if los:
        if d_2d <= d_BP:
            pl = (22.0 * np.log10(d_3d)
                  + 28.0
                  + 20.0 * np.log10(fc_ghz))
        else:
            pl = (40.0 * np.log10(d_3d)
                  + 28.0
                  + 20.0 * np.log10(fc_ghz)
                  - 9.0  * np.log10(d_BP**2 + (h_bs - h_ue)**2))
    else:  # NLOS
        pl_los = _path_loss_umi(d_2d, d_3d, fc_ghz, h_bs, h_ue, los=True)
        pl_nlos = (36.7 * np.log10(d_3d)
                   + 22.7
                   + 26.0 * np.log10(fc_ghz)
                   - 0.3  * (h_ue - 1.5))
        pl = max(pl_los, pl_nlos)

    return float(pl)


def _path_loss_indoor(d_3d: float, fc_ghz: float) -> float:
    """
    Indoor (femto) path loss (dB).
    3GPP TR 36.873 Table B.1.2.2-4 — Indoor hotspot.
    Always NLOS from the macro layer perspective.
    Penetration loss of 20 dB added to account for building walls.
    """
    # Decision: 20 dB penetration loss is standard for concrete/brick walls
    # (3GPP TR 36.873 Table B.1.2.6-1 for "high loss" building material)
    PENETRATION_LOSS_DB = 20.0
    pl = (16.9 * np.log10(d_3d)
          + 32.8
          + 20.0 * np.log10(fc_ghz)
          + PENETRATION_LOSS_DB)
    return float(pl)


def compute_path_loss(bs: BaseStation, ue_x: float, ue_y: float,
                      h_ue: float, fc_ghz: float,
                      rng: np.random.Generator,
                      ue_id: int) -> tuple[float, bool]:
    """
    Compute total path loss (dB) and LOS flag for a BS→UE link.

    Returns
    -------
    (path_loss_db, is_los)
    """
    d_2d = bs.distance_to(ue_x, ue_y)
    d_2d = max(d_2d, 1.0)   # avoid log(0); minimum 1m separation

    d_3d = float(np.sqrt(d_2d**2 + (bs.height_m - h_ue)**2))

    los = _determine_los(bs.bs_type, d_2d, rng)
    shadow = bs.shadow_fading_db(ue_id, rng, los)

    if bs.bs_type == BSType.MACRO:
        pl = _path_loss_uma(d_2d, d_3d, fc_ghz, bs.height_m, h_ue, los)
    elif bs.bs_type == BSType.PICO:
        pl = _path_loss_umi(d_2d, d_3d, fc_ghz, bs.height_m, h_ue, los)
    else:
        pl = _path_loss_indoor(d_3d, fc_ghz)

    return pl + shadow, los


# ─────────────────────────────────────────────────────────────────────────────
# RSRP / RSRQ / SINR computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsrp(tx_power_dbm: float, path_loss_db: float) -> float:
    """
    RSRP (dBm) = TX power − path loss.
    Clipped to the valid measurement range from config.
    """
    rsrp = tx_power_dbm - path_loss_db
    return float(np.clip(rsrp, cfg.RSRP_MIN, cfg.RSRP_MAX))


def compute_sinr(serving_rsrp_dbm: float,
                 interferer_rsrps_dbm: list[float]) -> float:
    """
    SINR (dB) for the serving link.

    SINR = S / (I + N)
    where S = serving received power (linear mW),
          I = sum of interferer received powers (linear mW),
          N = thermal noise floor (linear mW).

    Decision: all other BSs in the simulation are treated as interferers.
    This is a worst-case assumption — in practice, BSs use different
    frequency bands or ICIC to reduce inter-cell interference.
    """
    def to_mw(dbm: float) -> float:
        return 10.0 ** ((dbm - 30.0) / 10.0)   # dBm → Watts → mW equivalent

    signal_mw = to_mw(serving_rsrp_dbm)
    noise_mw  = to_mw(cfg.NOISE_FLOOR_DBM)

    interference_mw = sum(to_mw(r) for r in interferer_rsrps_dbm)

    sinr_linear = signal_mw / (interference_mw + noise_mw)
    sinr_db = 10.0 * np.log10(max(sinr_linear, 1e-10))

    return float(np.clip(sinr_db, cfg.SINR_MIN, cfg.SINR_MAX))


def compute_rsrq(serving_rsrp_dbm: float,
                 all_rsrps_dbm: list[float]) -> float:
    """
    RSRQ (dB) = N × RSRP / (RSSI)
    where N = number of resource blocks = 50 (10 MHz),
          RSSI = wideband received signal including all interference.

    Simplified model: RSRQ ≈ RSRP − (total received power in dB)
    Clipped to [RSRQ_MIN, RSRQ_MAX] from config.
    """
    def to_mw(dbm: float) -> float:
        return 10.0 ** ((dbm - 30.0) / 10.0)

    N = 50  # resource blocks in 10 MHz
    rsrp_mw  = to_mw(serving_rsrp_dbm)
    total_mw = sum(to_mw(r) for r in all_rsrps_dbm) + to_mw(cfg.NOISE_FLOOR_DBM)

    rsrq_linear = N * rsrp_mw / total_mw
    rsrq_db = 10.0 * np.log10(max(rsrq_linear, 1e-10))

    return float(np.clip(rsrq_db, cfg.RSRQ_MIN, cfg.RSRQ_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement function called by the simulation each TTI
# ─────────────────────────────────────────────────────────────────────────────

def measure_all_links(
    ue_id:        int,
    ue_x:         float,
    ue_y:         float,
    serving_bs_id: int,
    all_bs:       list[BaseStation],
    rng:          np.random.Generator,
    h_ue:         float = 1.5,
    fc_ghz:       float = None,
) -> UEMeasurement:
    """
    Compute RSRP for every BS→UE link, then derive RSRQ and SINR
    for the serving link.

    Parameters
    ----------
    ue_id         : UE identifier (for shadow fading cache lookup)
    ue_x, ue_y    : UE position (metres)
    serving_bs_id : current serving BS id
    all_bs        : list of all BaseStation objects in the topology
    rng           : numpy random Generator (seeded per run)
    h_ue          : UE antenna height (default 1.5m — pedestrian)
    fc_ghz        : carrier frequency; defaults to config.CARRIER_FREQ_GHZ

    Returns
    -------
    UEMeasurement with RSRP per BS, RSRQ and SINR for serving link.
    """
    if fc_ghz is None:
        fc_ghz = cfg.CARRIER_FREQ_GHZ

    links: list[LinkMeasurement] = []

    for bs in all_bs:
        pl, los = compute_path_loss(bs, ue_x, ue_y, h_ue, fc_ghz, rng, ue_id)
        rsrp = compute_rsrp(bs.tx_power_dbm, pl)
        links.append(LinkMeasurement(
            bs_id=bs.bs_id,
            bs_type=bs.bs_type,
            rsrp_dbm=rsrp,
            path_loss_db=pl,
            los=los,
        ))

    # Sort descending by RSRP so links[0] is always the strongest candidate
    links.sort(key=lambda lk: lk.rsrp_dbm, reverse=True)

    # Find serving link
    serving_link = next((lk for lk in links if lk.bs_id == serving_bs_id), links[0])

    # Interferers = all other BSs
    interferer_rsrps = [lk.rsrp_dbm for lk in links if lk.bs_id != serving_bs_id]
    all_rsrps        = [lk.rsrp_dbm for lk in links]

    rsrq = compute_rsrq(serving_link.rsrp_dbm, all_rsrps)
    sinr = compute_sinr(serving_link.rsrp_dbm, interferer_rsrps)

    return UEMeasurement(
        ue_id=ue_id,
        serving_bs_id=serving_bs_id,
        links=links,
        rsrq_db=rsrq,
        sinr_db=sinr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Network topology factory
# ─────────────────────────────────────────────────────────────────────────────

def build_network() -> list[BaseStation]:
    """
    Build the HetNet topology from config positions.
    Returns a list of BaseStation objects with sequential IDs.

    Decision D-3.7: 1 macro + 3 pico + 2 femto cells within 500×500m area.
    """
    stations = []
    bs_id = 0

    for (x, y) in cfg.MACRO_POSITIONS:
        stations.append(BaseStation(
            bs_id=bs_id, bs_type=BSType.MACRO,
            x=x, y=y,
            tx_power_dbm=cfg.MACRO_TX_POWER_DBM,
            height_m=cfg.MACRO_HEIGHT_M,
        ))
        bs_id += 1

    for (x, y) in cfg.PICO_POSITIONS:
        stations.append(BaseStation(
            bs_id=bs_id, bs_type=BSType.PICO,
            x=x, y=y,
            tx_power_dbm=cfg.PICO_TX_POWER_DBM,
            height_m=cfg.PICO_HEIGHT_M,
        ))
        bs_id += 1

    for (x, y) in cfg.FEMTO_POSITIONS:
        stations.append(BaseStation(
            bs_id=bs_id, bs_type=BSType.FEMTO,
            x=x, y=y,
            tx_power_dbm=cfg.FEMTO_TX_POWER_DBM,
            height_m=cfg.FEMTO_HEIGHT_M,
        ))
        bs_id += 1

    return stations
