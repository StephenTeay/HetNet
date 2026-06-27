"""
config.py — Central configuration for the HetNet fuzzy handover model.

Decision rationale for every parameter is documented inline.
All tunable values live here so Phase 5 sensitivity analysis can
vary them without touching any other module.
"""

# ─────────────────────────────────────────────────────────────────────────────
# INPUT UNIVERSES
# Decision 2.1 — RSRP range: −120 to −44 dBm
#   Source: 3GPP TS 38.133 Table 10.1.6.1-1 defines the UE RSRP measurement
#   range as −156 to −31 dBm. We clip to −120..−44 because values outside
#   this range never occur in realistic HetNet deployments (below −120 means
#   no service; above −44 means the UE is inside the BS antenna).
# ─────────────────────────────────────────────────────────────────────────────
RSRP_MIN = -120.0   # dBm — practical lower bound (no service below this)
RSRP_MAX = -44.0    # dBm — practical upper bound (extremely close to BS)

# Decision 2.2 — SINR range: −10 to +30 dB
#   Below −10 dB: connection cannot be maintained (physical layer failure).
#   Above +30 dB: essentially interference-free; rare in HetNets.
#   This range covers all relevant operating conditions.
SINR_MIN = -10.0    # dB
SINR_MAX = 30.0     # dB

# Decision 2.2b — RSRQ range: −19.5 to −3 dB
#   Source: 3GPP TS 36.133. RSRQ = N * RSRP / (LTE bandwidth * noise + interference)
#   Expressed in dB; −3 is excellent, below −15 is poor.
RSRQ_MIN = -19.5    # dB
RSRQ_MAX = -3.0     # dB

# Decision 2.3 — UE velocity: 0 to 120 km/h
#   Covers pedestrian (0–5), slow vehicular (15–60), fast vehicular (60–120).
#   3GPP mobility state estimation uses these boundaries (TS 36.331 s5.3.12.1).
#   We cap at 120 km/h — above this a UE crosses a pico cell too fast for
#   the handover algorithm to be meaningful regardless of model.
VELOCITY_MIN = 0.0    # km/h
VELOCITY_MAX = 120.0  # km/h

# Decision 2.4 — Cell load: 0 to 100 %
#   Expressed as Physical Resource Block (PRB) utilisation.
#   A fully loaded LTE cell is at 100%; typical operation is 40–70%.
LOAD_MIN = 0.0    # %
LOAD_MAX = 100.0  # %

# Decision 2.5 — Battery level: 0 to 100 %
#   Standard device battery percentage. Below 15% the device enters power-
#   saving mode in most Android/iOS implementations, which is the threshold
#   we use for the "critical" linguistic term.
BATTERY_MIN = 0.0    # %
BATTERY_MAX = 100.0  # %

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT UNIVERSE
# Decision 2.6 — HO score: 0.0 to 1.0 (normalised urgency)
#   0.0 = definitely stay on current cell
#   1.0 = immediate handover required
#   Threshold θ at 0.6 separates stay vs trigger (tuned in Phase 5).
# ─────────────────────────────────────────────────────────────────────────────
HO_SCORE_MIN = 0.0
HO_SCORE_MAX = 1.0

# Decision 2.9 — Handover trigger threshold θ
#   Starting value 0.6 based on literature midpoint (Balan et al. 2011 used 0.5;
#   Youssef et al. 2019 used 0.65). We start at 0.6 and sweep [0.4, 0.8]
#   in Phase 5 sensitivity analysis.
HO_THRESHOLD = 0.6

# ─────────────────────────────────────────────────────────────────────────────
# MEMBERSHIP FUNCTION BOUNDARIES
# Decision 2.7 — 3 linguistic terms per input: Low / Medium / High
#   (or equivalent: Poor/Medium/Good, Slow/Moderate/Fast, etc.)
#   Three terms chosen over five because:
#     a) Keeps rule base tractable (3^5 = 243 vs 5^5 = 3125 max rules)
#     b) Matching granularity to available 3GPP measurement precision
#     c) Standard in handover FIS literature (Mitew et al., Nkansah-Gyekye)
#
# Decision 2.7b — Boundary values derived from 3GPP specs and literature:
#   RSRP thresholds: 3GPP TR 36.902 Table 4.5 (cell range estimation)
#   SINR thresholds: align with MCS selection boundaries in TS 38.214
#   Velocity: 3GPP mobility state thresholds (TS 36.331 s5.3.12.1)
#   Load: operational guidelines (ETSI TR 136 902)
#   Battery: platform power-saving trigger values (Android/iOS documentation)
# ─────────────────────────────────────────────────────────────────────────────

# RSRP membership boundaries (dBm)
# Poor:   [-120, -120, -100, -85]  trapezoid (open left — no worse than -120)
# Medium: [-95,  -85,  -75]        triangle
# Good:   [-80,  -65,  -44, -44]   trapezoid (open right — no better than -44)
RSRP_POOR_BOUNDS   = (-120, -120, -100, -85)   # trapezoid
RSRP_MEDIUM_BOUNDS = (-95, -85, -75)            # triangle (peak at -85)
RSRP_GOOD_BOUNDS   = (-80, -65, -44, -44)       # trapezoid

# SINR membership boundaries (dB)
# Low:    [-10, -10, 0, 5]
# Medium: [2,   10,  18]
# High:   [15,  22,  30, 30]
SINR_LOW_BOUNDS    = (-10, -10, 0, 5)
SINR_MEDIUM_BOUNDS = (2, 10, 18)
SINR_HIGH_BOUNDS   = (15, 22, 30, 30)

# Velocity membership boundaries (km/h)
# Slow:     [0,  0,  15, 30]
# Moderate: [20, 45, 70]
# Fast:     [60, 80, 120, 120]
VELOCITY_SLOW_BOUNDS     = (0, 0, 15, 30)
VELOCITY_MODERATE_BOUNDS = (20, 45, 70)
VELOCITY_FAST_BOUNDS     = (60, 80, 120, 120)

# Cell load membership boundaries (%)
# Low:    [0,  0,  30, 50]
# Medium: [35, 55, 75]
# High:   [65, 80, 100, 100]
LOAD_LOW_BOUNDS    = (0, 0, 30, 50)
LOAD_MEDIUM_BOUNDS = (35, 55, 75)
LOAD_HIGH_BOUNDS   = (65, 80, 100, 100)

# Battery membership boundaries (%)
# Critical: [0,  0,  10, 20]
# Moderate: [15, 40, 65]
# Adequate: [55, 75, 100, 100]
BATTERY_CRITICAL_BOUNDS = (0, 0, 10, 20)
BATTERY_MODERATE_BOUNDS = (15, 40, 65)
BATTERY_ADEQUATE_BOUNDS = (55, 75, 100, 100)

# HO score output membership boundaries
# Low:    [0.0, 0.0, 0.2, 0.4]
# Medium: [0.3, 0.5, 0.7]
# High:   [0.6, 0.8, 1.0, 1.0]
HO_LOW_BOUNDS    = (0.0, 0.0, 0.2, 0.4)
HO_MEDIUM_BOUNDS = (0.3, 0.5, 0.7)
HO_HIGH_BOUNDS   = (0.6, 0.8, 1.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS  (populated in Phase 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────
SIM_DURATION_S    = 300      # seconds per run
NUM_UE            = 20       # user equipment per simulation run
NUM_MONTE_CARLO   = 30       # independent runs for statistical averaging
RANDOM_SEED_BASE  = 42       # base seed; run i uses seed BASE + i

# Network topology
MACRO_TX_POWER_DBM  = 46.0   # dBm  (typical macro BS)
PICO_TX_POWER_DBM   = 30.0   # dBm  (typical pico BS)
FEMTO_TX_POWER_DBM  = 20.0   # dBm  (typical femto BS)

MACRO_HEIGHT_M = 25.0    # m   antenna height
PICO_HEIGHT_M  = 10.0    # m
FEMTO_HEIGHT_M = 3.0     # m

CARRIER_FREQ_GHZ = 2.1   # GHz  (common LTE band)

# ─────────────────────────────────────────────────────────────────────────────
# A3 BASELINE PARAMETERS  (Decision 4.1 — documented in Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
A3_OFFSET_DB = 3.0       # dB   hysteresis offset
A3_TTT_MS    = 160       # ms   time-to-trigger


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — CHANNEL AND MOBILITY MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Decision D-3.5: Thermal noise floor
# N0 = kTB where k=Boltzmann, T=290K, B=10MHz
# = -174 dBm/Hz + 10*log10(10e6) + 9 dB NF = -95.0 dBm
THERMAL_NOISE_DBM   = -174.0   # dBm/Hz (thermal noise density at 290K)
NOISE_FIGURE_DB     =    9.0   # dB  (3GPP TS 36.101 Table 7.3-1 UE noise figure)
BANDWIDTH_MHZ       =   10.0   # MHz (10 MHz LTE channel = 50 PRBs)
NOISE_FLOOR_DBM     = THERMAL_NOISE_DBM + 10 * __import__('math').log10(BANDWIDTH_MHZ * 1e6) + NOISE_FIGURE_DB

# Decision D-3.2: Shadow fading std deviations (3GPP TR 36.873 Table B.1.2.1-1)
SHADOW_STD_UMA_LOS_DB    = 4.0   # dB
SHADOW_STD_UMA_NLOS_DB   = 6.0   # dB
SHADOW_STD_UMI_LOS_DB    = 3.0   # dB
SHADOW_STD_UMI_NLOS_DB   = 4.0   # dB
SHADOW_STD_INDOOR_DB      = 6.0   # dB

# Decision D-3.7: Simulation area
SIM_AREA_M = 500.0   # metres — square simulation region side length

# Decision D-3.8: Time step
TTI_S = 0.1          # seconds per Transmission Time Interval (measurement step)

# Network topology (positions in metres from origin)
# 1 macro cell at centre, 3 pico cells, 2 femto cells
MACRO_POSITIONS = [(250.0, 250.0)]   # centre of simulation area

PICO_POSITIONS  = [
    (100.0, 150.0),   # bottom-left cluster
    (380.0, 120.0),   # bottom-right cluster
    (220.0, 390.0),   # top-centre cluster
]

FEMTO_POSITIONS = [
    (150.0, 310.0),   # indoor — residential zone
    (340.0, 300.0),   # indoor — office zone
]

# Decision D-3.6: Random Waypoint pause time (seconds)
RWP_MIN_PAUSE_S  = 0.0    # minimum pause at waypoint
RWP_MAX_PAUSE_S  = 2.0    # maximum pause at waypoint

# Scenario velocity ranges (km/h) — converted to m/s in mobility.py
SCENARIO_VELOCITIES = {
    'pedestrian':     (0.5,   5.0),    # km/h  walking pace
    'slow_vehicle':  (15.0,  40.0),   # km/h  urban driving / cycling
    'fast_vehicle':  (60.0, 100.0),   # km/h  suburban/highway driving
}
