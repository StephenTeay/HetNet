"""
simulation.py — SimPy discrete-event simulation engine.

Wires together: channel_model, mobility, fuzzy_engine, baseline, metrics.

One simulation run:
  1. Build network topology (BaseStation objects)
  2. Initialise UEs with random positions, speeds, loads, batteries
  3. Every TTI (0.1s): move UEs, measure links, run HO algorithm, log KPIs
  4. Return MetricsCollector with all events for this run

Decision D-4.1 : Cell load static per run, drawn from U(20%, 85%)
Decision D-4.2 : Battery fixed per UE per run, drawn from U(10%, 100%)
Decision D-4.4 : HO execution instantaneous
Decision D-4.5 : QoS weight per UE: 0.3 / 0.7 / 1.0 (background/stream/voip)
"""

from __future__ import annotations
import numpy as np
import simpy
from typing import Literal

import config as cfg
from channel_model import build_network, measure_all_links, BaseStation
from mobility import MobilityModel
from fuzzy_engine import FastHandoverFIS as HandoverFIS
from baseline import A3Baseline
from metrics import MetricsCollector


# QoS service type weights (Decision D-4.5)
QOS_WEIGHTS = {
    'background': 0.3,
    'streaming':  0.7,
    'voip':       1.0,
}
QOS_TYPES = list(QOS_WEIGHTS.keys())


def _assign_ue_context(num_ue: int, rng: np.random.Generator) -> dict:
    """
    Draw static per-UE attributes for one simulation run.

    Returns dict with keys:
      battery[ue_id]  : float ∈ [10, 100] %
      qos[ue_id]      : float weight
      load[bs_id]     : float ∈ [20, 85] % (per BS, not per UE)
    """
    battery = {i: float(rng.uniform(10.0, 100.0)) for i in range(num_ue)}

    qos_types = rng.choice(QOS_TYPES, size=num_ue)
    qos = {i: QOS_WEIGHTS[qos_types[i]] for i in range(num_ue)}

    # One load value per BS (Decision D-4.1)
    network = build_network()
    load = {bs.bs_id: float(rng.uniform(20.0, 85.0)) for bs in network}

    return {'battery': battery, 'qos': qos, 'load': load}


def _initial_serving_bs(ue_x: float, ue_y: float,
                         network: list[BaseStation],
                         rng: np.random.Generator,
                         ue_id: int) -> int:
    """
    Assign the initial serving BS as the one with the strongest RSRP
    at the UE's starting position.
    """
    meas = measure_all_links(
        ue_id=ue_id, ue_x=ue_x, ue_y=ue_y,
        serving_bs_id=network[0].bs_id,
        all_bs=network, rng=rng,
    )
    return meas.links[0].bs_id   # links sorted strongest-first


# ─────────────────────────────────────────────────────────────────────────────
# UE process (SimPy generator)
# ─────────────────────────────────────────────────────────────────────────────

def _ue_process(
    env:        simpy.Environment,
    ue_id:      int,
    mobility:   MobilityModel,
    network:    list[BaseStation],
    context:    dict,
    algorithm:  Literal['fis', 'baseline'],
    fis:        HandoverFIS | None,
    a3:         A3Baseline | None,
    collector:  MetricsCollector,
    rng:        np.random.Generator,
):
    """
    SimPy process for one UE — runs for the full simulation duration.

    Each iteration:
      1. Wait one TTI
      2. Read current position from mobility model (already stepped globally)
      3. Measure all BS links
      4. Run handover algorithm
      5. Execute HO if triggered
      6. Log SINR sample; check for drop
    """
    ue_state = mobility.ues[ue_id]
    serving_bs_id = _initial_serving_bs(
        ue_state.x, ue_state.y, network, rng, ue_id
    )

    if algorithm == 'baseline' and a3 is not None:
        a3.register_ue(ue_id, serving_bs_id)

    while True:
        yield env.timeout(cfg.TTI_S)

        # Current position (mobility.step() called by the global stepper)
        ux, uy = ue_state.x, ue_state.y
        velocity_kmh = ue_state.velocity_kmh

        # Measure all links
        meas = measure_all_links(
            ue_id=ue_id, ue_x=ux, ue_y=uy,
            serving_bs_id=serving_bs_id,
            all_bs=network, rng=rng,
        )

        serving_link = next(
            (lk for lk in meas.links if lk.bs_id == serving_bs_id),
            meas.links[0]
        )
        serving_rsrp = serving_link.rsrp_dbm

        # Log SINR and check for drop
        collector.record_sinr_sample(ue_id, meas.sinr_db)
        dropped = collector.check_drop(
            env.now, ue_id, serving_bs_id, serving_rsrp
        )
        if dropped:
            # Force handover to strongest available BS on drop
            serving_bs_id = meas.links[0].bs_id
            if algorithm == 'baseline' and a3:
                a3.update_serving(ue_id, serving_bs_id)
            continue

        # ── HO decision ───────────────────────────────────────────────────
        trigger_ho = False
        target_bs_id = -1
        ho_score = float('nan')

        if algorithm == 'fis' and fis is not None:
            battery  = context['battery'][ue_id]
            cell_load = context['load'][serving_bs_id]

            ho_score, trigger_ho = fis.compute_and_decide(
                rsrp=serving_rsrp,
                sinr=meas.sinr_db,
                velocity=velocity_kmh,
                load=cell_load,
                battery=battery,
            )

            if trigger_ho:
                # Target = strongest BS that is NOT the current serving BS
                candidates = [lk for lk in meas.links if lk.bs_id != serving_bs_id]
                if candidates:
                    target_bs_id = candidates[0].bs_id

        elif algorithm == 'baseline' and a3 is not None:
            neighbour_rsrps = {
                lk.bs_id: lk.rsrp_dbm
                for lk in meas.links
                if lk.bs_id != serving_bs_id
            }
            trigger_ho, target_bs_id = a3.evaluate(
                ue_id=ue_id,
                serving_rsrp=serving_rsrp,
                neighbour_rsrps=neighbour_rsrps,
            )

        # ── Execute HO (instantaneous — Decision D-4.4) ────────────────────
        if trigger_ho and target_bs_id != -1 and target_bs_id != serving_bs_id:
            collector.record_handover(
                time_s=env.now,
                ue_id=ue_id,
                from_bs_id=serving_bs_id,
                to_bs_id=target_bs_id,
                ho_score=ho_score,
            )
            serving_bs_id = target_bs_id
            if algorithm == 'baseline' and a3:
                a3.update_serving(ue_id, target_bs_id)


def _mobility_stepper(
    env:      simpy.Environment,
    mobility: MobilityModel,
):
    """
    Single SimPy process that advances the mobility model each TTI.
    All UE processes read positions AFTER this has run.
    SimPy's event ordering ensures the stepper runs before UE processes
    at each time step because it is registered first.
    """
    while True:
        yield env.timeout(cfg.TTI_S)
        mobility.step(dt=cfg.TTI_S)


# ─────────────────────────────────────────────────────────────────────────────
# Public: run one simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    scenario:  str,
    algorithm: Literal['fis', 'baseline'],
    seed:      int,
) -> MetricsCollector:
    """
    Run one complete simulation and return a MetricsCollector.

    Parameters
    ----------
    scenario  : 'pedestrian', 'slow_vehicle', or 'fast_vehicle'
    algorithm : 'fis' or 'baseline'
    seed      : random seed (use cfg.RANDOM_SEED_BASE + run_index)

    Returns
    -------
    MetricsCollector with all events from this run.
    """
    rng = np.random.default_rng(seed=seed)

    # Build shared components
    network  = build_network()
    context  = _assign_ue_context(cfg.NUM_UE, rng)
    mobility = MobilityModel(num_ue=cfg.NUM_UE, scenario=scenario, rng=rng)
    collector = MetricsCollector(algorithm=algorithm, num_ue=cfg.NUM_UE)

    # Algorithm-specific objects
    fis = HandoverFIS() if algorithm == 'fis' else None
    a3  = A3Baseline()  if algorithm == 'baseline' else None

    # SimPy environment
    env = simpy.Environment()

    # Start mobility stepper (must be first so UEs read updated positions)
    env.process(_mobility_stepper(env, mobility))

    # Start one process per UE
    for i in range(cfg.NUM_UE):
        env.process(_ue_process(
            env=env, ue_id=i,
            mobility=mobility, network=network, context=context,
            algorithm=algorithm, fis=fis, a3=a3,
            collector=collector, rng=rng,
        ))

    # Run for the full simulation duration
    env.run(until=cfg.SIM_DURATION_S)

    return collector


# ─────────────────────────────────────────────────────────────────────────────
# Public: run full Monte Carlo experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    scenario:    str,
    algorithm:   Literal['fis', 'baseline'],
    num_runs:    int = None,
    seed_base:   int = None,
    verbose:     bool = False,
) -> pd.DataFrame:
    """
    Run num_runs independent simulations and return aggregated KPI DataFrame.

    Each run uses seed = seed_base + run_index, ensuring the same trajectory
    is used for both 'fis' and 'baseline' when called with identical seed_base.

    Returns
    -------
    pd.DataFrame with one row per run and columns = KPI names.
    """
    import pandas as pd

    if num_runs  is None: num_runs  = cfg.NUM_MONTE_CARLO
    if seed_base is None: seed_base = cfg.RANDOM_SEED_BASE

    rows = []
    for i in range(num_runs):
        seed = seed_base + i
        if verbose:
            print(f"  [{algorithm}] {scenario} run {i+1}/{num_runs} (seed={seed})")
        collector = run_simulation(scenario=scenario, algorithm=algorithm, seed=seed)
        row = collector.summary()
        row['run'] = i
        row['scenario'] = scenario
        rows.append(row)

    import pandas as pd
    return pd.DataFrame(rows)
