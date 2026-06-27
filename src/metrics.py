"""
metrics.py — KPI collection for HetNet handover simulation.

Collects per-UE events and aggregates them into DataFrames
suitable for Streamlit dashboard display and statistical analysis.

Decision D-4.3 : Ping-pong window = 2.0 seconds.
Decision D-6.1 : Outputs are pandas DataFrames (Streamlit-ready).
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import pandas as pd
import numpy as np

RSRP_DROP_THRESHOLD_DBM = -110.0
PING_PONG_WINDOW_S = 2.0


@dataclass
class HOEvent:
    time_s: float; ue_id: int; from_bs_id: int; to_bs_id: int
    algorithm: str; ho_score: float


@dataclass
class DropEvent:
    time_s: float; ue_id: int; bs_id: int; rsrp_dbm: float; algorithm: str


class MetricsCollector:
    """
    Collects handover and drop events during one simulation run.
    Call summary() at end for scalar KPIs.
    Call to_dataframes() for Streamlit-ready DataFrames.
    """

    def __init__(self, algorithm: str, num_ue: int):
        self.algorithm = algorithm
        self.num_ue    = num_ue
        self._ho_events:   list[HOEvent]   = []
        self._drop_events: list[DropEvent] = []
        self._sinr_samples: dict[int, list[float]] = {i: [] for i in range(num_ue)}
        self._recent_ho: dict[int, deque] = {i: deque(maxlen=2) for i in range(num_ue)}
        self._ping_pong_count = 0

    def record_handover(self, time_s, ue_id, from_bs_id, to_bs_id, ho_score=float('nan')):
        ev = HOEvent(time_s=time_s, ue_id=ue_id, from_bs_id=from_bs_id,
                     to_bs_id=to_bs_id, algorithm=self.algorithm, ho_score=ho_score)
        self._ho_events.append(ev)
        hist = self._recent_ho[ue_id]
        if hist:
            last_time, last_to = hist[-1]
            if (time_s - last_time) <= PING_PONG_WINDOW_S and last_to == from_bs_id:
                self._ping_pong_count += 1
        self._recent_ho[ue_id].append((time_s, to_bs_id))

    def record_drop(self, time_s, ue_id, bs_id, rsrp_dbm):
        self._drop_events.append(DropEvent(
            time_s=time_s, ue_id=ue_id, bs_id=bs_id,
            rsrp_dbm=rsrp_dbm, algorithm=self.algorithm))

    def record_sinr_sample(self, ue_id: int, sinr_db: float):
        self._sinr_samples[ue_id].append(sinr_db)

    def check_drop(self, time_s, ue_id, bs_id, rsrp_dbm) -> bool:
        if rsrp_dbm < RSRP_DROP_THRESHOLD_DBM:
            self.record_drop(time_s, ue_id, bs_id, rsrp_dbm)
            return True
        return False

    def summary(self) -> dict:
        n_ho = len(self._ho_events)
        n_drops = len(self._drop_events)
        pp_rate = self._ping_pong_count / n_ho if n_ho > 0 else 0.0
        unnecessary = sum(1 for e in self._ho_events
                          if not pd.isna(e.ho_score) and e.ho_score < 0.65)
        unnecessary_rate = unnecessary / n_ho if n_ho > 0 else 0.0
        all_sinr = [s for samples in self._sinr_samples.values() for s in samples]
        mean_sinr = float(np.mean(all_sinr)) if all_sinr else float('nan')
        return {
            'algorithm':           self.algorithm,
            'ho_count':            n_ho,
            'ping_pong_count':     self._ping_pong_count,
            'ping_pong_rate':      round(pp_rate, 4),
            'unnecessary_ho_rate': round(unnecessary_rate, 4),
            'call_drop_count':     n_drops,
            'call_drop_rate':      round(n_drops / self.num_ue, 4),
            'mean_sinr_db':        round(mean_sinr, 3),
        }

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        ho_df = pd.DataFrame([
            {'time_s': e.time_s, 'ue_id': e.ue_id, 'from_bs': e.from_bs_id,
             'to_bs': e.to_bs_id, 'ho_score': e.ho_score, 'algorithm': e.algorithm}
            for e in self._ho_events
        ]) if self._ho_events else pd.DataFrame(
            columns=['time_s','ue_id','from_bs','to_bs','ho_score','algorithm'])
        drop_df = pd.DataFrame([
            {'time_s': e.time_s, 'ue_id': e.ue_id, 'bs_id': e.bs_id,
             'rsrp_dbm': e.rsrp_dbm, 'algorithm': e.algorithm}
            for e in self._drop_events
        ]) if self._drop_events else pd.DataFrame(
            columns=['time_s','ue_id','bs_id','rsrp_dbm','algorithm'])
        sinr_df = pd.DataFrame([
            {'ue_id': uid, 'mean_sinr_db': float(np.mean(s)) if s else float('nan'),
             'algorithm': self.algorithm}
            for uid, s in self._sinr_samples.items()
        ])
        return {
            'handovers': ho_df, 'drops': drop_df,
            'sinr': sinr_df, 'summary': pd.DataFrame([self.summary()])
        }
