"""
baseline.py — 3GPP A3 event handover algorithm.

The A3 event fires when:
    RSRP(neighbour) - RSRP(serving) > A3_OFFSET_DB
for a continuous duration of A3_TTT_MS milliseconds (time-to-trigger).

This is the dominant handover algorithm in deployed LTE networks and the
standard comparison baseline in HetNet handover research.

Decision D-4.6 : A3 offset = 3 dB, TTT = 160 ms (3GPP TS 36.331 defaults).
Decision D-4.4 : HO execution modelled as instantaneous.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import config as cfg


@dataclass
class A3State:
    """
    Per-UE state for A3 event tracking.
    The TTT timer must run continuously — if the condition drops before
    TTT expires, the timer resets.
    """
    ue_id:              int
    serving_bs_id:      int
    ttt_remaining_s:    float = 0.0      # seconds remaining in TTT
    candidate_bs_id:    int   = -1       # BS currently being timed


class A3Baseline:
    """
    3GPP A3 event handover algorithm.

    One A3State object per UE. Call evaluate() each TTI.

    Decision D-4.6:
        offset = config.A3_OFFSET_DB  = 3.0 dB
        TTT    = config.A3_TTT_MS     = 160 ms  → 0.16 s
    """

    def __init__(self):
        self._states: dict[int, A3State] = {}   # ue_id → A3State
        self._offset_db = cfg.A3_OFFSET_DB
        self._ttt_s     = cfg.A3_TTT_MS / 1000.0

    def register_ue(self, ue_id: int, initial_bs_id: int):
        """Call once per UE at simulation start."""
        self._states[ue_id] = A3State(
            ue_id=ue_id,
            serving_bs_id=initial_bs_id,
        )

    def update_serving(self, ue_id: int, new_bs_id: int):
        """Called after a handover to update the serving BS record."""
        if ue_id in self._states:
            st = self._states[ue_id]
            st.serving_bs_id   = new_bs_id
            st.ttt_remaining_s = 0.0
            st.candidate_bs_id = -1

    def evaluate(
        self,
        ue_id:          int,
        serving_rsrp:   float,
        neighbour_rsrps: dict[int, float],   # {bs_id: rsrp_dbm}
        dt:             float = None,
    ) -> tuple[bool, int]:
        """
        Evaluate the A3 condition for one UE at one TTI.

        Parameters
        ----------
        ue_id           : UE identifier
        serving_rsrp    : RSRP of current serving BS (dBm)
        neighbour_rsrps : dict mapping candidate BS ids to their RSRP (dBm)
        dt              : time step in seconds (default config.TTI_S)

        Returns
        -------
        (trigger_handover: bool, target_bs_id: int)
        target_bs_id is -1 if no handover triggered.
        """
        if dt is None:
            dt = cfg.TTI_S

        st = self._states[ue_id]

        # Find best neighbour
        if not neighbour_rsrps:
            return False, -1

        best_nb_id   = max(neighbour_rsrps, key=neighbour_rsrps.get)
        best_nb_rsrp = neighbour_rsrps[best_nb_id]

        # A3 condition: neighbour RSRP exceeds serving by offset
        condition_met = (best_nb_rsrp - serving_rsrp) > self._offset_db

        if condition_met and best_nb_id == st.candidate_bs_id:
            # Same candidate — continue counting down TTT
            st.ttt_remaining_s -= dt
            if st.ttt_remaining_s <= 0.0:
                # TTT expired — trigger handover
                return True, best_nb_id

        elif condition_met and best_nb_id != st.candidate_bs_id:
            # New candidate — start TTT timer
            st.candidate_bs_id   = best_nb_id
            st.ttt_remaining_s   = self._ttt_s - dt

        else:
            # Condition not met — reset
            st.ttt_remaining_s = 0.0
            st.candidate_bs_id = -1

        return False, -1
