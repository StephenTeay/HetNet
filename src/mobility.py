"""
mobility.py — Random Waypoint (RWP) mobility model for HetNet simulation.

Decision D-3.6 : Random Waypoint model chosen for academic reproducibility.
Decision D-3.7 : Simulation area 500×500m (SIM_AREA_M from config).
Decision D-3.8 : Position updated every TTI_S = 0.1 seconds.

The RWP model works as follows:
  1. UE starts at a random position in the simulation area.
  2. UE picks a random destination (waypoint) uniformly within the area.
  3. UE travels toward that waypoint at constant speed drawn from the
     scenario velocity range.
  4. On arrival, UE pauses for a random duration [RWP_MIN_PAUSE_S, RWP_MAX_PAUSE_S].
  5. Repeat from step 2.

Known RWP limitation: the model produces a non-uniform spatial distribution
(UEs spend more time near the centre). This is acknowledged in the literature
and is standard for comparative simulation — both the FIS and baseline run
on the same traces, so the bias cancels out.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UEState:
    """
    Complete state of a single UE at one time step.
    Updated in-place each TTI by MobilityModel.step().
    """
    ue_id:       int
    x:           float          # current position (metres)
    y:           float          # current position (metres)
    speed_ms:    float          # current speed (m/s)
    heading_rad: float          # current direction of travel (radians)
    is_paused:   bool = False   # True when UE is stationary at waypoint
    pause_remaining_s: float = 0.0

    # Waypoint target
    wp_x: float = 0.0
    wp_y: float = 0.0

    @property
    def velocity_kmh(self) -> float:
        """Speed in km/h (used as FIS input)."""
        return self.speed_ms * 3.6

    @property
    def velocity_ms(self) -> float:
        return self.speed_ms


@dataclass
class MobilityModel:
    """
    Manages position updates for all UEs in the simulation.

    Parameters
    ----------
    num_ue   : number of UE devices
    scenario : 'pedestrian', 'slow_vehicle', or 'fast_vehicle'
    rng      : seeded numpy Generator for reproducibility
    """
    num_ue:   int
    scenario: str
    rng:      np.random.Generator
    ues:      list[UEState] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.scenario not in cfg.SCENARIO_VELOCITIES:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. "
                f"Choose from {list(cfg.SCENARIO_VELOCITIES.keys())}"
            )
        self._init_ues()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _random_position(self) -> tuple[float, float]:
        """Uniform random position within the simulation area."""
        x = float(self.rng.uniform(0.0, cfg.SIM_AREA_M))
        y = float(self.rng.uniform(0.0, cfg.SIM_AREA_M))
        return x, y

    def _random_speed_ms(self) -> float:
        """
        Draw a speed uniformly from the scenario velocity range.
        Decision: uniform distribution within the range is simplest and
        avoids introducing an additional distributional assumption.
        Converted from km/h to m/s.
        """
        v_min_kmh, v_max_kmh = cfg.SCENARIO_VELOCITIES[self.scenario]
        speed_kmh = float(self.rng.uniform(v_min_kmh, v_max_kmh))
        return speed_kmh / 3.6   # km/h → m/s

    def _init_ues(self):
        """Initialise all UEs with random positions, speeds, and waypoints."""
        for i in range(self.num_ue):
            x, y   = self._random_position()
            wp_x, wp_y = self._random_position()
            speed  = self._random_speed_ms()
            dx, dy = wp_x - x, wp_y - y
            heading = float(np.arctan2(dy, dx))

            self.ues.append(UEState(
                ue_id=i,
                x=x, y=y,
                speed_ms=speed,
                heading_rad=heading,
                wp_x=wp_x, wp_y=wp_y,
            ))

    # ── Per-TTI update ────────────────────────────────────────────────────────

    def _new_waypoint(self, ue: UEState):
        """
        Assign a new random waypoint to a UE that has reached its destination.
        Also draws a new speed (UE may change speed at each waypoint —
        standard RWP variant that produces more realistic velocity variation).
        """
        ue.wp_x, ue.wp_y = self._random_position()
        ue.speed_ms = self._random_speed_ms()
        dx = ue.wp_x - ue.x
        dy = ue.wp_y - ue.y
        ue.heading_rad = float(np.arctan2(dy, dx))

    def _start_pause(self, ue: UEState):
        """Put a UE into the pause state at its current position."""
        ue.is_paused = True
        ue.pause_remaining_s = float(
            self.rng.uniform(cfg.RWP_MIN_PAUSE_S, cfg.RWP_MAX_PAUSE_S)
        )
        ue.speed_ms = 0.0

    def step(self, dt: float = None) -> list[UEState]:
        """
        Advance all UE positions by one time step (dt seconds).
        Default dt = config.TTI_S = 0.1s.

        Returns the updated list of UEState objects.
        """
        if dt is None:
            dt = cfg.TTI_S

        for ue in self.ues:
            if ue.is_paused:
                ue.pause_remaining_s -= dt
                if ue.pause_remaining_s <= 0.0:
                    # Pause over — pick new waypoint and resume
                    ue.is_paused = False
                    self._new_waypoint(ue)
                continue

            # Distance to waypoint
            dx = ue.wp_x - ue.x
            dy = ue.wp_y - ue.y
            dist_to_wp = float(np.sqrt(dx**2 + dy**2))

            step_size = ue.speed_ms * dt   # metres this TTI

            if step_size >= dist_to_wp:
                # UE reaches waypoint this step — snap to it and pause
                ue.x = ue.wp_x
                ue.y = ue.wp_y
                self._start_pause(ue)
            else:
                # Move toward waypoint
                # Decision: recompute heading each step in case UE drifted
                # (numerical stability for very small steps)
                ue.heading_rad = float(np.arctan2(dy, dx))
                ue.x += step_size * np.cos(ue.heading_rad)
                ue.y += step_size * np.sin(ue.heading_rad)

                # Boundary reflection: if UE exits the simulation area,
                # reflect it back. This keeps UEs inside the coverage zone
                # without needing wrap-around or respawning logic.
                # Decision: reflection is the simplest boundary condition and
                # avoids UEs clustering at edges (which wrap-around can cause).
                ue.x = _reflect(ue.x, 0.0, cfg.SIM_AREA_M)
                ue.y = _reflect(ue.y, 0.0, cfg.SIM_AREA_M)

        return self.ues

    def positions(self) -> list[tuple[float, float]]:
        """Return current (x, y) for all UEs — convenience for plotting."""
        return [(ue.x, ue.y) for ue in self.ues]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reflect(value: float, lo: float, hi: float) -> float:
    """
    Reflect a value that has gone outside [lo, hi] back into range.
    e.g. _reflect(510, 0, 500) → 490
    """
    if value < lo:
        return 2.0 * lo - value
    if value > hi:
        return 2.0 * hi - value
    return value
