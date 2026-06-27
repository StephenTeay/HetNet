# Design Decision Log — HetNet Fuzzy Handover Optimization

All decisions are numbered by phase. Every decision records:
- what was decided
- what the alternatives were
- why this choice was made
- where in the codebase it is implemented

---

## Phase 1 — Environment and project setup

### D-1.1 Python version: 3.12 (target: 3.10+)
**Decided:** Use Python 3.12 (the version available in the build environment).
**Alternatives:** 3.10 (most stable for older packages), 3.11.
**Rationale:** 3.10 was the original target but the build machine runs 3.12. scikit-fuzzy 0.4.2 breaks on 3.12 due to the removed `imp` module, which was addressed by D-1.9. No other compatibility issues encountered.
**Implemented in:** requirements.txt (note), README.md.

### D-1.2 Environment isolation: venv
**Decided:** Use Python's built-in `venv`.
**Alternatives:** conda, pipenv, poetry.
**Rationale:** venv is zero-dependency, ships with Python, and produces a `requirements.txt` that is straightforward to reproduce. conda adds unnecessary overhead for a pure-Python research project.
**Implemented in:** README.md setup instructions.

### D-1.3 Notebook platform: local Jupyter Lab
**Decided:** Canonical environment is local Jupyter Lab; Colab acceptable for prototyping.
**Alternatives:** Google Colab exclusively, VS Code notebooks.
**Rationale:** Simulation runs (30 Monte Carlo × 300s) exceed Colab's ~90 minute idle timeout. Local Jupyter has no such limitation. VS Code is also excellent but requires an extension; Jupyter Lab is the de facto standard in academic ML research.
**Implemented in:** README.md.

### D-1.4 Project structure: modular .py files
**Decided:** Logic lives in importable Python modules under `src/`; notebooks are thin orchestration layers.
**Alternatives:** Single monolithic notebook.
**Rationale:** Modular files are independently testable with pytest. A monolithic notebook cannot be unit-tested cleanly. The modular approach also makes it easier to reuse (e.g. import `fuzzy_engine` directly into a new experiment notebook without copy-pasting).
**Implemented in:** Directory structure (`src/config.py`, `src/fuzzy_engine.py`, etc.).

### D-1.5 FIS type: Mamdani
**Decided:** Use Mamdani inference system.
**Alternatives:** Sugeno (Takagi-Sugeno), Tsukamoto.
**Rationale:**
- Mamdani output membership functions are linguistic fuzzy sets — interpretable in plain language ("the urgency is high"). Sugeno outputs are crisp weighted averages — more efficient but harder to explain.
- PhD examiners will ask "what does the system think?". Mamdani allows you to plot the output fuzzy set and show the CoG visually. Sugeno does not.
- Mamdani is the dominant choice in published handover FIS research (Youssef et al. 2019; Balan et al. 2011; Nkansah-Gyekye & Agbinya 2014) — important for direct comparability.
**Implemented in:** `src/fuzzy_engine.py` (ctrl.Consequent, build_fis()).

### D-1.6 Defuzzification: Centroid (CoG)
**Decided:** Centre of Gravity defuzzification.
**Alternatives:** Bisector, Mean of Maximum (MOM), Smallest of Maximum, Largest of Maximum.
**Rationale:** CoG is the most widely used method in the literature this work builds on. It considers the entire output fuzzy set (not just the peak), making it robust to partial rule firing. MOM only considers the peak region, losing information from partially fired rules. Bisector is theoretically cleaner but computationally equivalent and less common.
**Implemented in:** `src/fuzzy_engine.py` (`defuzzify_method='centroid'`).

### D-1.7 Membership function shape: triangular + trapezoidal
**Decided:** Triangular MFs for middle terms; trapezoidal for boundary terms.
**Alternatives:** Gaussian (bell-shaped), sigmoidal, uniform.
**Rationale:**
- Triangular MFs have exactly one parameter (peak) plus two shoulders. Simple to tune.
- Boundary terms (e.g. "Poor RSRP" at the extreme low end) have no natural left shoulder — trapezoidal functions allow a flat top that extends to the universe boundary.
- Gaussian MFs never reach exactly 0 at the boundary, which causes rule firing at extremes — undesirable for hard physical limits like −120 dBm (no service) or 0% battery.
**Implemented in:** `src/fuzzy_engine.py` (_trimf, _trapmf helpers); boundaries in `src/config.py`.

### D-1.8 Package versioning: pinned
**Decided:** All packages pinned to specific versions in requirements.txt.
**Alternatives:** Unpinned (>=).
**Rationale:** Research reproducibility requires identical environments. An unpinned `numpy` could resolve to a version with different floating-point rounding, producing different simulation outputs for the same random seed. All versions are frozen.
**Implemented in:** `requirements.txt`.

### D-1.9 scikit-fuzzy version: 0.5.0 (dev/git) instead of 0.4.2
**Decided:** Install scikit-fuzzy from the GitHub main branch (resolves to 0.5.0).
**Alternatives:** scikit-fuzzy 0.4.2 (PyPI stable).
**Rationale:** 0.4.2 imports the `imp` module which was removed in Python 3.12, causing an immediate ImportError. The GitHub main branch has patched this. No functional differences in the FIS API were encountered.
**Risk:** The dev branch could change. Mitigated by pinning to the specific commit hash in a production deployment.
**Implemented in:** requirements.txt comment; install command in README.md.

---

## Phase 1 — Fuzzy system design (membership functions and rules)

### D-2.1 RSRP universe: −120 to −44 dBm
**Decided:** [−120, −44] dBm.
**Source:** 3GPP TS 38.133 Table 10.1.6.1-1 (full range −156 to −31); clipped to practical deployment range.
**Rationale:** Below −120 dBm the UE cannot maintain any connection. Above −44 dBm the UE is essentially inside the BS antenna enclosure — not a realistic scenario in simulation.
**Implemented in:** `src/config.py` (RSRP_MIN, RSRP_MAX).

### D-2.2 SINR universe: −10 to +30 dB
**Decided:** [−10, +30] dB.
**Rationale:** Below −10 dB the physical layer cannot decode data frames; the connection drops. Above +30 dB the channel is interference-free; this ceiling covers all realistic HetNet interference scenarios including heavy co-channel interference between macro and pico layers.
**Implemented in:** `src/config.py` (SINR_MIN, SINR_MAX).

### D-2.3 Velocity universe: 0 to 120 km/h
**Decided:** [0, 120] km/h.
**Source:** 3GPP TS 36.331 s5.3.12.1 defines mobility state estimation using TCRmax thresholds. The "High Mobility" state begins at ~60 km/h.
**Rationale:** Covers pedestrian (0–5), slow vehicular (15–60), fast vehicular (60–120). Above 120 km/h a UE crosses a pico cell (200–300m radius) in under 9 seconds — faster than the handover signalling latency, making any model's decisions moot.
**Implemented in:** `src/config.py` (VELOCITY_MIN, VELOCITY_MAX).

### D-2.4 Cell load universe: 0 to 100 %
**Decided:** PRB utilisation percentage [0, 100].
**Rationale:** Physical Resource Block utilisation is the standard LTE/NR load metric, directly observable by the network management layer and commonly logged in OSS data.
**Implemented in:** `src/config.py` (LOAD_MIN, LOAD_MAX).

### D-2.5 Battery universe: 0 to 100 %
**Decided:** Device battery percentage [0, 100].
**Rationale:** Battery level is reported via the UE capability information element in LTE (TS 36.306). The "Critical" boundary at 15% aligns with Android/iOS power-saving mode activation thresholds.
**Implemented in:** `src/config.py` (BATTERY_MIN, BATTERY_MAX).

### D-2.6 Output universe: 0.0 to 1.0 (normalised)
**Decided:** HO urgency score on [0.0, 1.0].
**Alternatives:** Raw dB scale, categorical output (0/1/2).
**Rationale:** Normalised output decouples the inference engine from the trigger mechanism. The threshold θ can be tuned independently (Phase 5) without touching the FIS. A categorical output (discrete classes) loses the graduation — a score of 0.59 vs 0.61 is physically different and the threshold should be tunable.
**Implemented in:** `src/fuzzy_engine.py` (Consequent definition); threshold in `src/config.py` HO_THRESHOLD.

### D-2.7 Linguistic terms: 3 per input
**Decided:** Three terms per input (Low/Medium/High or equivalent).
**Alternatives:** 5 terms (Very Low, Low, Medium, High, Very High), 7 terms.
**Rationale:**
- 3 terms → max 3^5 = 243 rules; expert-curated to 33.
- 5 terms → max 5^5 = 3,125 rules; impractical for manual expert curation.
- 3 terms matches the granularity of 3GPP measurement quantisation (RSRP reported in 1 dBm steps; the absolute range meaningful variation spans ~3 quality zones).
- Published comparators use 3 terms (Nkansah-Gyekye 2014; Balan 2011).
**Implemented in:** `src/fuzzy_engine.py` (3 MFs per antecedent); boundaries in `src/config.py`.

### D-2.7b MF boundary values
**Decided:** Specific boundary values for all MFs (see config.py).
**Source:** 3GPP TS 36.902 (cell range); TS 38.214 (MCS selection SINR thresholds); TS 36.331 (mobility state estimation velocity thresholds); ETSI TR 136 902 (operational load guidelines).
**Rationale:** Using 3GPP-sourced boundaries is a research contribution in itself — most published FIS work uses arbitrary boundaries without justification. This makes the boundary choices defensible in a thesis.
**Implemented in:** `src/config.py` (all *_BOUNDS constants with inline citations).

### D-2.8 Rule count: 33 curated rules (from 243 theoretical maximum)
**Decided:** 33 expert-curated rules.
**Alternatives:** Full 243-rule Cartesian product, genetic algorithm rule generation, neuro-fuzzy learning.
**Rationale:**
- The Cartesian product contains many physically unrealistic or redundant combinations (e.g. RSRP 'good' + SINR 'low' rarely co-occur in a macro cell; slow UE + good signal + low load is three different rules that all conclude 'stay').
- Expert reduction follows a 3-tier priority (critical signal → mobility → context) that is easy to explain and defend.
- Neuro-fuzzy (ANFIS) learning would require a labelled training dataset of real handover decisions, which is not available without network operator access. Expert rules are the methodologically correct choice given this constraint.
**Implemented in:** `src/fuzzy_engine.py` (rules list, R01–R33 with comments).

### D-2.9 Handover threshold θ: 0.6
**Decided:** θ = 0.6 as the starting value for Phase 1.
**Alternatives:** 0.5 (Balan et al.), 0.65 (Youssef et al.).
**Rationale:** 0.6 is the midpoint of the range reported in literature. Treated as a hyperparameter; Phase 5 sweeps [0.4, 0.8] in steps of 0.05 and selects θ that minimises a combined KPI loss function.
**Implemented in:** `src/config.py` (HO_THRESHOLD); used in `src/fuzzy_engine.py` (decide()).

### D-2.10 R33 added: unconditional rule for poor RSRP + low SINR
**Decided:** Add R33 as a reinforcing rule when both primary signal metrics are definitively poor.
**Trigger:** Unit test failure — test_critical_battery_poor_signal_still_triggers returned score=0.557.
**Root cause:** R19 (battery_critical & rsrp_medium → low) fired at μ=0.667 via partial membership overlap even though rsrp_medium at −112 dBm was 0.0. Investigation showed this was NOT R19 — see D-2.11.
**Implemented in:** `src/fuzzy_engine.py` (R33 in rules list).

### D-2.11 R26 refined: added RSRP signal quality guard
**Decided:** R26 ("slow UE + low load → stay") was modified to require signal quality to be at least 'medium' or 'good'.
**Trigger:** Unit test failure — same test as D-2.10. Investigation revealed R26 fired at μ=0.667 because velocity=20 km/h has partial membership (0.667) in the 'slow' term ([0, 0, 15, 30] trapezoid — at 20, the UE is 33% of the way from the 'slow' plateau to the 'moderate' peak, so slow membership = (30−20)/(30−15) = 0.667). This competing "low" consequent pulled the CoG below θ=0.6.
**Physical justification:** It is correct that a slow UE at low load should stay — but only when the signal quality is acceptable. When RSRP is critically poor (−112 dBm), the slow/low-load combination is irrelevant; the physical link is failing and the UE must handover. The RSRP guard captures this correctly.
**Implemented in:** `src/fuzzy_engine.py` (R26, rule condition updated).

---

## Phases 3–5 decisions (to be added as each phase is built)

Placeholders for future decisions:
- D-3.x: 3GPP path loss model selection (UMa vs UMi)
- D-3.x: Shadow fading standard deviation value
- D-3.x: Random Waypoint pause time distribution
- D-4.x: SimPy event scheduling strategy
- D-4.x: A3 baseline TTT and offset values
- D-5.x: Statistical test selection (Mann-Whitney U vs t-test)
- D-5.x: Final θ value after sensitivity sweep
