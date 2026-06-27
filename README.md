# Fuzzy Logic-Based Handover Optimization for HetNets

## Research overview
Design, implementation, and simulation-based evaluation of a Fuzzy Inference System
for handover optimization in Heterogeneous Cellular Networks (HetNets), comparing
against the 3GPP A3 event baseline across three mobility scenarios.

## Project structure
```
hetnet_fuzzy/
├── src/
│   ├── config.py          # All constants and tunable parameters (single source of truth)
│   ├── fuzzy_engine.py    # Fuzzy Inference System: membership functions + rule base
│   ├── channel_model.py   # 3GPP path loss, RSRP/RSRQ/SINR computation  [Phase 3]
│   ├── mobility.py        # Random Waypoint mobility model                [Phase 3]
│   ├── baseline.py        # 3GPP A3 event handover algorithm              [Phase 4]
│   ├── simulation.py      # SimPy discrete-event simulation engine        [Phase 4]
│   └── metrics.py         # KPI collection and statistical analysis       [Phase 5]
├── tests/
│   ├── test_fuzzy.py      # Unit tests for FIS (sanity checks)
│   ├── test_channel.py    # Unit tests for path loss computation          [Phase 3]
│   └── test_mobility.py   # Unit tests for mobility model                 [Phase 3]
├── notebooks/
│   └── 01_phase1_fuzzy_system.ipynb
├── results/
│   ├── plots/
│   └── data/
├── requirements.txt
└── README.md
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Key decisions
See docs/decisions.md for full log.
