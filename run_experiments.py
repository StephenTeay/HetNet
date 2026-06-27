"""
run_experiments.py — Monte Carlo experiment runner for Phase 5.

Runs all combinations of:
  scenarios  × 3 (pedestrian, slow_vehicle, fast_vehicle)
  algorithms × 2 (fis, baseline)
  runs       × 30 (Monte Carlo seeds)
  = 180 total simulation runs

Also runs θ sensitivity sweep:
  θ values × 7 (0.4 → 0.7 step 0.05)
  scenarios × 3
  runs      × 10 per θ
  = 210 sensitivity runs

All results saved to results/data/ as CSVs.
Decision D-5.5: pre-compute and save; dashboard reads CSVs.

Usage:
    python3 run_experiments.py            # full experiment (~30 min)
    python3 run_experiments.py --quick    # 5 runs per combo (~3 min, for testing)
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import config as cfg
from simulation import run_simulation, run_monte_carlo

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS  = ['pedestrian', 'slow_vehicle', 'fast_vehicle']
ALGORITHMS = ['fis', 'baseline']
THETA_VALUES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def run_main_experiment(num_runs: int):
    """Run the primary 30-run Monte Carlo across all scenarios and algorithms."""
    print(f"\n{'='*60}")
    print(f"MAIN EXPERIMENT  ({num_runs} runs × 3 scenarios × 2 algorithms)")
    print(f"{'='*60}\n")

    all_rows = []
    t_total = time.time()

    for scenario in SCENARIOS:
        for algo in ALGORITHMS:
            print(f"  ► {algo:8s} | {scenario} ...", end=' ', flush=True)
            t0 = time.time()
            df = run_monte_carlo(
                scenario=scenario,
                algorithm=algo,
                num_runs=num_runs,
                seed_base=cfg.RANDOM_SEED_BASE,
            )
            elapsed = time.time() - t0
            all_rows.append(df)
            s = df.describe().loc['mean']
            print(f"done in {elapsed:.0f}s  |  "
                  f"HOs={s['ho_count']:.0f}  "
                  f"pp={s['ping_pong_rate']:.3f}  "
                  f"drops={s['call_drop_count']:.1f}  "
                  f"SINR={s['mean_sinr_db']:.1f}dB")

    results = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, 'main_results.csv')
    results.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {len(results)} rows → {out_path}")
    print(f"  Total time: {time.time()-t_total:.0f}s")
    return results


def run_sensitivity_experiment(num_runs: int):
    """Sweep θ threshold and record KPI changes."""
    print(f"\n{'='*60}")
    print(f"SENSITIVITY SWEEP  (θ ∈ {THETA_VALUES}, {num_runs} runs each)")
    print(f"{'='*60}\n")

    rows = []
    t_total = time.time()

    for theta in THETA_VALUES:
        # Temporarily override threshold in config
        cfg.HO_THRESHOLD = theta

        for scenario in SCENARIOS:
            print(f"  θ={theta:.2f} | {scenario} ...", end=' ', flush=True)
            t0 = time.time()
            df = run_monte_carlo(
                scenario=scenario,
                algorithm='fis',
                num_runs=num_runs,
                seed_base=cfg.RANDOM_SEED_BASE,
            )
            df['theta'] = theta
            rows.append(df)
            s = df.describe().loc['mean']
            print(f"done {time.time()-t0:.0f}s  "
                  f"HOs={s['ho_count']:.0f}  pp={s['ping_pong_rate']:.3f}  "
                  f"SINR={s['mean_sinr_db']:.1f}dB")

    # Restore default
    cfg.HO_THRESHOLD = 0.6

    sensitivity = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, 'sensitivity_results.csv')
    sensitivity.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {len(sensitivity)} rows → {out_path}")
    print(f"  Total time: {time.time()-t_total:.0f}s")
    return sensitivity


def run_statistical_tests(results: pd.DataFrame):
    """
    Mann-Whitney U test comparing FIS vs baseline per scenario and KPI.
    Decision D-5.4: non-parametric test; report p-value and effect size.
    """
    from scipy import stats

    print(f"\n{'='*60}")
    print("STATISTICAL TESTS  (Mann-Whitney U, α=0.05)")
    print(f"{'='*60}\n")

    kpis = ['ho_count', 'ping_pong_rate', 'unnecessary_ho_rate',
            'call_drop_count', 'mean_sinr_db']

    stat_rows = []

    for scenario in SCENARIOS:
        fis_df  = results[(results.scenario==scenario) & (results.algorithm=='fis')]
        base_df = results[(results.scenario==scenario) & (results.algorithm=='baseline')]

        print(f"  Scenario: {scenario}")
        for kpi in kpis:
            fis_vals  = fis_df[kpi].dropna().values
            base_vals = base_df[kpi].dropna().values

            if len(fis_vals) < 2 or len(base_vals) < 2:
                continue

            stat, p = stats.mannwhitneyu(fis_vals, base_vals, alternative='two-sided')

            # Rank-biserial correlation as effect size
            n1, n2 = len(fis_vals), len(base_vals)
            r = 1 - (2 * stat) / (n1 * n2)

            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            direction = ('FIS↓' if fis_vals.mean() < base_vals.mean() else 'FIS↑')

            print(f"    {kpi:25s}  p={p:.4f} {sig}  r={r:+.3f}  {direction}  "
                  f"FIS={fis_vals.mean():.3f}  Base={base_vals.mean():.3f}")

            stat_rows.append({
                'scenario': scenario, 'kpi': kpi,
                'fis_mean': fis_vals.mean(), 'baseline_mean': base_vals.mean(),
                'p_value': p, 'effect_size_r': r,
                'significant': p < 0.05, 'direction': direction,
            })
        print()

    stat_df = pd.DataFrame(stat_rows)
    out_path = os.path.join(RESULTS_DIR, 'statistical_tests.csv')
    stat_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved → {out_path}")
    return stat_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Run 5 iterations instead of 30 (for testing)')
    args = parser.parse_args()

    num_runs_main = 5 if args.quick else cfg.NUM_MONTE_CARLO
    num_runs_sens = 3 if args.quick else 10

    print(f"\nHetNet Fuzzy Handover — Phase 5 Experiments")
    print(f"Mode: {'QUICK (5 runs)' if args.quick else 'FULL (30 runs)'}")

    results    = run_main_experiment(num_runs=num_runs_main)
    sensitivity = run_sensitivity_experiment(num_runs=num_runs_sens)
    stat_df    = run_statistical_tests(results)

    print(f"\n{'='*60}")
    print("All experiments complete. Run the dashboard with:")
    print("  streamlit run dashboard.py")
    print(f"{'='*60}\n")
