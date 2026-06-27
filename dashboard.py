"""
dashboard.py — Streamlit dashboard for HetNet Fuzzy Handover results.

Tabs:
  1. Overview      — headline KPI cards and bar charts
  2. Scenarios     — per-scenario box plots and SINR histograms
  3. Sensitivity   — θ sweep line charts
  4. Run Experiments — in-app simulation runner with live progress
  5. Raw Data      — downloadable tables

Decision D-5.6 : Four result tabs + one runner tab.
Decision D-5.5 : Results read from results/data/ CSVs.
Decision D-6.2 : CSVs committed to repo for instant cloud load.
Decision D-6.3 : Cloud-safe runner — warns user about resource limits,
                 exposes Quick/Standard/Full modes, streams live progress.
"""

import os, sys, time, threading
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HetNet Fuzzy Handover",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = os.path.join(_HERE, 'results', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_FIS      = '#185FA5'
C_BASELINE = '#E24B4A'
C_ACCENT   = '#0F6E56'
C_LIGHT    = '#F1EFE8'

SCENARIO_LABELS = {
    'pedestrian':   'Pedestrian\n(0–5 km/h)',
    'slow_vehicle': 'Slow Vehicle\n(15–40 km/h)',
    'fast_vehicle': 'Fast Vehicle\n(60–100 km/h)',
}

KPI_META = {
    'ho_count':            ('Handover Count',      'lower is better',  False),
    'ping_pong_rate':      ('Ping-Pong Rate',       'lower is better',  False),
    'unnecessary_ho_rate': ('Unnecessary HO Rate',  'lower is better',  False),
    'call_drop_count':     ('Call Drop Count',      'lower is better',  False),
    'mean_sinr_db':        ('Mean SINR (dB)',       'higher is better', True),
}

SCENARIOS  = ['pedestrian', 'slow_vehicle', 'fast_vehicle']
ALGORITHMS = ['fis', 'baseline']


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    main_path = os.path.join(RESULTS_DIR, 'main_results.csv')
    sens_path = os.path.join(RESULTS_DIR, 'sensitivity_results.csv')
    stat_path = os.path.join(RESULTS_DIR, 'statistical_tests.csv')
    main = pd.read_csv(main_path) if os.path.exists(main_path) else None
    sens = pd.read_csv(sens_path) if os.path.exists(sens_path) else None
    stat = pd.read_csv(stat_path) if os.path.exists(stat_path) else None
    return main, sens, stat


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _box_plot(ax, df, kpi, scenario, title):
    fv = df[(df.scenario==scenario) & (df.algorithm=='fis')][kpi].dropna()
    bv = df[(df.scenario==scenario) & (df.algorithm=='baseline')][kpi].dropna()
    bp = ax.boxplot([fv, bv], labels=['FIS','Baseline'], patch_artist=True,
                   medianprops=dict(color='white', linewidth=2),
                   whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                   flierprops=dict(marker='o', markersize=4, alpha=0.5))
    bp['boxes'][0].set_facecolor(C_FIS)
    bp['boxes'][1].set_facecolor(C_BASELINE)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_facecolor(C_LIGHT)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.spines[['top','right']].set_visible(False)


def _bar_comparison(ax, df, kpi, ylabel):
    x = np.arange(len(SCENARIOS)); w = 0.35
    fm = [df[(df.scenario==s)&(df.algorithm=='fis')][kpi].mean() for s in SCENARIOS]
    bm = [df[(df.scenario==s)&(df.algorithm=='baseline')][kpi].mean() for s in SCENARIOS]
    fs = [df[(df.scenario==s)&(df.algorithm=='fis')][kpi].std() for s in SCENARIOS]
    bs = [df[(df.scenario==s)&(df.algorithm=='baseline')][kpi].std() for s in SCENARIOS]
    ax.bar(x-w/2, fm, w, yerr=fs, label='FIS',      color=C_FIS,      capsize=4, alpha=0.9, error_kw={'linewidth':1.2})
    ax.bar(x+w/2, bm, w, yerr=bs, label='Baseline', color=C_BASELINE, capsize=4, alpha=0.9, error_kw={'linewidth':1.2})
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_facecolor(C_LIGHT)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner (runs in the same process — Streamlit-safe)
# Decision D-6.3: inline runner avoids subprocess and file-locking issues
# ─────────────────────────────────────────────────────────────────────────────

def _run_experiments_inline(num_runs_main, num_runs_sens, theta_values,
                             sim_duration, num_ue, log_fn, progress_fn):
    """
    Run the full experiment pipeline inline (no subprocess).
    Calls log_fn(msg) for live log lines and progress_fn(fraction) for the bar.
    Saves CSVs to RESULTS_DIR when done.
    Returns (success: bool, error_msg: str)
    """
    try:
        import config as cfg
        from simulation import run_monte_carlo
        from scipy import stats as scipy_stats

        # Override simulation parameters
        cfg.SIM_DURATION_S  = sim_duration
        cfg.NUM_UE          = num_ue
        cfg.NUM_MONTE_CARLO = num_runs_main

        total_steps = (
            len(SCENARIOS) * len(ALGORITHMS) * num_runs_main +   # main
            len(SCENARIOS) * len(theta_values) * num_runs_sens    # sensitivity
        )
        completed = [0]

        def tick(n=1):
            completed[0] += n
            progress_fn(min(completed[0] / total_steps, 0.99))

        # ── Main experiment ───────────────────────────────────────────────────
        log_fn("▶ Starting main experiment...")
        all_rows = []
        for scenario in SCENARIOS:
            for algo in ALGORITHMS:
                log_fn(f"   {algo:8s} | {scenario} ({num_runs_main} runs)...")
                t0 = time.time()
                df = run_monte_carlo(
                    scenario=scenario, algorithm=algo,
                    num_runs=num_runs_main,
                    seed_base=cfg.RANDOM_SEED_BASE,
                )
                elapsed = time.time() - t0
                s = df.mean(numeric_only=True)
                log_fn(f"   ✓ done {elapsed:.0f}s | "
                       f"HOs={s['ho_count']:.0f} "
                       f"pp={s['ping_pong_rate']:.3f} "
                       f"SINR={s['mean_sinr_db']:.1f}dB")
                all_rows.append(df)
                tick(num_runs_main)

        main_results = pd.concat(all_rows, ignore_index=True)
        main_results.to_csv(os.path.join(RESULTS_DIR, 'main_results.csv'), index=False)
        log_fn(f"✓ main_results.csv saved ({len(main_results)} rows)")

        # ── Sensitivity sweep ─────────────────────────────────────────────────
        log_fn("\n▶ Starting θ sensitivity sweep...")
        sens_rows = []
        for theta in theta_values:
            cfg.HO_THRESHOLD = theta
            for scenario in SCENARIOS:
                log_fn(f"   θ={theta:.2f} | {scenario} ({num_runs_sens} runs)...")
                df = run_monte_carlo(
                    scenario=scenario, algorithm='fis',
                    num_runs=num_runs_sens,
                    seed_base=cfg.RANDOM_SEED_BASE,
                )
                df['theta'] = theta
                sens_rows.append(df)
                s = df.mean(numeric_only=True)
                log_fn(f"   ✓ HOs={s['ho_count']:.0f} pp={s['ping_pong_rate']:.3f}")
                tick(num_runs_sens)

        cfg.HO_THRESHOLD = 0.6   # restore default
        sensitivity = pd.concat(sens_rows, ignore_index=True)
        sensitivity.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_results.csv'), index=False)
        log_fn(f"✓ sensitivity_results.csv saved ({len(sensitivity)} rows)")

        # ── Statistical tests ─────────────────────────────────────────────────
        log_fn("\n▶ Running statistical tests (Mann-Whitney U)...")
        kpis = list(KPI_META.keys())
        stat_rows = []
        for scenario in SCENARIOS:
            fdf = main_results[(main_results.scenario==scenario)&(main_results.algorithm=='fis')]
            bdf = main_results[(main_results.scenario==scenario)&(main_results.algorithm=='baseline')]
            for kpi in kpis:
                fv = fdf[kpi].dropna().values
                bv = bdf[kpi].dropna().values
                if len(fv) < 2 or len(bv) < 2:
                    continue
                stat, p = scipy_stats.mannwhitneyu(fv, bv, alternative='two-sided')
                r = 1 - (2 * stat) / (len(fv) * len(bv))
                stat_rows.append({
                    'scenario': scenario, 'kpi': kpi,
                    'fis_mean': round(fv.mean(), 4),
                    'baseline_mean': round(bv.mean(), 4),
                    'p_value': round(p, 6),
                    'effect_size_r': round(r, 4),
                    'significant': p < 0.05,
                    'direction': 'FIS lower' if fv.mean() < bv.mean() else 'FIS higher',
                })

        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(os.path.join(RESULTS_DIR, 'statistical_tests.csv'), index=False)
        log_fn(f"✓ statistical_tests.csv saved ({len(stat_df)} rows)")

        progress_fn(1.0)
        log_fn("\n✅ All experiments complete. Reloading dashboard...")
        return True, ""

    except Exception as e:
        import traceback
        return False, traceback.format_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📡 HetNet FIS")
    st.markdown("**Fuzzy Logic Handover Optimization**")
    st.markdown("---")
    st.markdown("**Model:** Mamdani FIS · 5 inputs · 33 rules")
    st.markdown("**Baseline:** 3GPP A3 event")
    st.markdown("**Scenarios:** Pedestrian · Slow vehicle · Fast vehicle")
    st.markdown("---")
    st.markdown("#### HetNet topology")
    st.markdown("1 macro + 3 pico + 2 femto cells")
    st.markdown("500 × 500 m simulation area")
    st.markdown("---")
    st.caption("Open the **▶ Run Experiments** tab to regenerate results.")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Fuzzy Logic-Based Handover Optimization in HetNets")
st.markdown("*Comparative evaluation: Mamdani FIS vs 3GPP A3 event baseline*")

main_df, sens_df, stat_df = load_data()

tab_overview, tab_scenarios, tab_sensitivity, tab_runner, tab_raw = st.tabs([
    "📊 Overview",
    "📶 Scenarios",
    "🎛️ Sensitivity (θ)",
    "▶ Run Experiments",
    "📋 Raw Data",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_overview:
    if main_df is None:
        st.info("No results yet. Go to **▶ Run Experiments** to generate them.")
    else:
        st.header("Overall Performance Summary")

        fis_all  = main_df[main_df.algorithm=='fis']
        base_all = main_df[main_df.algorithm=='baseline']
        ho_reduction = (1 - fis_all.ho_count.mean() / base_all.ho_count.mean()) * 100
        pp_reduction = (1 - fis_all.ping_pong_rate.mean() / base_all.ping_pong_rate.mean()) * 100
        sinr_delta   = fis_all.mean_sinr_db.mean() - base_all.mean_sinr_db.mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("HO Count Reduction", f"{ho_reduction:.0f}%",
                      delta=f"FIS {fis_all.ho_count.mean():.0f} vs {base_all.ho_count.mean():.0f}",
                      delta_color="inverse")
        with c2:
            st.metric("Ping-Pong Reduction", f"{pp_reduction:.0f}%",
                      delta=f"FIS {fis_all.ping_pong_rate.mean():.3f} vs {base_all.ping_pong_rate.mean():.3f}",
                      delta_color="inverse")
        with c3:
            st.metric("Mean SINR Δ", f"{sinr_delta:+.1f} dB",
                      delta=f"FIS {fis_all.mean_sinr_db.mean():.1f} dB",
                      delta_color="normal" if sinr_delta >= 0 else "inverse")
        with c4:
            st.metric("Call Drops (FIS)", f"{fis_all.call_drop_count.mean():.1f}",
                      delta_color="off")

        st.markdown("---")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor('white')
        _bar_comparison(axes[0], main_df, 'ho_count', 'Mean Handover Count')
        axes[0].set_title('Handover Count by Scenario', fontweight='bold')
        _bar_comparison(axes[1], main_df, 'ping_pong_rate', 'Mean Ping-Pong Rate')
        axes[1].set_title('Ping-Pong Rate by Scenario', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        if stat_df is not None:
            st.subheader("Statistical Significance (Mann-Whitney U, α=0.05)")
            disp = stat_df.copy()
            disp['p_value'] = disp['p_value'].map(lambda p:
                f"{p:.4f} ***" if p<0.001 else
                f"{p:.4f} **"  if p<0.01  else
                f"{p:.4f} *"   if p<0.05  else f"{p:.4f} ns")
            disp['fis_mean']      = disp['fis_mean'].map('{:.3f}'.format)
            disp['baseline_mean'] = disp['baseline_mean'].map('{:.3f}'.format)
            disp['effect_size_r'] = disp['effect_size_r'].map('{:+.3f}'.format)
            st.dataframe(disp[['scenario','kpi','fis_mean','baseline_mean',
                               'p_value','effect_size_r','direction']],
                         use_container_width=True, hide_index=True)
            st.caption("*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

with tab_scenarios:
    if main_df is None:
        st.info("No results yet. Go to **▶ Run Experiments** to generate them.")
    else:
        st.header("Per-Scenario Analysis")
        sel = st.selectbox("Select scenario", SCENARIOS,
                           format_func=lambda s: SCENARIO_LABELS[s].replace('\n',' '))

        sfis  = main_df[(main_df.scenario==sel)&(main_df.algorithm=='fis')]
        sbase = main_df[(main_df.scenario==sel)&(main_df.algorithm=='baseline')]

        cols = st.columns(5)
        for col, (kpi, (label, hint, hi)) in zip(cols, KPI_META.items()):
            fv = sfis[kpi].mean(); bv = sbase[kpi].mean(); dv = fv - bv
            with col:
                st.metric(label, f"{fv:.3f}", delta=f"{dv:+.3f} vs baseline",
                          delta_color="normal" if (hi == (dv>0)) else "inverse", help=hint)

        st.markdown("---")
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        fig.patch.set_facecolor('white')
        fig.suptitle(f"KPI Distributions — {SCENARIO_LABELS[sel].replace(chr(10),' ')}",
                     fontweight='bold', fontsize=12, y=1.02)
        for ax, (kpi, (label,_,__)) in zip(axes, KPI_META.items()):
            _box_plot(ax, main_df, kpi, sel, label)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("SINR Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        fig2.patch.set_facecolor('white'); ax2.set_facecolor(C_LIGHT)
        for algo, color, lbl in [('fis',C_FIS,'FIS'),('baseline',C_BASELINE,'Baseline')]:
            vals = main_df[(main_df.scenario==sel)&(main_df.algorithm==algo)]['mean_sinr_db']
            ax2.hist(vals, bins=8, alpha=0.7, color=color, label=lbl, edgecolor='white')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='0 dB')
        ax2.set_xlabel('Mean SINR (dB)', fontsize=10); ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Mean SINR across Monte Carlo runs', fontweight='bold')
        ax2.legend(); ax2.grid(axis='y', alpha=0.4); ax2.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

with tab_sensitivity:
    if sens_df is None:
        st.info("No sensitivity results yet. Go to **▶ Run Experiments** to generate them.")
    else:
        st.header("Threshold Sensitivity Analysis (θ sweep)")
        st.markdown("Lower θ = more aggressive handover triggering. Higher θ = more conservative.")

        theta_vals = sorted(sens_df['theta'].unique())
        colors = [C_FIS, C_ACCENT, C_BASELINE]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor('white')
        for ax, kpi, ylabel in zip(axes,
                                   ['ho_count','ping_pong_rate'],
                                   ['Mean Handover Count','Mean Ping-Pong Rate']):
            ax.set_facecolor(C_LIGHT)
            for scenario, color in zip(SCENARIOS, colors):
                means = [sens_df[(sens_df.theta==t)&(sens_df.scenario==scenario)][kpi].mean() for t in theta_vals]
                stds  = [sens_df[(sens_df.theta==t)&(sens_df.scenario==scenario)][kpi].std()  for t in theta_vals]
                ax.plot(theta_vals, means, 'o-', color=color,
                        label=SCENARIO_LABELS[scenario].replace('\n',' '), linewidth=2)
                ax.fill_between(theta_vals, [m-s for m,s in zip(means,stds)],
                                [m+s for m,s in zip(means,stds)], color=color, alpha=0.15)
            ax.axvline(x=0.6, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='θ=0.6 default')
            ax.set_xlabel('θ'); ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs θ', fontweight='bold')
            ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        fig2, ax = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor('white'); ax.set_facecolor(C_LIGHT)
        for scenario, color in zip(SCENARIOS, colors):
            means = [sens_df[(sens_df.theta==t)&(sens_df.scenario==scenario)]['mean_sinr_db'].mean() for t in theta_vals]
            ax.plot(theta_vals, means, 'o-', color=color,
                    label=SCENARIO_LABELS[scenario].replace('\n',' '), linewidth=2)
        ax.axvline(x=0.6, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='θ=0.6')
        ax.axhline(y=0,   color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel('θ'); ax.set_ylabel('Mean SINR (dB)')
        ax.set_title('Mean SINR vs θ', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.4); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RUN EXPERIMENTS  (Decision D-6.3)
# ─────────────────────────────────────────────────────────────────────────────

with tab_runner:
    st.header("▶ Run Experiments")
    st.markdown(
        "Regenerate all simulation results from scratch. "
        "Results are saved to `results/data/` and charts update automatically when done."
    )

    # ── Environment detection ──────────────────────────────────────────────────
    on_cloud = os.environ.get('STREAMLIT_SHARING_MODE') == '1' or \
               'HOSTNAME' in os.environ and 'streamlit' in os.environ.get('HOSTNAME','').lower()

    # ── Mode selector ─────────────────────────────────────────────────────────
    st.subheader("1. Choose a run mode")

    mode_col1, mode_col2, mode_col3 = st.columns(3)

    with mode_col1:
        st.markdown("""
        <div style="border:2px solid #0F6E56; border-radius:10px; padding:16px; min-height:160px">
        <h4 style="color:#0F6E56; margin:0">⚡ Quick</h4>
        <p style="font-size:13px; color:#444; margin-top:8px">
        5 Monte Carlo runs<br>
        3 θ values<br>
        30s simulation · 5 UEs<br><br>
        <strong>~2 minutes</strong><br>
        ✅ Safe on Streamlit Cloud
        </p>
        </div>
        """, unsafe_allow_html=True)

    with mode_col2:
        st.markdown("""
        <div style="border:2px solid #185FA5; border-radius:10px; padding:16px; min-height:160px">
        <h4 style="color:#185FA5; margin:0">📊 Standard</h4>
        <p style="font-size:13px; color:#444; margin-top:8px">
        15 Monte Carlo runs<br>
        5 θ values<br>
        60s simulation · 10 UEs<br><br>
        <strong>~8 minutes</strong><br>
        ⚠️ May be slow on Cloud
        </p>
        </div>
        """, unsafe_allow_html=True)

    with mode_col3:
        st.markdown("""
        <div style="border:2px solid #E24B4A; border-radius:10px; padding:16px; min-height:160px">
        <h4 style="color:#E24B4A; margin:0">🔬 Full</h4>
        <p style="font-size:13px; color:#444; margin-top:8px">
        30 Monte Carlo runs<br>
        7 θ values<br>
        300s simulation · 20 UEs<br><br>
        <strong>~45 minutes</strong><br>
        🖥️ Run locally only
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    mode = st.radio(
        "Select mode",
        options=["Quick", "Standard", "Full"],
        horizontal=True,
        label_visibility="collapsed",
    )

    MODE_PARAMS = {
        "Quick":    dict(num_runs_main=5,  num_runs_sens=3,  sim_duration=30,  num_ue=5,
                         theta_values=[0.4, 0.6, 0.8]),
        "Standard": dict(num_runs_main=15, num_runs_sens=5,  sim_duration=60,  num_ue=10,
                         theta_values=[0.4, 0.5, 0.6, 0.7, 0.8]),
        "Full":     dict(num_runs_main=30, num_runs_sens=10, sim_duration=300, num_ue=20,
                         theta_values=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]),
    }
    params = MODE_PARAMS[mode]

    # ── Summary of what will run ───────────────────────────────────────────────
    st.subheader("2. Confirm parameters")

    total_main = params['num_runs_main'] * len(SCENARIOS) * len(ALGORITHMS)
    total_sens = params['num_runs_sens'] * len(params['theta_values']) * len(SCENARIOS)

    p1, p2, p3, p4 = st.columns(4)
    with p1: st.metric("Monte Carlo runs", params['num_runs_main'])
    with p2: st.metric("Sim duration", f"{params['sim_duration']}s")
    with p3: st.metric("UEs per run", params['num_ue'])
    with p4: st.metric("Total simulations", total_main + total_sens)

    if mode == "Full":
        st.warning(
            "⚠️ **Full mode** is designed to run locally, not on Streamlit Cloud. "
            "On Cloud, this will likely time out or hit memory limits. "
            "Run `python3 run_experiments.py` on your machine instead.",
            icon="⚠️"
        )
    elif mode == "Standard" and on_cloud:
        st.info(
            "ℹ️ Standard mode may take 8–10 minutes on Streamlit Cloud. "
            "The app will stay responsive — results save automatically when done.",
            icon="ℹ️"
        )

    # ── Run button ─────────────────────────────────────────────────────────────
    st.subheader("3. Run")

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'run_log' not in st.session_state:
        st.session_state.run_log = []
    if 'run_progress' not in st.session_state:
        st.session_state.run_progress = 0.0
    if 'run_complete' not in st.session_state:
        st.session_state.run_complete = False
    if 'run_error' not in st.session_state:
        st.session_state.run_error = ''

    btn_label = "🔄 Running..." if st.session_state.running else f"▶ Run {mode} Experiment"
    run_clicked = st.button(
        btn_label,
        disabled=st.session_state.running,
        type="primary",
        use_container_width=False,
    )

    # ── Live output area ───────────────────────────────────────────────────────
    progress_bar  = st.progress(0.0, text="Waiting to start...")
    log_container = st.container()
    log_box       = log_container.empty()

    if run_clicked and not st.session_state.running:
        st.session_state.running     = True
        st.session_state.run_log     = []
        st.session_state.run_progress = 0.0
        st.session_state.run_complete = False
        st.session_state.run_error   = ''
        st.rerun()

    if st.session_state.running:
        progress_bar.progress(st.session_state.run_progress,
                              text="Running simulations...")

        # Accumulate log
        log_lines = st.session_state.run_log
        if log_lines:
            log_box.code('\n'.join(log_lines), language=None)

        # Run synchronously — update progress via session state callbacks
        log_buf  = []
        prog_val = [0.0]

        def log_fn(msg):
            log_buf.append(msg)
            st.session_state.run_log = list(log_buf)

        def progress_fn(frac):
            prog_val[0] = frac
            st.session_state.run_progress = frac

        success, err = _run_experiments_inline(
            num_runs_main=params['num_runs_main'],
            num_runs_sens=params['num_runs_sens'],
            theta_values=params['theta_values'],
            sim_duration=params['sim_duration'],
            num_ue=params['num_ue'],
            log_fn=log_fn,
            progress_fn=progress_fn,
        )

        st.session_state.running      = False
        st.session_state.run_complete = success
        st.session_state.run_error    = err
        st.session_state.run_progress = 1.0 if success else prog_val[0]

        # Clear cache so charts reload from new CSVs
        st.cache_data.clear()
        st.rerun()

    # ── Post-run status ────────────────────────────────────────────────────────
    if st.session_state.run_complete:
        progress_bar.progress(1.0, text="Complete!")
        st.success(
            "✅ Experiments complete. Switch to the **📊 Overview** tab to see updated results.",
            icon="✅"
        )
        if st.session_state.run_log:
            with st.expander("View run log"):
                st.code('\n'.join(st.session_state.run_log), language=None)
        if st.button("↺ Reset and run again"):
            st.session_state.run_complete = False
            st.session_state.run_log = []
            st.rerun()

    elif st.session_state.run_error:
        progress_bar.progress(st.session_state.run_progress, text="Error — see below")
        st.error("❌ Experiment failed with an error:")
        st.code(st.session_state.run_error)
        if st.button("↺ Clear error and retry"):
            st.session_state.run_error = ''
            st.session_state.run_log = []
            st.rerun()

    elif not st.session_state.running and not st.session_state.run_complete:
        progress_bar.progress(0.0, text="Ready")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — RAW DATA
# ─────────────────────────────────────────────────────────────────────────────

with tab_raw:
    st.header("Raw Simulation Data")

    if main_df is None:
        st.info("No results yet. Go to **▶ Run Experiments** to generate them.")
    else:
        st.subheader("Main Results")
        st.dataframe(main_df.round(4), use_container_width=True, hide_index=True)
        ca, cb, cc = st.columns(3)
        with ca:
            st.download_button("⬇ main_results.csv",
                               main_df.to_csv(index=False),
                               "main_results.csv", "text/csv")
        with cb:
            if stat_df is not None:
                st.download_button("⬇ statistical_tests.csv",
                                   stat_df.to_csv(index=False),
                                   "statistical_tests.csv", "text/csv")
        with cc:
            if sens_df is not None:
                st.download_button("⬇ sensitivity_results.csv",
                                   sens_df.to_csv(index=False),
                                   "sensitivity_results.csv", "text/csv")

        if stat_df is not None:
            st.subheader("Statistical Test Results")
            st.dataframe(stat_df.round(4), use_container_width=True, hide_index=True)

        if sens_df is not None:
            st.subheader("Sensitivity Analysis Data")
            st.dataframe(sens_df.round(4), use_container_width=True, hide_index=True)
