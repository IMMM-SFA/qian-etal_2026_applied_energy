# -*- coding: utf-8 -*-
"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

import os
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.1_two_sample_comparisons"
import glob
import zlib
import numpy as np
import pandas as pd
from scipy import stats

import re
import matplotlib
matplotlib.use("Agg")  # safe on HPC / no display
import matplotlib.pyplot as plt

# -----------------------------
# User paths
# -----------------------------
SYSTEM_CSV = PACKAGE_ROOT / "shared" / "data" / "system_daily_table.csv"
BA_DIR = PACKAGE_ROOT / "shared" / "data" / "BA_daily_tables"
OUT_DIR = REPRODUCED_ROOT / "3.1" / "statistics"

# -----------------------------
# Settings (keep simple)
# -----------------------------
Q_LOW  = 0.10  # P10
Q_HIGH = 0.90  # P90
MIN_SUPPORT_MEAN  = 30  # minimum days in anomaly/base groups for mean-OFV tests
MIN_SUPPORT_RATES = 30  # minimum days in anomaly/base groups for rate (RR) tests (initial attempt)
DROP_MILD = True  # drop days with CDD==0 and HDD==0

ALPHAS = [97.0, 99.0, 99.5]   # extreme stress cutoffs
MIN_STRESS_EVENTS = 10         # require at least this many total stress events across both groups to test
CONTINUITY = 0.5              # continuity correction used ONLY for RiskRatio_cc (optional diagnostic)

# Bootstrap uncertainty for the uncorrected RiskRatio. The default i.i.d. paired-day
# bootstrap matches the approach in the "scatter cut ... (boot tau mean spaced)"
# analysis: resample rows, then re-estimate all data-derived cutoffs in each replicate.
DO_BOOTSTRAP_RR_SYSTEM = True
DO_BOOTSTRAP_RR_BA = False     # many BA files; enable explicitly if BA-level CIs are needed
RR_BOOT_B = 500
RR_BOOT_SEED = 123
RR_BOOT_METHOD = "iid"         # "iid" (reference-compatible) or "moving_block"
RR_BOOT_BLOCK_LENGTH = 7       # used only when RR_BOOT_METHOD == "moving_block"
RR_BOOT_CI = (2.5, 97.5)       # percentile 95% confidence interval

SYSTEM_VARS = ["Wind", "Solar", "WindFrac", "SolarFrac", "CDD", "HDD", "Q2", "WSPD", "SWDOWN", "GLW", "CoverageFrac"]
BA_VARS     = ["Wind", "Solar", "WindFrac", "SolarFrac", "CDD", "HDD", "Q2", "WSPD", "SWDOWN", "GLW"]

# Coverage fraction columns expected in system_daily_table.csv (edit if your column names differ)
COV_ABOVE_COL = "above_avg"   # for HOT
COV_BELOW_COL = "below_avg"   # for COLD

# -----------------------------
# Residual diagnostics settings
# -----------------------------
DO_RESID_DIAGNOSTICS = True

# Save plots only for a limited subset (avoid thousands of PNGs)
PLOT_RESID_SYSTEM = True
PLOT_RESID_BA = False

RESID_ACF_LAGS = 14       # lags for ACF plot and Ljung-Box-style test
RESID_LB_LAGS = 7         # use first 7 lags for a simple Ljung-Box Q statistic
MAX_RESID_PLOTS = 30      # cap number of (regime,var,tail) plots per label

# Choose which cases to plot (system): "significant" or "top_effect"
RESID_PLOT_MODE = "significant"   # or "top_effect"
RESID_SIG_Q = 0.05

# -----------------------------
# Helpers
# -----------------------------
def add_wspd_from_u_v(df):
    # System table: WSPD computed from U10/V10 if WSPD not already present
    if "WSPD" not in df.columns:
        if "U10" in df.columns and "V10" in df.columns:
            df["WSPD"] = np.sqrt(df["U10"].astype(float)**2 + df["V10"].astype(float)**2)
    else:
        # keep as is
        pass
    return df

def assign_regime(df):
    # Regime defined by system CDD/HDD columns
    cdd = df["CDD"].fillna(0)
    hdd = df["HDD"].fillna(0)

    regime = np.where(cdd > 0, "HOT", np.where(hdd > 0, "COLD", "MILD"))
    df["Regime"] = regime

    if DROP_MILD:
        df = df[df["Regime"].isin(["HOT", "COLD"])].copy()
    return df

def cohen_d(a, b):
    # classic pooled SD Cohen's d (OK for descriptive effect size)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa2 = np.var(a, ddof=1)
    sb2 = np.var(b, ddof=1)
    sp = np.sqrt(((len(a)-1)*sa2 + (len(b)-1)*sb2) / (len(a)+len(b)-2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction; returns q-values aligned to pvals order."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    qvals = np.empty(n, dtype=float)
    qvals[order] = np.clip(q, 0, 1)
    return qvals

def add_stress_flags_within_regime(df, alphas):
    df = df.copy()
    for regime in ["HOT", "COLD"]:
        m = df["Regime"] == regime
        y = pd.to_numeric(df.loc[m, "ObjectiveValue"], errors="coerce")
        for a in alphas:
            thr = y.quantile(a / 100.0)
            col = f"Stress_{str(a).replace('.','p')}"
            df.loc[m, col] = (pd.to_numeric(df.loc[m, "ObjectiveValue"], errors="coerce") >= thr).astype(int)
    return df

def run_two_sample(df, vars_list, label=None):
    rows = []
    for regime in ["HOT", "COLD"]:
        sub = df[df["Regime"] == regime].copy()
        if sub.empty:
            continue

        for var in vars_list:
            if var not in sub.columns:
                continue

            x = pd.to_numeric(sub[var], errors="coerce")
            y = pd.to_numeric(sub["ObjectiveValue"], errors="coerce")

            mask = (~x.isna()) & (~y.isna())
            x = x[mask]
            y = y[mask]

            if len(x) < 2 * MIN_SUPPORT_MEAN:
                continue

            ql = x.quantile(Q_LOW)
            qh = x.quantile(Q_HIGH)

            for tail, flag in [
                ("LOW",  x <= ql),
                ("HIGH", x >= qh),
            ]:
                y_anom = y[flag]
                y_base = y[~flag]

                if len(y_anom) < MIN_SUPPORT_MEAN or len(y_base) < MIN_SUPPORT_MEAN:
                    continue

                mean_anom = float(np.mean(y_anom))
                mean_base = float(np.mean(y_base))
                dmean = mean_anom - mean_base
                pct = (dmean / mean_base * 100.0) if mean_base != 0 else np.nan

                # Welch t-test
                t_res = stats.ttest_ind(y_anom, y_base, equal_var=False, nan_policy="omit")
                p_t = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else np.nan

                # Mann-Whitney U (robust-ish)
                try:
                    u_res = stats.mannwhitneyu(y_anom, y_base, alternative="two-sided")
                    p_u = float(u_res.pvalue)
                except Exception:
                    p_u = np.nan

                rows.append({
                    "Label": label,
                    "Regime": regime,
                    "Variable": var,
                    "Tail": tail,
                    "Q_low": float(ql),
                    "Q_high": float(qh),
                    "N_anom": int(len(y_anom)),
                    "N_base": int(len(y_base)),
                    "OFV_mean_anom": mean_anom,
                    "OFV_mean_base": mean_base,
                    "Delta_mean": float(dmean),
                    "Delta_mean_pct": float(pct),
                    "p_welch_t": p_t,
                    "p_mannwhitney": p_u,
                    "cohen_d": float(cohen_d(y_anom, y_base)),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        # FDR on Welch p-values within this run
        out["q_fdr_welch_t"] = bh_fdr(out["p_welch_t"].fillna(1.0).values)
        out["q_fdr_mannwhitney"] = bh_fdr(out["p_mannwhitney"].fillna(1.0).values)
    return out


# =============================================================================
# 
# def run_two_sample_rates(df, vars_list, label=None):
#     rows = []
#     for regime in ["HOT", "COLD"]:
#         sub = df[df["Regime"] == regime].copy()
#         if sub.empty:
#             continue
# 
#         for var in vars_list:
#             if var not in sub.columns:
#                 continue
# 
#             x = pd.to_numeric(sub[var], errors="coerce")
#             mask_x = ~x.isna()
# 
#             if mask_x.sum() < 2 * MIN_SUPPORT_RATES:
#                 continue
# 
#             ql = x[mask_x].quantile(Q_LOW)
#             qh = x[mask_x].quantile(Q_HIGH)
# 
#             for tail, flag in [("LOW", x <= ql), ("HIGH", x >= qh)]:
#                 # Keep only rows with valid X and stress flags
#                 keep = mask_x & flag.notna()
#                 idx_anom = keep & flag
#                 idx_base = keep & (~flag)
# 
#                 n_anom = int(idx_anom.sum())
#                 n_base = int(idx_base.sum())
#                 if n_anom < MIN_SUPPORT_RATES or n_base < MIN_SUPPORT_RATES:
#                     continue
# 
#                 for a in ALPHAS:
#                     scol = f"Stress_{str(a).replace('.','p')}"
#                     if scol not in sub.columns:
#                         continue
# 
#                     s = pd.to_numeric(sub[scol], errors="coerce")
#                     idx_anom2 = idx_anom & (~s.isna())
#                     idx_base2 = idx_base & (~s.isna())
# 
#                     s_anom = s[idx_anom2]
#                     s_base = s[idx_base2]
# 
#                     # counts
#                     e_anom = int(s_anom.sum())
#                     e_base = int(s_base.sum())
#                     ne_anom = int(len(s_anom) - e_anom)
#                     ne_base = int(len(s_base) - e_base)
# 
#                     if (e_anom + e_base) < MIN_STRESS_EVENTS:
#                         continue
# 
#                     rate_anom = e_anom / (e_anom + ne_anom) if (e_anom + ne_anom) else np.nan
#                     rate_base = e_base / (e_base + ne_base) if (e_base + ne_base) else np.nan
#                     riskratio = (rate_anom / rate_base) if (np.isfinite(rate_anom) and np.isfinite(rate_base) and rate_base > 0) else np.nan
# 
#                     # effect sizes (with continuity correction for stability)
#                     ra = (e_anom + CONTINUITY) / (e_anom + ne_anom + 2*CONTINUITY)
#                     rb = (e_base + CONTINUITY) / (e_base + ne_base + 2*CONTINUITY)
#                     rr = ra / rb if rb > 0 else np.nan
#                     rd = rate_anom - rate_base
# 
#                     # Fisher exact test on 2x2 table
#                     table = np.array([[e_anom, ne_anom],
#                                       [e_base, ne_base]], dtype=int)
#                     try:
#                         or_f, p_f = stats.fisher_exact(table, alternative="two-sided")
#                     except Exception:
#                         or_f, p_f = np.nan, np.nan
# 
#                     rows.append({
#                         "Label": label,
#                         "Regime": regime,
#                         "Variable": var,
#                         "Tail": tail,
#                         "Alpha": a,
#                         "Q_low": float(ql),
#                         "Q_high": float(qh),
#                         "N_anom": n_anom,
#                         "N_base": n_base,
#                         "Events_anom": e_anom,
#                         "Events_base": e_base,
#                         "Rate_anom": float(rate_anom),
#                         "Rate_base": float(rate_base),
#                         "RiskDiff": float(rd),
#                         "RiskRatio": float(riskratio),
#                         "RiskRatio_cc": float(rr),
#                         "OddsRatio_fisher": float(or_f) if np.isfinite(or_f) else np.nan,
#                         "p_fisher": float(p_f) if np.isfinite(p_f) else np.nan,
#                         "Direction": "Higher" if rate_anom > rate_base else "Lower",
#                     })
# 
#     out = pd.DataFrame(rows)
#     if not out.empty:
#         out["q_fdr_fisher"] = bh_fdr(out["p_fisher"].fillna(1.0).values)
#     return out
# 
# 
# =============================================================================


def _stable_bootstrap_seed(base_seed, label, regime):
    """Return a reproducible per-label/per-regime seed (independent of call order)."""
    tag = f"{label}|{regime}".encode("utf-8")
    return (int(base_seed) + zlib.crc32(tag)) % (2**32)


def _bootstrap_indices(n, B, rng, method="iid", block_length=7):
    """Generate paired-row bootstrap indices, optionally using circular moving blocks."""
    if n <= 0 or B <= 0:
        return np.empty((0, 0), dtype=int)

    method = str(method).lower()
    if method == "iid":
        return rng.integers(0, n, size=(B, n))
    if method != "moving_block":
        raise ValueError("RR_BOOT_METHOD must be 'iid' or 'moving_block'")

    block_length = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(B, n_blocks))
    offsets = np.arange(block_length, dtype=int)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(B, -1)[:, :n]


def _bootstrap_rr_for_tail(x, ov, tail, alphas, bootstrap_idx):
    """Bootstrap RR after re-estimating X and OFV quantile cutoffs each replicate.

    Rows are resampled as paired (X, OFV) observations. X P10/P90 is estimated from
    finite X values, while each stress threshold is estimated from finite OFV values,
    matching the full-sample construction of the two-sample comparison.
    """
    x = np.asarray(x, dtype=float)
    ov = np.asarray(ov, dtype=float)
    draws = {float(a): [] for a in alphas}

    for idx in bootstrap_idx:
        xb = x[idx]
        ovb = ov[idx]
        finite_x = np.isfinite(xb)
        finite_ov = np.isfinite(ovb)
        if finite_x.sum() < 2 or finite_ov.sum() < 2:
            continue

        qx = Q_LOW if tail == "LOW" else Q_HIGH
        x_cut = float(np.nanquantile(xb[finite_x], qx))
        anomaly = (xb <= x_cut) if tail == "LOW" else (xb >= x_cut)
        valid = finite_x & finite_ov
        anomaly &= valid
        base = valid & (~anomaly)
        n_anom = int(anomaly.sum())
        n_base = int(base.sum())
        if n_anom == 0 or n_base == 0:
            continue

        for a in alphas:
            stress_cut = float(np.nanquantile(ovb[finite_ov], float(a) / 100.0))
            stress = ovb >= stress_cut
            rate_anom = float(stress[anomaly].mean())
            rate_base = float(stress[base].mean())
            if np.isfinite(rate_anom) and np.isfinite(rate_base) and rate_base > 0:
                draws[float(a)].append(rate_anom / rate_base)

    lo_pct, hi_pct = RR_BOOT_CI
    summaries = {}
    for a in alphas:
        vals = np.asarray(draws[float(a)], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            summaries[float(a)] = {
                "RiskRatio_boot_mean": float(np.mean(vals)),
                "RiskRatio_boot_median": float(np.median(vals)),
                "RiskRatio_boot_ci_low": float(np.percentile(vals, lo_pct)),
                "RiskRatio_boot_ci_high": float(np.percentile(vals, hi_pct)),
                "RiskRatio_boot_n_valid": int(vals.size),
            }
        else:
            summaries[float(a)] = {
                "RiskRatio_boot_mean": np.nan,
                "RiskRatio_boot_median": np.nan,
                "RiskRatio_boot_ci_low": np.nan,
                "RiskRatio_boot_ci_high": np.nan,
                "RiskRatio_boot_n_valid": 0,
            }
    return summaries


def run_two_sample_rates(
    df,
    vars_list,
    label=None,
    bootstrap_rr=False,
    bootstrap_B=RR_BOOT_B,
    bootstrap_seed=RR_BOOT_SEED,
    bootstrap_method=RR_BOOT_METHOD,
    bootstrap_block_length=RR_BOOT_BLOCK_LENGTH,
):
    # 1. NEW: Ensure the dataframe is strictly sorted by time before splitting
    df = _sort_time(df) 
    
    rows = []
    for regime in ["HOT", "COLD"]:
        sub = df[df["Regime"] == regime].copy()
        if sub.empty:
            continue

        bootstrap_idx = None
        if bootstrap_rr:
            rng = np.random.default_rng(_stable_bootstrap_seed(bootstrap_seed, label, regime))
            bootstrap_idx = _bootstrap_indices(
                len(sub), bootstrap_B, rng, bootstrap_method, bootstrap_block_length
            )
        ov_for_bootstrap = pd.to_numeric(sub["ObjectiveValue"], errors="coerce").to_numpy(dtype=float)

        for var in vars_list:
            if var not in sub.columns:
                continue

            x = pd.to_numeric(sub[var], errors="coerce")
            mask_x = ~x.isna()

            if mask_x.sum() < 2 * MIN_SUPPORT_RATES:
                continue

            ql = x[mask_x].quantile(Q_LOW)
            qh = x[mask_x].quantile(Q_HIGH)

            for tail, flag in [("LOW", x <= ql), ("HIGH", x >= qh)]:
                boot_summary = None
                if bootstrap_rr:
                    boot_summary = _bootstrap_rr_for_tail(
                        x.to_numpy(dtype=float),
                        ov_for_bootstrap,
                        tail,
                        ALPHAS,
                        bootstrap_idx,
                    )

                # Keep only rows with valid X and stress flags
                keep = mask_x & flag.notna()
                idx_anom = keep & flag
                idx_base = keep & (~flag)

                n_anom = int(idx_anom.sum())
                n_base = int(idx_base.sum())
                if n_anom < MIN_SUPPORT_RATES or n_base < MIN_SUPPORT_RATES:
                    continue

                for a in ALPHAS:
                    scol = f"Stress_{str(a).replace('.','p')}"
                    if scol not in sub.columns:
                        continue

                    s = pd.to_numeric(sub[scol], errors="coerce")
                    idx_anom2 = idx_anom & (~s.isna())
                    idx_base2 = idx_base & (~s.isna())

                    s_anom = s[idx_anom2]
                    s_base = s[idx_base2]

                    # 2. NEW: Calculate i.i.d. diagnostics for both groups
                    iid_anom = check_binary_iid(s_anom.values)
                    iid_base = check_binary_iid(s_base.values)

                    # counts
                    e_anom = int(s_anom.sum())
                    e_base = int(s_base.sum())
                    ne_anom = int(len(s_anom) - e_anom)
                    ne_base = int(len(s_base) - e_base)

                    if (e_anom + e_base) < MIN_STRESS_EVENTS:
                        continue

                    rate_anom = e_anom / (e_anom + ne_anom) if (e_anom + ne_anom) else np.nan
                    rate_base = e_base / (e_base + ne_base) if (e_base + ne_base) else np.nan
                    riskratio = (rate_anom / rate_base) if (np.isfinite(rate_anom) and np.isfinite(rate_base) and rate_base > 0) else np.nan

                    # effect sizes (with continuity correction for stability)
                    ra = (e_anom + CONTINUITY) / (e_anom + ne_anom + 2*CONTINUITY)
                    rb = (e_base + CONTINUITY) / (e_base + ne_base + 2*CONTINUITY)
                    rr = ra / rb if rb > 0 else np.nan
                    rd = rate_anom - rate_base

                    # Fisher exact test on 2x2 table
                    table = np.array([[e_anom, ne_anom],
                                      [e_base, ne_base]], dtype=int)
                    try:
                        or_f, p_f = stats.fisher_exact(table, alternative="two-sided")
                    except Exception:
                        or_f, p_f = np.nan, np.nan

                    # 3. NEW: Add the diagnostic results to the output row
                    row = {
                        "Label": label,
                        "Regime": regime,
                        "Variable": var,
                        "Tail": tail,
                        "Alpha": a,
                        "Q_low": float(ql),
                        "Q_high": float(qh),
                        "N_anom": n_anom,
                        "N_base": n_base,
                        "Events_anom": e_anom,
                        "Events_base": e_base,
                        "Rate_anom": float(rate_anom),
                        "Rate_base": float(rate_base),
                        "RiskDiff": float(rd),
                        "RiskRatio": float(riskratio),
                        "RiskRatio_cc": float(rr),
                        "OddsRatio_fisher": float(or_f) if np.isfinite(or_f) else np.nan,
                        "p_fisher": float(p_f) if np.isfinite(p_f) else np.nan,
                        "Direction": "Higher" if rate_anom > rate_base else "Lower",
                        "ACF_lag1_anom": float(iid_anom['lag1_acf']),
                        "Runs_p_anom": float(iid_anom['runs_p']),
                        "ACF_lag1_base": float(iid_base['lag1_acf']),
                        "Runs_p_base": float(iid_base['runs_p'])
                    }
                    if bootstrap_rr:
                        row.update(boot_summary[float(a)])
                        row.update({
                            "RiskRatio_boot_B": int(bootstrap_B),
                            "RiskRatio_boot_seed": int(bootstrap_seed),
                            "RiskRatio_boot_method": str(bootstrap_method),
                            "RiskRatio_boot_ci_level": float(RR_BOOT_CI[1] - RR_BOOT_CI[0]),
                            "RiskRatio_boot_block_length": (
                                int(bootstrap_block_length)
                                if str(bootstrap_method).lower() == "moving_block"
                                else np.nan
                            ),
                        })
                    rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr_fisher"] = bh_fdr(out["p_fisher"].fillna(1.0).values)
    return out


def add_system_coverage_frac(df, above_col=COV_ABOVE_COL, below_col=COV_BELOW_COL):
    """
    Create a single system-level CoverageFrac variable:
      - HOT: use above-average coverage fraction
      - COLD: use below-average coverage fraction
    Assumes df already has df["Regime"].
    """
    if above_col not in df.columns or below_col not in df.columns:
        print(f"[WARN] Coverage fraction columns not found in system table: "
              f"{above_col} and/or {below_col}. Skipping CoverageFrac.")
        return df

    df = df.copy()
    df["CoverageFrac"] = np.nan
    df.loc[df["Regime"] == "HOT", "CoverageFrac"]  = pd.to_numeric(df.loc[df["Regime"] == "HOT",  above_col], errors="coerce")
    df.loc[df["Regime"] == "COLD", "CoverageFrac"] = pd.to_numeric(df.loc[df["Regime"] == "COLD", below_col], errors="coerce")
    return df


# -----------------------------
# Helpers (Residuals for mean OFV)
# -----------------------------

def _sanitize(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s[:180]

def _sort_time(df):
    # Prefer Date if present; else Year/Day
    if "Date" in df.columns:
        tmp = df.copy()
        tmp["Date_dt"] = pd.to_datetime(tmp["Date"], errors="coerce")
        return tmp.sort_values(["Date_dt", "Year", "Day"]).drop(columns=["Date_dt"])
    elif "Year" in df.columns and "Day" in df.columns:
        return df.sort_values(["Year", "Day"])
    return df

def _acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < (nlags + 2):
        return np.full(nlags, np.nan)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.full(nlags, np.nan)
    ac = []
    for k in range(1, nlags + 1):
        ac.append(np.dot(x[:-k], x[k:]) / denom)
    return np.array(ac, dtype=float)

def _ljung_box_pvalue(x, m=7):
    """
    Simple Ljung-Box Q test approximation:
    Q = n(n+2) * sum_{k=1..m} r_k^2/(n-k) ~ chi2(df=m)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < (m + 5):
        return np.nan, np.nan
    r = _acf(x, m)
    if np.any(~np.isfinite(r)):
        return np.nan, np.nan
    Q = n * (n + 2) * np.sum((r**2) / (n - np.arange(1, m + 1)))
    p = stats.chi2.sf(Q, df=m)
    return float(Q), float(p)





# -----------------------------
# Helpers (Residuals for Rates)
# -----------------------------

def check_binary_iid(binary_series):
    """
    Evaluates the i.i.d assumption for a sequence of binary events (1s and 0s).
    Returns a dictionary with the Lag-1 Autocorrelation and Runs Test results.
    """
    # Ensure it's a 1D numpy array of integers
    arr = np.asarray(binary_series, dtype=int)
    
    # Require at least 3 days and variation (both 1s and 0s must exist)
    if len(arr) < 3 or len(np.unique(arr)) < 2:
        return {"lag1_acf": np.nan, "runs_actual": np.nan, "runs_z": np.nan, "runs_p": np.nan}
    
    # ----------------------------------------------------
    # 1. Lag-1 Autocorrelation (interpretable metric)
    # ----------------------------------------------------
    y_t = arr[1:]
    y_t_1 = arr[:-1]
    
    # Calculate Pearson correlation (Phi coefficient for binary data)
    # Catching warnings in case of zero variance in the slices
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(y_t_1, y_t)
        lag1_acf = corr_matrix[0, 1] if np.shape(corr_matrix) == (2, 2) else np.nan
    
    # ----------------------------------------------------
    # 2. Wald-Wolfowitz Runs Test (formal statistical check)
    # ----------------------------------------------------
    runs = 1
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            runs += 1
            
    n1 = np.sum(arr == 1)
    n0 = np.sum(arr == 0)
    n = n1 + n0
    
    # Expected number of runs if perfectly i.i.d
    expected_runs = ((2 * n1 * n0) / n) + 1
    
    # Variance of the number of runs
    variance_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / ((n ** 2) * (n - 1))
    
    if variance_runs <= 0:
        z_stat = np.nan
        p_value = np.nan
    else:
        # Z-statistic (Negative Z means fewer runs = clustering)
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
    return {
        "lag1_acf": lag1_acf,
        "runs_actual": runs,
        "runs_z": z_stat,
        "runs_p": p_value
    }


# -----------------------------
# System run
# -----------------------------
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    sys_df = pd.read_csv(SYSTEM_CSV)
    sys_df = add_wspd_from_u_v(sys_df)
    sys_df = assign_regime(sys_df)

    # NEW: build CoverageFrac (HOT uses above-average coverage; COLD uses below-average coverage)
    sys_df = add_system_coverage_frac(sys_df)

    sys_df = add_stress_flags_within_regime(sys_df, ALPHAS)

    sys_out = run_two_sample(sys_df, SYSTEM_VARS, label="SYSTEM")
    sys_out_path = os.path.join(OUT_DIR, "system_two_sample_singlevar_anoms.csv")
    sys_out.to_csv(sys_out_path, index=False)
    print("[SAVE]", sys_out_path, "rows:", len(sys_out))

    sys_out_rates = run_two_sample_rates(
        sys_df,
        SYSTEM_VARS,
        label="SYSTEM",
        bootstrap_rr=DO_BOOTSTRAP_RR_SYSTEM,
    )
    sys_out_rates_path = os.path.join(OUT_DIR, "system_two_sample_singlevar_anoms (rates).csv")
    sys_out_rates.to_csv(sys_out_rates_path, index=False)
    print("[SAVE]", sys_out_rates_path, "rows:", len(sys_out_rates))

if __name__ == "__main__":
    main()
