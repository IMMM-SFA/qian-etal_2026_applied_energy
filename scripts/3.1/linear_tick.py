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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Batch figure generation; no desktop GUI required.
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import matplotlib.font_manager as fm

# -----------------------------
# User paths (keep consistent)
# -----------------------------
BASE_DIR = SECTION_DIR / "data"
OUT_DIR = REPRODUCED_ROOT / "3.1" / "rates"
IN_CSV   = os.path.join(BASE_DIR, "system_two_sample_singlevar_anoms (rates).csv")

# -----------------------------
# Font Loading (Aptos Cloud Fonts)
# -----------------------------
APTOS_DIR = PACKAGE_ROOT / "shared" / "data" / "Aptos"

FONT_FAMILY = "DejaVu Sans"
if os.path.exists(APTOS_DIR):
    font_files = glob.glob(os.path.join(APTOS_DIR, "*.ttf"))
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    if font_files:
        FONT_FAMILY = "Aptos"
    else:
        print(f"[WARN] Aptos folder exists but contains no .ttf files: {APTOS_DIR}; using Matplotlib default font.")
else:
    print(f"[WARN] Cannot find Aptos folder: {APTOS_DIR}; using Matplotlib default font.")


# -----------------------------
# User knobs
# -----------------------------
ALPHAS = [97.0, 99.0, 99.5]      # OFV threshold percentiles
P_COL = "p_fisher"               # choose: "p_fisher" or "q_fdr_fisher"
RR_COL = "RiskRatio_boot_mean"   # bootstrap mean of uncorrected RiskRatio (NOT RiskRatio_cc)
DELTA_COL = "RiskDiff"           # direction: RiskDiff = Rate_anom - Rate_base

MIN_EVENTS = 10                   # support rule for rare support
COLOR_CAP_LOG2RR = 5.0             # fixed color saturation cap for log2(RR)

# Visuals
HOT_COLOR = "tab:red"
COLD_COLOR = "tab:blue"
REGIME_COLORS = {"HOT": HOT_COLOR, "COLD": COLD_COLOR}
REGIME_TITLES = {"HOT": "Hot", "COLD": "Cold"}

MARKER_S = 1400
EDGE_LW = 8.0

MARKER_EDGE_COLOR = "black"
RARE_DASH = (0, (2, 1))

X_LIM = (-2.45, 2.45)

# Layout knobs
BOTTOM_MARGIN = 0.22
BOTTOM_TITLE_Y = 0.070

BASE_FONTSIZE = 50
LEGEND_FONTSIZE = 44
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE + 4,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE + 4,
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
})

# Row order, top -> bottom. GLW / Longwave Radiation removed.
MET_ROWS = ["CDD/HDD", "CoverageFrac", "Q2", "WSPD", "SWDOWN"]
REN_ROWS = ["Wind", "Solar"]

DISPLAY_NAMES = {
    "CoverageFrac": "Temperature\nCoverage",
    "Q2": "Specific\nHumidity",
    "WSPD": "Wind Speed",
    "SWDOWN": "Shortwave\nRadiation",
    "Wind": "Wind\nProduction",
    "Solar": "Solar\nProduction",
    "": "",
}

# Add an empty row as a visual separator. Reverse because y=0 is the bottom.
VISUAL_ORDER = MET_ROWS + [""] + REN_ROWS
PLOT_ROWS = list(reversed(VISUAL_ORDER))


# -----------------------------
# Helpers
# -----------------------------
def significance_level(p):
    """Return 2, 1, or 0 for the configured significance thresholds."""
    if p is None or not np.isfinite(p):
        return 0
    if p <= 0.01:
        return 2
    if p <= 0.05:
        return 1
    return 0


def blend_white(base_rgb, t):
    """Blend from white (t=0) to base color (t=1)."""
    base = np.array(base_rgb, dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(white * (1 - t) + base * t)


def pick_var_for_row(row_label, regime):
    """Special case: CDD for HOT, HDD for COLD."""
    if row_label == "CDD/HDD":
        return "CDD" if regime == "HOT" else "HDD"
    return row_label


def get_capped_log2_rr(row, max_cap):
    """Return capped log2 bootstrap-mean RR; complete separation maps to cap."""
    e_base = row.get("Events_base", np.nan)
    e_anom = row.get("Events_anom", np.nan)
    if np.isfinite(e_base) and np.isfinite(e_anom) and e_base == 0 and e_anom > 0:
        return max_cap

    rr = row.get(RR_COL, np.nan)

    if rr is None or not np.isfinite(rr) or float(rr) <= 1:
        return 0.0
    return min(float(np.log2(float(rr))), max_cap)


def risk_ratio_rank(row):
    """Comparable bootstrap-mean RR; complete separation ranks as +infinity."""
    e_base = row.get("Events_base", np.nan)
    e_anom = row.get("Events_anom", np.nan)
    if np.isfinite(e_base) and np.isfinite(e_anom) and e_base == 0 and e_anom > 0:
        return np.inf

    rr = row.get(RR_COL, np.nan)
    return float(rr) if rr is not None and np.isfinite(rr) else -np.inf


def is_one_sided_rare(events_anom, events_base):
    """Rare support if either comparison group has fewer than MIN_EVENTS events."""
    if not np.isfinite(events_anom) or not np.isfinite(events_base):
        return False
    a = int(events_anom) < MIN_EVENTS
    b = int(events_base) < MIN_EVENTS
    return (a or b)


def collect_points(df, rows, alpha, regime, max_log2rr):
    """Select one signed or neutral stress-increase point per plotted row."""
    pts = {}
    for row in rows:
        if row == "":
            continue

        var = pick_var_for_row(row, regime)
        sub = df[
            (df["Regime"].astype(str).str.upper() == regime)
            & (df["Variable"] == var)
            & (df["Tail"].astype(str).str.upper().isin(["LOW", "HIGH"]))
            & (df["Alpha"].astype(float) == float(alpha))
        ]
        if sub.empty:
            continue

        # Low renewable production is the only resource-stress direction.
        eligible_tails = ["LOW"] if row in {"Wind", "Solar"} else ["LOW", "HIGH"]
        sub = sub[sub["Tail"].astype(str).str.upper().isin(eligible_tails)]
        if sub.empty:
            continue

        candidates = []
        for _, r in sub.iterrows():
            p = r.get(P_COL, np.nan)
            rr = r.get(RR_COL, np.nan)
            delta = r.get(DELTA_COL, np.nan)
            tail = str(r.get("Tail", "")).upper()
            increases_stress = bool(np.isfinite(delta) and delta > 0)
            level = significance_level(p) if increases_stress else 0
            candidates.append({
                "score": (-level if tail == "LOW" else level) if level else 0,
                "level": level,
                "rr": rr,
                "rr_rank": risk_ratio_rank(r),
                "log2rr": get_capped_log2_rr(r, max_log2rr) if increases_stress else 0.0,
                "p": p,
                "delta": delta,
                "tail": tail,
                "increases_stress": increases_stress,
                "rare": is_one_sided_rare(
                    r.get("Events_anom", np.nan), r.get("Events_base", np.nan)
                ),
            })

        significant = [point for point in candidates if point["level"] > 0]
        if significant:
            # Prefer stronger significance, then smaller p, then larger RR.
            selected = max(
                significant,
                key=lambda point: (
                    point["level"],
                    -point["p"] if np.isfinite(point["p"]) else -np.inf,
                    point["rr_rank"],
                ),
            )
        else:
            # At zero, use the largest stress-increasing RR. If neither tail
            # increases stress, the largest available RR still selects support
            # status while its color remains at log2(RR)=0.
            increasing = [point for point in candidates if point["increases_stress"]]
            pool = increasing or candidates
            selected = max(pool, key=lambda point: point["rr_rank"])
            selected = selected.copy()
            selected["score"] = 0

        pts[row] = selected
    return pts


def _robust_cap(vals, pct=99):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return 1.0
    if vals.size < 10:
        cap = float(np.max(vals))
    else:
        cap = float(np.nanpercentile(vals, pct))
    return cap if cap > 0 else 1.0


def plotted_variable_names():
    """Variables used in the revised figures, excluding GLW."""
    names = set()
    for regime in ["HOT", "COLD"]:
        for row in PLOT_ROWS:
            if row:
                names.add(pick_var_for_row(row, regime))
    return names


def make_shared_legend_figure(max_log2rr, out_prefix):
    """Save a standalone shared legend figure for all alphas/regimes."""
    fig = plt.figure(figsize=(36.0, 14.0))
    axL = fig.add_axes([0.04, 0.04, 0.53, 0.92])
    axR = fig.add_axes([0.62, 0.04, 0.34, 0.92])
    axL.axis("off")
    axR.axis("off")

    axL.set_xlim(-0.12, 1.0)
    axL.set_ylim(0, 1)
    axR.set_xlim(-0.02, 1.12)
    axR.set_ylim(0, 1)

    # Keep the standalone legend readily legible when placed beside the panels.
    LFONT = LEGEND_FONTSIZE

    axL.text(0.00, 0.95, "Legend", fontsize=LFONT + 8, fontweight="bold", va="top")
    axL.text(0.00, 0.82, "Tick labels (direction + significance):",
             fontsize=LFONT + 2, fontweight="bold", va="top")

    sig_lines = [
        r"$\mathbf{(-)}$, $p \leq 0.01$: Low-end anomaly ($\leq 10^{th}$ pct) increases RA stress",
        r"$\mathbf{(-)}$, $p \leq 0.05$: Low-end anomaly ($\leq 10^{th}$ pct) increases RA stress",
        r"$\mathbf{NS}$: No significant stress increase at either extreme",
        r"$\mathbf{(+)}$, $p \leq 0.05$: High-end anomaly ($\geq 90^{th}$ pct) increases RA stress",
        r"$\mathbf{(+)}$, $p \leq 0.01$: High-end anomaly ($\geq 90^{th}$ pct) increases RA stress",
    ]

    y0 = 0.73
    dy = 0.105
    for i, s in enumerate(sig_lines):
        axL.text(0.02, y0 - i * dy, s, fontsize=LFONT, va="top")

    axL.text(
        0.02, 0.16,
        "Ticks encode direction + statistical significance (confidence), not magnitude.",
        fontsize=LFONT - 6, va="top",
    )

    axR.text(0.00, 0.82, "Color intensity:", fontsize=LFONT + 2, fontweight="bold", va="top")
    axR.text(
        0.00, 0.74,
        r"encodes $\log_2(\mathrm{Risk\ Ratio})$ for RR > 1",
        fontsize=LFONT, va="top",
    )

    grad = np.linspace(0, 1, 256).reshape(1, -1)

    for label, color, y_mid in [("HOT", HOT_COLOR, 0.60), ("COLD", COLD_COLOR, 0.46)]:
        base = to_rgb(color)
        cmap = LinearSegmentedColormap.from_list(
            f"{label.lower()}_grad",
            [blend_white(base, 0), blend_white(base, 1)],
            N=256,
        )
        axR.text(0.00, y_mid, label, fontsize=LFONT, fontweight="bold", va="center", color=color)
        axR.imshow(grad, aspect="auto", cmap=cmap, extent=[0.18, 0.95, y_mid - 0.03, y_mid + 0.03])
        if label == "COLD":
            axR.text(0.18, y_mid - 0.06, "0", fontsize=LFONT - 2, ha="left", va="top")
            axR.text(0.95, y_mid - 0.06, f"{max_log2rr:.2f}", fontsize=LFONT - 2, ha="right", va="top")

    axR.text(0.00, 0.31, "Marker examples:", fontsize=LFONT + 2, fontweight="bold", va="top")

    x_marker, x_text = 0.07, 0.18
    y_supported, y_no_support = 0.20, 0.08
    axR.scatter([x_marker], [y_supported], s=MARKER_S, marker="o",
                facecolor="none",
                edgecolors=MARKER_EDGE_COLOR, linewidths=EDGE_LW, linestyles="solid", zorder=3)
    axR.text(x_text, y_supported, "Support: solid black outline", fontsize=LFONT, va="center")
    axR.scatter([x_marker], [y_no_support], s=MARKER_S, marker="o",
                facecolor="none",
                edgecolors=MARKER_EDGE_COLOR, linewidths=EDGE_LW, linestyles=RARE_DASH, zorder=3)
    axR.text(
        x_text, y_no_support,
        f"No support: dashed black outline\nEvents < {MIN_EVENTS} in at least one group",
        fontsize=LFONT, va="center", linespacing=1.05,
    )

    png = os.path.join(OUT_DIR, f"{out_prefix}.png")
    pdf = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("[SAVE]", png)


def plot_hot_cold(df_alpha, rows, out_prefix, alpha, max_log2rr):
    """Plot HOT and COLD side by side in one figure for one alpha."""
    points = {
        regime: collect_points(
            df_alpha, rows, alpha=alpha, regime=regime,
            max_log2rr=max_log2rr,
        )
        for regime in ["HOT", "COLD"]
    }

    n = len(rows)
    y = np.arange(n)
    fig, axes = plt.subplots(
        1, 2, figsize=(31.0, 2.45 * n + 5.8), sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=BOTTOM_MARGIN)

    def draw_panel(ax, regime, show_ylabels):
        pts = points[regime]
        base_color = REGIME_COLORS[regime]
        base_rgb = to_rgb(base_color)

        ax.axvline(0, color="0.82", linestyle="--", linewidth=2.0, alpha=1.0, zorder=1)
        for yi, row_label in zip(y, rows):
            if row_label == "":
                ax.hlines(yi, X_LIM[0], X_LIM[1], color="black", linewidth=1.5,
                          linestyle="--", alpha=0.4, zorder=1)

        ax.set_yticks(y)
        if show_ylabels:
            display_labels = [
                "CDD / HDD" if row == "CDD/HDD" else DISPLAY_NAMES.get(row, row)
                for row in rows
            ]
            ax.set_yticklabels(display_labels, fontweight="bold")
        else:
            ax.tick_params(axis="y", labelleft=False)

        ax.set_xlim(*X_LIM)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels([
            r"$(-)$" "\n" r"$p \leq 0.01$",
            r"$(-)$" "\n" r"$p \leq 0.05$",
            r"$\mathrm{NS}$" "\n ",
            r"$(+)$" "\n" r"$p \leq 0.05$",
            r"$(+)$" "\n" r"$p \leq 0.01$",
        ], fontsize=BASE_FONTSIZE - 18, linespacing=1.0)
        ax.set_ylim(-0.9, n - 0.3)
        ax.grid(False)

        for i, row in enumerate(rows):
            if row == "" or row not in pts:
                continue

            point = pts[row]
            sc = float(point["score"])
            t = (point["log2rr"] / max_log2rr) if np.isfinite(point["log2rr"]) else 0.0
            t = float(np.clip(t, 0.0, 1.0))
            fc = blend_white(base_rgb, t)
            ls = RARE_DASH if point["rare"] else "solid"

            ax.scatter(sc, i, s=MARKER_S, marker="o",
                       facecolor=fc, edgecolors=MARKER_EDGE_COLOR,
                       linewidths=EDGE_LW, linestyles=ls, zorder=3)

        ax.set_title(
            REGIME_TITLES[regime], fontsize=BASE_FONTSIZE + 8,
            fontweight="bold", color=base_color, pad=16,
        )
        ax.set_xlabel("Significance", fontsize=BASE_FONTSIZE,
                      fontweight="bold", color="0.4", labelpad=30)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    draw_panel(axes[0], "HOT", show_ylabels=True)
    draw_panel(axes[1], "COLD", show_ylabels=False)

    met_indices = [i for i, row in enumerate(rows) if row in MET_ROWS]
    ren_indices = [i for i, row in enumerate(rows) if row in REN_ROWS]
    for label, indices in [
        ("Meteorological", met_indices),
        ("Renewables", ren_indices),
    ]:
        axes[0].text(
            -0.55, float(np.mean(indices)), label,
            transform=axes[0].get_yaxis_transform(), rotation=90,
            va="center", ha="center", fontsize=BASE_FONTSIZE,
            fontweight="bold", color="0.25", clip_on=False,
        )

    fig.text(0.5, BOTTOM_TITLE_Y, f"RA Stress Occurrence Rates ($\\alpha$ = {alpha:g})",
             ha="center", va="center", fontsize=BASE_FONTSIZE + 4,
             fontweight="bold")

    png = os.path.join(OUT_DIR, f"{out_prefix}.png")
    pdf = os.path.join(OUT_DIR, f"{out_prefix}.pdf")
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("[SAVE]", png)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"Cannot find {IN_CSV}\n"
            "Please generate the rates table first: system_two_sample_singlevar_anoms (rates).csv"
        )

    df = pd.read_csv(IN_CSV)

    if "Label" in df.columns:
        df = df[df["Label"].astype(str).str.upper().eq("SYSTEM")].copy()

    need = {"Regime", "Variable", "Tail", "Alpha", P_COL, RR_COL, DELTA_COL,
            "Events_anom", "Events_base"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {IN_CSV}: {missing}")

    df["Regime"] = df["Regime"].astype(str).str.upper()
    df["Tail"] = df["Tail"].astype(str).str.upper()

    df = (
        df.sort_values(["Alpha", "Regime", "Variable", "Tail"])
          .drop_duplicates(subset=["Alpha", "Regime", "Variable", "Tail"], keep="first")
          .copy()
    )

    # Fixed color cap shared by all output figures.
    # A value of 5 means any stress-increasing bootstrap-mean RiskRatio >= 32
    # receives the darkest color.
    global_max_log2rr = COLOR_CAP_LOG2RR

    make_shared_legend_figure(global_max_log2rr, "linear_ticks_rates_shared_legend_split_regimes")

    for alpha in ALPHAS:
        df_alpha = df[df["Alpha"].astype(float).eq(float(alpha))].copy()
        if df_alpha.empty:
            print(f"[WARN] No rows found for Alpha={alpha}. Skipping.")
            continue

        alpha_tag = str(alpha).replace(".", "p")
        missing_regimes = [
            regime for regime in ["HOT", "COLD"]
            if df_alpha[df_alpha["Regime"].eq(regime)].empty
        ]
        if missing_regimes:
            print(f"[WARN] Missing regimes for Alpha={alpha}: {missing_regimes}. Skipping.")
            continue

        plot_hot_cold(
            df_alpha,
            PLOT_ROWS,
            out_prefix=f"linear_ticks_rates_alpha{alpha_tag}_hot_cold_system",
            alpha=alpha,
            max_log2rr=global_max_log2rr,
        )


if __name__ == "__main__":
    main()
