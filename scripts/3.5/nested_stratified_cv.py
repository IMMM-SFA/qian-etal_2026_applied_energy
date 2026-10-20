#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-package script. Output scope is documented in PAPER_OUTPUTS.md at the package root."""

import os
from pathlib import Path

import os
STAGING_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PAPER1_DATA_ROOT", str(STAGING_ROOT / "MSD-LIVE"))).resolve()
PACKAGE_ROOT = DATA_ROOT / "datasets_and_scripts"
REPRODUCED_ROOT = Path(os.environ.get("PAPER1_OUTPUT_ROOT", str(STAGING_ROOT / "reproduced"))).resolve()
SECTION_DIR = PACKAGE_ROOT / "3.5_ml_coverage"
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import loguniform

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

import json
import re
import matplotlib
matplotlib.use("Agg")  # headless on HPC
import matplotlib.pyplot as plt

# Silence convergence spam (toggle off if you want to debug)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------- Optional Group Lasso -----------------------
_HAS_GROUP_LASSO = False
try:
    from group_lasso import LogisticGroupLasso  # pip install group-lasso
    _HAS_GROUP_LASSO = True
except Exception:
    _HAS_GROUP_LASSO = False

# ----------------------- Paths / CLI / HPC knobs -----------------------

parser = argparse.ArgumentParser(description="Layer 1 Nested Stratified CV with BA-level features (relative-path defaults)")
parser.add_argument("--output-dir", default=str(REPRODUCED_ROOT / "3.5" / "training"))
parser.add_argument("--alphas", type=str, default="80,82,84,86,88,90,92,95,97,99,99.5")
# default models: elastic-net logistic, rf, gb, svm, knn, mlp
parser.add_argument("--models", type=str, default="logreg_en,rf,gb,svc,knn,mlp")
parser.add_argument("--no-shap", action="store_true", help="Disable SHAP computation")
parser.add_argument("--threads", type=int, default=1, help="Max BLAS/numexpr threads")
parser.add_argument("--jobs", type=int, default=-1, help="n_jobs for RandomizedSearchCV (-1 = all)")
parser.add_argument("--random-state", type=int, default=42)
parser.add_argument("--out-split", choices=["combined","alpha"], default="alpha",
                    help="Write combined CSVs (default) or split outputs by alpha subfolders (good for array jobs).")
parser.add_argument("--shap-plot-max", type=int, default=200,
                    help="Max samples for SHAP plotting (per fold).")

parser.add_argument("--ba-dir", type=Path, default=PACKAGE_ROOT / "shared/data/BA_daily_tables")
parser.add_argument("--check-inputs", action="store_true", help="Validate BA inputs and report dimensions without fitting models.")
args = parser.parse_args()

# Limit thread oversubscription
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, str(args.threads))

# Expose knobs
RANDOM_STATE = args.random_state
N_JOBS = args.jobs

output_dir = args.output_dir

# Results dirs (same structure as before)
high_stress_dir = os.path.join(output_dir, 'results', 'high_grid_stress')
ba_weather_dir  = os.path.join(output_dir, 'results', 'ba_grid_stress')
os.makedirs(ba_weather_dir, exist_ok=True)

# ----------------------- Global knobs for CV -----------------------
N_OUTER_FOLDS = 5
INNER_FOLDS = 3
N_SSS_SPLITS = 30
SSS_TEST_SIZE = 0.25
N_RANDOM_SEARCH = 50

# ----------------------- Utilities -----------------------

def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Convert Kelvin to Fahrenheit
_kelvin_to_f = lambda k: (k - 273.15) * 9/5 + 32

# HH: Helper to compute day index 1..365 even in leap years (drops Feb 29)
def _compute_day_index_no_leap(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates).sort_values()
    d = d[~((d.dt.month == 2) & (d.dt.day == 29))]
    uniq = d.dt.normalize().drop_duplicates()
    mapper = {dt: i+1 for i, dt in enumerate(uniq)}
    return d.dt.normalize().map(mapper).to_numpy()

# BA tables use degree days calculated AFTER averaging hourly temperature.
# Year/Day keys are preserved as stored; blank Date columns are not reconstructed.
def load_ba_features(directory):
    files = sorted(Path(directory).glob("*_daily_table.csv"))
    if not files:
        raise FileNotFoundError(f"No BA daily tables in {directory}")
    result = None
    excluded = []
    for path in files:
        ba = path.name.removesuffix("_daily_table.csv")
        d = pd.read_csv(path).sort_values(["Year", "Day"]).reset_index(drop=True)
        if d.duplicated(["Year", "Day"]).any():
            raise ValueError(f"Duplicate Year/Day keys: {path}")
        common = d[["Year", "Day", "ObjectiveValue", "CDD", "HDD"]]
        if result is None:
            result = common.copy()
        else:
            pd.testing.assert_frame_equal(result[common.columns], common, check_exact=False, rtol=1e-12, atol=1e-10)
        for source, prefix in [("CDD_BA", "CDD"), ("HDD_BA", "HDD"), ("Solar", "Solar"), ("Wind", "Wind")]:
            values = pd.to_numeric(d[source], errors="raise")
            if values.isna().all() or (prefix in ("Solar", "Wind") and values.fillna(0).eq(0).all()):
                excluded.append(f"{prefix}_{ba}")
                continue
            if prefix in ("CDD", "HDD"):
                expected = (d.T2 - 65).clip(lower=0) if prefix == "CDD" else (65 - d.T2).clip(lower=0)
                if not np.allclose(values, expected, equal_nan=True, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Degree-day definition differs from daily mean T2: {path}")
            result[f"{prefix}_{ba}"] = values.to_numpy()
    print("Excluded missing or zero-generation features:", ", ".join(excluded))
    return result

all_features_df = load_ba_features(args.ba_dir)
print("[METHOD CHANGE] BA daily-mean temperature degree days; archived ML results require retraining.")

# ----------------------- Build hot/cold subsets -----------------------

hot_data_all = all_features_df[all_features_df["CDD"] > 0].copy()
hot_data_all.drop(columns=[c for c in hot_data_all.columns if c.startswith("HDD_")], inplace=True, errors="ignore")
hot_data_all.drop(columns=["HDD"], inplace=True, errors="ignore")

cold_data_all = all_features_df[all_features_df["HDD"] > 0].copy()
cold_data_all.drop(columns=[c for c in cold_data_all.columns if c.startswith("CDD_")], inplace=True, errors="ignore")
cold_data_all.drop(columns=["CDD"], inplace=True, errors="ignore")

if args.check_inputs:
    report = {"rows": len(all_features_df), "columns": len(all_features_df.columns)}
    for name, frame, degree in [("hot", hot_data_all, "CDD_"), ("cold", cold_data_all, "HDD_")]:
        cols = [c for c in frame if c.startswith((degree, "Solar_", "Wind_"))]
        report[name] = {"rows": len(frame), "features": len(cols), "complete_rows": len(frame[["ObjectiveValue"] + cols].dropna())}
    print(json.dumps(report, indent=2))
    raise SystemExit(0)

# ----------------------- Modeling helpers -----------------------

@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    search_space: dict
    is_tree: bool = False

def label_by_percentile(df: pd.DataFrame, percentile: float) -> np.ndarray:
    cutoff = np.percentile(df["ObjectiveValue"], percentile)
    return (df["ObjectiveValue"] >= cutoff).astype(int).to_numpy()

def _compute_sample_weight_balanced(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight(class_weight="balanced", y=y)

def _supports_sample_weight(estimator) -> bool:
    import inspect
    try:
        sig = inspect.signature(estimator.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False

def _safe_probabilities(est, X: pd.DataFrame) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        elif proba.ndim == 1:
            return proba
    if hasattr(est, "decision_function"):
        scores = est.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    preds = est.predict(X)
    return preds.astype(float)

def _pick_tau_from_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0,
        )
    if thresholds.size == 0:
        return 0.5
    prec_thr = precisions[1:]
    rec_thr  = recalls[1:]
    f1_thr   = f1s[1:]
    diffs = np.abs(prec_thr - rec_thr)
    idxs = np.where(diffs == np.nanmin(diffs))[0]
    if idxs.size > 1:
        best = idxs[np.argmax(f1_thr[idxs])]
    else:
        best = idxs[0]
    return float(thresholds[best])

def _mean_sd_ci(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    half = 1.96 * (sd / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}

def _json_safe(obj):
    """
    Make objects JSON-serializable (handles numpy types/arrays, tuples, dicts).
    Long 1D arrays are summarized to avoid giant JSON blobs.
    """
    import numpy as _np
    import json as _json

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        if obj.ndim == 1 and obj.size > 50:
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        _json.dumps(obj)  # will succeed if already serializable
        return obj
    except TypeError:
        return str(obj)

def compute_builtin_importances(estimator, feature_names, subset, alpha, model_key, outer_fold):
    cols = ["subset", "alpha", "model", "outer_fold", "feature",
            "importance", "importance_norm", "rank"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=cols)
    imp = np.asarray(estimator.feature_importances_, dtype=float)
    feature_names = list(feature_names)
    n = min(len(imp), len(feature_names))
    imp = imp[:n]
    feature_names = feature_names[:n]
    maxv = float(imp.max()) if imp.size else 0.0
    imp_norm = (imp / maxv) if maxv > 0 else imp
    df = pd.DataFrame({
        "subset": [subset] * n,
        "alpha":  [alpha]  * n,
        "model":  [model_key] * n,
        "outer_fold": [outer_fold] * n,
        "feature": feature_names,
        "importance": imp,
        "importance_norm": imp_norm,
    })
    df["rank"] = df["importance"].rank(ascending=False, method="dense").astype(int)
    return df

# SHAP helper (tree models + linear + svm)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Which models get SHAP (skip MLP per request)
SHAP_MODELS = {"logreg_en", "logreg_l1", "group_lasso", "rf", "gb", "svc"}


def _pipeline_parts(best_est):
    """Return (preprocessor, clf) from a sklearn Pipeline or (None, model)."""
    try:
        if hasattr(best_est, "named_steps"):
            return best_est.named_steps.get("scaler", None), best_est.named_steps.get("clf", best_est)
    except Exception:
        pass
    return None, best_est


def _transform_if_scaler(scaler, X: pd.DataFrame) -> pd.DataFrame:
    if scaler is None:
        return X
    Xt = scaler.transform(X)
    return pd.DataFrame(Xt, index=X.index, columns=X.columns)


def _tree_explainer_path_dep(clf):
    # Prefer path-dependent; fall back gracefully if SHAP version differs
    try:
        return shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    except Exception:
        return shap.TreeExplainer(clf)


def _compute_shap_generic(best_est, model_key: str, X_tr: pd.DataFrame, X_te: pd.DataFrame,
                          max_samples: int = 200, random_state: int = 42):
    """
    Returns (shap_values, X_eval_for_plot).
      - Tree: TreeExplainer with path-dependent perturbation
      - Linear: LinearExplainer on scaled features
      - SVM: KernelExplainer on P(y=1) with small background
    """
    if not _HAS_SHAP:
        raise RuntimeError("SHAP not available (import failed).")

    # Subsample for background/eval to bound cost
    n_bg  = min(max_samples, len(X_tr))
    n_eval = min(max_samples, len(X_te))
    X_bg   = X_tr.sample(n_bg, random_state=random_state) if len(X_tr) > n_bg else X_tr.copy()
    X_eval = X_te.sample(n_eval, random_state=random_state) if len(X_te) > n_eval else X_te.copy()

    scaler, clf = _pipeline_parts(best_est)

    if model_key in {"rf", "gb"}:
        explainer = _tree_explainer_path_dep(clf)
        sv = explainer.shap_values(X_eval)
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]
        return sv, X_eval

    if model_key in {"logreg_en", "logreg_l1", "group_lasso"}:
        X_bg_s   = _transform_if_scaler(scaler, X_bg)
        X_eval_s = _transform_if_scaler(scaler, X_eval)
        explainer = shap.LinearExplainer(clf, X_bg_s)
        sv = explainer.shap_values(X_eval_s)
        return sv, X_eval

    if model_key == "svc":
        # Kernel SHAP on proba of class 1
        def predict_pos_proba(Xarray):
            Xdf = pd.DataFrame(Xarray, columns=X_tr.columns)
            Xs  = _transform_if_scaler(scaler, Xdf)
            return clf.predict_proba(Xs)[:, 1]        # call the classifier, not the pipeline
        bg = X_bg.sample(min(100, len(X_bg)), random_state=random_state) if len(X_bg) > 100 else X_bg
        explainer = shap.KernelExplainer(predict_pos_proba, bg, link="logit")
        sv = explainer.shap_values(X_eval, nsamples=100)
        return sv, X_eval

    raise ValueError(f"SHAP not implemented for model '{model_key}'")




def _useful_df(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    return df.dropna(how="all", axis=1).shape[1] > 0


def safe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    good = [d for d in dfs if _useful_df(d)]
    return pd.concat(good, ignore_index=True) if len(good) else pd.DataFrame()

# ----------------------- Group building for Group Lasso -----------------------

def _extract_ba_from_feat(feat: str) -> str:
    # Features like CDD_PJM, HDD_PJM, Solar_PJM, Wind_PJM
    # Take the suffix after last underscore as BA key
    if '_' in feat:
        return feat.rsplit('_', 1)[-1]
    return feat


def _groups_for_features(feature_names: List[str]) -> np.ndarray:
    # Same BA across variables gets same group id
    bas = [_extract_ba_from_feat(f) for f in feature_names]
    uniq = {ba:i for i, ba in enumerate(sorted(set(bas)))}
    return np.array([uniq[ba] for ba in bas], dtype=int)

# ----------------------- Model specs -----------------------

def get_model_specs() -> Dict[str, ModelSpec]:
    specs: Dict[str, ModelSpec] = {}

    # Elastic-Net Logistic Regression (replaces old L1)
    pipe_lr_en = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            l1_ratio=0.2,
        )),
    ])
    specs["logreg_en"] = ModelSpec(
        name="logreg_en",
        pipeline=pipe_lr_en,
        search_space={
            "clf__C": loguniform(1e-3, 1e2),
            "clf__l1_ratio": [0.1, 0.2, 0.3],
            "clf__max_iter": [5000, 10000],
            "clf__tol": [1e-3, 1e-4],
            "clf__class_weight": ["balanced"],
        },
        is_tree=False,
    )
    
    # NEW: Logistic Regression (L1)
    pipe_lr_l1 = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    specs["logreg_l1"] = ModelSpec(
        name="logreg_l1",
        pipeline=pipe_lr_l1,
        search_space={
            "clf__C": loguniform(1e-3, 1e2),
            "clf__max_iter": [5000, 10000],
            "clf__tol": [1e-3, 1e-4],
            "clf__class_weight": ["balanced"],
        },
        is_tree=False,
    )

    # Random Forest
    pipe_rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=1,
        ))
    ])
    specs["rf"] = ModelSpec(
        name="rf",
        pipeline=pipe_rf,
        search_space={
            "clf__n_estimators": [200,300,400,500,600,700,800,900,1000],
            "clf__max_depth": [None, 5, 10, 20, 40],
            "clf__min_samples_split": [2,5,10],
            "clf__min_samples_leaf": [1,2,4],
            "clf__max_features": ["sqrt","log2", 0.3, 0.6, 1.0],
            "clf__class_weight": [None, "balanced", "balanced_subsample"],
        },
        is_tree=True,
    )

    # Gradient Boosting
    pipe_gb = Pipeline([
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    specs["gb"] = ModelSpec(
        name="gb",
        pipeline=pipe_gb,
        search_space={
            "clf__n_estimators": [100,200,300,400,500,600,700,800],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__max_depth": [2,3,4,5],
            "clf__subsample": [0.6, 0.8, 1.0],
        },
        is_tree=True,
    )

    # Support Vector Classifier
    pipe_svc = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    specs["svc"] = ModelSpec(
        name="svc",
        pipeline=pipe_svc,
        search_space={
            "clf__C": loguniform(1e-1, 1e2),
            "clf__gamma": list(np.logspace(-4, -1, 4)) + ["scale", "auto"],
        },
        is_tree=False,
    )

    # k-NN
    pipe_knn = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", KNeighborsClassifier())
    ])
    specs["knn"] = ModelSpec(
        name="knn",
        pipeline=pipe_knn,
        search_space={
            "clf__n_neighbors": list(range(3, 26)),
            "clf__weights": ["uniform","distance"],
            "clf__metric": ["euclidean","manhattan"],
        },
        is_tree=False,
    )

    # MLP (kept in model list; no SHAP)
    pipe_mlp = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(
            solver="adam",
            learning_rate="adaptive",
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            max_iter=5000,
            tol=1e-4,
            random_state=RANDOM_STATE
        ))
    ])
    specs["mlp"] = ModelSpec(
        name="mlp",
        pipeline=pipe_mlp,
        search_space={
            "clf__hidden_layer_sizes": [(50,), (100,), (50,50), (100,50)],
            "clf__alpha": loguniform(1e-5, 1e-2),
            "clf__learning_rate_init": [0.0005, 0.001, 0.003],
            "clf__activation": ["relu", "tanh"],
            "clf__batch_size": ["auto", 128, 256],
            "clf__max_iter": [2000, 5000],
            "clf__tol": [1e-3, 1e-4],
        },
        is_tree=False
    )

    # Optional: Group Lasso (if available)
    if _HAS_GROUP_LASSO:
        pipe_gl = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticGroupLasso(
                groups=None,           # to be injected later per feature list
                group_reg=0.05,        # tuned by search
                l1_reg=0.0,            # pure group penalty by default
                n_iter=1000,
                tol=1e-3,
                supress_warning=True,
                random_state=RANDOM_STATE,
            ))
        ])
        specs["group_lasso"] = ModelSpec(
            name="group_lasso",
            pipeline=pipe_gl,
            search_space={
                "clf__group_reg": [0.01, 0.03, 0.05, 0.1, 0.2],
                "clf__l1_reg": [0.0, 1e-6, 1e-5, 1e-4],
                # groups will be injected before search.fit
            },
            is_tree=False,
        )

    return specs

# ----------------------- Adaptive CV builders -----------------------

def _class_counts(y: np.ndarray) -> Tuple[int,int,int]:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_pos, n_neg, min(n_pos, n_neg)

def _build_outer_cv_safe(y: np.ndarray, desired_folds: int = 5, base_test_size: float = 0.25, random_state: int = 42):
    n_pos, n_neg, n_min = _class_counts(y)
    if n_min == 0:
        return None, f"skip: only one class present (pos={n_pos}, neg={n_neg})"
    n_splits = min(desired_folds, n_min)
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), None
    p, n = n_pos, n_neg
    low  = max(1.0/p if p >= 2 else 0.5, 1.0/n if n >= 2 else 0.5)
    high = min(1.0 - 1.0/p if p >= 2 else 0.5, 1.0 - 1.0/n if n >= 2 else 0.5)
    if low >= high:
        return None, f"skip: too few positives/negatives for any valid split (pos={n_pos}, neg={n_neg})"
    test_size = min(max(base_test_size, low + 1e-6), high - 1e-6)
    return StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state), None

def _build_inner_cv_safe(y: np.ndarray, alpha: float, base_inner_folds: int = 3, base_test_size: float = 0.25, random_state: int = 42):
    n_pos, n_neg, n_min = _class_counts(y)
    if n_min == 0:
        return None
    if alpha >= 99:
        p, n = n_pos, n_neg
        low  = max(1.0/p if p >= 2 else 0.5, 1.0/n if n >= 2 else 0.5)
        high = min(1.0 - 1.0/p if p >= 2 else 0.5, 1.0 - 1.0/n if n >= 2 else 0.5)
        if low >= high:
            return None
        test_size = min(max(base_test_size, low + 1e-6), high - 1e-6)
        return StratifiedShuffleSplit(n_splits=30, test_size=test_size, random_state=random_state)
    else:
        n_splits = min(base_inner_folds, n_min)
        if n_splits < 2:
            return None
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# ----------------------- Nested CV core -----------------------

def _logreg_params_rows(best_est, feature_names: List[str], subset: str, alpha: float, model_key: str, outer_fold: int) -> List[dict]:
    rows = []
    try:
        scaler, clf = _pipeline_parts(best_est)
        if not hasattr(clf, "coef_"):
            return rows
        w_scaled = clf.coef_.ravel()
        b_scaled = float(clf.intercept_.ravel()[0]) if hasattr(clf, "intercept_") else 0.0
        # Unscale back to original feature units if StandardScaler used
        if scaler is not None and hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
            scale = np.asarray(scaler.scale_, dtype=float)
            mean  = np.asarray(scaler.mean_, dtype=float)
            w_unscaled = w_scaled / scale
            b_unscaled = b_scaled - np.sum(w_scaled * mean / scale)
        else:
            w_unscaled = w_scaled.copy()
            b_unscaled = b_scaled
        for feat, ws, wu in zip(feature_names, w_scaled, w_unscaled):
            rows.append({
                "subset": subset,
                "alpha": alpha,
                "model": model_key,
                "outer_fold": outer_fold,
                "feature": feat,
                "coef_scaled": float(ws),
                "coef_unscaled": float(wu),
                "intercept_scaled": float(b_scaled),
                "intercept_unscaled": float(b_unscaled),
            })
    except Exception:
        pass
    return rows


def nested_cv_one_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_key: str,
    alpha: float,
    subset_name: str,
    compute_shap: bool = False,
    shap_max_samples: int = 200,
    random_state: int = RANDOM_STATE,
    outer_cv=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    specs = get_model_specs()
    if model_key not in specs:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(specs)}")
    spec = specs[model_key]

    if outer_cv is None:
        outer_cv, reason = _build_outer_cv_safe(y, desired_folds=N_OUTER_FOLDS, base_test_size=SSS_TEST_SIZE, random_state=random_state)
    else:
        reason = None
    if outer_cv is None:
        warnings.warn(f"Skipping model {model_key} (alpha {alpha}, subset {subset_name}): {reason}")
        empty_shap_cols = ["subset","alpha","model","outer_fold","feature","mean_abs_shap"]
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=empty_shap_cols), pd.DataFrame(), pd.DataFrame()

    rows_metrics = []
    rows_pred = []
    shap_rows = []
    builtin_importances_list = []
    logreg_param_rows = []

    fold_shap_summaries = []
    fold_shap_arrays    = []
    fold_X_test_list    = []

    # If group lasso: inject groups into search space / estimator now
    groups = None
    if model_key == "group_lasso":
        groups = _groups_for_features(list(X.columns))

    for outer_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        inner_cv = _build_inner_cv_safe(y_tr, alpha, base_inner_folds=INNER_FOLDS, base_test_size=SSS_TEST_SIZE, random_state=random_state)

        search = RandomizedSearchCV(
            estimator=spec.pipeline,
            param_distributions=dict(spec.search_space),
            n_iter=N_RANDOM_SEARCH,
            scoring="average_precision",
            n_jobs=N_JOBS,
            cv=inner_cv if inner_cv is not None else 3,
            refit=True,
            random_state=random_state,
            verbose=0,
        )
        # Inject GL groups so cloned estimators during search carry them
        if model_key == "group_lasso" and groups is not None:
            search.param_distributions["clf__groups"] = [groups]

        fit_params = {}
        try:
            final_est = spec.pipeline.named_steps.get("clf")
            if _supports_sample_weight(final_est) and not hasattr(final_est, "class_weight"):
                sw = _compute_sample_weight_balanced(y_tr)
                fit_params = {"clf__sample_weight": sw}
        except Exception:
            pass

        if inner_cv is None:
            best_est = spec.pipeline
            # Ensure GL groups present even without search
            if model_key == "group_lasso" and groups is not None:
                best_est.set_params(**{"clf__groups": groups})
            best_est.fit(X_tr, y_tr, **fit_params)
            best_inner_pr_auc = float('nan')
            best_params_dict = {}
        else:
            search.fit(X_tr, y_tr, **fit_params)
            best_est = search.best_estimator_
            best_inner_pr_auc = float(search.best_score_)
            best_params_dict = dict(search.best_params_)
            # summarize giant arrays (e.g., clf__groups from group-lasso) before dumping to JSON
            if "clf__groups" in best_params_dict:
                _grp = best_params_dict["clf__groups"]
                glen = len(_grp) if hasattr(_grp, "__len__") else "?"
                best_params_dict["clf__groups"] = f"groups(len={glen})"

        # τ* tuning
        try:
            if inner_cv is None:
                y_scores_tr = _safe_probabilities(best_est, X_tr)
                tau_star = _pick_tau_from_pr(y_tr, y_scores_tr)
            else:
                y_scores_oof = np.zeros_like(y_tr, dtype=float)
                for tr_sub, val_sub in inner_cv.split(X_tr, y_tr):
                    Xt_sub, Xv_sub = X_tr.iloc[tr_sub], X_tr.iloc[val_sub]
                    yt_sub = y_tr[tr_sub]
                    est = clone(best_est)
                    fp = {}
                    try:
                        fin = est.named_steps.get("clf")
                        if _supports_sample_weight(fin) and not hasattr(fin, "class_weight"):
                            fp = {"clf__sample_weight": _compute_sample_weight_balanced(yt_sub)}
                    except Exception:
                        pass
                    est.fit(Xt_sub, yt_sub, **fp)
                    y_scores_oof[val_sub] = _safe_probabilities(est, Xv_sub)
                tau_star = _pick_tau_from_pr(y_tr, y_scores_oof)
        except Exception:
            warnings.warn("Threshold tuning failed; defaulting τ* = 0.5")
            tau_star = 0.5

        # Final fit on all outer-train
        best_est.fit(X_tr, y_tr, **fit_params)

        # Evaluate on outer-test
        y_score_te = _safe_probabilities(best_est, X_te)
        y_pred_te  = (y_score_te >= tau_star).astype(int)

        pr_auc = average_precision_score(y_te, y_score_te)
        prec = precision_score(y_te, y_pred_te, zero_division=0)
        rec  = recall_score(y_te, y_pred_te, zero_division=0)
        f1   = f1_score(y_te, y_pred_te, zero_division=0)
        acc  = accuracy_score(y_te, y_pred_te)

        # Built-in impurity importances for tree models (RF/GB)
        model_for_imp = best_est.named_steps.get("clf", best_est)
        feat_imp_df_fold = compute_builtin_importances(
            estimator=model_for_imp,
            feature_names=X.columns,
            subset=subset_name,
            alpha=alpha,
            model_key=model_key,
            outer_fold=outer_i,
        )
        builtin_importances_list.append(feat_imp_df_fold)

        rows_metrics.append({
            "subset": subset_name,
            "alpha": alpha,
            "model": model_key,
            "outer_fold": outer_i,
            "PR_AUC": pr_auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "tau_star": tau_star,
            "best_inner_pr_auc": best_inner_pr_auc,
            "best_params_json": json.dumps(_json_safe(best_params_dict), sort_keys=True),
        })
        rows_pred.append({
            "subset": subset_name,
            "alpha": alpha,
            "model": model_key,
            "outer_fold": outer_i,
            "y_true": y_te.tolist(),
            "y_score": y_score_te.tolist(),
            "y_pred": y_pred_te.tolist(),
        })

        # Save logistic params (both logreg_en and legacy key)
        if model_key in {"logreg_en", "logreg_l1", "group_lasso"}:
            logreg_param_rows.extend(
                _logreg_params_rows(best_est, list(X.columns), subset_name, alpha, model_key, outer_i)
            )

        # SHAP (optional) for: LogReg, RF, GB, SVM (skip MLP)
        if compute_shap and (model_key in SHAP_MODELS) and _HAS_SHAP:
            try:
                sv, X_eval_for_plots = _compute_shap_generic(
                    best_est=best_est,
                    model_key=model_key,
                    X_tr=X_tr,
                    X_te=X_te,
                    max_samples=args.shap_plot_max,
                    random_state=RANDOM_STATE
                )
                abs_mean = np.mean(np.abs(sv), axis=0)
                for feat, val in zip(X.columns, abs_mean):
                    shap_rows.append({
                        "subset": subset_name,
                        "alpha": alpha,
                        "model": model_key,
                        "outer_fold": outer_i,
                        "feature": feat,
                        "mean_abs_shap": float(val),
                    })

                fold_shap_summaries.append(
                    pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": abs_mean})
                )
                fold_shap_arrays.append(sv)
                X_eval_df = X_eval_for_plots if isinstance(X_eval_for_plots, pd.DataFrame) \
                            else pd.DataFrame(X_eval_for_plots, columns=X.columns)
                fold_X_test_list.append(X_eval_df)


            except Exception as e:
                warnings.warn(f"SHAP computation failed on fold {outer_i} for {model_key}: {e}")

    # Aggregate SHAP across folds (combined bar + beeswarm)

    fold_metrics_df = pd.DataFrame(rows_metrics)
    fold_predictions_df = pd.DataFrame(rows_pred)
    shap_summary_df = pd.DataFrame(shap_rows) if shap_rows else pd.DataFrame(columns=["subset","alpha","model","outer_fold","feature","mean_abs_shap"])
    builtin_importances_df = pd.concat(builtin_importances_list, ignore_index=True) if builtin_importances_list else pd.DataFrame()
    logreg_params_df = pd.DataFrame(logreg_param_rows) if logreg_param_rows else pd.DataFrame(
        columns=["subset","alpha","model","outer_fold","feature","coef_scaled","coef_unscaled","intercept_scaled","intercept_unscaled"]
    )

    return fold_metrics_df, fold_predictions_df, shap_summary_df, builtin_importances_df, logreg_params_df

# ----------------------- Runner & aggregation -----------------------

def run_nested_cv_layer1(
    hot_data: pd.DataFrame,
    cold_data: pd.DataFrame,
    alphas: List[float],
    model_keys: List[str],
    features_hot: List[str],
    features_cold: List[str],
    compute_shap: bool,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:

    def _prep(df: pd.DataFrame, feat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = ["ObjectiveValue"] + feat_cols
        dfc = df[cols].dropna().copy()
        X = dfc[feat_cols]
        return dfc, X

    hot_clean, X_hot = _prep(hot_data, features_hot)
    cold_clean, X_cold = _prep(cold_data, features_cold)

    all_metrics = []
    all_preds = []
    all_shap = []
    skips = []
    all_builtin_imps = []
    all_logreg_params = []

    for subset_name, base_df, X in [("hot", hot_clean, X_hot), ("cold", cold_clean, X_cold)]:
        for alpha in alphas:
            y = label_by_percentile(base_df, alpha)
            outer_cv, skip_reason = _build_outer_cv_safe(y, desired_folds=N_OUTER_FOLDS, base_test_size=SSS_TEST_SIZE, random_state=RANDOM_STATE)
            if outer_cv is None:
                skips.append({"subset": subset_name, "alpha": alpha, "reason": skip_reason})
                continue

            for m in model_keys:
                fold_metrics, fold_preds, shap_df, feat_imp_df, params_df = nested_cv_one_model(
                    X=X,
                    y=y,
                    model_key=m,
                    alpha=alpha,
                    subset_name=subset_name,
                    compute_shap=compute_shap,
                    outer_cv=outer_cv,
                )
                if not fold_metrics.empty:
                    all_metrics.append(fold_metrics)
                if not fold_preds.empty:
                    all_preds.append(fold_preds)
                if not shap_df.empty:
                    all_shap.append(shap_df)
                if not feat_imp_df.empty:
                    all_builtin_imps.append(feat_imp_df)
                if not params_df.empty:
                    all_logreg_params.append(params_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    preds_df   = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    shap_df    = pd.concat(all_shap, ignore_index=True) if all_shap else pd.DataFrame()
    skips_df   = pd.DataFrame(skips) if skips else pd.DataFrame(columns=["subset","alpha","reason"])
    builtin_df = pd.concat(all_builtin_imps, ignore_index=True) if all_builtin_imps else pd.DataFrame()
    logreg_params_all = pd.concat(all_logreg_params, ignore_index=True) if all_logreg_params else pd.DataFrame()

    # Aggregate metrics by subset/alpha/model
    summaries = []
    if not metrics_df.empty:
        for (subset, alpha, model), g in metrics_df.groupby(["subset","alpha","model"], sort=False):
            def colstat(c):
                arr = g[c].astype(float).tolist()
                s = _mean_sd_ci(arr)
                return s["mean"], s["sd"], s["ci95_low"], s["ci95_high"]
            pr_mean, pr_sd, pr_lo, pr_hi = colstat("PR_AUC")
            p_mean, p_sd, p_lo, p_hi     = colstat("precision")
            r_mean, r_sd, r_lo, r_hi     = colstat("recall")
            f1_mean, f1_sd, f1_lo, f1_hi = colstat("f1")
            a_mean, a_sd, a_lo, a_hi     = colstat("accuracy")
            t_mean, t_sd, t_lo, t_hi     = colstat("tau_star")
            summaries.append({
                "subset": subset,
                "alpha": alpha,
                "model": model,
                "PR_AUC_mean": pr_mean, "PR_AUC_sd": pr_sd, "PR_AUC_ci95_low": pr_lo, "PR_AUC_ci95_high": pr_hi,
                "precision_mean": p_mean, "precision_sd": p_sd, "precision_ci95_low": p_lo, "precision_ci95_high": p_hi,
                "recall_mean": r_mean, "recall_sd": r_sd, "recall_ci95_low": r_lo, "recall_ci95_high": r_hi,
                "f1_mean": f1_mean, "f1_sd": f1_sd, "f1_ci95_low": f1_lo, "f1_ci95_high": f1_hi,
                "accuracy_mean": a_mean, "accuracy_sd": a_sd, "accuracy_ci95_low": a_lo, "accuracy_ci95_high": a_hi,
                "tau_star_mean": t_mean, "tau_star_sd": t_sd, "tau_star_ci95_low": t_lo, "tau_star_ci95_high": t_hi,
            })
    metrics_summary = pd.DataFrame(summaries)

    if output_dir:
        # Only the four result tables consumed by the paper's ML analyses.
        tables = {"fold_metrics": metrics_df, "metrics_summary": metrics_summary,
                  "predictions": preds_df, "shap_summary": shap_df}
        for name, frame in tables.items():
            if frame.empty:
                continue
            groups = frame.groupby("alpha") if args.out_split == "alpha" else [(None, frame)]
            for alpha, group in groups:
                folder = Path(output_dir) if alpha is None else Path(output_dir) / ("alpha_" + str(alpha).replace(".", "_"))
                folder.mkdir(parents=True, exist_ok=True)
                group.to_csv(folder / f"layer1_nestedcv_{name}.csv", index=False)
        if not skips_df.empty:
            print("[SKIPPED]", skips_df.to_string(index=False))

    return {
        "metrics": metrics_df,
        "metrics_summary": metrics_summary,
        "predictions": preds_df,
        "shap": shap_df,
        "builtin_importances": builtin_df,
        "logreg_params": logreg_params_all,
        "skips": skips_df,
    }

# ----------------------- Main -----------------------

if __name__ == "__main__":
    alpha_list = [float(a) for a in args.alphas.split(",")]
    model_keys = tuple(m.strip() for m in args.models.split(",") if m.strip())

    # Dynamic feature lists from columns
    hot_feature_cols  = [c for c in hot_data_all.columns  if c.startswith("CDD_") or c.startswith("Solar_") or c.startswith("Wind_")]
    cold_feature_cols = [c for c in cold_data_all.columns if c.startswith("HDD_") or c.startswith("Solar_") or c.startswith("Wind_")]

    # Run nested CV
    results = run_nested_cv_layer1(
        hot_data=hot_data_all,
        cold_data=cold_data_all,
        alphas=alpha_list,
        model_keys=model_keys,
        features_hot=hot_feature_cols,
        features_cold=cold_feature_cols,
        compute_shap=(not args.no_shap),
        output_dir=ba_weather_dir,
    )

    # Tiny preview
    try:
        print("\n[Summary preview]\n", results["metrics_summary"].head())
        if not results["skips"].empty:
            print("\n[Notice] Skipped (subset, alpha):\n", results["skips"])
    except Exception:
        pass

