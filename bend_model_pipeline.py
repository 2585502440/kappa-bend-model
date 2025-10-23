# -*- coding: utf-8 -*-
"""
PNIPAM:PDMS bending — training-only pipeline with paper-friendly baseline summaries

What this script does
---------------------
- Train θ_tip (deg) model from (ratio, temperature) and evaluate on a held-out test set.
- Train curvature-shape branch (PCA + regressors) to enable θ(s) auto-generation after training.
- Compute two geometry baselines under unified preprocessing:
    (1) Chord–Sagitta (constant κ), (2) Polyline-κ (curvature from resampled XY).
- Save the usual figures (parity/residual/learning curve/robustness/response surface).
- Paper-friendly summaries under outputs/paper/:
    1) baseline_mae_ci.png — MAE with 95% bootstrap CIs (same test set for both baselines; optional ML).
    2) baseline_parity_combined.png — both baselines on one parity axis (truth vs prediction).
    3) baseline_theta_delta_ci.png — mean ± 95% band of [θ_baseline(s) − θ_ref(s)] over s∈[0,1],
       aggregated across all available files (unit-consistent).

NEW in this version
-------------------
- Cumulative Simpson integration and Trapezoid-vs-Simpson consistency check for discretization uncertainty.
- 2D rigid registration (Kabsch/Procrustes) on (s_norm, θ(s)) curves; export RMS & Hausdorff distances.
- Paired permutation test (two-sided) comparing methods' MAE/RMSE; plus bootstrap 95% CIs for MAE/RMSE/R².
- GUM-style uncertainty (MC propagation): Class A (digitization/quantization), Class B (calibration k),
  and discretization term (trap vs simpson). Per-file CSV and overall JSON summaries.

Inputs & calibration
--------------------
- Place your files under data_root/train/ and data_root/test/.
- CSV/XLSX should contain either curvature columns ("Point Curvature", optional "Sign") and/or XY columns.


Usage
-----
python bend_model_pipeline.py --data_root <folder with train/ and test/> --out_dir <outputs/> --um_per_px 0.62
"""

from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
import joblib

# ----------------------- Tunables -----------------------
RANDOM_SEED = 42
N_FOLDS = 5

TOL_TEMP_STD = 0.5
TOL_RATIO_STD = 0.1
N_MONTE_CARLO = 2000

GRID_T_MIN, GRID_T_MAX, GRID_T_STEP = 20, 60, 1
GRID_R_MIN, GRID_R_MAX, GRID_R_STEP = 1.0, 5.0, 0.05

N_POINTS_CURV = 200          # resampling points for curvature/θ(s)
N_POINTS_BASELINE = 400      # resampling points for XY-based baselines

# Global pixel→micron fallback (μm/px) used only if not found in files
UM_PER_PX = 1.0

# Bootstrap settings for paper figures
N_BOOT = 10000
ALPHA = 0.05

# NEW: Uncertainty & statistics tunables
N_MC_UNCERT = 2000           # Monte Carlo samples for GUM-style uncertainty
REL_K_STD = 0.0             # relative standard uncertainty for calibration factor k (Type B), e.g., 2%
N_PERM = 20000               # permutations (sign-flips) for paired permutation tests

# ----------------------- Paths -----------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"
OUT_DIR   = ROOT / "outputs"
FIG_DIR   = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)

# Paper-friendly outputs (figures + csv)
PAPER_FIG_DIR = OUT_DIR / "paper" / "figs"
PAPER_CSV_DIR = OUT_DIR / "paper" / "csv"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_CSV_DIR.mkdir(parents=True, exist_ok=True)

# NEW: statistics outputs
STATS_DIR = OUT_DIR / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

_ratio_re = re.compile(r"(\d+)\s*vs\s*(\d+)", re.IGNORECASE)
_temp_re  = re.compile(r"(\d+)\s*C", re.IGNORECASE)

# ----------------------- Small IO utils -----------------------
def configure_paths(data_root: Optional[str], out_dir: Optional[str]) -> None:
    """Set data/output directories at runtime."""
    global DATA_ROOT, TRAIN_DIR, TEST_DIR, OUT_DIR, FIG_DIR, PAPER_FIG_DIR, PAPER_CSV_DIR, STATS_DIR
    if data_root:
        DATA_ROOT = Path(data_root).resolve()
        TRAIN_DIR = DATA_ROOT / "train"; TEST_DIR = DATA_ROOT / "test"
        print(f"[INFO] Using data_root = {DATA_ROOT}")
    if out_dir:
        OUT_DIR = Path(out_dir).resolve()
        print(f"[INFO] Using out_dir = {OUT_DIR}")
    FIG_DIR = OUT_DIR / "figures"
    OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR = OUT_DIR / "paper" / "figs"
    PAPER_CSV_DIR = OUT_DIR / "paper" / "csv"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_CSV_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR = OUT_DIR / "stats"
    STATS_DIR.mkdir(parents=True, exist_ok=True)

def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV/XLSX with a few fallback encodings."""
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def _find_col_like(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    """Find the first column whose lowercase name contains any of the keys."""
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None

def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized cumulative trapezoidal integral."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    if len(y) < 2: return np.zeros_like(y)
    dx = np.diff(x); mid = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y, float); out[1:] = np.cumsum(mid)
    return out

# NEW: cumulative Simpson integration (non-uniform x) via local quadratic fits
def _cumsimpson(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Cumulative 'Simpson-like' integral for possibly non-uniform x by integrating a local quadratic
    through triples (x[i-2],x[i-1],x[i]) over [x[i-2], x[i]]. The last leftover interval (odd length)
    is integrated with trapezoid. Falls back to trapezoid if <3 points.
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    if n < 3:
        return _cumtrapz(y, x)
    out = np.zeros(n, float)
    # first step: use trapezoid for i=1
    out[1] = out[0] + 0.5 * (y[0] + y[1]) * (x[1] - x[0])
    # then accumulate by quadratic segments for i>=2
    for i in range(2, n):
        # quadratic through (x[i-2], y[i-2]), (x[i-1], y[i-1]), (x[i], y[i])
        xs = x[i-2:i+1]; ys = y[i-2:i+1]
        # fit quadratic a t^2 + b t + c
        try:
            a, b, c = np.polyfit(xs, ys, 2)
            # integral from x[i-1] to x[i]
            F = lambda t: (a/3.0)*t**3 + 0.5*b*t**2 + c*t
            # add the last slice incrementally
            inc = F(xs[2]) - F(xs[1])
            out[i] = out[i-1] + inc
        except Exception:
            # fallback: trapezoid for the last small step
            out[i] = out[i-1] + 0.5 * (ys[1] + ys[2]) * (xs[2] - xs[1])
    return out

def _trapz_safe(y: np.ndarray, x: np.ndarray) -> float:
    """Safe trapezoid area (NumPy ≥1.26 uses trapezoid)."""
    try:    return float(np.trapezoid(y, x))
    except AttributeError: return float(np.trapz(y, x))

def _interp_fill_nans(v: np.ndarray) -> np.ndarray:
    """Linear interpolate to fill NaNs, keep length unchanged."""
    v = np.asarray(v, float); idx = np.flatnonzero(np.isfinite(v))
    if len(idx) == 0: return np.zeros_like(v)
    return np.interp(np.arange(len(v)), idx, v[idx])

# ----------------------- Calibration helpers -----------------------
def _str_has_um(s: str) -> bool:
    s = s.lower()
    return ("um" in s) or ("µm" in s)

def _infer_um_per_px_from_df(df: pd.DataFrame) -> Optional[float]:
    """Try to infer μm/px from typical column names or inline strings."""
    cand_cols = [c for c in df.columns if any(k in c.lower() for k in
                   ["um per pixel", "um/pixel", "pixel size", "scale", "calibration", "um_per_px"])]
    for c in cand_cols:
        try:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                v = float(vals.iloc[0])
                if np.isfinite(v) and v > 0: return v
        except Exception:
            pass
    if len(df) > 0:
        first_row = " ".join(str(x) for x in df.iloc[0].tolist())
        m = re.search(r"([\d.]+)\s*(um|µm)\s*/\s*(px|pixel)", first_row, re.I)
        if m:
            v = float(m.group(1))
            if v > 0: return v
    return None

def _get_xy_in_um(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, bool]:
    """Return XY in microns (x_um,y_um), along with scale (μm/px) and a boolean telling if XY already in μm."""
    x_col = _find_col_like(df, ["x-coordinate", "x (", "x)", " x", "x_um", "x"])
    y_col = _find_col_like(df, ["y-coordinate", "y (", "y)", " y", "y_um", "y"])
    if x_col is None or y_col is None:
        return None, None, 1.0, False

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    x = _interp_fill_nans(x); y = _interp_fill_nans(y)

    if _str_has_um(x_col) or _str_has_um(y_col):
        return x, y, 1.0, True

    scale = _infer_um_per_px_from_df(df)
    if scale is None: scale = UM_PER_PX
    if scale <= 0: scale = 1.0
    return x * scale, y * scale, scale, False

def _arc_length(x_um: np.ndarray, y_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative arc length s and segment lengths ds (both in μm)."""
    ds = np.hypot(np.diff(x_um), np.diff(y_um))
    s = np.r_[0.0, np.cumsum(ds)]
    return s, np.r_[ds, ds[-1] if len(ds) else 0.0]

def _resample_equal_s(x_um: np.ndarray, y_um: np.ndarray, N: int = N_POINTS_BASELINE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample XY to equal arc-length spacing for stable curvature estimation."""
    s, _ = _arc_length(x_um, y_um)
    if s[-1] <= 0: s = np.arange(len(x_um), dtype=float)
    s_new = np.linspace(s[0], s[-1], max(2, N))
    x_new = np.interp(s_new, s, x_um)
    y_new = np.interp(s_new, s, y_um)
    return x_new, y_new, s_new

# ----------------------- Parsing helpers -----------------------
def parse_ratio_from_path(path: Path) -> Optional[float]:
    m = _ratio_re.search(str(path))
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a / b if b else None
    name = path.name.lower()
    if name.startswith(("c_", "c-", "c ")): return 1.0
    if name.startswith(("d_", "d-", "d ")): return 3.0
    if name.startswith(("e_", "e-", "e ")): return 5.0
    for p in path.parents:
        m = _ratio_re.search(p.name)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return a / b if b else None
    return None

def parse_temp_from_path(path: Path) -> Optional[float]:
    m = _temp_re.search(path.name)
    if m: return float(m.group(1))
    for p in [path.parent, *path.parents]:
        m = _temp_re.search(p.name)
        if m: return float(m.group(1))
    return None

# ----------------------- Tip angle & baselines (unit-consistent) -----------------------
def integrate_tip_angle(df: pd.DataFrame) -> Dict[str, float]:
    """Integrate κ(s) to get θ_tip (deg); s is always in μm. Also returns Simpson/trapezoid consistency."""
    k_point = _find_col_like(df, ["point curvature"])
    sign_c  = _find_col_like(df, ["point curvature sign", "sign"])

    if k_point is not None:
        kappa = pd.to_numeric(df[k_point], errors="coerce").to_numpy()
        if sign_c is not None:
            sign = pd.to_numeric(df[sign_c], errors="coerce").to_numpy()
            nz = np.abs(sign) > 0
            if nz.any():
                last = -1
                for i in range(len(sign)):
                    if nz[i]: last = i
                    elif last >= 0: sign[i] = np.sign(sign[last])
                if last < 0:
                    sign[:] = 1.0
            else:
                sign = np.ones_like(kappa)
            kappa = kappa * np.sign(sign)
    else:
        k_avg = _find_col_like(df, ["average curvature", "curvature"])
        if k_avg is None: raise ValueError("No curvature column found.")
        kappa = pd.to_numeric(df[k_avg], errors="coerce").to_numpy()

    # s in μm (from XY if available, otherwise uniform spacing scaled by UM_PER_PX)
    x_um, y_um, _, _ = _get_xy_in_um(df)
    if x_um is not None and y_um is not None:
        ds = np.sqrt(np.diff(x_um)**2 + np.diff(y_um)**2)
        s  = np.concatenate([[0.0], np.cumsum(ds)])
    else:
        s = np.arange(len(kappa), dtype=float) * UM_PER_PX

    mask = np.isfinite(kappa) & np.isfinite(s)
    kappa = kappa[mask]; s = s[mask]
    if len(s) < 3: raise ValueError("Too few valid points to integrate.")

    theta_trap = _cumtrapz(kappa, s)
    theta_simp = _cumsimpson(kappa, s)  # NEW: Simpson cumulative
    tip_angle_deg_trap = float(np.degrees(theta_trap[-1]))
    tip_angle_deg_simp = float(np.degrees(theta_simp[-1]))
    disc_delta_deg = float(abs(tip_angle_deg_simp - tip_angle_deg_trap))

    L = float(s[-1] - s[0])
    mean_kappa = _trapz_safe(kappa, s) / max(L, 1e-9)
    max_abs_kappa = float(np.nanmax(np.abs(kappa)))
    return dict(tip_angle_deg=tip_angle_deg_trap,
                tip_angle_deg_simp=tip_angle_deg_simp,   # NEW
                disc_delta_deg=disc_delta_deg,           # NEW (for discretization uncertainty)
                L=L, mean_kappa=float(mean_kappa), max_abs_kappa=max_abs_kappa)

def _polyline_kappa_from_xy(x_um: np.ndarray, y_um: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute curvature along polyline sampled at equal arc-length spacing (units 1/μm)."""
    s, _ = _arc_length(x_um, y_um)
    if not np.all(np.diff(s) > 0):
        s = s + np.linspace(0, 1e-9, len(s))
    dx = np.gradient(x_um, s); dy = np.gradient(y_um, s)
    ddx = np.gradient(dx, s);  ddy = np.gradient(dy, s)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom[denom == 0] = np.finfo(float).eps
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa, s

def _baseline_chord_theta_deg(x_um: np.ndarray, y_um: np.ndarray) -> float:
    """Chord–Sagitta θ_tip estimate (constant κ circular arc) with sign from sagitta orientation."""
    x0, y0 = x_um[0], y_um[0]; x1, y1 = x_um[-1], y_um[-1]
    vx, vy = x1 - x0, y1 - y0
    c = float(np.hypot(vx, vy))
    if c <= 0: return 0.0
    cross = (x_um - x0) * vy - (y_um - y0) * vx
    dist = np.abs(cross) / c
    idx = int(np.argmax(dist)); h = float(np.sign(cross[idx]) * dist[idx])
    if np.isclose(h, 0.0): return 0.0
    R = (c * c) / (8.0 * h) + 0.5 * h
    arg = min(1.0, max(0.0, c / (2.0 * abs(R))))
    theta = 2.0 * np.arcsin(arg)
    return float(np.degrees(np.sign(h) * theta))

def _extract_xy_kappa(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (x_um, y_um, kappa_if_present)."""
    x_um, y_um, _, _ = _get_xy_in_um(df)
    k_point = _find_col_like(df, ["point curvature"])
    k_avg   = _find_col_like(df, ["average curvature", "curvature"])
    sign_c  = _find_col_like(df, ["point curvature sign", "sign"])
    kappa = None
    if k_point is not None:
        kappa = pd.to_numeric(df[k_point], errors="coerce").to_numpy()
        if sign_c is not None:
            sign = pd.to_numeric(df[sign_c], errors="coerce").to_numpy()
            nz = np.abs(sign) > 0
            if nz.any():
                last = -1
                for i in range(len(sign)):
                    if nz[i]: last = i
                    elif last >= 0: sign[i] = np.sign(sign[last])
                if last < 0: sign[:] = 1.0
            else:
                sign = np.ones_like(kappa)
            kappa = kappa * np.sign(sign)
    elif k_avg is not None:
        kappa = pd.to_numeric(df[k_avg], errors="coerce").to_numpy()
    return x_um, y_um, kappa

def _compute_baselines_from_file(path: Path) -> Tuple[float, float]:
    """Return (θ_chord_deg, θ_poly_deg). If XY is unavailable/too short, return NaN."""
    try:
        df = _read_table(path)
        x_um, y_um, _ = _extract_xy_kappa(df)[:3]
        if x_um is None or y_um is None or len(x_um) < 3:
            return float("nan"), float("nan")
        x_rs, y_rs, _ = _resample_equal_s(x_um, y_um, N=N_POINTS_BASELINE)
        th_chord = _baseline_chord_theta_deg(x_rs, y_rs)
        k_poly, s_poly = _polyline_kappa_from_xy(x_rs, y_rs)
        theta_poly = _cumtrapz(k_poly, s_poly)
        th_poly = float(np.degrees(theta_poly[-1]))
        return th_chord, th_poly
    except Exception as e:
        print(f"[WARN] Baseline failed for {path}: {e}")
        return float("nan"), float("nan")

# ----------------------- Dataset collection -----------------------
def collect_dataset(split_dir: Path, split_name: str) -> pd.DataFrame:
    rows, n_total, n_ok = [], 0, 0
    for path in list(split_dir.rglob("*.csv")) + list(split_dir.rglob("*.xlsx")):
        n_total += 1
        try:
            df = _read_table(path)
            feats = integrate_tip_angle(df)  # includes trap/simpson & disc_delta
            ratio = parse_ratio_from_path(path); temp = parse_temp_from_path(path)
            if ratio is None or temp is None:
                print(f"[WARN] Skip (cannot parse ratio/temp): {path}"); continue
            th_chord, th_poly = _compute_baselines_from_file(path)
            rows.append({
                "split": split_name,
                "specimen": "/".join(path.parts[-3:]),
                "ratio": float(ratio),
                "ratio_frac": float(ratio/(1.0+ratio)),
                "temp_C": float(temp),
                **feats,
                "theta_chord_deg": th_chord,
                "theta_poly_deg": th_poly,
                "file": str(path)
            }); n_ok += 1
        except Exception as e:
            print(f"[ERROR] Read failed: {path} -> {e}")
    df = pd.DataFrame(rows)
    print(f"[INFO] {split_name}: found {n_total} files, loaded {n_ok} usable samples.")
    return df

# ----------------------- Generic plotting helpers -----------------------
def _save_json(obj: dict, name: str) -> None:
    with open(OUT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _parity_residual(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> None:
    plt.figure(figsize=(5,5))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, '--', lw=1); plt.scatter(y_true, y_pred, alpha=0.9)
    plt.xlabel("True θ_tip (deg)"); plt.ylabel("Predicted θ_tip (deg)")
    plt.title(f"Parity plot ({tag})"); plt.tight_layout()
    plt.savefig(FIG_DIR / f"parity_{tag}.png", dpi=300); plt.close()
    resid = y_pred - y_true
    plt.figure(figsize=(5,4)); plt.axhline(0, ls="--", lw=1)
    plt.scatter(y_pred, resid, alpha=0.9)
    plt.xlabel("Predicted θ_tip (deg)"); plt.ylabel("Residual (pred - true)")
    plt.title(f"Residual vs Pred ({tag})"); plt.tight_layout()
    plt.savefig(FIG_DIR / f"residual_{tag}.png", dpi=300); plt.close()

def _learning_curve_plot(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    sizes, tr_scores, cv_scores = learning_curve(
        estimator, X, y, cv=min(N_FOLDS, max(2, len(y)//2)),
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    plt.figure(figsize=(6,4))
    plt.plot(sizes, -tr_scores.mean(1), 'o-', label="Train MAE")
    plt.plot(sizes, -cv_scores.mean(1), 'o-', label="CV MAE")
    plt.xlabel("Training Samples"); plt.ylabel("MAE (deg)")
    plt.title("Learning Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curve.png", dpi=300); plt.close()

def _importance(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, names: List[str]) -> None:
    r = permutation_importance(estimator, X, y, n_repeats=20, random_state=RANDOM_SEED, n_jobs=-1)
    imp, std = r.importances_mean, r.importances_std; idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(6,4)); plt.bar(range(len(names)), imp[idx], yerr=std[idx])
    plt.xticks(range(len(names)), [names[i] for i in idx])
    plt.ylabel("Permutation importance (ΔMAE)"); plt.title("Input importance")
    plt.tight_layout(); plt.savefig(FIG_DIR / "feature_importance.png", dpi=300); plt.close()

def _response_surface(model: Pipeline) -> None:
    Ts = np.arange(GRID_T_MIN, GRID_T_MAX + GRID_T_STEP, GRID_T_STEP)
    Rs = np.arange(GRID_R_MIN, GRID_R_MAX + GRID_R_STEP, GRID_R_STEP)
    TT, RR = np.meshgrid(Ts, Rs)
    df = pd.DataFrame({"ratio": RR.ravel(), "ratio_frac": (RR/(1.0+RR)).ravel(), "temp_C": TT.ravel()})
    Z = model.predict(df).reshape(RR.shape)
    plt.figure(figsize=(6.3,4.8)); cs = plt.contourf(TT, RR, Z, levels=24)
    plt.colorbar(cs, label="Predicted θ_tip (deg)")
    plt.xlabel("Temperature (°C)"); plt.ylabel("r = PNIPAM:PDMS")
    plt.title("Response surface of θ_tip"); plt.tight_layout()
    plt.savefig(FIG_DIR / "response_surface.png", dpi=300); plt.close()

def _robustness(model: Pipeline) -> None:
    Ts = np.arange(GRID_T_MIN, GRID_T_MAX + 1, 1)
    for r_nom in [1.0, 3.0, 5.0]:
        means, stds = [], []
        for T_nom in Ts:
            T = np.random.normal(T_nom, TOL_TEMP_STD, size=N_MONTE_CARLO)
            R = np.random.normal(r_nom, TOL_RATIO_STD, size=N_MONTE_CARLO)
            df = pd.DataFrame({"ratio": R, "ratio_frac": R/(1.0+R), "temp_C": T})
            y = model.predict(df); means.append(y.mean()); stds.append(y.std())
        means, stds = np.array(means), np.array(stds)
        plt.figure(figsize=(6,4))
        plt.plot(Ts, means, label=f"r={r_nom}")
        plt.fill_between(Ts, means-stds, means+stds, alpha=0.3)
        plt.xlabel("Temperature (°C)"); plt.ylabel("Predicted θ_tip (deg)")
        plt.title(f"Robustness (σ_T={TOL_TEMP_STD}°C, σ_r={TOL_RATIO_STD})")
        plt.legend(); plt.tight_layout()
        plt.savefig(FIG_DIR / f"robustness_r{int(r_nom)}.png", dpi=300); plt.close()

# ----------------------- Train & eval (θ_tip) -----------------------
def train_and_eval(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    target = "tip_angle_deg"; feats = ["ratio", "ratio_frac", "temp_C"]
    Xtr, ytr = df_tr[feats], df_tr[target]; Xte, yte = df_te[feats], df_te[target]
    cv = KFold(n_splits=min(N_FOLDS, max(2, len(df_tr)//2)), shuffle=True, random_state=RANDOM_SEED)
    candidates = []

    poly = Pipeline([("ct", ColumnTransformer([("num", "passthrough", feats)])),
                     ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                     ("sc", StandardScaler()), ("reg", LinearRegression())])
    candidates.append(("PolyLinear", poly, {}))

    rf = Pipeline([("ct", ColumnTransformer([("num", "passthrough", feats)])),
                   ("reg", RandomForestRegressor(random_state=RANDOM_SEED))])
    grid_rf = {"reg__n_estimators":[300,600], "reg__max_depth":[None,6,10], "reg__min_samples_leaf":[1,2,4]}
    candidates.append(("RandomForest", rf, grid_rf))

    gbr = Pipeline([("ct", ColumnTransformer([("num", "passthrough", feats)])),
                    ("reg", GradientBoostingRegressor(random_state=RANDOM_SEED))])
    grid_gbr = {"reg__n_estimators":[400,800], "reg__learning_rate":[0.05,0.1],
                "reg__max_depth":[2,3,4], "reg__subsample":[0.8,1.0]}
    candidates.append(("GradBoost", gbr, grid_gbr))

    best_name = None; best_model = None; best_metrics = None
    all_metrics: Dict[str, dict] = {}; leaderboard: Dict[str, dict] = {}

    for name, pipe, grid in candidates:
        gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr); y_pred = gs.best_estimator_.predict(Xte)
        m = {"type":"ML","cv_best_params":gs.best_params_, "cv_best_mae":-float(gs.best_score_),
             "test_r2": float(r2_score(yte, y_pred)),
             "test_rmse": float(math.sqrt(mean_squared_error(yte, y_pred))),
             "test_mae": float(mean_absolute_error(yte, y_pred)),
             "n_train": int(len(df_tr)), "n_test": int(len(df_te))}
        all_metrics[name] = m; leaderboard[name] = {"test_mae": m["test_mae"], "kind": "ML"}
        if best_metrics is None or m["test_mae"] < best_metrics["test_mae"]:
            best_metrics, best_model, best_name = m, gs.best_estimator_, name

    _save_json(all_metrics, "metrics.json"); joblib.dump(best_model, OUT_DIR / "best_model.joblib")
    print(f"[INFO] Best ML model: {best_name}"); print(json.dumps(best_metrics, indent=2, ensure_ascii=False))

    _parity_residual(ytr.to_numpy(), best_model.predict(Xtr), "train")
    _parity_residual(yte.to_numpy(), best_model.predict(Xte), "test")
    Xall = pd.concat([Xtr, Xte], ignore_index=True); yall = pd.concat([ytr, yte], ignore_index=True)
    _parity_residual(yall.to_numpy(), best_model.predict(Xall), "all")
    _learning_curve_plot(best_model, Xall, yall)
    _importance(best_model, Xtr, ytr, feats); _response_surface(best_model); _robustness(best_model)

    # Evaluate both baselines on the same test set
    def _eval_baseline(colname: str, tag: str) -> Optional[dict]:
        mask = np.isfinite(df_te[colname].to_numpy()) & np.isfinite(yte.to_numpy())
        if mask.sum() < 1:
            print(f"[WARN] No valid samples for baseline {tag}."); return None
        y_true = yte.to_numpy()[mask]; y_pred = df_te[colname].to_numpy()[mask]
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        _parity_residual(y_true, y_pred, f"baseline_{tag}")
        return {"type":"Baseline","test_mae":mae,"test_rmse":rmse,"test_r2":r2,"n_eval":int(mask.sum())}

    b_ch = _eval_baseline("theta_chord_deg", "chord")
    b_po = _eval_baseline("theta_poly_deg",  "poly")
    if b_ch is not None: all_metrics["BaselineChord"]=b_ch; leaderboard["BaselineChord"]={"test_mae": b_ch["test_mae"], "kind":"Baseline"}
    if b_po is not None: all_metrics["BaselinePoly"]=b_po; leaderboard["BaselinePoly"]={"test_mae": b_po["test_mae"], "kind":"Baseline"}
    _save_json(all_metrics, "metrics.json"); _save_json(leaderboard, "leaderboard.json")

    # Best overall by Test MAE (info only)
    best_overall_name, best_overall_mae, best_overall_kind = None, float("inf"), None
    for k,v in leaderboard.items():
        if np.isfinite(v["test_mae"]) and v["test_mae"] < best_overall_mae:
            best_overall_name, best_overall_mae, best_overall_kind = k, v["test_mae"], v["kind"]
    best_overall = {"name":best_overall_name, "test_mae":best_overall_mae, "kind":best_overall_kind,
                    "inference_supported": bool(best_overall_kind=="ML"),
                    "note":"If kind=='Baseline', it cannot do (r,T)->θ_tip; best_model.joblib remains for inference."}
    _save_json(best_overall, "best_overall.json")
    print("[INFO] Leaderboard (by Test MAE):"); print(json.dumps(leaderboard, indent=2, ensure_ascii=False))
    print("[INFO] Best overall by Test MAE:"); print(json.dumps(best_overall, indent=2, ensure_ascii=False))
    return best_model, all_metrics

# ----------------------- Curvature-shape models -----------------------
def _signed_kappa_and_s(df: pd.DataFrame):
    """Return signed κ(s) (1/μm) and s (μm)."""
    k_point = _find_col_like(df, ["point curvature"])
    sign_c  = _find_col_like(df, ["point curvature sign", "sign"])
    if k_point is not None:
        kappa = pd.to_numeric(df[k_point], errors="coerce").to_numpy()
        if sign_c is not None:
            sign = pd.to_numeric(df[sign_c], errors="coerce").to_numpy()
            nz = np.abs(sign) > 0
            if nz.any():
                last=-1
                for i in range(len(sign)):
                    if nz[i]: last=i
                    elif last>=0: sign[i]=np.sign(sign[last])
                if last<0: sign[:] = 1.0
            else:
                sign = np.ones_like(kappa)
            kappa = kappa * np.sign(sign)
    else:
        k_avg = _find_col_like(df, ["average curvature", "curvature"])
        if k_avg is None: raise ValueError("No curvature column found for shape model.")
        kappa = pd.to_numeric(df[k_avg], errors="coerce").to_numpy()

    x_um, y_um, _, _ = _get_xy_in_um(df)
    if x_um is not None and y_um is not None:
        ds = np.sqrt(np.diff(x_um)**2 + np.diff(y_um)**2)
        s  = np.concatenate([[0.0], np.cumsum(ds)])
    else:
        s = np.arange(len(kappa), dtype=float) * UM_PER_PX
    mask = np.isfinite(kappa) & np.isfinite(s)
    return kappa[mask], s[mask]

def _resample_kappa_from_file(filepath: str, n_points: int = N_POINTS_CURV):
    df = _read_table(Path(filepath))
    kappa, s = _signed_kappa_and_s(df)
    if len(s) < 3: raise ValueError("Too few points for curvature resampling.")
    L = float(s[-1] - s[0]) if s[-1] > s[0] else float(len(s)-1) * UM_PER_PX
    s_norm_src = (s - s[0]) / max(L, 1e-9)
    s_norm_tar = np.linspace(0.0, 1.0, n_points)
    kappa_rs = np.interp(s_norm_tar, s_norm_src, kappa)
    return s_norm_tar, kappa_rs, L

def train_curvature_shape_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """PCA on curvature shapes and regression from (ratio,temp) to PCA scores and absolute length."""
    feature_cols = ["ratio", "ratio_frac", "temp_C"]
    Xtr = df_train[feature_cols].to_numpy(dtype=float)
    files_tr = df_train["file"].tolist(); Ktr, Ltr = [], []
    for fp in files_tr:
        try:
            _, k_rs, L = _resample_kappa_from_file(fp, N_POINTS_CURV)
            Ktr.append(k_rs); Ltr.append(L)
        except Exception as e:
            print(f"[WARN] Curvature resampling failed (train): {fp} -> {e}")
    if len(Ktr) < 3:
        print("[WARN] Too few curves for shape model; skipping."); return
    Ktr = np.vstack(Ktr); Ltr = np.asarray(Ltr, float)

    pca_tmp = PCA(n_components=min(10, Ktr.shape[0], Ktr.shape[1]), svd_solver="full", random_state=RANDOM_SEED)
    pca_tmp.fit(Ktr); evr_cum = np.cumsum(pca_tmp.explained_variance_ratio_)
    n_comp = int(np.searchsorted(evr_cum, 0.99) + 1); n_comp = max(1, min(n_comp, 10))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=RANDOM_SEED).fit(Ktr)
    Ztr = pca.transform(Ktr)

    base_reg = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=600, max_depth=3, learning_rate=0.06, subsample=0.9)
    score_reg = MultiOutputRegressor(base_reg).fit(Xtr, Ztr)
    L_reg = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=600, max_depth=3, learning_rate=0.06, subsample=0.9)
    L_reg.fit(Xtr, Ltr)

    joblib.dump(pca, OUT_DIR / "curvature_pca.joblib")
    joblib.dump(score_reg, OUT_DIR / "curvature_reg.joblib")
    joblib.dump(L_reg, OUT_DIR / "length_model.joblib")
    meta = {"n_points": int(N_POINTS_CURV), "n_components": int(n_comp), "explained_variance_ratio_cum": float(evr_cum[n_comp-1])}
    _save_json(meta, "curvature_meta.json")
    print(f"[INFO] Saved curvature-shape models to {OUT_DIR} (n_points={N_POINTS_CURV}, n_components={n_comp})")

# ----------------------- θ(s) prediction (for auto-generation only) -----------------------
def _find_artifact(filename: str) -> Optional[Path]:
    for base in [OUT_DIR, ROOT, ROOT/"models", ROOT/"artifacts", ROOT/"trained", ROOT/"outputs", ROOT/"checkpoints"]:
        p = base / filename
        if p.exists(): return p
    return None

def _load_curve_models() -> Dict[str, object]:
    pca_p = _find_artifact("curvature_pca.joblib")
    reg_p = _find_artifact("curvature_reg.joblib")
    len_p = _find_artifact("length_model.joblib")
    missing = []
    if pca_p is None: missing.append("curvature_pca.joblib")
    if reg_p is None: missing.append("curvature_reg.joblib")
    if missing:
        raise FileNotFoundError("Missing required curvature-shape artifact(s): " + ", ".join(missing))
    return {"pca": joblib.load(pca_p), "reg": joblib.load(reg_p), "len": joblib.load(len_p) if len_p is not None else None}

def _predict_kappa_and_s(models: Dict[str, object], ratio: float, temp_C: float) -> Tuple[np.ndarray, np.ndarray, float]:
    pca, reg, len_reg = models["pca"], models["reg"], models["len"]
    r = float(ratio); X = np.array([[r, r/(1.0+r), float(temp_C)]], dtype=float)
    y_pred = np.asarray(reg.predict(X)).reshape(1, -1)
    n_components = getattr(pca, "n_components_", None) or pca.components_.shape[0]
    kappa = pca.inverse_transform(y_pred)[0] if y_pred.shape[1]==n_components else y_pred[0]
    N = kappa.shape[0]; s_norm = np.linspace(0.0, 1.0, N)
    if len_reg is not None:
        L_abs = float(len_reg.predict(X).ravel()[0])
        if not np.isfinite(L_abs) or L_abs <= 0: L_abs = 1.0
    else:
        L_abs = 1.0
    return kappa, s_norm, L_abs

def _kappa_to_theta_deg(kappa: np.ndarray, s_abs: np.ndarray, theta0_deg: float = 0.0) -> np.ndarray:
    theta_rad = _cumtrapz(kappa, s_abs)
    return np.degrees(theta_rad) + float(theta0_deg)

def run_predict_curve(ratio: float, temps: List[float], theta0_deg: float = 0.0,
                      outdir_override: Optional[str] = None, no_individual_plots: bool = False) -> None:
    """Batch-generate θ(s) for all temps at the given ratio (used post-training for your dataset)."""
    theta_root = Path(outdir_override).resolve() if outdir_override else (OUT_DIR / "theta_curves")
    (theta_root/"csv").mkdir(parents=True, exist_ok=True)
    (theta_root/"figs").mkdir(parents=True, exist_ok=True)
    try:
        models = _load_curve_models()
    except FileNotFoundError as e:
        print(f"[WARN] {e}"); print("[WARN] Skip θ(s) generation. Train curvature-shape models first."); return
    overlay_curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for T in temps:
        kappa, s_norm, L_abs = _predict_kappa_and_s(models, float(ratio), float(T))
        s_abs = s_norm * L_abs
        theta_deg = _kappa_to_theta_deg(kappa, s_abs, theta0_deg=theta0_deg)
        tag = f"r{ratio:g}_T{T:g}"
        pd.DataFrame({"s_norm": s_norm, "s_abs": s_abs, "kappa_pred": kappa, "theta_pred_deg": theta_deg})\
            .to_csv(theta_root/"csv"/f"theta_curve_{tag}.csv", index=False, encoding="utf-8-sig")
        if not no_individual_plots:
            fig = plt.figure(figsize=(6,4.2), dpi=150)
            plt.plot(s_abs, theta_deg, lw=2)
            plt.xlabel("Arc length s (units)"); plt.ylabel("Tangent angle θ(s) [deg]")
            plt.title(f"θ(s) — r={ratio:g}:1, T={T:g}°C"); plt.grid(True, alpha=0.3); fig.tight_layout()
            plt.savefig(theta_root/"figs"/f"theta_curve_{tag}.png"); plt.close(fig)
        overlay_curves.append((s_abs, theta_deg, f"r={ratio:g}:1, T={T:g}°C"))
    fig = plt.figure(figsize=(7,4.5), dpi=150)
    for s_abs, theta_deg, lbl in overlay_curves: plt.plot(s_abs, theta_deg, lw=2, label=lbl)
    plt.xlabel("Arc length s (units)"); plt.ylabel("Tangent angle θ(s) [deg]")
    plt.title(f"θ(s) overlay — ratio {ratio:g}:1"); plt.grid(True, alpha=0.3)
    if len(overlay_curves) > 1: plt.legend()
    plt.tight_layout(); plt.savefig(theta_root/"figs"/f"theta_overlay_r{ratio:g}.png"); plt.close(fig)
    print(f"[OK] Saved θ(s) CSVs -> {theta_root/'csv'}"); print(f"[OK] Saved θ(s) figures -> {theta_root/'figs'}")

# ----------------------- Auto-generate θ(s) after training -----------------------
def auto_generate_theta_curves(df_all: pd.DataFrame, theta0_deg: float = 0.0) -> None:
    """Produce θ(s) for all (ratio,temp) combinations seen in the dataset using the trained shape models."""
    pca_ok = _find_artifact("curvature_pca.joblib") is not None
    reg_ok = _find_artifact("curvature_reg.joblib") is not None
    if not (pca_ok and reg_ok):
        print("[WARN] Curvature-shape artifacts not found; skip θ(s) auto-generation."); return
    by_ratio = (df_all[["ratio","temp_C"]].dropna()
                .sort_values(["ratio","temp_C"]).groupby("ratio")["temp_C"]
                .apply(lambda s: sorted(set(float(x) for x in s.tolist()))).to_dict())
    if not by_ratio:
        print("[WARN] No (ratio,temp) combinations found for θ(s) curves."); return
    print("[INFO] Auto-generating θ(s) curves for dataset combos:")
    for r, temps in by_ratio.items():
        print(f"  - r={r:g}: temps={temps}")
        run_predict_curve(ratio=float(r), temps=temps, theta0_deg=theta0_deg,
                          outdir_override=None, no_individual_plots=False)

# ----------------------- File-level θ(s) overlays for debugging -----------------------
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _theta_curve_from_reference_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """θ_ref(s) from integrating the provided κ(s). Returns (s_abs, θ_deg, label)."""
    df = _read_table(Path(file_path))
    kappa, s = _signed_kappa_and_s(df)
    if len(s) < 3: raise ValueError("Too few points to form reference θ(s).")
    theta_ref = _cumtrapz(kappa, s); s_abs = s - s[0]
    return s_abs, np.degrees(theta_ref), "Reference κ(s) integral"

def _theta_curve_from_polyline_xy(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """θ_polyline(s) from XY by estimating κ(s) on a polyline and integrating."""
    df = _read_table(Path(file_path))
    x_um, y_um, _ = _extract_xy_kappa(df)[:3]
    if x_um is None or y_um is None or len(x_um) < 3: return None
    x_rs, y_rs, s_rs = _resample_equal_s(x_um, y_um, N=N_POINTS_BASELINE)
    k_poly, s_poly = _polyline_kappa_from_xy(x_rs, y_rs)
    theta_poly = _cumtrapz(k_poly, s_poly)
    return s_poly - s_poly[0], np.degrees(theta_poly), "Polyline-κ baseline"

def _theta_curve_from_chord(file_path: str, N: int = N_POINTS_CURV) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """θ_chord(s) assuming constant curvature equal to the chord–sagitta θ_tip, linearly distributed over s."""
    df = _read_table(Path(file_path))
    x_um, y_um, _ = _extract_xy_kappa(df)[:3]
    if x_um is None or y_um is None or len(x_um) < 3: return None
    x_rs, y_rs, s_rs = _resample_equal_s(x_um, y_um, N=N_POINTS_BASELINE)
    L = float(s_rs[-1] - s_rs[0])
    theta_tip = _baseline_chord_theta_deg(x_rs, y_rs)
    s_abs = np.linspace(0.0, L, N); theta_chord = np.linspace(0.0, theta_tip, N)
    return s_abs, theta_chord, "Chord–Sagitta baseline (const κ)"

def make_baseline_theta_overlay_for_file(file_path: str, include_predicted: bool = False) -> None:
    """Save per-file overlay for debugging (not meant for the paper)."""
    comp_dir = FIG_DIR / "baseline_theta"; _ensure_dir(comp_dir)
    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
    s_ref, th_ref, lbl_ref = _theta_curve_from_reference_file(file_path); curves.append((lbl_ref, s_ref, th_ref))
    poly = _theta_curve_from_polyline_xy(file_path);
    if poly is not None: s_poly, th_poly, lbl_poly = poly; curves.append((lbl_poly, s_poly, th_poly))
    chord = _theta_curve_from_chord(file_path)
    if chord is not None: s_ch, th_ch, lbl_ch = chord; curves.append((lbl_ch, s_ch, th_ch))

    # Optionally add model-predicted θ(s) if you want to inspect it alongside baselines.
    if include_predicted:
        try:
            ratio = parse_ratio_from_path(Path(file_path)); temp = parse_temp_from_path(Path(file_path))
            if (ratio is not None) and (temp is not None):
                models = _load_curve_models()
                k_pred, s_norm, L_abs = _predict_kappa_and_s(models, float(ratio), float(temp))
                s_abs_pred = s_norm * L_abs; theta_pred = _cumtrapz(k_pred, s_abs_pred)
                curves.append((f"Predicted θ(s) r={ratio:g},T={temp:g}°C", s_abs_pred, np.degrees(theta_pred)))
        except Exception as e:
            print(f"[WARN] Predicted θ(s) not drawn for {file_path}: {e}")

    if len(curves) < 2:
        print(f"[WARN] Not enough curves to compare for {file_path}"); return

    fig = plt.figure(figsize=(7,4.5), dpi=150)
    for lbl, sx, th in curves: plt.plot(sx, th, lw=2, label=lbl)
    plt.xlabel("Arc length s (absolute units)"); plt.ylabel("θ(s) [deg]")
    sp = Path(file_path); title_stub = f"{sp.parent.name}/{sp.name}"
    plt.title(f"θ(s) comparison — {title_stub}"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(comp_dir / (sp.stem + "_theta_compare.png")); plt.close(fig)

    # Export to CSV for reproducibility
    try:
        s_max = max(sx.max() for _, sx, _ in curves if len(sx) > 0)
        s_common = np.linspace(0.0, float(s_max), N_POINTS_CURV)
        out = {"s_abs": s_common}
        for lbl, sx, th in curves:
            out[lbl] = np.interp(s_common, sx, th)
        pd.DataFrame(out).to_csv(comp_dir / (sp.stem + "_theta_compare.csv"), index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[WARN] CSV export failed for {file_path}: {e}")

def generate_baseline_theta_overlays(df_all: pd.DataFrame, max_plots: Optional[int] = None, include_predicted: bool = False) -> None:
    files = [f for f in df_all["file"].dropna().tolist() if Path(f).exists()]
    if max_plots is not None: files = files[:max_plots]
    print(f"[INFO] Generating θ(s) overlays for {len(files)} files...")
    for fp in files: make_baseline_theta_overlay_for_file(fp, include_predicted=include_predicted)

# ----------------------- Paper-friendly SUMMARIES -----------------------
def _bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = RANDOM_SEED) -> Tuple[float,float]:
    """Return (low, high) percentile bootstrap CI for the mean of 'values'."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = np.mean(values[idx], axis=1)
    low = float(np.percentile(boots, 100 * (alpha / 2.0)))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2.0)))
    return low, high

def plot_baseline_error_summaries(df_te: pd.DataFrame, best_model, include_ml_bar: bool = True) -> None:
    """(Kept from previous version) — boxplot + Bland–Altman for test set."""
    feats = ["ratio","ratio_frac","temp_C"]
    if df_te.empty:
        print("[WARN] Empty test set; skip error summaries."); return
    y_true = df_te["tip_angle_deg"].to_numpy()
    y_ml   = best_model.predict(df_te[feats])
    y_ch   = df_te["theta_chord_deg"].to_numpy()
    y_po   = df_te["theta_poly_deg"].to_numpy()

    def _mask_valid(y): return np.isfinite(y_true) & np.isfinite(y)
    m_ml, m_ch, m_po = _mask_valid(y_ml), _mask_valid(y_ch), _mask_valid(y_po)

    # Boxplot of |error|
    fig = plt.figure(figsize=(6,4), dpi=150)
    labels, data = [], []
    if include_ml_bar and m_ml.sum()>0: labels.append("ML");       data.append(np.abs(y_ml[m_ml]-y_true[m_ml]))
    if m_ch.sum()>0:                     labels.append("Chord");    data.append(np.abs(y_ch[m_ch]-y_true[m_ch]))
    if m_po.sum()>0:                     labels.append("Polyline"); data.append(np.abs(y_po[m_po]-y_true[m_po]))
    if len(data) >= 2:
        try:
            plt.boxplot(data, tick_labels=labels, showfliers=False) # Matplotlib ≥3.9
        except TypeError:
            plt.boxplot(data, labels=labels, showfliers=False)       # Older versions
        plt.ylabel("|Error| (deg)"); plt.title("Absolute error comparison (TEST)")
        plt.tight_layout(); plt.savefig(FIG_DIR / "error_boxplot_test.png"); plt.close(fig)
    else:
        plt.close(fig); print("[WARN] Not enough methods for error boxplot.")

    # Bland–Altman for each baseline
    def bland(y_pred, name, mask):
        mean = (y_pred[mask] + y_true[mask]) / 2.0; diff = y_pred[mask] - y_true[mask]
        md = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if diff.size>1 else 0.0
        fig = plt.figure(figsize=(6,4), dpi=150)
        plt.scatter(mean, diff, alpha=0.85)
        plt.axhline(0, ls="--", lw=1, label="zero")
        plt.axhline(md, ls="-.", lw=1, label=f"mean={md:.2f}")
        plt.axhline(md+1.96*sd, ls=":", lw=1, label=f"+1.96σ={md+1.96*sd:.2f}")
        plt.axhline(md-1.96*sd, ls=":", lw=1, label=f"-1.96σ={md-1.96*sd:.2f}")
        plt.xlabel("Mean of prediction and truth (deg)"); plt.ylabel("Pred - Truth (deg)")
        plt.title(f"Bland–Altman: {name} vs truth"); plt.legend()
        plt.tight_layout(); plt.savefig(FIG_DIR / f"bland_altman_{name.lower()}.png"); plt.close(fig)
    if m_ch.sum()>1: bland(y_ch, "Chord", m_ch)
    if m_po.sum()>1: bland(y_po, "Polyline", m_po)

def make_paper_baseline_summary(df_te: pd.DataFrame, best_model, df_all: pd.DataFrame, include_ml_bar: bool = True) -> None:
    """
    Create three paper-friendly figures + CSV/JSON:
      (1) baseline_mae_ci.png   — MAE bars with 95% bootstrap CIs (same TEST set).
      (2) baseline_parity_combined.png — truth vs prediction for both baselines on one axis.
      (3) baseline_theta_delta_ci.png  — mean ± 95% band of [θ_baseline(s) − θ_ref(s)] over s∈[0,1].
    """
    feats = ["ratio","ratio_frac","temp_C"]
    if df_te.empty:
        print("[WARN] Empty test set; skip paper summary."); return

    # ----- (1) MAE with 95% bootstrap CI -----
    y_true = df_te["tip_angle_deg"].to_numpy()
    y_ch   = df_te["theta_chord_deg"].to_numpy()
    y_po   = df_te["theta_poly_deg"].to_numpy()
    def _valid_err(y):
        m = np.isfinite(y_true) & np.isfinite(y)
        return np.abs(y[m] - y_true[m])
    err_ch = _valid_err(y_ch)
    err_po = _valid_err(y_po)
    bars = []
    ci_low = []
    ci_high = []
    labels = []
    if err_ch.size>0:
        labels.append("Chord")
        bars.append(float(np.mean(err_ch)))
        lo, hi = _bootstrap_ci(err_ch); ci_low.append(bars[-1]-lo); ci_high.append(hi-bars[-1])
    if err_po.size>0:
        labels.append("Polyline")
        bars.append(float(np.mean(err_po)))
        lo, hi = _bootstrap_ci(err_po); ci_low.append(bars[-1]-lo); ci_high.append(hi-bars[-1])
    if include_ml_bar:
        y_ml = best_model.predict(df_te[feats])
        err_ml = _valid_err(y_ml)
        if err_ml.size>0:
            labels.append("ML")
            bars.append(float(np.mean(err_ml)))
            lo, hi = _bootstrap_ci(err_ml); ci_low.append(bars[-1]-lo); ci_high.append(hi-bars[-1])
    # Plot
    if len(bars)>=2:
        x = np.arange(len(bars))
        fig = plt.figure(figsize=(5.5,4.2), dpi=150)
        plt.bar(x, bars, yerr=[ci_low, ci_high], capsize=4)
        plt.xticks(x, labels); plt.ylabel("MAE (deg)")
        plt.title("Baseline comparison (MAE ± 95% CI, TEST)")
        plt.tight_layout(); plt.savefig(PAPER_FIG_DIR / "baseline_mae_ci.png"); plt.close(fig)
        # Export summary JSON for reproducibility
        summary = {labels[i]: {"mae": bars[i], "ci_low": bars[i]-ci_low[i], "ci_high": bars[i]+ci_high[i]} for i in range(len(labels))}
        with open(PAPER_CSV_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        print("[WARN] Not enough methods to draw MAE CI bar chart.")

    # ----- (2) Combined parity for both baselines -----
    mask_ch = np.isfinite(y_true) & np.isfinite(y_ch)
    mask_po = np.isfinite(y_true) & np.isfinite(y_po)
    if mask_ch.sum()>0 and mask_po.sum()>0:
        fig = plt.figure(figsize=(5.8,5.3), dpi=150)
        t_all = np.r_[y_true[mask_ch], y_true[mask_po]]
        p_all = np.r_[y_ch[mask_ch],   y_po[mask_po]]
        lims = [min(t_all.min(), p_all.min()), max(t_all.max(), p_all.max())]
        plt.plot(lims, lims, '--', lw=1, label="y=x")
        plt.scatter(y_true[mask_ch], y_ch[mask_ch], alpha=0.85, marker='o', label="Chord")
        plt.scatter(y_true[mask_po], y_po[mask_po], alpha=0.85, marker='s', label="Polyline")
        plt.xlabel("True θ_tip (deg)"); plt.ylabel("Predicted θ_tip (deg)")
        plt.title("Parity — two baselines on one axis (TEST)")
        plt.legend(); plt.tight_layout()
        plt.savefig(PAPER_FIG_DIR / "baseline_parity_combined.png"); plt.close(fig)
    else:
        print("[WARN] Not enough valid points for combined parity.")

    # ----- (3) Aggregated θ(s) deviation band: baseline − reference -----
    def _theta_curves_snorm(file_path: str) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray]]:
        # Return s_norm, theta_poly_deg, theta_chord_deg (both aligned to θ_ref s-grid).
        try:
            s_ref, th_ref, _ = _theta_curve_from_reference_file(file_path)
        except Exception:
            return None
        sn_ref = s_ref / max(float(s_ref[-1]) if len(s_ref)>0 else 1.0, 1e-9)
        # Polyline baseline
        poly = _theta_curve_from_polyline_xy(file_path)
        th_poly = None
        if poly is not None:
            s_p, th_p, _ = poly
            sn_p = s_p / max(float(s_p[-1]) if len(s_p)>0 else 1.0, 1e-9)
            th_poly = np.interp(sn_ref, sn_p, th_p)
        # Chord baseline (linear ramp up to its θ_tip)
        chord = _theta_curve_from_chord(file_path, N=len(sn_ref))
        th_ch = None
        if chord is not None:
            s_c, th_c, _ = chord
            sn_c = s_c / max(float(s_c[-1]) if len(s_c)>0 else 1.0, 1e-9)
            th_ch = np.interp(sn_ref, sn_c, th_c)
        return sn_ref, (th_poly - th_ref if th_poly is not None else None), (th_ch - th_ref if th_ch is not None else None)

    files = [f for f in df_all["file"].dropna().tolist() if Path(f).exists()]
    if len(files)==0:
        print("[WARN] No files for θ(s) delta aggregation."); return
    # Build matrices of delta-theta over common s-grid
    s_common = np.linspace(0.0, 1.0, N_POINTS_CURV)
    deltas_poly, deltas_ch = [], []
    for fp in files:
        out = _theta_curves_snorm(fp)
        if out is None: continue
        sn, d_poly, d_ch = out
        if d_poly is not None:
            deltas_poly.append(np.interp(s_common, sn, d_poly))
        if d_ch is not None:
            deltas_ch.append(np.interp(s_common, sn, d_ch))
    deltas_poly = np.array(deltas_poly) if len(deltas_poly)>0 else None
    deltas_ch   = np.array(deltas_ch)   if len(deltas_ch)>0 else None

    if deltas_poly is None and deltas_ch is None:
        print("[WARN] No valid θ(s) deltas for aggregation."); return

    fig = plt.figure(figsize=(7.2,4.6), dpi=150)
    if deltas_poly is not None and deltas_poly.shape[0] >= 1:
        m = np.nanmean(deltas_poly, axis=0)
        lo = np.nanpercentile(deltas_poly, 2.5, axis=0)
        hi = np.nanpercentile(deltas_poly, 97.5, axis=0)
        plt.plot(s_common, m, lw=2, label="Polyline − Ref")
        plt.fill_between(s_common, lo, hi, alpha=0.25)
        # export
        pd.DataFrame({"s_norm": s_common, "mean": m, "p2.5": lo, "p97.5": hi}).to_csv(PAPER_CSV_DIR/"theta_delta_polyline.csv", index=False, encoding="utf-8-sig")
    if deltas_ch is not None and deltas_ch.shape[0] >= 1:
        m = np.nanmean(deltas_ch, axis=0)
        lo = np.nanpercentile(deltas_ch, 2.5, axis=0)
        hi = np.nanpercentile(deltas_ch, 97.5, axis=0)
        plt.plot(s_common, m, lw=2, label="Chord − Ref")
        plt.fill_between(s_common, lo, hi, alpha=0.25)
        # export
        pd.DataFrame({"s_norm": s_common, "mean": m, "p2.5": lo, "p97.5": hi}).to_csv(PAPER_CSV_DIR/"theta_delta_chord.csv", index=False, encoding="utf-8-sig")
    plt.axhline(0, lw=1, ls="--")
    plt.xlabel("Normalized arc length s (0→1)")
    plt.ylabel("Δθ(s) [deg] (baseline − reference)")
    plt.title("Baseline bias along s (mean ± 95% band)")
    plt.tight_layout(); plt.savefig(PAPER_FIG_DIR / "baseline_theta_delta_ci.png"); plt.close(fig)

# ----------------------- NEW: Paired permutation tests & bootstrap metrics -----------------------
def _paired_permutation_pvalue(a_vals: np.ndarray, b_vals: np.ndarray, n_perm: int = N_PERM, seed: int = RANDOM_SEED) -> float:
    """
    Two-sided paired permutation (sign-flip) test on paired samples.
    Returns p-value for mean(a - b) under H0: mean diff == 0.
    """
    a_vals = np.asarray(a_vals, float); b_vals = np.asarray(b_vals, float)
    mask = np.isfinite(a_vals) & np.isfinite(b_vals)
    d = a_vals[mask] - b_vals[mask]
    if d.size < 2: return float("nan")
    obs = abs(np.mean(d))
    rng = np.random.default_rng(seed)
    # sign flips
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null_means = np.mean(signs * d, axis=1)
    p = float((np.sum(np.abs(null_means) >= obs) + 1) / (n_perm + 1))
    return p

def _bootstrap_metric_cis(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA, seed: int = RANDOM_SEED) -> Dict[str, Dict[str, float]]:
    """Bootstrap CIs for MAE / RMSE / R2."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size < 2:
        return {"MAE":{"mean":float("nan"),"lo":float("nan"),"hi":float("nan")},
                "RMSE":{"mean":float("nan"),"lo":float("nan"),"hi":float("nan")},
                "R2":{"mean":float("nan"),"lo":float("nan"),"hi":float("nan")}}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, yt.size, size=(n_boot, yt.size))
    mae = np.mean(np.abs(yp - yt))
    rmse = math.sqrt(np.mean((yp - yt)**2))
    r2 = r2_score(yt, yp)
    maes = np.mean(np.abs(yp[idx] - yt[idx]), axis=1)
    rmses = np.sqrt(np.mean((yp[idx] - yt[idx])**2, axis=1))
    # R² bootstrap by resampling pairs
    r2s = np.array([r2_score(yt[i], yp[i]) for i in idx])
    def CI(arr):
        lo = float(np.percentile(arr, 100 * (alpha/2)))
        hi = float(np.percentile(arr, 100 * (1 - alpha/2)))
        return lo, hi
    lo_mae, hi_mae = CI(maes); lo_rmse, hi_rmse = CI(rmses); lo_r2, hi_r2 = CI(r2s)
    return {"MAE":{"mean":float(mae),"lo":lo_mae,"hi":hi_mae},
            "RMSE":{"mean":float(rmse),"lo":lo_rmse,"hi":hi_rmse},
            "R2":{"mean":float(r2),"lo":lo_r2,"hi":hi_r2}}

def compute_tests_and_bootstrap(df_te: pd.DataFrame, best_model) -> None:
    """Compute paired permutation p-values and bootstrap CIs; save under outputs/stats/"""
    if df_te.empty:
        print("[WARN] Empty test set; skip stats tests."); return
    feats = ["ratio","ratio_frac","temp_C"]
    y_true = df_te["tip_angle_deg"].to_numpy()
    y_ml   = best_model.predict(df_te[feats])
    y_ch   = df_te["theta_chord_deg"].to_numpy()
    y_po   = df_te["theta_poly_deg"].to_numpy()

    # Bootstrap CIs for all three
    out_ci = {
        "ML": _bootstrap_metric_cis(y_true, y_ml),
        "Chord": _bootstrap_metric_cis(y_true, y_ch),
        "Polyline": _bootstrap_metric_cis(y_true, y_po)
    }
    with open(STATS_DIR / "bootstrap_metrics_ci.json", "w", encoding="utf-8") as f:
        json.dump(out_ci, f, indent=2, ensure_ascii=False)

    # Paired permutation on per-sample |error| (MAE) and squared error (RMSE surrogate)
    def errs(y_pred):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        return np.abs(y_pred[mask]-y_true[mask]), (y_pred[mask]-y_true[mask])**2

    abs_ml, sq_ml = errs(y_ml)
    abs_ch, sq_ch = errs(y_ch)
    abs_po, sq_po = errs(y_po)

    pvals = {
        "MAE": {
            "ML_vs_Chord": _paired_permutation_pvalue(abs_ml, abs_ch),
            "ML_vs_Polyline": _paired_permutation_pvalue(abs_ml, abs_po),
            "Polyline_vs_Chord": _paired_permutation_pvalue(abs_po, abs_ch)
        },
        "RMSE": {
            "ML_vs_Chord": _paired_permutation_pvalue(sq_ml, sq_ch),
            "ML_vs_Polyline": _paired_permutation_pvalue(sq_ml, sq_po),
            "Polyline_vs_Chord": _paired_permutation_pvalue(sq_po, sq_ch)
        }
    }
    with open(STATS_DIR / "permutation_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(pvals, f, indent=2, ensure_ascii=False)
    print("[OK] Stats tests ->", STATS_DIR)

# ----------------------- NEW: GUM-style uncertainty (MC propagation) -----------------------
def _mc_quantization_theta(kappa: np.ndarray, s: np.ndarray, df: pd.DataFrame, n_mc: int = N_MC_UNCERT) -> float:
    """
    Type A: digitization/quantization.
    If XY available, add U(-0.5*scale, +0.5*scale) to every XY point, recompute s and integrate κ(s) by trapezoid.
    Else, jitter ds with U(-0.5*UM_PER_PX, +0.5*UM_PER_PX).
    Returns std of θ_tip (deg) over MC samples.
    """
    x_um, y_um, scale, _ = _get_xy_in_um(df)
    rng = np.random.default_rng(RANDOM_SEED)
    thetas = []
    if x_um is not None and y_um is not None and len(x_um) >= 3:
        for _ in range(n_mc):
            nx = x_um + rng.uniform(-0.5*scale, 0.5*scale, size=len(x_um))
            ny = y_um + rng.uniform(-0.5*scale, 0.5*scale, size=len(y_um))
            ds = np.hypot(np.diff(nx), np.diff(ny))
            s_mc = np.concatenate([[0.0], np.cumsum(ds)])
            th = _cumtrapz(kappa, s_mc)
            thetas.append(float(np.degrees(th[-1])))
    else:
        # jitter s spacing directly
        ds = np.diff(s)
        for _ in range(n_mc):
            noise = rng.uniform(-0.5*UM_PER_PX, 0.5*UM_PER_PX, size=ds.size)
            ds_mc = np.maximum(1e-9, ds + noise)
            s_mc = np.concatenate([[0.0], np.cumsum(ds_mc)])
            th = _cumtrapz(kappa, s_mc)
            thetas.append(float(np.degrees(th[-1])))
    return float(np.std(thetas, ddof=1)) if len(thetas) > 1 else 0.0

def _mc_calibration_theta(kappa: np.ndarray, s: np.ndarray, rel_std: float = REL_K_STD, n_mc: int = N_MC_UNCERT) -> float:
    """
    Type B: calibration factor k uncertainty. Sample k ~ N(1, rel_std), scale s' = k*s and integrate.
    Returns std of θ_tip (deg).
    """
    if rel_std <= 0 or n_mc <= 0:
        return 0.0
    rng = np.random.default_rng(RANDOM_SEED+1)
    thetas = []
    for _ in range(n_mc):
        k = rng.normal(1.0, rel_std)
        s_mc = s * max(k, 1e-9)
        th = _cumtrapz(kappa, s_mc)
        thetas.append(float(np.degrees(th[-1])))
    return float(np.std(thetas, ddof=1)) if len(thetas) > 1 else 0.0

def _uncertainty_for_one_file(path: str) -> Optional[Dict[str, float]]:
    """
    Compute per-file uncertainties:
      u_A (quantization), u_B (calibration), u_disc (trap-vs-simpson), combined u_c, expanded U_k2.
    """
    try:
        df = _read_table(Path(path))
        kappa, s = _signed_kappa_and_s(df)
        if len(s) < 3: return None
        # nominal
        th_trap = _cumtrapz(kappa, s); th_simp = _cumsimpson(kappa, s)
        tip_deg = float(np.degrees(th_trap[-1]))
        u_disc = abs(float(np.degrees(th_simp[-1]) - tip_deg))  # take absolute diff (deg)
        u_A = _mc_quantization_theta(kappa, s, df, n_mc=N_MC_UNCERT)
        u_B = _mc_calibration_theta(kappa, s, rel_std=REL_K_STD, n_mc=N_MC_UNCERT)
        u_c = float(math.sqrt(u_A*u_A + u_B*u_B + u_disc*u_disc))
        U_k2 = 2.0 * u_c
        return {"file": path, "theta_tip_deg": tip_deg, "u_A_deg": u_A, "u_B_deg": u_B,
                "u_disc_deg": u_disc, "u_c_deg": u_c, "U_k2_deg": U_k2,
                "n_points": int(len(s))}
    except Exception as e:
        print(f"[WARN] Uncertainty failed for {path}: {e}")
        return None

def compute_gum_uncertainties(df_all: pd.DataFrame) -> None:
    """Compute GUM-like uncertainties for all files; write CSV and overall JSON."""
    files = [f for f in df_all["file"].dropna().tolist() if Path(f).exists()]
    rows = []
    for fp in files:
        r = _uncertainty_for_one_file(fp)
        if r is not None: rows.append(r)
    if not rows:
        print("[WARN] No uncertainty rows computed."); return
    dfu = pd.DataFrame(rows)
    dfu.to_csv(STATS_DIR / "uncertainty_per_file.csv", index=False, encoding="utf-8-sig")
    overall = {
        "count": int(len(dfu)),
        "theta_tip_deg_mean": float(dfu["theta_tip_deg"].mean()),
        "u_A_deg_mean": float(dfu["u_A_deg"].mean()),
        "u_B_deg_mean": float(dfu["u_B_deg"].mean()),
        "u_disc_deg_mean": float(dfu["u_disc_deg"].mean()),
        "u_c_deg_mean": float(dfu["u_c_deg"].mean()),
        "U_k2_deg_mean": float(dfu["U_k2_deg"].mean())
    }
    with open(STATS_DIR / "uncertainty_overall.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print("[OK] Uncertainty ->", STATS_DIR)

# ----------------------- NEW: Kabsch/Procrustes (rigid) on (s_norm, θ) curves -----------------------
def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve R,t that minimize ||P - (R Q + t)||_F for 2D point sets (no reflection).
    Returns (R, t). R is 2x2 rotation, t is 2x1 translation vector.
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    if P.shape != Q.shape or P.shape[1] != 2:
        raise ValueError("P and Q must be (N,2) with equal N.")
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # avoid reflection
        Vt[1,:] *= -1
        R = Vt.T @ U.T
    t = P.mean(axis=0) - (R @ Q.mean(axis=0))
    return R, t

def _hausdorff(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two 2D point sets."""
    def directed(A, B):
        # for each a in A, min distance to B
        d = []
        for a in A:
            d.append(np.min(np.linalg.norm(B - a, axis=1)))
        return np.max(d) if d else 0.0
    return float(max(directed(P,Q), directed(Q,P)))

def _kabsch_metrics_for_file(file_path: str) -> List[Dict[str, float]]:
    """
    Build (s_norm, θ) for reference and for each baseline, run rigid registration, report RMS & Hausdorff.
    """
    try:
        s_ref, th_ref, _ = _theta_curve_from_reference_file(file_path)
    except Exception:
        return []
    if len(s_ref) < 3: return []
    s_norm = s_ref / max(float(s_ref[-1]), 1e-9)
    ref = np.stack([s_norm, th_ref], axis=1)

    results = []
    # polyline baseline
    poly = _theta_curve_from_polyline_xy(file_path)
    if poly is not None:
        s_p, th_p, _ = poly
        sp = (s_p / max(float(s_p[-1]), 1e-9))
        poly2 = np.stack([np.interp(s_norm, sp, sp), np.interp(s_norm, sp, th_p)], axis=1)
        R, t = _kabsch(ref, poly2)
        Qhat = (R @ poly2.T).T + t
        rms = float(np.sqrt(np.mean(np.sum((ref - Qhat)**2, axis=1))))
        hd = _hausdorff(ref, Qhat)
        rot_deg = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
        results.append({"file":file_path, "method":"Polyline", "rms":rms, "hausdorff":hd, "rot_deg":rot_deg, "tx":float(t[0]), "ty":float(t[1])})
    # chord baseline
    chord = _theta_curve_from_chord(file_path, N=len(s_ref))
    if chord is not None:
        s_c, th_c, _ = chord
        sc = (s_c / max(float(s_c[-1]), 1e-9))
        ch2 = np.stack([np.interp(s_norm, sc, sc), np.interp(s_norm, sc, th_c)], axis=1)
        R, t = _kabsch(ref, ch2)
        Qhat = (R @ ch2.T).T + t
        rms = float(np.sqrt(np.mean(np.sum((ref - Qhat)**2, axis=1))))
        hd = _hausdorff(ref, Qhat)
        rot_deg = float(np.degrees(np.arctan2(R[1,0], R[0,0])))
        results.append({"file":file_path, "method":"Chord", "rms":rms, "hausdorff":hd, "rot_deg":rot_deg, "tx":float(t[0]), "ty":float(t[1])})
    return results

# --- REPLACE the whole function with this bugfixed version ---
def compute_kabsch_over_dataset(df_all: pd.DataFrame) -> None:
    """Run Kabsch/Procrustes metrics over all files; write CSV & JSON summary."""
    files = [f for f in df_all["file"].dropna().tolist() if Path(f).exists()]
    rows = []
    for fp in files:
        rows.extend(_kabsch_metrics_for_file(fp))

    if not rows:
        print("[WARN] No Kabsch metrics computed."); return

    dfk = pd.DataFrame(rows)
    dfk.to_csv(STATS_DIR / "kabsch_metrics.csv", index=False, encoding="utf-8-sig")

    # Build a JSON-safe nested summary: {method: {metric: {mean,std,median,min,max,n}}}
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    metrics = ["rms", "hausdorff", "rot_deg"]

    for method, g in dfk.groupby("method"):
        method = str(method)
        summary[method] = {}
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                # If no valid numbers for this metric, store None
                summary[method][metric] = {
                    "mean": None, "std": None, "median": None, "min": None, "max": None, "n": 0
                }
                continue
            summary[method][metric] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n":   int(vals.size),
            }

    with open(STATS_DIR / "kabsch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Kabsch metrics ->", STATS_DIR)


# ----------------------- Training entry -----------------------
def run_train():
    df_tr = collect_dataset(TRAIN_DIR, "train")
    df_te = collect_dataset(TEST_DIR,  "test")
    if df_tr.empty and df_te.empty: raise RuntimeError("No usable CSV/XLSX found in train/ or test/.")
    if df_te.empty:
        df = df_tr.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        n = max(1, int(0.8 * len(df))); df_tr, df_te = df.iloc[:n].copy(), df.iloc[n:].copy()
        print("[WARN] No test/ directory found. Used 80/20 split from training set.")
    df_tr.to_csv(OUT_DIR / "dataset_train_clean.csv", index=False, encoding="utf-8-sig")
    df_te.to_csv(OUT_DIR / "dataset_test_clean.csv",  index=False, encoding="utf-8-sig")
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all.to_csv(OUT_DIR / "dataset_all_clean.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Train samples: {len(df_tr)}, Test samples: {len(df_te)}")

    best_model, _ = train_and_eval(df_tr, df_te)
    train_curvature_shape_models(df_tr, df_te)
    auto_generate_theta_curves(df_all, theta0_deg=0.0)

    # Keep the previous diagnostic visuals:
    plot_baseline_error_summaries(df_te, best_model, include_ml_bar=True)
    generate_baseline_theta_overlays(df_all, include_predicted=False)

    # Compact, paper-ready summaries (figures + CSV)
    make_paper_baseline_summary(df_te, best_model, df_all, include_ml_bar=True)

    # NEW: stats tests & bootstrap CIs
    compute_tests_and_bootstrap(df_te, best_model)

    # NEW: GUM-style uncertainties
    compute_gum_uncertainties(df_all)

    # NEW: Kabsch/Procrustes metrics on θ(s) curves in (s_norm, θ) space
    compute_kabsch_over_dataset(df_all)

    print(f"[OK] Figures -> {FIG_DIR} and {PAPER_FIG_DIR}")
    print(f"[OK] Model & metrics -> {OUT_DIR}")
    print(f"[OK] Extra stats/uncertainty -> {STATS_DIR}")

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-only pipeline with paper-friendly baseline summaries + uncertainties/tests")
    parser.add_argument("--data_root", type=str, default=None, help="Folder containing train/ and test/ subfolders")
    parser.add_argument("--out_dir",   type=str, default=None,    help="Output folder (default: ./outputs)")
    parser.add_argument("--um_per_px", type=float, default=None,  help="Global pixel size in μm/px (fallback if not found in files)")
    # Optional knobs for uncertainty
    parser.add_argument("--u_k_rel", type=float, default=REL_K_STD, help="Relative std uncertainty for calibration factor k (Type B). Default 0.0")
    parser.add_argument("--mc_n", type=int, default=N_MC_UNCERT, help="MC samples for uncertainty propagation (Type A/B).")
    args = parser.parse_args()

    if args.um_per_px is not None:
        UM_PER_PX = float(args.um_per_px)
        if UM_PER_PX <= 0: UM_PER_PX = 1.0
        print(f"[INFO] Global UM_PER_PX = {UM_PER_PX} μm/px (fallback)")

    # allow runtime override of a couple of tunables
    if args.u_k_rel is not None and args.u_k_rel > 0:
        REL_K_STD = float(args.u_k_rel)
        print(f"[INFO] REL_K_STD (Type B) = {REL_K_STD:.4f}")
    if args.mc_n is not None and args.mc_n > 10:
        N_MC_UNCERT = int(args.mc_n)
        print(f"[INFO] N_MC_UNCERT (MC) = {N_MC_UNCERT}")

    configure_paths(args.data_root, args.out_dir)
    run_train()

# ----------------------- Curvature figures (train/test; unitless) -----------------------
from typing import Optional, List, Dict, Tuple

# The following feature is additive: it does not modify existing training/evaluation logic.
# It generates κ(s) overlay plots and CSVs for train/ and test/ splits under OUT_DIR/curvature/.
# Axis labels and CSV headers are unitless ("s", "kappa"), and per-file plots mirror the source
# relative paths inside outputs for easy cross-reference.

def _curv_collect_rows(split_dir: Path, fallback_um_per_px: float) -> Tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    notes: List[str] = []
    files = list(split_dir.rglob("*.csv")) + list(split_dir.rglob("*.xlsx"))
    for fp in files:
        try:
            df = _read_table(fp)
            kappa, s = _signed_kappa_and_s(df)  # uses UM_PER_PX inside if XY missing
            # If XY was missing, _signed_kappa_and_s already used UM_PER_PX; allow override:
            if (s.size >= 3) and (not np.all(np.diff(s) > 0)):
                s = np.arange(len(kappa), dtype=float) * fallback_um_per_px
            ratio = parse_ratio_from_path(fp)
            temp  = parse_temp_from_path(fp)
            rows.append({
                "file": str(fp),
                "rel": str(fp.relative_to(split_dir)),
                "ratio": float(ratio) if ratio is not None else None,
                "temp_C": float(temp) if temp is not None else None,
                "s": s - s[0],
                "kappa": kappa
            })
        except Exception as e:
            notes.append(f"[WARN] Skip {fp}: {e}")
    return rows, notes

def _curv_export_csv(rows: List[Dict], out_dir: Path, split_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for r in rows:
        n = min(len(r["s"]), len(r["kappa"]))
        parts.append(pd.DataFrame({
            "file": [r["file"]]*n,
            "rel":  [r["rel"]]*n,
            "ratio":[r["ratio"]]*n,
            "temp_C":[r["temp_C"]]*n,
            "s": r["s"][:n],
            "kappa": r["kappa"][:n],
        }))
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(out_dir / f"curvature_all_{split_name}.csv", index=False, encoding="utf-8-sig")

def _curv_plot_overlay(rows: List[Dict], out_dir: Path, split_name: str) -> None:
    if not rows:
        print(f"[WARN] No curvature curves for split={split_name}."); return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.6), dpi=150)
    for r in rows:
        lbl = r.get("rel", Path(r["file"]).name)
        plt.plot(r["s"], r["kappa"], lw=1.3, alpha=0.85, label=lbl)
    plt.xlabel("s"); plt.ylabel("kappa")
    plt.title(f"Curvature overlay ({split_name})")
    if len(rows) <= 15: plt.legend(fontsize=8, frameon=False)
    plt.grid(True, alpha=0.3); fig.tight_layout()
    plt.savefig(out_dir / f"curvature_overlay_{split_name}.png"); plt.close(fig)

def _curv_save_per_file(rows: List[Dict], out_dir: Path, split_name: str) -> None:
    root = out_dir / f"per_file_{split_name}"
    root.mkdir(parents=True, exist_ok=True)
    for r in rows:
        fig = plt.figure(figsize=(5.2, 3.6), dpi=150)
        plt.plot(r["s"], r["kappa"], lw=1.6)
        plt.xlabel("s"); plt.ylabel("kappa")
        lbl = r.get("rel", Path(r["file"]).name)
        ratio = r.get("ratio"); temp = r.get("temp_C")
        title = lbl
        if ratio is not None or temp is not None:
            title = f"{lbl} — r={ratio if ratio is not None else '?'}:1, T={temp if temp is not None else '?'}"
        plt.title(title); plt.grid(True, alpha=0.3); fig.tight_layout()
        # Mirror the relative path under outputs
        dst_dir = root / Path(lbl).parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        fname = Path(lbl).name
        fname = Path(fname).with_suffix(".png")
        plt.savefig(dst_dir / fname); plt.close(fig)

def run_curvature_plots(per_file: bool = True, um_per_px: Optional[float] = None) -> None:
    """Generate unitless κ(s) overlays/CSVs for train/ and test/ under OUT_DIR/curvature/ (additive feature)."""
    base = OUT_DIR / "curvature"
    fig_dir = base / "figs"; csv_dir = base / "csv"; paths_dir = base / "paths"
    for d in (fig_dir, csv_dir, paths_dir): d.mkdir(parents=True, exist_ok=True)
    fallback = float(um_per_px) if um_per_px is not None else float(UM_PER_PX)

    for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        if not split_dir.exists():
            print(f"[WARN] {split_name}/ not found under {DATA_ROOT}, skip."); continue
        rows, notes = _curv_collect_rows(split_dir, fallback)
        for m in notes: print(m)
        _curv_export_csv(rows, csv_dir, split_name)
        _curv_plot_overlay(rows, fig_dir, split_name)
        if per_file: _curv_save_per_file(rows, fig_dir, split_name)
        # Save paths list for traceability
        files = sorted([str(fp.relative_to(DATA_ROOT)) for fp in split_dir.rglob("*.csv")] +
                       [str(fp.relative_to(DATA_ROOT)) for fp in split_dir.rglob("*.xlsx")])
        with open(paths_dir / f"{split_name}_paths.txt", "w", encoding="utf-8") as f:
            for it in files: f.write(it + "\n")

# Register an atexit hook so this runs after training, without touching existing CLI/flow.
import atexit
def _curvature_atexit():
    try:
        run_curvature_plots(per_file=True, um_per_px=UM_PER_PX)
        print("[OK] Curvature figures ->", OUT_DIR / "curvature" / "figs")
        print("[OK] Curvature CSV ->", OUT_DIR / "curvature" / "csv")
    except Exception as e:
        print(f"[WARN] Curvature plotting skipped: {e}")
atexit.register(_curvature_atexit)

# ----------------------- Reconstruction from θ(s) (PNG only) -----------------------
# Additive feature: reconstruct (x,y) from a results CSV containing s and θ(s).
# This mirrors the standalone utility we use elsewhere (PNG only; no extra CSVs),
# and saves outputs under OUT_DIR/reconstruction/figs. Original XY (if present)
# are overlaid after shifting to start at (0,0).

def _recon_pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    """Pick the first column name from candidates (case/format tolerant)."""
    norm = {c.lower().strip(): c for c in df.columns}
    def simp(s: str) -> str:
        return (s.lower()
                .replace("(", " ").replace(")", " ")
                .replace("-", " ").replace("_", " ")
                .replace("/", " ").replace(".", " ").replace(",", " ")
                .strip())
    fuzz = {simp(c): c for c in df.columns}
    for cand in cands:
        key = cand.lower().strip()
        if key in norm: return norm[key]
    for cand in cands:
        key = simp(cand)
        if key in fuzz: return fuzz[key]
    return None

def reconstruct_from_results_csv(path: str, save_dir: Optional[str] = None) -> Optional[Path]:
    """
    Read a CSV with (s, θ_deg) and reconstruct XY via:
      dx = ds * cos(θ),  dy = ds * sin(θ).
    If original XY are present, also overlay for visual comparison.
    PNG only; saved under OUT_DIR/reconstruction/figs (or save_dir if provided).
    """
    try:
        import os
        p = Path(path)
        if not p.exists(): 
            print(f"[WARN] Reconstruction input not found: {path}")
            return None
        df = pd.read_csv(p)

        # Columns for s and θ(s) (accept several variants).
        s_col = _recon_pick_col(df, ["s", "s_abs", "s_um", "arc_length", "arc length s um", "arc_length_s_um"])
        th_col = _recon_pick_col(df, ["theta_deg", "theta (deg)", "theta_pred_deg", "tangent angle θ (deg)", "theta"])
        if not s_col or not th_col:
            raise KeyError("CSV must contain columns for 's' and 'theta_deg' (or compatible headers).")

        s = pd.to_numeric(df[s_col], errors="coerce").to_numpy(float)
        theta_deg = pd.to_numeric(df[th_col], errors="coerce").to_numpy(float)

        # Reconstruct relative (x, y) from θ(s), starting at (0,0)
        theta_rad = np.radians(theta_deg)
        ds = np.diff(s, prepend=s[0])
        dx = ds * np.cos(theta_rad)
        dy = ds * np.sin(theta_rad)
        x_rec = np.cumsum(dx)
        y_rec = np.cumsum(dy)

        # Optional original XY overlay (if present)
        x_col = _recon_pick_col(df, ["x_um", "x (um)", "x", "x_abs"])
        y_col = _recon_pick_col(df, ["y_um", "y (um)", "y", "y_abs"])
        has_xy = (x_col is not None and y_col is not None)

        # Save figure
        base_dir = Path(save_dir).resolve() if save_dir else (OUT_DIR / "reconstruction" / "figs")
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = p.stem

        fig = plt.figure(figsize=(5.6, 4.8), dpi=150)
        if has_xy:
            x0 = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
            y0 = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
            plt.plot(x0 - x0[0], y0 - y0[0], "--", label="Original (shifted to start)")
            plt.plot(x_rec, y_rec, lw=2, label="Reconstructed from θ(s)")
            out_png = base_dir / f"{stem}_original_vs_reconstructed.png"
        else:
            plt.plot(x_rec, y_rec, lw=2, label="Reconstructed from θ(s)")
            out_png = base_dir / f"{stem}_reconstructed.png"

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x (µm)"); plt.ylabel("y (µm)")
        plt.title("Reconstructed Curve from θ(s)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"[OK] Saved reconstruction: {out_png}")
        return out_png
    except Exception as e:
        print(f"[WARN] Reconstruction failed for {path}: {e}")
        return None

def batch_reconstruct_from_dir(dir_path: str) -> None:
    """
    Scan a directory for candidate results CSVs and run reconstruction on each.
    Heuristics: files under OUT_DIR/theta_curves/csv or files whose name contains
    'kappa_to_angle' or 'theta_curve'.
    """
    root = Path(dir_path).resolve()
    if not root.exists():
        print(f"[WARN] Reconstruction dir not found: {dir_path}"); return
    # Gather candidate CSVs
    cands = []
    for fp in root.rglob("*.csv"):
        name = fp.name.lower()
        if ("kappa_to_angle" in name) or ("theta_curve" in name) or ("theta" in name and "curve" in name):
            cands.append(fp)
    # Fallback: if nothing matched, try all CSVs (but cap at 50 to be safe)
    if not cands:
        cands = list(root.rglob("*.csv"))[:50]
    if not cands:
        print(f"[WARN] No reconstruction candidates under {dir_path}."); return
    print(f"[INFO] Reconstruction: {len(cands)} candidate CSV(s) found under {dir_path}.")
    for fp in cands:
        reconstruct_from_results_csv(str(fp))

# Register a separate atexit hook (non-invasive).
def _reconstruction_atexit():
    try:
        # Prefer OUT_DIR/theta_curves/csv (if exists), else OUT_DIR
        pref = OUT_DIR / "theta_curves" / "csv"
        target = pref if pref.exists() else OUT_DIR
        batch_reconstruct_from_dir(str(target))
        print("[OK] Reconstruction PNGs ->", OUT_DIR / "reconstruction" / "figs")
    except Exception as e:
        print(f"[WARN] Reconstruction skipped: {e}")

import atexit as _recon_atexit_mod
_recon_atexit_mod.register(_reconstruction_atexit)
