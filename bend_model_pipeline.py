# -*- coding: utf-8 -*-
"""
End-to-end training pipeline for predicting final bending angle θ_tip (deg)
from PNIPAM:PDMS ratio r and temperature T (°C), PLUS an additional
curvature-shape modeling branch (PCA + regression) that enables predicting
an entire curvature profile κ(s) during inference.

UPDATED:
- Train step now also saves parity plots for TRAIN, TEST, and ALL (train+test).
- After training curvature-shape models, the script automatically generates full
  θ(s) curves for every (ratio, temp) that appears in your dataset:
    outputs/theta_curves/figs/*.png and outputs/theta_curves/csv/*.csv
- A CLI subcommand `predict_curve` remains available for ad-hoc θ(s) generation.

Data assumption:
- Each file (CSV/XLSX) corresponds to ONE specimen at ONE ratio & ONE temperature.
- The file contains a curvature series; preferred columns:
    "Point Curvature (um-1)" and "Point Curvature Sign" (+/-1).
  X/Y coordinates are used to reconstruct arc-length s; otherwise unit spacing.

Outputs (under ./outputs by default):
  - dataset_*_clean.csv, best_model.joblib, metrics.json
  - figures/: parity_train.png, parity_test.png, parity_all.png,
              residual_*.png, learning_curve.png,
              feature_importance.png, response_surface.png,
              robustness_r{1,3,5}.png
  - curvature_pca.joblib, curvature_reg.joblib, length_model.joblib, curvature_meta.json
  - theta_curves/csv/*.csv, theta_curves/figs/*.png  (auto-generated after train)
"""

from __future__ import annotations
import argparse
import json
import math
import re
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


# ------------------- Tunables -------------------
RANDOM_SEED = 42
N_FOLDS = 5

# Input tolerance (for robustness plots)
TOL_TEMP_STD = 0.5   # °C (std. dev.)
TOL_RATIO_STD = 0.1  # absolute std. dev. of r
N_MONTE_CARLO = 2000

# Response surface grid
GRID_T_MIN, GRID_T_MAX, GRID_T_STEP = 20, 60, 1
GRID_R_MIN, GRID_R_MAX, GRID_R_STEP = 1.0, 5.0, 0.05

# Curvature shape model resampling
N_POINTS_CURV = 200   # number of resampled points along normalized arc-length [0..1]

# ------------------- Paths -------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"
OUT_DIR   = ROOT / "outputs"
FIG_DIR   = OUT_DIR / "figures"

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Regex helpers
_ratio_re = re.compile(r"(\d+)\s*vs\s*(\d+)", re.IGNORECASE)
_temp_re  = re.compile(r"(\d+)\s*C", re.IGNORECASE)


# ------------------- Small utilities -------------------
def configure_paths(data_root: Optional[str], out_dir: Optional[str]) -> None:
    global DATA_ROOT, TRAIN_DIR, TEST_DIR, OUT_DIR, FIG_DIR
    if data_root:
        DATA_ROOT = Path(data_root).resolve()
        print(f"[INFO] Using data_root = {DATA_ROOT}")
        TRAIN_DIR = DATA_ROOT / "train"
        TEST_DIR  = DATA_ROOT / "test"
    if out_dir:
        OUT_DIR = Path(out_dir).resolve()
        print(f"[INFO] Using out_dir = {OUT_DIR}")
    FIG_DIR = OUT_DIR / "figures"
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _find_col_like(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if len(y) < 2:
        return np.zeros_like(y)
    dx = np.diff(x)
    mid = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y, float)
    out[1:] = np.cumsum(mid)
    return out


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


def integrate_tip_angle(df: pd.DataFrame) -> Dict[str, float]:
    k_point = _find_col_like(df, ["point curvature"])
    sign_c  = _find_col_like(df, ["point curvature sign", "sign"])
    x_col   = _find_col_like(df, ["x-coordinate", "x (", "x)", " x"])
    y_col   = _find_col_like(df, ["y-coordinate", "y (", "y)", " y"])

    if k_point is not None:
        kappa = pd.to_numeric(df[k_point], errors="coerce").to_numpy()
        if sign_c is not None:
            sign = pd.to_numeric(df[sign_c], errors="coerce").to_numpy()
            nz = np.abs(sign) > 0
            if nz.any():
                last = -1
                for i in range(len(sign)):
                    if nz[i]:
                        last = i
                    elif last >= 0:
                        sign[i] = np.sign(sign[last])
                if last < 0:
                    sign[:] = 1.0
            else:
                sign = np.ones_like(kappa)
            kappa = kappa * np.sign(sign)
    else:
        k_avg = _find_col_like(df, ["average curvature", "curvature"])
        if k_avg is None:
            raise ValueError("No curvature column found.")
        kappa = pd.to_numeric(df[k_avg], errors="coerce").to_numpy()

    if x_col is not None and y_col is not None:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()

        def _interp(v):
            idx = np.flatnonzero(np.isfinite(v))
            if len(idx) == 0: return np.zeros_like(v)
            return np.interp(np.arange(len(v)), idx, v[idx])

        x = _interp(x)
        y = _interp(y)
        ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s  = np.concatenate([[0.0], np.cumsum(ds)])
    else:
        s = np.arange(len(kappa), dtype=float)

    mask = np.isfinite(kappa) & np.isfinite(s)
    kappa = kappa[mask]; s = s[mask]
    if len(s) < 3:
        raise ValueError("Too few valid points to integrate.")

    theta_rad = _cumtrapz(kappa, s)
    tip_angle_deg = float(np.degrees(theta_rad[-1]))
    L = float(s[-1] - s[0])
    mean_kappa = float(np.trapz(kappa, s) / max(L, 1e-9))
    max_abs_kappa = float(np.nanmax(np.abs(kappa)))
    return dict(tip_angle_deg=tip_angle_deg, L=L,
                mean_kappa=mean_kappa, max_abs_kappa=max_abs_kappa)


def collect_dataset(split_dir: Path, split_name: str) -> pd.DataFrame:
    rows = []
    n_total, n_ok = 0, 0
    for path in list(split_dir.rglob("*.csv")) + list(split_dir.rglob("*.xlsx")):
        n_total += 1
        try:
            df = _read_table(path)
            feats = integrate_tip_angle(df)
            ratio = parse_ratio_from_path(path)
            temp  = parse_temp_from_path(path)
            if ratio is None or temp is None:
                print(f"[WARN] Skip (cannot parse ratio/temp): {path}")
                continue
            rows.append({
                "split": split_name,
                "specimen": "/".join(path.parts[-3:]),
                "ratio": float(ratio),
                "ratio_frac": float(ratio/(1.0+ratio)),
                "temp_C": float(temp),
                **feats,
                "file": str(path)
            })
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] Read failed: {path} -> {e}")
    df = pd.DataFrame(rows)
    print(f"[INFO] {split_name}: found {n_total} files, loaded {n_ok} usable samples.")
    return df


# ------------------- Plots -------------------
def _save_metrics(all_metrics: Dict[str, dict]) -> None:
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)


def _parity_residual(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> None:
    # Parity
    plt.figure(figsize=(5, 5))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, '--', lw=1)
    plt.scatter(y_true, y_pred, alpha=0.9)
    plt.xlabel("True θ_tip (deg)")
    plt.ylabel("Predicted θ_tip (deg)")
    plt.title(f"Parity plot ({tag})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"parity_{tag}.png", dpi=300)
    plt.close()

    # Residual
    resid = y_pred - y_true
    plt.figure(figsize=(5, 4))
    plt.axhline(0, ls="--", lw=1)
    plt.scatter(y_pred, resid, alpha=0.9)
    plt.xlabel("Predicted θ_tip (deg)")
    plt.ylabel("Residual (pred - true)")
    plt.title(f"Residual vs Pred ({tag})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"residual_{tag}.png", dpi=300)
    plt.close()


def _learning_curve(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    sizes, tr_scores, cv_scores = learning_curve(
        estimator, X, y,
        cv=min(N_FOLDS, max(2, len(y)//2)),
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_SEED
    )
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, -tr_scores.mean(1), 'o-', label="Train MAE")
    plt.plot(sizes, -cv_scores.mean(1), 'o-', label="CV MAE")
    plt.xlabel("Training Samples")
    plt.ylabel("MAE (deg)")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curve.png", dpi=300)
    plt.close()


def _importance(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, names: List[str]) -> None:
    r = permutation_importance(estimator, X, y, n_repeats=20, random_state=RANDOM_SEED, n_jobs=-1)
    imp, std = r.importances_mean, r.importances_std
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(names)), imp[idx], yerr=std[idx])
    plt.xticks(range(len(names)), [names[i] for i in idx])
    plt.ylabel("Permutation importance (ΔMAE)")
    plt.title("Input importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=300)
    plt.close()


def _response_surface(model: Pipeline) -> None:
    Ts = np.arange(GRID_T_MIN, GRID_T_MAX + GRID_T_STEP, GRID_T_STEP)
    Rs = np.arange(GRID_R_MIN, GRID_R_MAX + GRID_R_STEP, GRID_R_STEP)
    TT, RR = np.meshgrid(Ts, Rs)
    df = pd.DataFrame({
        "ratio": RR.ravel(),
        "ratio_frac": (RR/(1.0+RR)).ravel(),
        "temp_C": TT.ravel()
    })
    Z = model.predict(df).reshape(RR.shape)
    plt.figure(figsize=(6.3, 4.8))
    cs = plt.contourf(TT, RR, Z, levels=24)
    plt.colorbar(cs, label="Predicted θ_tip (deg)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("r = PNIPAM:PDMS")
    plt.title("Response surface of θ_tip")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "response_surface.png", dpi=300)
    plt.close()


def _robustness(model: Pipeline) -> None:
    Ts = np.arange(GRID_T_MIN, GRID_T_MAX + 1, 1)
    for r_nom in [1.0, 3.0, 5.0]:
        means, stds = [], []
        for T_nom in Ts:
            T = np.random.normal(T_nom, TOL_TEMP_STD, size=N_MONTE_CARLO)
            R = np.random.normal(r_nom, TOL_RATIO_STD, size=N_MONTE_CARLO)
            df = pd.DataFrame({"ratio": R, "ratio_frac": R/(1.0+R), "temp_C": T})
            y = model.predict(df)
            means.append(y.mean())
            stds.append(y.std())
        means = np.array(means); stds = np.array(stds)
        plt.figure(figsize=(6, 4))
        plt.plot(Ts, means, label=f"r={r_nom}")
        plt.fill_between(Ts, means - stds, means + stds, alpha=0.3)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Predicted θ_tip (deg)")
        plt.title(f"Robustness under tolerance (σ_T={TOL_TEMP_STD}°C, σ_r={TOL_RATIO_STD})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"robustness_r{int(r_nom)}.png", dpi=300)
        plt.close()


# ------------------- Training & evaluation (θ_tip model) -------------------
def train_and_eval(df_tr: pd.DataFrame, df_te: pd.DataFrame):
    target = "tip_angle_deg"
    feats = ["ratio", "ratio_frac", "temp_C"]

    Xtr, ytr = df_tr[feats], df_tr[target]
    Xte, yte = df_te[feats], df_te[target]

    cv = KFold(n_splits=min(N_FOLDS, max(2, len(df_tr)//2)),
               shuffle=True, random_state=RANDOM_SEED)

    candidates = []

    # (1) Polynomial + Linear regression (degree=2)
    poly = Pipeline([
        ("ct", ColumnTransformer([("num", "passthrough", feats)])),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("sc", StandardScaler()),
        ("reg", LinearRegression())
    ])
    candidates.append(("PolyLinear", poly, {}))

    # (2) RandomForest
    rf = Pipeline([
        ("ct", ColumnTransformer([("num", "passthrough", feats)])),
        ("reg", RandomForestRegressor(random_state=RANDOM_SEED))
    ])
    grid_rf = {"reg__n_estimators": [300, 600],
               "reg__max_depth": [None, 6, 10],
               "reg__min_samples_leaf": [1, 2, 4]}
    candidates.append(("RandomForest", rf, grid_rf))

    # (3) GradientBoosting
    gbr = Pipeline([
        ("ct", ColumnTransformer([("num", "passthrough", feats)])),
        ("reg", GradientBoostingRegressor(random_state=RANDOM_SEED))
    ])
    grid_gbr = {"reg__n_estimators": [400, 800],
                "reg__learning_rate": [0.05, 0.1],
                "reg__max_depth": [2, 3, 4],
                "reg__subsample": [0.8, 1.0]}
    candidates.append(("GradBoost", gbr, grid_gbr))

    best_name, best_model, best_metrics = None, None, None
    all_metrics: Dict[str, dict] = {}

    for name, pipe, grid in candidates:
        gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                          cv=cv, n_jobs=-1, verbose=0)
        gs.fit(Xtr, ytr)
        y_pred = gs.best_estimator_.predict(Xte)
        m = {
            "cv_best_params": gs.best_params_,
            "cv_best_mae": -float(gs.best_score_),
            "test_r2": float(r2_score(yte, y_pred)),
            "test_rmse": float(math.sqrt(mean_squared_error(yte, y_pred))),
            "test_mae": float(mean_absolute_error(yte, y_pred)),
            "n_train": int(len(df_tr)),
            "n_test": int(len(df_te))
        }
        all_metrics[name] = m
        if best_metrics is None or m["test_mae"] < best_metrics["test_mae"]:
            best_metrics, best_model, best_name = m, gs.best_estimator_, name

    _save_metrics(all_metrics)
    joblib.dump(best_model, OUT_DIR / "best_model.joblib")
    print(f"[INFO] Best model: {best_name}")
    print(json.dumps(best_metrics, indent=2, ensure_ascii=False))

    # Plots
    _parity_residual(ytr.to_numpy(), best_model.predict(Xtr), "train")
    _parity_residual(yte.to_numpy(), best_model.predict(Xte), "test")

    # NEW: parity on ALL samples (train+test)
    Xall = pd.concat([Xtr, Xte], ignore_index=True)
    yall = pd.concat([ytr, yte], ignore_index=True)
    _parity_residual(yall.to_numpy(), best_model.predict(Xall), "all")

    _learning_curve(best_model, Xall, yall)
    _importance(best_model, Xtr, ytr, feats)
    _response_surface(best_model)
    _robustness(best_model)

    return best_model, all_metrics


# ------------------- Curvature shape models (PCA + regression) -------------------
def _signed_kappa_and_s(df: pd.DataFrame):
    k_point = _find_col_like(df, ["point curvature"])
    sign_c  = _find_col_like(df, ["point curvature sign", "sign"])
    x_col   = _find_col_like(df, ["x-coordinate", "x (", "x)", " x"])
    y_col   = _find_col_like(df, ["y-coordinate", "y (", "y)", " y"])

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
    else:
        k_avg = _find_col_like(df, ["average curvature", "curvature"])
        if k_avg is None:
            raise ValueError("No curvature column found for shape model.")
        kappa = pd.to_numeric(df[k_avg], errors="coerce").to_numpy()

    if x_col is not None and y_col is not None:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        def _interp(v):
            idx = np.flatnonzero(np.isfinite(v))
            if len(idx) == 0: return np.zeros_like(v)
            return np.interp(np.arange(len(v)), idx, v[idx])
        x = _interp(x); y = _interp(y)
        ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s  = np.concatenate([[0.0], np.cumsum(ds)])
    else:
        s = np.arange(len(kappa), dtype=float)

    mask = np.isfinite(kappa) & np.isfinite(s)
    return kappa[mask], s[mask]


def _resample_kappa_from_file(filepath: str, n_points: int = N_POINTS_CURV):
    df = _read_table(Path(filepath))
    kappa, s = _signed_kappa_and_s(df)
    if len(s) < 3:
        raise ValueError("Too few points for curvature resampling.")
    L = float(s[-1] - s[0]) if s[-1] > s[0] else float(len(s) - 1)
    s_norm_src = (s - s[0]) / max(L, 1e-9)
    s_norm_tar = np.linspace(0.0, 1.0, n_points)
    kappa_rs = np.interp(s_norm_tar, s_norm_src, kappa)
    return s_norm_tar, kappa_rs, L


def train_curvature_shape_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    feature_cols = ["ratio", "ratio_frac", "temp_C"]

    Xtr = df_train[feature_cols].to_numpy(dtype=float)
    files_tr = df_train["file"].tolist()
    Ktr, Ltr = [], []
    for fp in files_tr:
        try:
            _, k_rs, L = _resample_kappa_from_file(fp, N_POINTS_CURV)
            Ktr.append(k_rs); Ltr.append(L)
        except Exception as e:
            print(f"[WARN] Curvature resampling failed (train): {fp} -> {e}")
    if len(Ktr) < 3:
        print("[WARN] Too few curves for shape model; skipping.")
        return
    Ktr = np.vstack(Ktr)
    Ltr = np.asarray(Ltr, float)

    pca_tmp = PCA(n_components=min(10, Ktr.shape[0], Ktr.shape[1]), svd_solver="full", random_state=RANDOM_SEED)
    pca_tmp.fit(Ktr)
    evr_cum = np.cumsum(pca_tmp.explained_variance_ratio_)
    n_comp = int(np.searchsorted(evr_cum, 0.99) + 1)
    n_comp = max(1, min(n_comp, 10))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=RANDOM_SEED).fit(Ktr)
    Ztr = pca.transform(Ktr)

    base_reg = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=600,
                                         max_depth=3, learning_rate=0.06, subsample=0.9)
    score_reg = MultiOutputRegressor(base_reg).fit(Xtr, Ztr)

    L_reg = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=600,
                                      max_depth=3, learning_rate=0.06, subsample=0.9)
    L_reg.fit(Xtr, Ltr)

    joblib.dump(pca, OUT_DIR / "curvature_pca.joblib")
    joblib.dump(score_reg, OUT_DIR / "curvature_reg.joblib")
    joblib.dump(L_reg, OUT_DIR / "length_model.joblib")
    meta = {
        "n_points": int(N_POINTS_CURV),
        "n_components": int(n_comp),
        "explained_variance_ratio_cum": float(evr_cum[n_comp-1]),
    }
    with open(OUT_DIR / "curvature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved curvature-shape models to {OUT_DIR} (n_points={N_POINTS_CURV}, n_components={n_comp})")


# ------------------- Full θ(s) prediction utilities -------------------
def _find_artifact(filename: str) -> Optional[Path]:
    candidates = [
        OUT_DIR,
        ROOT,
        ROOT / "models",
        ROOT / "artifacts",
        ROOT / "trained",
        ROOT / "outputs",
        ROOT / "checkpoints",
    ]
    for base in candidates:
        p = base / filename
        if p.exists():
            return p
    return None


def _load_curve_models() -> Dict[str, object]:
    pca_p = _find_artifact("curvature_pca.joblib")
    reg_p = _find_artifact("curvature_reg.joblib")
    len_p = _find_artifact("length_model.joblib")
    missing = []
    if pca_p is None: missing.append("curvature_pca.joblib")
    if reg_p is None: missing.append("curvature_reg.joblib")
    if missing:
        raise FileNotFoundError(
            "Missing required curvature-shape artifact(s): "
            + ", ".join(missing)
            + f". Looked under OUT_DIR={OUT_DIR} and common folders near this script."
        )
    models = {
        "pca": joblib.load(pca_p),
        "reg": joblib.load(reg_p),
        "len": joblib.load(len_p) if len_p is not None else None
    }
    return models


def _predict_kappa_and_s(models: Dict[str, object], ratio: float, temp_C: float) -> Tuple[np.ndarray, np.ndarray, float]:
    pca = models["pca"]
    reg = models["reg"]
    len_reg = models["len"]

    r = float(ratio)
    X = np.array([[r, r / (1.0 + r), float(temp_C)]], dtype=float)

    y_pred = reg.predict(X)
    y_pred = np.asarray(y_pred).reshape(1, -1)

    n_components = getattr(pca, "n_components_", None) or pca.components_.shape[0]
    if y_pred.shape[1] == n_components:
        kappa = pca.inverse_transform(y_pred)[0]
    else:
        kappa = y_pred[0]

    N = kappa.shape[0]
    s_norm = np.linspace(0.0, 1.0, N)

    if len_reg is not None:
        L_abs = float(len_reg.predict(X).ravel()[0])
        if not np.isfinite(L_abs) or L_abs <= 0:
            L_abs = 1.0
    else:
        L_abs = 1.0

    return kappa, s_norm, L_abs


def _kappa_to_theta_deg(kappa: np.ndarray, s_abs: np.ndarray, theta0_deg: float = 0.0) -> np.ndarray:
    theta_rad = _cumtrapz(kappa, s_abs)
    return np.degrees(theta_rad) + float(theta0_deg)


def _make_label(ratio: float, temp_C: float) -> str:
    return f"r={ratio:g}:1, T={temp_C:g}°C"


def run_predict_curve(ratio: Optional[float], ratio_str: Optional[str],
                      temps: List[float], theta0_deg: float,
                      outdir_override: Optional[str], no_individual_plots: bool) -> None:
    if ratio is None and ratio_str:
        m = _ratio_re.search(ratio_str)
        if not m:
            raise ValueError("--ratio_str must look like '5vs1' or '5:1'")
        a, b = float(m.group(1)), float(m.group(2))
        if b == 0:
            raise ValueError("ratio denominator cannot be zero.")
        ratio = a / b
    if ratio is None:
        raise ValueError("Provide --ratio or --ratio_str")

    theta_root = Path(outdir_override).resolve() if outdir_override else (OUT_DIR / "theta_curves")
    (theta_root / "csv").mkdir(parents=True, exist_ok=True)
    (theta_root / "figs").mkdir(parents=True, exist_ok=True)

    models = _load_curve_models()

    overlay_curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for T in temps:
        kappa, s_norm, L_abs = _predict_kappa_and_s(models, float(ratio), float(T))
        s_abs = s_norm * L_abs
        theta_deg = _kappa_to_theta_deg(kappa, s_abs, theta0_deg=theta0_deg)

        tag = f"r{ratio:g}_T{T:g}"
        df = pd.DataFrame({
            "s_norm": s_norm,
            "s_abs": s_abs,
            "kappa_pred": kappa,
            "theta_pred_deg": theta_deg,
        })
        csv_path = theta_root / "csv" / f"theta_curve_{tag}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        if not no_individual_plots:
            fig = plt.figure(figsize=(6, 4.2), dpi=150)
            plt.plot(s_abs, theta_deg, lw=2)
            plt.xlabel("Arc length s (absolute units)")
            plt.ylabel("Tangent angle θ(s) [deg]")
            plt.title(f"θ(s) — {_make_label(ratio, T)}")
            plt.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_path = theta_root / "figs" / f"theta_curve_{tag}.png"
            plt.savefig(fig_path)
            plt.close(fig)

        overlay_curves.append((s_abs, theta_deg, _make_label(ratio, T)))

    fig = plt.figure(figsize=(7, 4.5), dpi=150)
    for s_abs, theta_deg, lbl in overlay_curves:
        plt.plot(s_abs, theta_deg, lw=2, label=lbl)
    plt.xlabel("Arc length s (absolute units)")
    plt.ylabel("Tangent angle θ(s) [deg]")
    plt.title(f"θ(s) overlay — ratio {ratio:g}:1")
    plt.grid(True, alpha=0.3)
    if len(overlay_curves) > 1:
        plt.legend()
    fig.tight_layout()
    overlay_path = theta_root / "figs" / f"theta_overlay_r{ratio:g}.png"
    plt.savefig(overlay_path)
    plt.close(fig)

    print(f"[OK] Saved θ(s) CSVs -> {theta_root / 'csv'}")
    print(f"[OK] Saved θ(s) figures -> {theta_root / 'figs'}")


# ------------------- Helpers to auto-generate θ(s) after train -------------------
def auto_generate_theta_curves(df_all: pd.DataFrame, theta0_deg: float = 0.0) -> None:
    """
    For each distinct ratio in the dataset, gather all temps and produce θ(s) curves.
    """
    # Ensure artifacts exist
    pca_ok = _find_artifact("curvature_pca.joblib") is not None
    reg_ok = _find_artifact("curvature_reg.joblib") is not None
    if not (pca_ok and reg_ok):
        print("[WARN] Curvature-shape artifacts not found; skip θ(s) auto-generation.")
        return

    by_ratio = (
        df_all[["ratio", "temp_C"]]
        .dropna()
        .sort_values(["ratio", "temp_C"])
        .groupby("ratio")["temp_C"]
        .apply(lambda s: sorted(set(float(x) for x in s.tolist())))
        .to_dict()
    )
    if not by_ratio:
        print("[WARN] No (ratio, temp) combinations found for θ(s) curves.")
        return

    print("[INFO] Auto-generating θ(s) curves for dataset combos:")
    for r, temps in by_ratio.items():
        print(f"  - r={r:g}: temps={temps}")
        run_predict_curve(ratio=float(r), ratio_str=None,
                          temps=temps, theta0_deg=theta0_deg,
                          outdir_override=None, no_individual_plots=False)


# ------------------- Entry points -------------------
def run_train():
    df_tr = collect_dataset(TRAIN_DIR, "train")
    df_te = collect_dataset(TEST_DIR,  "test")

    if df_tr.empty and df_te.empty:
        raise RuntimeError("No usable CSV/XLSX found in train/ or test/.")

    if df_te.empty:
        df = df_tr.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        n = max(1, int(0.8 * len(df)))
        df_tr, df_te = df.iloc[:n].copy(), df.iloc[n:].copy()
        print("[WARN] No test/ directory found. Used 80/20 split from training set.")

    df_tr.to_csv(OUT_DIR / "dataset_train_clean.csv", index=False, encoding="utf-8-sig")
    df_te.to_csv(OUT_DIR / "dataset_test_clean.csv",  index=False, encoding="utf-8-sig")
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all.to_csv(OUT_DIR / "dataset_all_clean.csv", index=False, encoding="utf-8-sig")

    print(f"[INFO] Train samples: {len(df_tr)}, Test samples: {len(df_te)}")
    train_and_eval(df_tr, df_te)

    # Train curvature-shape models & auto-generate θ(s)
    train_curvature_shape_models(df_tr, df_te)
    auto_generate_theta_curves(df_all, theta0_deg=0.0)

    print(f"[OK] Figures -> {FIG_DIR}")
    print(f"[OK] Model & metrics -> {OUT_DIR}")


def run_predict(ratio: Optional[float], ratio_str: Optional[str], temp: float):
    if ratio is None and ratio_str:
        m = _ratio_re.search(ratio_str)
        if not m:
            raise ValueError("--ratio_str must look like '5vs1'")
        a, b = int(m.group(1)), int(m.group(2))
        ratio = a / b
    if ratio is None:
        raise ValueError("Provide --ratio or --ratio_str")

    model = joblib.load(OUT_DIR / "best_model.joblib")
    df = pd.DataFrame({
        "ratio": [ratio],
        "ratio_frac": [ratio / (1 + ratio)],
        "temp_C": [temp]
    })
    pred = float(model.predict(df)[0])
    print(json.dumps({
        "inputs": {"ratio": ratio, "temp_C": temp},
        "prediction": {"theta_tip_deg": pred}
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PNIPAM:PDMS bending angle model — training, prediction, and full-curve inference"
    )
    parser.add_argument("--data_root", type=str, default=None,
                        help="Folder that contains train/ and test/ (default: this file's directory)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output folder (default: ./outputs relative to this file)")
    sub = parser.add_subparsers(dest="cmd")

    # Single-point θ_tip prediction
    p_pred = sub.add_parser("predict", help="Single-point θ_tip prediction (after training)")
    p_pred.add_argument("--ratio", type=float, default=None,
                        help="r = PNIPAM:PDMS front term (e.g., 5 for 5:1)")
    p_pred.add_argument("--ratio_str", type=str, default=None,
                        help="String like '5vs1' or '5:1'")
    p_pred.add_argument("--temp", type=float, required=False,
                        help="Temperature in °C")

    # Full curve θ(s) prediction
    p_curve = sub.add_parser("predict_curve", help="Predict full θ(s) curve(s) from saved curvature-shape models")
    p_curve.add_argument("--ratio", type=float, default=None,
                         help="r = PNIPAM:PDMS front term (e.g., 5 for 5:1)")
    p_curve.add_argument("--ratio_str", type=str, default=None,
                         help="String like '5vs1' or '5:1'")
    p_curve.add_argument("--temp", type=float, nargs="+", required=True,
                         help="One or more temperatures in °C, e.g., --temp 20 30 40")
    p_curve.add_argument("--theta0_deg", type=float, default=0.0,
                         help="θ(0) offset in degrees (default 0)")
    p_curve.add_argument("--outdir", type=str, default=None,
                         help="Override output folder for θ(s) curves (default: OUT_DIR/theta_curves)")
    p_curve.add_argument("--no_individual_plots", action="store_true",
                         help="Only write overlay figure, skip per-(r,T) figures")

    args = parser.parse_args()
    configure_paths(args.data_root, args.out_dir)

    if args.cmd == "predict":
        if args.temp is None:
            raise ValueError("--temp is required for prediction")
        run_predict(args.ratio, args.ratio_str, args.temp)
    elif args.cmd == "predict_curve":
        run_predict_curve(args.ratio, args.ratio_str, args.temp, args.theta0_deg,
                          args.outdir, args.no_individual_plots)
    else:
        # default: train (and auto-generate θ(s) curves)
        run_train()
