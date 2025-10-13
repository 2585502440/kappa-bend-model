# predict_curvature.py
# -*- coding: utf-8 -*-
"""
Predict an entire curvature profile κ(s) for a given (ratio, temp_C),
using PCA + multi-output regression models trained by bend_model_pipeline.py.

This "Method A" version:
- Edit DEFAULT_RATIO / DEFAULT_TEMP / DEFAULT_PRED_DIR (and optionally DEFAULT_FILE_STEM) below.
- Running `python predict_curvature.py` with NO CLI flags will use those defaults.
- The script saves ALL outputs (CSV + PNGs) into a unified folder, by default: ./outputs/predictions/

Outputs (all under the chosen prediction directory):
  - <stem>_curvature_prediction.csv        (s_norm, s_abs, kappa_pred, theta_pred_deg)
  - <stem>_theta_vs_s.png                  (θ(s) vs arc-length)
  - <stem>_kappa_vs_s.png                  (κ(s) vs arc-length)
  - <stem>_xy_from_theta.png               (XY curve reconstructed from θ(s))

Usage examples:
  # 1) Use defaults written in this file
  python predict_curvature.py

  # 2) Override from CLI
  python predict_curvature.py --ratio 5 --temp 30
  python predict_curvature.py --ratio_str 3vs1 --temp 40 --out_dir outputs/predictions_run2 --stem custom_r3_T40
  python predict_curvature.py --ratio 5 --temp 30 --out_dir outputs/predictions --stem r5_T30
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Default settings you can edit ----------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

# Default inputs for prediction
DEFAULT_RATIO    = 6.0        # r = PNIPAM:PDMS front term (e.g., 5 means 5:1)
DEFAULT_TEMP     = 60.0       # Temperature in °C

# Unified folder to put ALL prediction outputs (CSV + PNGs)
DEFAULT_PRED_DIR = OUT_DIR / "predictions"   # <- change if you want

# Optional default file name stem; if None, auto-generated as r{ratio}_T{temp}
DEFAULT_FILE_STEM = None
# ---------------------------------------------------------------

# Artifacts produced by the training step (keep as-is unless you changed output folder in training)
PCA_PATH    = OUT_DIR / "curvature_pca.joblib"
SCORE_PATH  = OUT_DIR / "curvature_reg.joblib"
LREG_PATH   = OUT_DIR / "length_model.joblib"
META_PATH   = OUT_DIR / "curvature_meta.json"

_ratio_re = re.compile(r"(\d+)\s*vs\s*(\d+)", re.IGNORECASE)


def _parse_ratio(ratio: float | None, ratio_str: str | None) -> float:
    """
    Resolve numeric ratio:
      - If ratio_str like '5vs1' is provided, it takes precedence.
      - Else use ratio numeric value.
      - If both are missing, fall back to DEFAULT_RATIO.
    """
    if ratio_str:
        m = _ratio_re.search(ratio_str)
        if not m:
            raise ValueError("ratio_str must look like '5vs1' (e.g., 3vs1, 5vs1)")
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            raise ValueError("ratio denominator cannot be 0")
        return a / b
    if ratio is not None:
        return float(ratio)
    return float(DEFAULT_RATIO)


def _load_artifacts():
    """
    Load curvature-shape artifacts produced by bend_model_pipeline.py
    (curvature_pca.joblib, curvature_reg.joblib, length_model.joblib).
    """
    if not PCA_PATH.exists() or not SCORE_PATH.exists() or not LREG_PATH.exists():
        raise FileNotFoundError(
            "Curvature-shape artifacts not found.\n"
            "Train them first by running bend_model_pipeline.py "
            "(with the curvature-shape training included)."
        )
    pca   = joblib.load(PCA_PATH)
    reg   = joblib.load(SCORE_PATH)
    lreg  = joblib.load(LREG_PATH)
    meta  = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    n_pts = int(meta.get("n_points", 200))
    return pca, reg, lreg, n_pts


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral (numpy-only), initial value 0."""
    if len(y) < 2:
        return np.zeros_like(y)
    dx = np.diff(x)
    mid = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y, float)
    out[1:] = np.cumsum(mid)
    return out


def _reconstruct_xy_from_theta(s_abs: np.ndarray, theta_deg: np.ndarray):
    """
    Reconstruct XY curve from θ(s):
      dx = ds * cos(theta), dy = ds * sin(theta), starting at (0,0).
    """
    theta_rad = np.radians(theta_deg)
    ds = np.diff(s_abs, prepend=s_abs[0])
    dx = ds * np.cos(theta_rad)
    dy = ds * np.sin(theta_rad)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return x, y


def _plot_theta_vs_s(s_abs: np.ndarray, theta_deg: np.ndarray, save_path: Path):
    plt.figure()
    plt.plot(s_abs, theta_deg, linewidth=1.5)
    plt.xlabel("Arc length s (µm)")
    plt.ylabel("Tangent angle θ (deg)")
    plt.title("Tangent angle θ(s) from predicted curvature κ(s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_kappa_vs_s(s_abs: np.ndarray, kappa: np.ndarray, save_path: Path):
    plt.figure()
    plt.plot(s_abs, kappa, linewidth=1.5)
    plt.xlabel("Arc length s (µm)")
    plt.ylabel("Curvature κ (1/µm)")
    plt.title("Predicted curvature κ(s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_xy(x: np.ndarray, y: np.ndarray, save_path: Path):
    plt.figure()
    plt.plot(x, y, linewidth=2.0, label="Reconstructed from θ(s)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Reconstructed XY curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def predict_curvature_and_save_all(ratio: float, temp_c: float, pred_dir: Path, file_stem: str | None = None):
    """
    Predict the full curvature profile and save CSV + PNGs into pred_dir with a consistent stem.
    Returns a dict with all output paths.
    """
    pca, reg, lreg, n_pts = _load_artifacts()

    # Prepare outputs dir and naming
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not file_stem:
        file_stem = f"r{ratio:.3g}_T{temp_c:.3g}"

    csv_path   = pred_dir / f"{file_stem}_curvature_prediction.csv"
    theta_png  = pred_dir / f"{file_stem}_theta_vs_s.png"
    kappa_png  = pred_dir / f"{file_stem}_kappa_vs_s.png"
    xy_png     = pred_dir / f"{file_stem}_xy_from_theta.png"

    # Features expected by the training pipeline: [ratio, ratio_frac, temp_C]
    X = np.array([[ratio, ratio / (1.0 + ratio), temp_c]], dtype=float)

    # 1) Predict PCA scores and reconstruct curvature on normalized arc
    Z = reg.predict(X)                      # shape (1, n_components)
    kappa_rs = pca.inverse_transform(Z)[0]  # shape (n_points,)

    # 2) Predict total arc-length to de-normalize s
    L = float(lreg.predict(X)[0])
    s_norm = np.linspace(0.0, 1.0, n_pts)
    s_abs  = s_norm * max(L, 1e-9)

    # 3) Integrate curvature to get theta(s) and the final tip angle
    theta_rad = _cumtrapz(kappa_rs, s_abs)
    theta_deg = np.degrees(theta_rad)

    # 4) Save CSV
    df = pd.DataFrame({
        "s_norm": s_norm,
        "s_abs": s_abs,
        "kappa_pred": kappa_rs,
        "theta_pred_deg": theta_deg
    })
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 5) Save figures
    _plot_theta_vs_s(s_abs, theta_deg, theta_png)
    _plot_kappa_vs_s(s_abs, kappa_rs, kappa_png)
    x_rec, y_rec = _reconstruct_xy_from_theta(s_abs, theta_deg)
    _plot_xy(x_rec, y_rec, xy_png)

    print("[OK] Saved files:")
    print(" CSV :", csv_path)
    print(" IMG :", theta_png)
    print(" IMG :", kappa_png)
    print(" IMG :", xy_png)

    return {
        "csv": csv_path,
        "theta_png": theta_png,
        "kappa_png": kappa_png,
        "xy_png": xy_png,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict curvature CSV + plots from (ratio, temp_C), saving ALL into a unified folder."
    )
    # Defaults so running with no args just works (Method A)
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                        help="PNIPAM:PDMS r (e.g., 5 for 5:1). Ignored if --ratio_str is provided.")
    parser.add_argument("--ratio_str", type=str, default=None,
                        help="e.g., '5vs1' (takes precedence over --ratio if given).")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP,
                        help="Temperature in °C (default uses DEFAULT_TEMP).")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_PRED_DIR),
                        help="Folder to save ALL outputs (CSV + PNGs). Default: ./outputs/predictions/")
    parser.add_argument("--stem", type=str, default=DEFAULT_FILE_STEM,
                        help="File name stem for outputs (e.g., 'r5_T30'). If None, auto-generated.")

    args = parser.parse_args()

    # Resolve inputs
    r = _parse_ratio(args.ratio, args.ratio_str)
    t = float(args.temp)

    # Resolve output folder and stem
    pred_dir = Path(args.out_dir)
    stem = args.stem if args.stem else None

    # Predict and save
    predict_curvature_and_save_all(r, t, pred_dir, stem)


if __name__ == "__main__":
    main()
