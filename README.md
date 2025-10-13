# kappa-bend-model

End-to-end pipeline that learns **tip angle** and **full curvature shape κ(s)** from ImageJ **Kappa** data, then integrates to **θ(s)** and reconstructs **XY** for thin-strip bending.

**Quick links:** [Install](#install) · [Data](#data) · [Train](#train) · [Predict](#predict) · [Outputs](#outputs) · [Examples](#examples)

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/2a40a26d-fded-4935-b0f8-4e05ad3f58ac" alt="Curvature map example (20 °C)" width="420">
  <br/>
  <sub>Curvature map example (20 °C)</sub>
</div>

---

## What it does

- Parses curvature CSVs exported from ImageJ/Fiji **Kappa**.
- Learns **θ_tip** regression and a **PCA + regression** model for full κ(s).
- Integrates κ(s) → θ(s) and reconstructs XY.
- Exports metrics and publication-ready figures.

## Data

- Place training CSVs under `train/` (and optional test CSVs under `test/`).
- Recommended columns: `Point Curvature (um-1)`, `Point Curvature Sign`, `X-Coordinate`, `Y-Coordinate`.
- Folder/file names should include tokens like `5vs1` (ratio 5:1) and `20C` (temperature 20 °C).

## Examples

<table>
  <tr>
    <td align="center">
      <img src="examples/r5_T20_xy_from_theta.png" alt="r5_T20_xy_from_theta" width="180"><br/>
      <sub>ratio = 5:1 · temp = 20 °C</sub>
    </td>
    <td align="center">
      <img src="examples/r5_T30_xy_from_theta.png" alt="r5_T30_xy_from_theta" width="180"><br/>
      <sub>ratio = 5:1 · temp = 30 °C</sub>
    </td>
    <td align="center">
      <img src="examples/r5_T40_xy_from_theta.png" alt="r5_T40_xy_from_theta" width="180"><br/>
      <sub>ratio = 5:1 · temp = 40 °C</sub>
    </td>
    <td align="center">
      <img src="examples/r5_T50_xy_from_theta.png" alt="r5_T50_xy_from_theta" width="180"><br/>
      <sub>ratio = 5:1 · temp = 50 °C</sub>
    </td>
    <td align="center">
      <img src="examples/r5_T60_xy_from_theta.png" alt="r5_T60_xy_from_theta" width="180"><br/>
      <sub>ratio = 5:1 · temp = 60 °C</sub>
    </td>
  </tr>
</table>
