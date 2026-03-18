#!/usr/bin/env python3
"""
Real Robot — Single Model Error Log Analysis
=============================================
Reads:
  <LOG_DIR>/case_<k>_trajectory.csv   (per-step, prefix="trajectory")
  <LOG_DIR>/case_<k>_final_gt.csv     (single row, prefix="final_gt")

Outputs to:
  <LOG_DIR>/analysis/
    csv/
    per_case/
      error_trends/
      point_errors/
      missing_count/
      control_effort/
    cross_case/
      final_gt/
      trajectory/
    heatmaps/
    notes/
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# CONFIG — edit these two lines only
# ─────────────────────────────────────────────
LOG_DIR    = r"<SET_THIS_TO_YOUR_LOG_DIR>"   # e.g. results/real/logs/rbf/2D
MODEL_NAME = "rbf"                            # label used in chart titles
OUT_DIR = r"<SET_THIS_TO_YOUR_OUTPUT_DIR>"        # e.g. results/real/analysis/rbf/2D

# ─────────────────────────────────────────────
# Thresholds for settling / success
# ─────────────────────────────────────────────
THRESHOLD_M = 0.05      # 5 cm
DWELL_S     = 1.0       # must stay below threshold for 1 s


# ═════════════════════════════════════════════
# Directory setup
# ═════════════════════════════════════════════

def make_dirs(base):
    dirs = {
        "root":            base,
        "csv":             os.path.join(base, "csv"),
        "notes":           os.path.join(base, "notes"),
        "error_trends":    os.path.join(base, "per_case", "error_trends"),
        "point_errors":    os.path.join(base, "per_case", "point_errors"),
        "missing_count":   os.path.join(base, "per_case", "missing_count"),
        "control_effort":  os.path.join(base, "per_case", "control_effort"),
        "cross_final":     os.path.join(base, "cross_case", "final_gt"),
        "cross_traj":      os.path.join(base, "cross_case", "trajectory"),
        "heatmaps":        os.path.join(base, "heatmaps"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ═════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════

def meta(png, caption, desc=""):
    with open(png + ".meta.json", "w") as f:
        json.dump({"caption": caption, "description": desc}, f)


def settling_time(t, e, threshold=THRESHOLD_M, dwell=DWELL_S):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    for i in range(len(t)):
        if not np.isfinite(e[i]) or e[i] > threshold:
            continue
        t0, ok = t[i], True
        j = i
        while j < len(t) and (t[j] - t0) <= dwell:
            if not np.isfinite(e[j]) or e[j] > threshold:
                ok = False
                break
            j += 1
        if ok and j < len(t):
            return float(t0)
    return np.nan


def first_hit(t, e, threshold=THRESHOLD_M):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    idx = np.where(np.isfinite(e) & (e < threshold))[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def auc(t, e):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    mask = np.isfinite(t) & np.isfinite(e)
    return float(np.trapz(e[mask], t[mask])) if mask.sum() > 1 else np.nan


def line(df, x, y, color, title, xlab, ylab, png, threshold_line=None):
    fig = px.line(df, x=x, y=y, color=color)
    if threshold_line is not None:
        fig.add_hline(y=threshold_line, line_dash="dot",
                      line_color="red", annotation_text=f"threshold {threshold_line}m")
    fig.update_layout(
        title={"text": title},
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(png)
    meta(png, os.path.basename(png).replace(".png", ""), title)


def bar(df, x, y, color, title, xlab, ylab, png):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group")
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        title={"text": title},
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(png)
    meta(png, os.path.basename(png).replace(".png", ""), title)


def heatmap(mat_df, title, png):
    fig = px.imshow(mat_df, aspect='auto', color_continuous_scale='Viridis')
    fig.update_layout(title={"text": title})
    fig.update_xaxes(title_text="Feature pt")
    fig.update_yaxes(title_text="Case")
    fig.write_image(png)
    meta(png, os.path.basename(png).replace(".png", ""), title)


# ═════════════════════════════════════════════
# Load logs
# ═════════════════════════════════════════════

def load_trajectory_logs(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, "case_*_trajectory.csv")))
    if not files:
        raise FileNotFoundError(f"No trajectory CSVs found in {log_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                print(f"  [WARN] Empty: {f}")
                continue
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Skipping {f}: {e}")
    if not dfs:
        raise ValueError("All trajectory CSVs were empty or unreadable.")
    return pd.concat(dfs, ignore_index=True)


def load_final_gt_logs(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, "case_*_final_gt.csv")))
    if not files:
        print("  [INFO] No final_gt CSVs found — skipping final GT charts.")
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Skipping {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ═════════════════════════════════════════════
# Log validation
# ═════════════════════════════════════════════

REQUIRED_TRAJ_COLS = [
    "t", "case", "model", "prefix",
    "missing_count", "n_visible",
    "rmse_visible", "mean_visible",
    "rmse_all", "mean_err", "max_err",
    "inner_mean_err", "endpoint_mean_err",
]
REQUIRED_GT_COLS = [
    "t", "case", "model", "prefix",
    "rmse_all", "inner_mean_err", "max_err",
]

def validate(df, required, label):
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [WARN] {label} missing columns: {missing}")
    else:
        print(f"  [OK]  {label} — all required columns present")
    return df


# ═════════════════════════════════════════════
# Summary builders
# ═════════════════════════════════════════════

def build_traj_summary(df):
    rows = []
    for case_id, g in df.groupby("case"):
        g = g.sort_values("t").reset_index(drop=True)
        metric = g["rmse_visible"] if "rmse_visible" in g.columns else g["mean_visible"]
        end_w  = max(1, min(10, len(g)))
        rows.append({
            "case":            int(case_id),
            "model":           g["model"].iloc[0] if "model" in g.columns else MODEL_NAME,
            "duration_s":      float(g["t"].max()),
            "samples":         len(g),
            "missing_avg":     float(g["missing_count"].mean()) if "missing_count" in g.columns else np.nan,
            "n_visible_avg":   float(g["n_visible"].mean()) if "n_visible" in g.columns else np.nan,
            "rmse_vis_avg":    float(g["rmse_visible"].mean()) if "rmse_visible" in g.columns else np.nan,
            "rmse_vis_final":  float(g["rmse_visible"].iloc[-1]) if "rmse_visible" in g.columns else np.nan,
            "inner_avg":       float(g["inner_mean_err"].mean()) if "inner_mean_err" in g.columns else np.nan,
            "inner_final":     float(g["inner_mean_err"].iloc[-1]) if "inner_mean_err" in g.columns else np.nan,
            "max_err_peak":    float(g["max_err"].max()) if "max_err" in g.columns else np.nan,
            "auc_rmse_vis":    auc(g["t"], g["rmse_visible"]) if "rmse_visible" in g.columns else np.nan,
            "first_hit_s":     first_hit(g["t"], metric),
            "settling_s":      settling_time(g["t"], metric),
            "success":         bool(np.all(metric.tail(end_w) < THRESHOLD_M)),
        })
    return pd.DataFrame(rows)


def build_gt_summary(df):
    rows = []
    for case_id, g in df.groupby("case"):
        rows.append({
            "case":           int(case_id),
            "model":          g["model"].iloc[0] if "model" in g.columns else MODEL_NAME,
            "rmse_all":       float(g["rmse_all"].iloc[0]),
            "inner_mean_err": float(g["inner_mean_err"].iloc[0]),
            "max_err":        float(g["max_err"].iloc[0]),
            "mean_err":       float(g["mean_err"].iloc[0]) if "mean_err" in g.columns else np.nan,
            "endpoint_err":   float(g["endpoint_mean_err"].iloc[0]) if "endpoint_mean_err" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════
# Per-case charts
# ═════════════════════════════════════════════

def per_case_charts(df, dirs):
    for case_id, g in df.groupby("case"):
        g   = g.sort_values("t").reset_index(drop=True)
        tag = f"case_{int(case_id)}"

        # ── 1. Error trends (summary metrics) ────────────────────────────
        metric_cols = {
            "rmse_visible":    "RMSE vis",
            "mean_visible":    "Mean vis",
            "inner_mean_err":  "Inner err",
            "max_err":         "Max err",
        }
        avail = {k: v for k, v in metric_cols.items() if k in g.columns}
        if avail:
            long = g[["t"] + list(avail.keys())].melt(
                id_vars="t", var_name="metric", value_name="value"
            )
            long["metric"] = long["metric"].map(avail)
            line(
                long, "t", "value", "metric",
                f'Error trends ({tag}, {MODEL_NAME})<br>'
                f'<span style="font-size:18px;font-weight:normal;">Visible-point metrics over time</span>',
                "Time (s)", "Error (m)",
                os.path.join(dirs["error_trends"], f"{tag}_error_trends.png"),
                threshold_line=THRESHOLD_M
            )

        # ── 2. Per-point error curves ─────────────────────────────────────
        pt_cols = [f"err_{i}_norm" for i in range(10) if f"err_{i}_norm" in g.columns]
        if pt_cols:
            # Apply is_visible mask where available
            vis_cols = [f"is_visible_{i}" for i in range(10)]
            long_pts = g[["t"] + pt_cols].melt(
                id_vars="t", var_name="point", value_name="value"
            )
            long_pts["pt_idx"] = long_pts["point"].str.extract(r"err_(\d+)_norm").astype(int)

            # Mask occluded values so they don't distort the chart
            if all(c in g.columns for c in vis_cols):
                vis_long = g[["t"] + vis_cols].melt(
                    id_vars="t", var_name="vis_col", value_name="is_vis"
                )
                vis_long["pt_idx"] = vis_long["vis_col"].str.extract(r"is_visible_(\d+)").astype(int)
                vis_long = vis_long[["t", "pt_idx", "is_vis"]]
                long_pts = long_pts.merge(vis_long, on=["t", "pt_idx"], how="left")
                long_pts.loc[long_pts["is_vis"] == 0, "value"] = np.nan

            long_pts["point"] = "p" + long_pts["pt_idx"].astype(str)
            line(
                long_pts, "t", "value", "point",
                f'Per-point errors ({tag}, {MODEL_NAME})<br>'
                f'<span style="font-size:18px;font-weight:normal;">Occluded points masked out</span>',
                "Time (s)", "Error (m)",
                os.path.join(dirs["point_errors"], f"{tag}_point_errors.png")
            )

        # ── 3. Missing count ──────────────────────────────────────────────
        if "missing_count" in g.columns:
            mc = g[["t", "missing_count"]].copy()
            mc["series"] = "missing"
            line(
                mc, "t", "missing_count", "series",
                f'Missing detections ({tag}, {MODEL_NAME})<br>'
                f'<span style="font-size:18px;font-weight:normal;">Vision dropouts over time</span>',
                "Time (s)", "Count",
                os.path.join(dirs["missing_count"], f"{tag}_missing_count.png")
            )

        # ── 4. Control effort ─────────────────────────────────────────────
        u_cols = [f"u_{j}" for j in range(6, 12) if f"u_{j}" in g.columns]
        if u_cols:
            g["u_norm"] = np.linalg.norm(g[u_cols].values, axis=1)
            eff = g[["t", "u_norm"]].copy()
            eff["series"] = "‖u‖"
            line(
                eff, "t", "u_norm", "series",
                f'Control effort ({tag}, {MODEL_NAME})<br>'
                f'<span style="font-size:18px;font-weight:normal;">Right arm velocity norm over time</span>',
                "Time (s)", "‖u‖ (rad/s)",
                os.path.join(dirs["control_effort"], f"{tag}_control_effort.png")
            )


# ═════════════════════════════════════════════
# Cross-case charts — trajectory
# ═════════════════════════════════════════════

def cross_case_traj_charts(summary, dirs):
    s = summary.sort_values("case")

    bar(s, "case", "rmse_vis_avg", None,
        f'Avg visible RMSE per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Lower = better tracking</span>',
        "Case", "Avg RMSE",
        os.path.join(dirs["cross_traj"], "cross_rmse_vis_avg.png"))

    bar(s, "case", "inner_avg", None,
        f'Avg inner error per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Shape quality during run</span>',
        "Case", "Inner err",
        os.path.join(dirs["cross_traj"], "cross_inner_avg.png"))

    bar(s, "case", "settling_s", None,
        f'Settling time per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">First stable crossing of {THRESHOLD_M}m</span>',
        "Case", "Settle (s)",
        os.path.join(dirs["cross_traj"], "cross_settling.png"))

    bar(s, "case", "auc_rmse_vis", None,
        f'Error area per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Transient + steady error combined</span>',
        "Case", "Err area",
        os.path.join(dirs["cross_traj"], "cross_auc.png"))

    bar(s, "case", "missing_avg", None,
        f'Avg missing detections per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Higher = harder occlusion</span>',
        "Case", "Avg missing",
        os.path.join(dirs["cross_traj"], "cross_missing_avg.png"))

    # Success rate as single bar
    success_df = pd.DataFrame({
        "model": [MODEL_NAME],
        "success_rate": [float(summary["success"].mean()) * 100]
    })
    bar(success_df, "model", "success_rate", None,
        f'Success rate ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">% cases below {THRESHOLD_M}m threshold</span>',
        "Model", "Success (%)",
        os.path.join(dirs["cross_traj"], "cross_success_rate.png"))


# ═════════════════════════════════════════════
# Cross-case charts — final GT
# ═════════════════════════════════════════════

def cross_case_gt_charts(gt_summary, dirs):
    s = gt_summary.sort_values("case")

    bar(s, "case", "rmse_all", None,
        f'Final true RMSE per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">All 10 pts after occlusion removed</span>',
        "Case", "RMSE (m)",
        os.path.join(dirs["cross_final"], "final_rmse_all.png"))

    bar(s, "case", "inner_mean_err", None,
        f'Final inner error per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">True shape quality at end of run</span>',
        "Case", "Inner err",
        os.path.join(dirs["cross_final"], "final_inner_err.png"))

    bar(s, "case", "max_err", None,
        f'Final max error per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Worst single point at end of run</span>',
        "Case", "Max err",
        os.path.join(dirs["cross_final"], "final_max_err.png"))

    bar(s, "case", "endpoint_err", None,
        f'Final endpoint error per case ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Endpoint accuracy at end of run</span>',
        "Case", "Endpt err",
        os.path.join(dirs["cross_final"], "final_endpoint_err.png"))


# ═════════════════════════════════════════════
# Heatmaps
# ═════════════════════════════════════════════

def heatmap_charts(df, dirs):
    pt_cols = [f"err_{i}_norm" for i in range(10) if f"err_{i}_norm" in df.columns]
    if not pt_cols:
        return

    # Mean per-point error per case (only visible timesteps)
    rows = []
    for case_id, g in df.groupby("case"):
        row = {"case": int(case_id)}
        for i, col in enumerate(pt_cols):
            pt_idx = i
            vis_col = f"is_visible_{pt_idx}"
            if vis_col in g.columns:
                vals = g.loc[g[vis_col] == 1, col]
            else:
                vals = g[col]
            row[f"p{pt_idx}"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)

    mat = pd.DataFrame(rows).set_index("case")
    heatmap(
        mat,
        f'Per-point mean error heatmap ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Visible steps only — cases × points</span>',
        os.path.join(dirs["heatmaps"], "heatmap_point_mean_error.png")
    )

    # Final per-point error per case (from trajectory last timestep)
    rows_final = []
    for case_id, g in df.groupby("case"):
        g = g.sort_values("t")
        last = g.iloc[-1]
        row = {"case": int(case_id)}
        for i in range(len(pt_cols)):
            col = f"err_{i}_norm"
            row[f"p{i}"] = float(last[col]) if col in last else np.nan
        rows_final.append(row)

    mat_final = pd.DataFrame(rows_final).set_index("case")
    heatmap(
        mat_final,
        f'Per-point final error heatmap ({MODEL_NAME})<br>'
        f'<span style="font-size:18px;font-weight:normal;">Error at last timestep — cases × points</span>',
        os.path.join(dirs["heatmaps"], "heatmap_point_final_error.png")
    )


# ═════════════════════════════════════════════
# Distribution charts
# ═════════════════════════════════════════════

def distribution_charts(traj_summary, gt_summary, dirs):

    # Violin: final inner error from GT
    if not gt_summary.empty and "inner_mean_err" in gt_summary.columns:
        fig = px.violin(gt_summary, y="inner_mean_err", box=True, points="all")
        fig.update_layout(
            title={
                "text": f'Final inner error distribution ({MODEL_NAME})<br>'
                        f'<span style="font-size:18px;font-weight:normal;">All cases — true shape quality</span>'
            }
        )
        fig.update_xaxes(title_text="Model")
        fig.update_yaxes(title_text="Inner err (m)")
        png = os.path.join(dirs["cross_final"], "dist_final_inner_violin.png")
        fig.write_image(png)
        meta(png, "Final inner error violin", "Distribution of final inner error across cases")

    # CDF: final RMSE from GT
    if not gt_summary.empty and "rmse_all" in gt_summary.columns:
        vals = np.sort(gt_summary["rmse_all"].dropna().values)
        cdf  = np.arange(1, len(vals) + 1) / len(vals) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vals, y=cdf, mode='lines',
                                 fill='tozeroy', fillcolor='rgba(99,110,250,0.15)',
                                 line=dict(width=2)))
        fig.add_vline(x=THRESHOLD_M, line_dash="dot",
                      line_color="red", annotation_text=f"{THRESHOLD_M}m")
        fig.update_layout(
            title={
                "text": f'CDF of final RMSE ({MODEL_NAME})<br>'
                        f'<span style="font-size:18px;font-weight:normal;">% of cases below each RMSE value</span>'
            },
            legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
        )
        fig.update_xaxes(title_text="RMSE (m)")
        fig.update_yaxes(title_text="Cases (%)")
        png = os.path.join(dirs["cross_final"], "cdf_final_rmse.png")
        fig.write_image(png)
        meta(png, "CDF of final RMSE", "Cumulative distribution of final RMSE across cases")

    # Violin: settling time from trajectory
    if "settling_s" in traj_summary.columns:
        fig = px.violin(traj_summary.dropna(subset=["settling_s"]),
                        y="settling_s", box=True, points="all")
        fig.update_layout(
            title={
                "text": f'Settling time distribution ({MODEL_NAME})<br>'
                        f'<span style="font-size:18px;font-weight:normal;">All cases — time to converge</span>'
            }
        )
        fig.update_xaxes(title_text="Model")
        fig.update_yaxes(title_text="Settle (s)")
        png = os.path.join(dirs["cross_traj"], "dist_settling_violin.png")
        fig.write_image(png)
        meta(png, "Settling time violin", "Distribution of settling times across cases")


# ═════════════════════════════════════════════
# Save notes
# ═════════════════════════════════════════════

def save_notes(dirs, n_traj_cases, n_gt_cases):
    notes = {
        "model": MODEL_NAME,
        "threshold_m": THRESHOLD_M,
        "dwell_s": DWELL_S,
        "n_trajectory_cases": n_traj_cases,
        "n_final_gt_cases": n_gt_cases,
        "metric_notes": {
            "rmse_visible":    "RMSE over visible points only — use for trajectory plots",
            "inner_mean_err":  "Mean error pts 1-8 — best shape quality proxy",
            "rmse_all":        "RMSE all 10 pts — reliable ONLY for final_gt rows",
            "settling_s":      f"First time metric < {THRESHOLD_M}m, stays for {DWELL_S}s",
            "auc_rmse_vis":    "Integral of rmse_visible over time — combined transient+steady",
            "is_visible_i":    "1=visible, 0=occluded — use to mask per-point charts",
        }
    }
    with open(os.path.join(dirs["notes"], "analysis_notes.json"), "w") as f:
        json.dump(notes, f, indent=2)
    print(f"  Notes saved.")


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════

def main():
    OUT_DIR = OUT_DIR
    dirs    = make_dirs(OUT_DIR)

    print(f"\n{'='*55}")
    print(f" Single Model Analysis — {MODEL_NAME}")
    print(f" Log dir : {LOG_DIR}")
    print(f" Out dir : {OUT_DIR}")
    print(f"{'='*55}\n")

    # ── Load ──────────────────────────────────────────────────
    print("[1/6] Loading trajectory logs...")
    traj_df = load_trajectory_logs(LOG_DIR)
    traj_df = validate(traj_df, REQUIRED_TRAJ_COLS, "trajectory")
    print(f"      {len(traj_df)} rows, {traj_df['case'].nunique()} cases\n")

    print("[2/6] Loading final GT logs...")
    gt_df = load_final_gt_logs(LOG_DIR)
    if not gt_df.empty:
        gt_df = validate(gt_df, REQUIRED_GT_COLS, "final_gt")
        print(f"      {len(gt_df)} rows, {gt_df['case'].nunique()} cases\n")
    else:
        print("      No final_gt files — skipping GT charts\n")

    # ── Summaries ─────────────────────────────────────────────
    print("[3/6] Building summaries...")
    traj_summary = build_traj_summary(traj_df)
    traj_summary.to_csv(os.path.join(dirs["csv"], "traj_case_summary.csv"), index=False)

    if not gt_df.empty:
        gt_summary = build_gt_summary(gt_df)
        gt_summary.to_csv(os.path.join(dirs["csv"], "final_gt_summary.csv"), index=False)
    else:
        gt_summary = pd.DataFrame()

    # Combined headline metrics
    headline = {
        "model":              MODEL_NAME,
        "n_cases":            int(traj_summary["case"].nunique()),
        "success_rate_pct":   float(traj_summary["success"].mean() * 100),
        "rmse_vis_avg_mean":  float(traj_summary["rmse_vis_avg"].mean()),
        "inner_avg_mean":     float(traj_summary["inner_avg"].mean()),
        "settling_mean_s":    float(traj_summary["settling_s"].mean()),
        "auc_mean":           float(traj_summary["auc_rmse_vis"].mean()),
    }
    if not gt_summary.empty:
        headline["final_rmse_all_mean"]  = float(gt_summary["rmse_all"].mean())
        headline["final_inner_err_mean"] = float(gt_summary["inner_mean_err"].mean())
        headline["final_max_err_mean"]   = float(gt_summary["max_err"].mean())

    pd.DataFrame([headline]).to_csv(
        os.path.join(dirs["csv"], "headline_metrics.csv"), index=False
    )
    print(f"      Summaries saved to {dirs['csv']}\n")

    # ── Charts ────────────────────────────────────────────────
    print("[4/6] Generating per-case charts...")
    per_case_charts(traj_df, dirs)
    print(f"      Done — {traj_df['case'].nunique()} cases processed\n")

    print("[5/6] Generating cross-case charts...")
    cross_case_traj_charts(traj_summary, dirs)
    if not gt_summary.empty:
        cross_case_gt_charts(gt_summary, dirs)
    distribution_charts(traj_summary, gt_summary, dirs)
    heatmap_charts(traj_df, dirs)
    print("      Done\n")

    print("[6/6] Saving notes...")
    save_notes(dirs, traj_df["case"].nunique(),
               gt_df["case"].nunique() if not gt_df.empty else 0)

    # ── Print headline ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f" HEADLINE METRICS — {MODEL_NAME}")
    print(f"{'='*55}")
    for k, v in headline.items():
        if k != "model":
            print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
    print(f"\n Analysis complete → {OUT_DIR}\n")


if __name__ == "__main__":
    main()
