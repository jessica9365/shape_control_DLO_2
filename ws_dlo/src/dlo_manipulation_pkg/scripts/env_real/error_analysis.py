#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import pandas as pd
import plotly.express as px

#

BASE_LOG_DIR = r"C:/Users/91990/Documents/GitHub/FYP_Object_Detection_Model/shape_control_DLO_2/results/real/logs/gnn_v1/2D"
OUT_DIR = os.path.join(BASE_LOG_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)

ERROR_COLS = [f"err_{i}_norm" for i in range(10)]
SUMMARY_COLS = [
    "t", "case", "model",
    "rmse_all", "mean_err", "inner_mean_err",
    "endpoint_mean_err", "max_err", "missing_count"
]

SETTLING_THRESHOLD_M = 0.02   # 2 cm
SETTLING_DWELL_S = 1.0        # remain below threshold for 1 s


def save_meta(png_name, caption, description=""):
    with open(png_name + ".meta.json", "w") as f:
        json.dump({"caption": caption, "description": description}, f)


def load_logs(base_dir):
    files = sorted(glob.glob(os.path.join(base_dir, "*.csv")))
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No CSV logs found in {base_dir}")

    return pd.concat(dfs, ignore_index=True)


def settling_time(series_t, series_e, threshold=0.02, dwell_s=1.0):
    t = np.asarray(series_t, dtype=float)
    e = np.asarray(series_e, dtype=float)

    if len(t) < 2:
        return np.nan

    for i in range(len(t)):
        if e[i] > threshold:
            continue

        t0 = t[i]
        j = i
        stable = True

        while j < len(t) and (t[j] - t0) <= dwell_s:
            if e[j] > threshold:
                stable = False
                break
            j += 1

        if stable and j < len(t):
            return float(t0)

    return np.nan


def save_line_chart(df, x, y, color, title, y_label, out_png):
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(
        title={
            "text": title
        },
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5
        )
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text=y_label)
    fig.write_image(out_png)
    save_meta(out_png, os.path.basename(out_png).replace(".png", ""), title)


def save_bar_chart(df, x, y, color, title, x_label, y_label, out_png):
    fig = px.bar(df, x=x, y=y, color=color)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        title={
            "text": title
        },
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5
        )
    )
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.write_image(out_png)
    save_meta(out_png, os.path.basename(out_png).replace(".png", ""), title)


def main():
    df = load_logs(BASE_LOG_DIR)

    keep_cols = [c for c in SUMMARY_COLS + ERROR_COLS if c in df.columns]
    df = df[keep_cols + [c for c in ["source_file"] if c in df.columns]].copy()

    case_rows = []
    for case_id, g in df.groupby("case"):
        g = g.sort_values("t")

        metric_series = g["inner_mean_err"] if "inner_mean_err" in g.columns else g["mean_err"]

        case_rows.append({
            "case": int(case_id),
            "model": g["model"].iloc[0] if "model" in g.columns else "unknown",
            "samples": len(g),
            "duration_s": float(g["t"].max()),
            "rmse_all_avg": float(g["rmse_all"].mean()) if "rmse_all" in g.columns else np.nan,
            "rmse_all_final": float(g["rmse_all"].iloc[-1]) if "rmse_all" in g.columns else np.nan,
            "mean_err_avg": float(g["mean_err"].mean()) if "mean_err" in g.columns else np.nan,
            "mean_err_final": float(g["mean_err"].iloc[-1]) if "mean_err" in g.columns else np.nan,
            "inner_mean_avg": float(g["inner_mean_err"].mean()) if "inner_mean_err" in g.columns else np.nan,
            "inner_mean_final": float(g["inner_mean_err"].iloc[-1]) if "inner_mean_err" in g.columns else np.nan,
            "endpoint_mean_avg": float(g["endpoint_mean_err"].mean()) if "endpoint_mean_err" in g.columns else np.nan,
            "max_err_peak": float(g["max_err"].max()) if "max_err" in g.columns else np.nan,
            "missing_avg": float(g["missing_count"].mean()) if "missing_count" in g.columns else np.nan,
            "settling_time_s": settling_time(
                g["t"], metric_series,
                threshold=SETTLING_THRESHOLD_M,
                dwell_s=SETTLING_DWELL_S
            )
        })

    case_summary = pd.DataFrame(case_rows)
    case_summary.to_csv(os.path.join(OUT_DIR, "case_summary.csv"), index=False)

    model_summary = (
        case_summary.groupby("model", dropna=False)
        .agg({
            "case": "count",
            "rmse_all_avg": "mean",
            "rmse_all_final": "mean",
            "mean_err_avg": "mean",
            "mean_err_final": "mean",
            "inner_mean_avg": "mean",
            "inner_mean_final": "mean",
            "endpoint_mean_avg": "mean",
            "max_err_peak": "mean",
            "missing_avg": "mean",
            "settling_time_s": "mean"
        })
        .rename(columns={"case": "n_cases"})
        .reset_index()
    )
    model_summary.to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)

    # Per-case charts
    for case_id, g in df.groupby("case"):
        g = g.sort_values("t")
        case_tag = f"case_{int(case_id)}"

        metric_cols = [c for c in ["mean_err", "inner_mean_err", "endpoint_mean_err", "max_err"] if c in g.columns]
        if metric_cols:
            long_df = g[["t"] + metric_cols].melt(id_vars="t", var_name="metric", value_name="value")
            save_line_chart(
                long_df,
                x="t",
                y="value",
                color="metric",
                title=f"Error trends ({case_tag})<br><span style='font-size:18px;font-weight:normal;'>Summary feature errors over time</span>",
                y_label="Error (m)",
                out_png=os.path.join(OUT_DIR, f"{case_tag}_error_trends.png")
            )

        point_cols = [c for c in ERROR_COLS if c in g.columns]
        if point_cols:
            long_df = g[["t"] + point_cols].melt(id_vars="t", var_name="point", value_name="value")
            save_line_chart(
                long_df,
                x="t",
                y="value",
                color="point",
                title=f"Per-point errors ({case_tag})<br><span style='font-size:18px;font-weight:normal;'>Feature-point error norms over time</span>",
                y_label="Error (m)",
                out_png=os.path.join(OUT_DIR, f"{case_tag}_point_errors.png")
            )

        if "missing_count" in g.columns:
            fig_df = g[["t", "missing_count"]].copy()
            fig = px.line(fig_df, x="t", y="missing_count")
            fig.update_layout(
                title={
                    "text": f"Missing detections ({case_tag})<br><span style='font-size:18px;font-weight:normal;'>Vision dropouts over time</span>"
                }
            )
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Missing cnt")
            out_png = os.path.join(OUT_DIR, f"{case_tag}_missing_detections.png")
            fig.write_image(out_png)
            save_meta(out_png, f"{case_tag} missing detections", "Missing feature detections over time")

    # Cross-case charts
    if len(case_summary) > 0:
        save_bar_chart(
            case_summary.sort_values("case"),
            x="case",
            y="rmse_all_avg",
            color="model" if "model" in case_summary.columns else None,
            title="Average RMSE by case<br><span style='font-size:18px;font-weight:normal;'>Lower is better</span>",
            x_label="Case",
            y_label="Avg RMSE",
            out_png=os.path.join(OUT_DIR, "cross_case_rmse.png")
        )

        save_bar_chart(
            case_summary.sort_values("case"),
            x="case",
            y="inner_mean_avg",
            color="model" if "model" in case_summary.columns else None,
            title="Inner-point error by case<br><span style='font-size:18px;font-weight:normal;'>Shape quality across cases</span>",
            x_label="Case",
            y_label="Inner err",
            out_png=os.path.join(OUT_DIR, "cross_case_inner_mean.png")
        )

        save_bar_chart(
            case_summary.sort_values("case"),
            x="case",
            y="settling_time_s",
            color="model" if "model" in case_summary.columns else None,
            title="Settling time by case<br><span style='font-size:18px;font-weight:normal;'>2 cm threshold, 1 s dwell</span>",
            x_label="Case",
            y_label="Settle (s)",
            out_png=os.path.join(OUT_DIR, "cross_case_settling_time.png")
        )

    print(f"Saved analysis outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
