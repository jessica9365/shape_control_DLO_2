#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import pandas as pd
import plotly.express as px

BASE_LOG_DIR = r"<SET_THIS_TO_YOUR_SIM_LOG_DIR>"
OUT_DIR = os.path.join(BASE_LOG_DIR, "analysis")
THRESHOLD_M = 0.05
DWELL_S = 1.0
USE_PREFIXES = ["gt", "obs"]


def ensure_dirs(base):
    dirs = {
        "root": base,
        "csv": os.path.join(base, "csv"),
        "notes": os.path.join(base, "notes"),
        "per_case": os.path.join(base, "per_case"),
        "cross_case": os.path.join(base, "cross_case"),
        "heatmaps": os.path.join(base, "heatmaps"),

        "per_case_gt_error_trends": os.path.join(base, "per_case", "gt_error_trends"),
        "per_case_obs_error_trends": os.path.join(base, "per_case", "obs_error_trends"),
        "per_case_gt_point_errors": os.path.join(base, "per_case", "gt_point_errors"),
        "per_case_obs_point_errors": os.path.join(base, "per_case", "obs_point_errors"),
        "per_case_gt_vs_obs": os.path.join(base, "per_case", "gt_vs_obs"),
        "per_case_missing_count": os.path.join(base, "per_case", "missing_count"),

        "cross_case_gt": os.path.join(base, "cross_case", "gt"),
        "cross_case_obs": os.path.join(base, "cross_case", "obs"),
        "cross_case_compare": os.path.join(base, "cross_case", "gt_vs_obs"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


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


def settling_time(t, e, threshold=0.05, dwell_s=1.0):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)

    if len(t) < 2:
        return np.nan

    for i in range(len(t)):
        if not np.isfinite(e[i]) or e[i] > threshold:
            continue

        t0 = t[i]
        j = i
        ok = True

        while j < len(t) and (t[j] - t0) <= dwell_s:
            if not np.isfinite(e[j]) or e[j] > threshold:
                ok = False
                break
            j += 1

        if ok and j < len(t):
            return float(t0)

    return np.nan


def first_hit_time(t, e, threshold=0.05):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    idx = np.where(np.isfinite(e) & (e < threshold))[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def auc_error(t, e):
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    mask = np.isfinite(t) & np.isfinite(e)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(e[mask], t[mask]))


def relabel_metric(m):
    mapping = {
        "mean_err": "Mean err",
        "inner_mean_err": "Inner err",
        "endpoint_mean_err": "Endpt err",
        "max_err": "Max err",
        "rmse_all": "RMSE all"
    }
    return mapping.get(m, m)


def save_line(df, x, y, color, title, xlab, ylab, out_png):
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(
        title={"text": title},
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(out_png)
    save_meta(out_png, os.path.basename(out_png).replace(".png", ""), title)


def save_bar(df, x, y, color, title, xlab, ylab, out_png):
    fig = px.bar(df, x=x, y=y, color=color)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(
        title={"text": title},
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(out_png)
    save_meta(out_png, os.path.basename(out_png).replace(".png", ""), title)


def save_heatmap(mat_df, title, out_png):
    fig = px.imshow(mat_df, aspect='auto', color_continuous_scale='Viridis')
    fig.update_layout(title={"text": title})
    fig.update_xaxes(title_text="Feature point")
    fig.update_yaxes(title_text="Case")
    fig.write_image(out_png)
    save_meta(out_png, os.path.basename(out_png).replace(".png", ""), title)


def build_case_summary(df, prefix):
    rows = []
    err_cols = [f"{prefix}_err_{i}_norm" for i in range(10) if f"{prefix}_err_{i}_norm" in df.columns]

    for case_id, g in df.groupby("case"):
        g = g.sort_values("t").reset_index(drop=True)
        metric = f"{prefix}_inner_mean_err" if f"{prefix}_inner_mean_err" in g.columns else f"{prefix}_mean_err"
        end_window = max(1, min(10, len(g)))

        row = {
            "case": int(case_id),
            "model": g["model"].iloc[0] if "model" in g.columns else "unknown",
            "samples": len(g),
            "duration_s": float(g["t"].max()),
            "success_final": bool(np.all(g[metric].tail(end_window) < THRESHOLD_M)) if metric in g.columns else False,
            "first_hit_s": first_hit_time(g["t"], g[metric], THRESHOLD_M) if metric in g.columns else np.nan,
            "settling_s": settling_time(g["t"], g[metric], THRESHOLD_M, DWELL_S) if metric in g.columns else np.nan,
            "auc_err": auc_error(g["t"], g[metric]) if metric in g.columns else np.nan,
            f"{prefix}_rmse_avg": float(g[f"{prefix}_rmse_all"].mean()) if f"{prefix}_rmse_all" in g.columns else np.nan,
            f"{prefix}_rmse_final": float(g[f"{prefix}_rmse_all"].iloc[-1]) if f"{prefix}_rmse_all" in g.columns else np.nan,
            f"{prefix}_mean_avg": float(g[f"{prefix}_mean_err"].mean()) if f"{prefix}_mean_err" in g.columns else np.nan,
            f"{prefix}_mean_final": float(g[f"{prefix}_mean_err"].iloc[-1]) if f"{prefix}_mean_err" in g.columns else np.nan,
            f"{prefix}_inner_avg": float(g[f"{prefix}_inner_mean_err"].mean()) if f"{prefix}_inner_mean_err" in g.columns else np.nan,
            f"{prefix}_inner_final": float(g[f"{prefix}_inner_mean_err"].iloc[-1]) if f"{prefix}_inner_mean_err" in g.columns else np.nan,
            f"{prefix}_endpoint_avg": float(g[f"{prefix}_endpoint_mean_err"].mean()) if f"{prefix}_endpoint_mean_err" in g.columns else np.nan,
            f"{prefix}_max_peak": float(g[f"{prefix}_max_err"].max()) if f"{prefix}_max_err" in g.columns else np.nan,
            "missing_avg": float(g["missing_count"].mean()) if "missing_count" in g.columns else np.nan,
        }

        for c in err_cols:
            row[c.replace("_err_", "_mean_")] = float(g[c].mean())
            row[c.replace("_err_", "_final_")] = float(g[c].iloc[-1])

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    dirs = ensure_dirs(OUT_DIR)
    df = load_logs(BASE_LOG_DIR)
    df.to_csv(os.path.join(dirs["csv"], "all_logs_combined.csv"), index=False)

    available_prefixes = [p for p in USE_PREFIXES if any(col.startswith(p + "_") for col in df.columns)]
    if not available_prefixes:
        raise ValueError("No gt_/obs_ columns found in logs.")

    summaries = {}

    for prefix in available_prefixes:
        cs = build_case_summary(df, prefix)
        cs.to_csv(os.path.join(dirs["csv"], f"case_summary_{prefix}.csv"), index=False)
        summaries[prefix] = cs

        ms = cs.groupby("model", dropna=False).agg({
            "case": "count",
            "success_final": "mean",
            "duration_s": "mean",
            "first_hit_s": "mean",
            "settling_s": "mean",
            "auc_err": "mean",
            f"{prefix}_rmse_avg": "mean",
            f"{prefix}_rmse_final": "mean",
            f"{prefix}_mean_avg": "mean",
            f"{prefix}_mean_final": "mean",
            f"{prefix}_inner_avg": "mean",
            f"{prefix}_inner_final": "mean",
            f"{prefix}_endpoint_avg": "mean",
            f"{prefix}_max_peak": "mean",
            "missing_avg": "mean"
        }).rename(columns={"case": "n_cases", "success_final": "success_rate"}).reset_index()

        ms.to_csv(os.path.join(dirs["csv"], f"model_summary_{prefix}.csv"), index=False)

    if "gt" in summaries and "obs" in summaries:
        gap = summaries["gt"][["case", "model", "gt_inner_final", "gt_rmse_final"]].merge(
            summaries["obs"][["case", "obs_inner_final", "obs_rmse_final"]],
            on="case", how="inner"
        )
        gap["inner_gap_final"] = gap["obs_inner_final"] - gap["gt_inner_final"]
        gap["rmse_gap_final"] = gap["obs_rmse_final"] - gap["gt_rmse_final"]
        gap.to_csv(os.path.join(dirs["csv"], "obs_vs_gt_gap.csv"), index=False)

    for case_id, g in df.groupby("case"):
        g = g.sort_values("t").reset_index(drop=True)
        case_tag = f"case_{int(case_id)}"

        for prefix in available_prefixes:
            metric_cols = [c for c in [
                f"{prefix}_mean_err",
                f"{prefix}_inner_mean_err",
                f"{prefix}_endpoint_mean_err",
                f"{prefix}_max_err"
            ] if c in g.columns]

            if metric_cols:
                long_df = g[["t"] + metric_cols].melt(id_vars="t", var_name="metric", value_name="value")
                long_df["metric"] = long_df["metric"].str.replace(prefix + "_", "", regex=False).map(relabel_metric)
                subdir = dirs[f"per_case_{prefix}_error_trends"]
                save_line(
                    long_df, "t", "value", "metric",
                    f'Error trends ({case_tag}, {prefix})<br><span style="font-size:18px;font-weight:normal;">Summary error signals over time</span>',
                    "Time (s)", "Error (m)",
                    os.path.join(subdir, f"{case_tag}_{prefix}_error_trends.png")
                )

            err_cols = [f"{prefix}_err_{i}_norm" for i in range(10) if f"{prefix}_err_{i}_norm" in g.columns]
            if err_cols:
                long_df = g[["t"] + err_cols].melt(id_vars="t", var_name="point", value_name="value")
                long_df["point"] = long_df["point"].str.replace(f"{prefix}_err_", "p", regex=False).str.replace("_norm", "", regex=False)
                subdir = dirs[f"per_case_{prefix}_point_errors"]
                save_line(
                    long_df, "t", "value", "point",
                    f'Point errors ({case_tag}, {prefix})<br><span style="font-size:18px;font-weight:normal;">Per-feature error norms over time</span>',
                    "Time (s)", "Error (m)",
                    os.path.join(subdir, f"{case_tag}_{prefix}_point_errors.png")
                )

        if all(c in g.columns for c in ["gt_inner_mean_err", "obs_inner_mean_err"]):
            dual = g[["t", "gt_inner_mean_err", "obs_inner_mean_err"]].melt(
                id_vars="t", var_name="source", value_name="value"
            )
            dual["source"] = dual["source"].str.replace("_inner_mean_err", "", regex=False)
            save_line(
                dual, "t", "value", "source",
                'GT vs observed ({})<br><span style="font-size:18px;font-weight:normal;">Occlusion effect on shaping error</span>'.format(case_tag),
                "Time (s)", "Error (m)",
                os.path.join(dirs["per_case_gt_vs_obs"], f"{case_tag}_gt_vs_obs_inner.png")
            )

        if "missing_count" in g.columns:
            fig_df = g[["t", "missing_count"]].copy()
            fig_df["series"] = "missing"
            save_line(
                fig_df, "t", "missing_count", "series",
                f'Missing detections ({case_tag})<br><span style="font-size:18px;font-weight:normal;">Observation loss over time</span>',
                "Time (s)", "Count",
                os.path.join(dirs["per_case_missing_count"], f"{case_tag}_missing_count.png")
            )

    for prefix, cs in summaries.items():
        target_dir = dirs[f"cross_case_{prefix}"]
        if len(cs) == 0:
            continue

        save_bar(
            cs.sort_values("case"), "case", f"{prefix}_rmse_avg", "model",
            f'Average RMSE by case ({prefix})<br><span style="font-size:18px;font-weight:normal;">Lower is better</span>',
            "Case", "Avg RMSE",
            os.path.join(target_dir, f"cross_case_{prefix}_rmse_avg.png")
        )

        save_bar(
            cs.sort_values("case"), "case", f"{prefix}_inner_final", "model",
            f'Final inner error by case ({prefix})<br><span style="font-size:18px;font-weight:normal;">Final shape quality</span>',
            "Case", "Final err",
            os.path.join(target_dir, f"cross_case_{prefix}_inner_final.png")
        )

        save_bar(
            cs.sort_values("case"), "case", "settling_s", "model",
            f'Settling time by case ({prefix})<br><span style="font-size:18px;font-weight:normal;">Threshold and dwell based</span>',
            "Case", "Settle (s)",
            os.path.join(target_dir, f"cross_case_{prefix}_settling.png")
        )

        save_bar(
            cs.sort_values("case"), "case", "auc_err", "model",
            f'Error area by case ({prefix})<br><span style="font-size:18px;font-weight:normal;">Transient plus steady error</span>',
            "Case", "Err area",
            os.path.join(target_dir, f"cross_case_{prefix}_auc.png")
        )

        err_mean_cols = [c for c in cs.columns if c.startswith(f"{prefix}_mean_") and c.endswith("_norm")]
        if err_mean_cols:
            heat = cs[["case"] + err_mean_cols].sort_values("case").set_index("case")
            heat.columns = [c.replace(f"{prefix}_mean_", "p").replace("_norm", "") for c in heat.columns]
            save_heatmap(
                heat,
                f'Per-point mean error heatmap ({prefix})<br><span style="font-size:18px;font-weight:normal;">Cases by feature points</span>',
                os.path.join(dirs["heatmaps"], f"heatmap_{prefix}_point_mean_error.png")
            )

    if "gt" in summaries and "obs" in summaries:
        merged = summaries["gt"][["case", "model", "gt_inner_final", "gt_rmse_final"]].merge(
            summaries["obs"][["case", "obs_inner_final", "obs_rmse_final"]],
            on="case", how="inner"
        )
        long = merged.melt(
            id_vars=["case", "model"],
            value_vars=["gt_inner_final", "obs_inner_final"],
            var_name="source",
            value_name="value"
        )
        long["source"] = long["source"].str.replace("_inner_final", "", regex=False)
        save_bar(
            long.sort_values("case"), "case", "value", "source",
            'GT vs observed final inner error<br><span style="font-size:18px;font-weight:normal;">Observation penalty across cases</span>',
            "Case", "Final err",
            os.path.join(dirs["cross_case_compare"], "cross_case_gt_vs_obs_inner_final.png")
        )

    notes = {
        "threshold_m": THRESHOLD_M,
        "dwell_s": DWELL_S,
        "folder_layout": {k: v for k, v in dirs.items() if k != "root"},
    }
    with open(os.path.join(dirs["notes"], "analysis_notes.json"), "w") as f:
        json.dump(notes, f, indent=2)

    tree_lines = [
        "analysis/",
        "  csv/",
        "  notes/",
        "  per_case/",
        "    gt_error_trends/",
        "    obs_error_trends/",
        "    gt_point_errors/",
        "    obs_point_errors/",
        "    gt_vs_obs/",
        "    missing_count/",
        "  cross_case/",
        "    gt/",
        "    obs/",
        "    gt_vs_obs/",
        "  heatmaps/"
    ]
    with open(os.path.join(dirs["notes"], "folder_structure.txt"), "w") as f:
        f.write("\n".join(tree_lines))

    print(f"Saved analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()
