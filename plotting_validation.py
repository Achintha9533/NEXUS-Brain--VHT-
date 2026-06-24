import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitting import estimate_patient_rho_D
from forecasting import simulate_trajectory_from_params, bootstrap_trajectory_band, counterfactual_trajectories

def generate_chunked_patient_plots(df_final, patient_objects, chunk_size=10):
    rep_all = df_final.dropna(subset=["Fisher_KPP_Rho_1_day", "Fisher_KPP_D_mm2_day"]).copy()
    rep_all = rep_all.sort_values("Optimization_Loss")

    patient_chunks = [rep_all.iloc[i : i + chunk_size] for i in range(0, len(rep_all), chunk_size)]
    print(f"Total Cohort Size: {len(rep_all)} patients. Splitting into {len(patient_chunks)} batches...")

    for chunk_idx, rep in enumerate(patient_chunks):
        num_rows = len(rep)
        v_spacing = 0.08 if num_rows <= 10 else max(0.005, 0.8 / num_rows)

        fig = make_subplots(
            rows=num_rows, cols=1,
            shared_xaxes=False,
            vertical_spacing=v_spacing,
            subplot_titles=[f"Patient {pid}" for pid in rep["Patient_ID"]],
        )

        for i, row in rep.reset_index(drop=True).iterrows():
            pid = row["Patient_ID"]
            obj = patient_objects[pid]
            V_max_ref = float(row["V_max_Used"]) if pd.notna(row["V_max_Used"]) else 500000.0

            records = []
            kin_df = obj.get("kinetic_features", None)
            day_mapping = {}
            if isinstance(kin_df, pd.DataFrame) and "Timepoint" in kin_df.columns and "Day" in kin_df.columns:
                day_mapping = dict(zip(kin_df["Timepoint"], kin_df["Day"]))

            for tp, sub_regions in obj.get("volumes", {}).items():
                if isinstance(sub_regions, dict):
                    v_nec = float(sub_regions.get("Necrotic", 0.0))
                    v_ede = float(sub_regions.get("Edema", 0.0))
                    v_enh = float(sub_regions.get("Enhancing", 0.0))
                    v_res = float(sub_regions.get("Resection", 0.0))
                    active = sum([v for v in [v_nec, v_ede, v_enh] if v > 0])
                    v_total = active - v_res if v_res > 0 else active
                    day_val = day_mapping.get(tp, None)
                    if day_val is None:
                        try: day_val = float(tp)
                        except Exception: continue
                    records.append({"Day": float(day_val), "Volume_Total": float(v_total)})

            sub_df = pd.DataFrame(records).sort_values("Day").reset_index(drop=True)
            if sub_df.empty:
                continue

            rho = float(row["Fisher_KPP_Rho_1_day"])
            D = float(row["Fisher_KPP_D_mm2_day"])

            traj_test = simulate_trajectory_from_params(sub_df, rho, D, V_max_ref, pid, patient_objects)
            is_failed = (
                traj_test.empty or 
                traj_test["Volume"].isna().all() or 
                np.allclose(traj_test["Volume"], traj_test["Volume"].iloc[0], atol=1e-2)
            )

            if is_failed:
                rho_opt, D_opt, success, _, _, _, _ = estimate_patient_rho_D(
                    sub_df, Vmax_ref=V_max_ref, pid=pid, patient_objects=patient_objects, n_starts=6
                )
                local_success = False
                if success and pd.notna(rho_opt) and pd.notna(D_opt):
                    traj_local_test = simulate_trajectory_from_params(sub_df, rho_opt, D_opt, V_max_ref, pid, patient_objects)
                    if not traj_local_test.empty and not np.allclose(traj_local_test["Volume"], traj_local_test["Volume"].iloc[0], atol=1e-2):
                        local_success = True
                        rho, D = rho_opt, D_opt

                if not local_success:
                    d_min_idx = sub_df["Volume_Total"].idxmin()
                    day_min = sub_df.loc[d_min_idx, "Day"]
                    vol_min = max(1.0, float(sub_df.loc[d_min_idx, "Volume_Total"]))
                    v_base_rescue = max(1.0, float(sub_df["Volume_Total"].iloc[0]))
                    delta_t = day_min - sub_df["Day"].iloc[0]
                    if delta_t > 0:
                        rho = np.log(vol_min / v_base_rescue) / delta_t
                        D = 0.0

            traj = simulate_trajectory_from_params(sub_df, rho, D, V_max_ref, pid, patient_objects)
            band = bootstrap_trajectory_band(sub_df, V_max_ref, pid, patient_objects, rho, D, n_boot=80, noise_sd=0.05)
            cf = counterfactual_trajectories(sub_df, rho, D, V_max_ref, pid, patient_objects)

            r = i + 1
            fig.add_trace(
                go.Scatter(
                    x=sub_df["Day"], y=sub_df["Volume_Total"], mode="markers+lines",
                    name=f"Observed {pid}", marker=dict(size=8, color="black"),
                    line=dict(color="black"), showlegend=(i == 0)
                ), row=r, col=1
            )

            if not traj.empty:
                fig.add_trace(
                    go.Scatter(
                        x=traj["Day"], y=traj["Volume"], mode="lines",
                        name=f"Predicted {pid}", line=dict(color="#1f77b4", width=2),
                        showlegend=(i == 0)
                    ), row=r, col=1
                )

            if band is not None and not band.empty:
                fig.add_trace(go.Scatter(x=band["Day"], y=band["p95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=r, col=1)
                fig.add_trace(
                    go.Scatter(
                        x=band["Day"], y=band["p05"], mode="lines", fill="tonexty",
                        line=dict(width=0), name=f"95% band {pid}",
                        fillcolor="rgba(31,119,180,0.20)", showlegend=(i == 0)
                    ), row=r, col=1
                )

            if not cf.empty:
                palette = {
                    "baseline": "#2ca02c", "reduced_growth_25%": "#d62728",
                    "reduced_diffusion_25%": "#9467bd", "both_reduced_25%": "#ff7f0e",
                    "increased_diffusion_25%": "#17becf", "slower_clearance_25%": "#ff7f0e",
                    "faster_clearance_25%": "#17becf"
                }
                for scen, grp in cf.groupby("scenario"):
                    fig.add_trace(
                        go.Scatter(
                            x=grp["Day"], y=grp["Volume"], mode="lines",
                            line=dict(color=palette.get(scen, "gray"), dash="dash", width=1.5),
                            name=scen, showlegend=(i == 0)
                        ), row=r, col=1
                    )

        fig.update_layout(
            height=max(350, 280 * num_rows), width=1100, template="plotly_white",
            title=f"Observed MRI Trajectory vs Digital Twin Predictions (Part {chunk_idx + 1})",
            legend_title_text="Series"
        )
        fig.update_xaxes(title_text="Days")
        fig.update_yaxes(title_text="Tumour volume (mm³)")
        fig.show()

def generate_cohort_distribution_plots(df_final):
    cohort = df_final.copy()
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=("rho histogram", "D histogram", "rho vs D", "rho and D boxplots"))
    fig2.add_trace(go.Histogram(x=cohort["Fisher_KPP_Rho_1_day"], nbinsx=20, name="rho"), row=1, col=1)
    fig2.add_trace(go.Histogram(x=cohort["Fisher_KPP_D_mm2_day"], nbinsx=20, name="D"), row=1, col=2)
    fig2.add_trace(go.Scatter(x=cohort["Fisher_KPP_Rho_1_day"], y=cohort["Fisher_KPP_D_mm2_day"], mode="markers", text=cohort["Patient_ID"], name="rho vs D"), row=2, col=1)
    fig2.add_trace(go.Box(y=cohort["Fisher_KPP_Rho_1_day"], name="rho"), row=2, col=2)
    fig2.add_trace(go.Box(y=cohort["Fisher_KPP_D_mm2_day"], name="D"), row=2, col=2)

    fig2.update_layout(height=900, width=1200, template="plotly_white", title="Cohort Distribution of Mechanistic Parameters")
    fig2.show()