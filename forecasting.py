# forecasting.py
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

from feature_building import tp_index
from fitting import estimate_fisher_kpp_velocity, estimate_patient_rho_D
from online_gp import StreamingResidualGP
from neural_ode_cnf import CNFWrapper

# ======================================================================================
# 1. PARABOLIC MECHANISTIC TRAJECTORY FORECASTERS (Your Original Logic)
# ======================================================================================

def simulate_trajectory_from_params(sub_df, rho, D, V_max_ref, pid, patient_objects, days_grid=None):
    sub = sub_df.sort_values("Day").reset_index(drop=True)
    if len(sub) < 2:
        return pd.DataFrame()

    if days_grid is None:
        d0 = float(sub["Day"].min())
        d1 = float(sub["Day"].max())
        days_grid = np.linspace(d0, d1, 60)

    sim_rows = []
    raw_v0 = float(sub.iloc[0]["Volume_Total"])
    day_start = float(sub.iloc[0]["Day"])

    # Adaptive Seeding Layer: Protect reaction-diffusion ODE from flatlining
    has_subsequent_growth = (sub["Volume_Total"] > 10.0).any()
    if raw_v0 <= 0.0 and has_subsequent_growth:
        V_start = 10.0  
    else:
        V_start = raw_v0

    V = V_start
    day_prev = day_start

    for day in days_grid:
        if day <= day_start:
            sim_rows.append({"Day": day, "Volume": raw_v0})
            continue

        dt = max(day - day_prev, 0.0)
        
        if rho < 0:
            dvdt = rho * V
        else:
            dvdt, _, _, _ = estimate_fisher_kpp_velocity(V, rho, D, {pid: V_max_ref}, pid, patient_objects)
        
        if pd.isna(dvdt):
            sim_rows.append({"Day": day, "Volume": V if (pd.notna(V) and V > 1.0) else V_start})
            continue
            
        V = max(1.0, V + dvdt * dt)
        sim_rows.append({"Day": day, "Volume": V})
        day_prev = day

    return pd.DataFrame(sim_rows)


def bootstrap_trajectory_band(sub_df, V_max_ref, pid, patient_objects, rho_hat, D_hat,
                              n_boot=100, noise_sd=0.05, drop_fraction=0.25):
    boot_trajs = []
    grid = np.linspace(float(sub_df["Day"].min()), float(sub_df["Day"].max()), 80)

    for b in range(n_boot):
        boot = sub_df.sort_values("Day").reset_index(drop=True).copy()

        if len(boot) > 2 and drop_fraction > 0:
            rng = np.random.default_rng(1000 + b)
            keep_n = max(2, int(np.ceil(len(boot) * (1.0 - drop_fraction))))
            keep_idx = np.sort(rng.choice(np.arange(len(boot)), size=keep_n, replace=False))
            boot = boot.iloc[keep_idx].reset_index(drop=True)

        if noise_sd and noise_sd > 0:
            rng = np.random.default_rng(5000 + b)
            scale = max(boot["Volume_Total"].median(), 1.0)
            boot["Volume_Total"] = (
                boot["Volume_Total"] + rng.normal(0.0, noise_sd * scale, size=len(boot))
            ).clip(lower=1.0)

        # Enforce safe denominator validation floors
        v_base = max(1.0, float(boot["Volume_Total"].iloc[0]))
        is_boot_shrinking = boot["Volume_Total"].iloc[-1] < boot["Volume_Total"].iloc[0]

        if is_boot_shrinking:
            d_min_idx = boot["Volume_Total"].idxmin()
            day_min = boot.loc[d_min_idx, "Day"]
            vol_min = max(1.0, float(boot.loc[d_min_idx, "Volume_Total"]))
            delta_t = day_min - boot["Day"].iloc[0]
            rho_b = np.log(vol_min / v_base) / delta_t if delta_t > 0 else -0.01
            D_b = 0.0
        else:
            rho_b, D_b, _, _, _, _, _ = estimate_patient_rho_D(
                boot, Vmax_ref=V_max_ref, pid=pid, patient_objects=patient_objects, n_starts=6
            )

        if len(boot) >= 2 and pd.notna(rho_b) and pd.notna(D_b):
            traj = simulate_trajectory_from_params(
                boot, rho_b, D_b, V_max_ref, pid, patient_objects, days_grid=grid
            )
            if not traj.empty:
                boot_trajs.append(traj["Volume"].to_numpy())

    if len(boot_trajs) == 0:
        return None

    arr = np.vstack(boot_trajs)
    q05 = np.nanpercentile(arr, 5, axis=0)
    q50 = np.nanpercentile(arr, 50, axis=0)
    q95 = np.nanpercentile(arr, 95, axis=0)

    return pd.DataFrame({"Day": grid, "p05": q05, "p50": q50, "p95": q95})


def counterfactual_trajectories(sub_df, rho, D, V_max_ref, pid, patient_objects):
    d0 = float(sub_df["Day"].min())
    d1 = float(sub_df["Day"].max()) + 180
    days = np.linspace(d0, d1, 120)

    if rho < 0:
        scenarios = {
            "baseline": (rho, D),
            "slower_clearance_25%": (rho * 0.75, D),
            "faster_clearance_25%": (rho * 1.25, D),
        }
    else:
        scenarios = {
            "baseline": (rho, D),
            "reduced_growth_25%": (rho * 0.75, D),
            "reduced_diffusion_25%": (rho, D * 0.75),
            "both_reduced_25%": (rho * 0.75, D * 0.75),
            "increased_diffusion_25%": (rho, D * 1.25),
        }

    out = []
    for name, (r, d) in scenarios.items():
        traj = simulate_trajectory_from_params(sub_df, r, d, V_max_ref, pid, patient_objects, days_grid=days)
        if not traj.empty:
            traj["scenario"] = name
            out.append(traj)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ======================================================================================
# 2. HYBRID BAYESIAN VHT FORECASTER MODULE (Newly Added System Blocks)
# ======================================================================================

def build_treatment_plan(context_cols, **kwargs):
    cidx = {c: i for i, c in enumerate(context_cols)}

    def plan(step, c_step):
        c = np.asarray(c_step, dtype=float).reshape(1, -1)
        for key, value in kwargs.items():
            if key in cidx:
                c[0, cidx[key]] = float(value)
        return c

    return plan


def _extract_tp_features(obj, tp):
    mask_path = next(
        (p for p in obj.get('images', {}).get(tp, []) if "mask" in Path(p).name.lower()),
        None
    )
    if mask_path is None:
        return None

    img = nib.load(mask_path)
    vol = img.get_fdata()
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))

    row = {}
    for lbl, name in [(1, "Necrotic"), (2, "Edema"), (3, "Enhancing"), (4, "Resection")]:
        binary = (vol == lbl)
        row[f"Volume_{name}"] = float(np.sum(binary)) * voxel_volume
        if np.sum(binary) > 0:
            vals = vol[binary]
            row[f"IntensityMean_{name}"] = float(np.mean(vals))
            row[f"IntensityStd_{name}"] = float(np.std(vals))
        else:
            row[f"IntensityMean_{name}"] = np.nan
            row[f"IntensityStd_{name}"] = np.nan
    return row


def build_scenario_plan(context_cols, scenario):
    cidx = {c: i for i, c in enumerate(context_cols)}

    def plan(step, c_step):
        c = np.asarray(c_step, dtype=float).reshape(1, -1)

        def set_if_present(name, value):
            if name in cidx:
                c[0, cidx[name]] = float(value)

        all_indicators = ["Post_Chemo", "Post_Rad", "Post_Additional1", "Post_Additional2", "Post_Immuno", "Post_Brachy"]
        for ind in all_indicators:
            set_if_present(ind, 0.0)
        set_if_present("TreatmentExposureFlag", 0.0)

        timeline_groups = [
            ["ChemoStartDay", "ChemoEndDay", "ChemoDurationDays"],
            ["RadStartDay", "RadEndDay", "RadDurationDays", "Radiation_Total_Dose", "Radiation_Fractions", "Rad_Dose_Per_Fraction"],
            ["Additional1_StartDay", "Additional1_EndDay", "Additional1_DurationDays", "Additional1_CycleLength", "Additional1_CycleNumber"],
            ["Additional2_StartDay", "Additional2_EndDay", "Additional2_DurationDays", "Additional2_CycleLength", "Additional2_CycleNumber"],
            ["ImmunoStartDay", "ImmunoEndDay", "ImmunoDurationDays", "ImmunoCycleLength", "ImmunoCycleNumber"],
            ["BrachyStartDay"]
        ]
        for grp in timeline_groups:
            for field in grp:
                set_if_present(field, 0.0)

        if scenario == "baseline":
            pass
        elif scenario == "chemo_on":
            set_if_present("Post_Chemo", 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("ChemoStartDay", 1.0)
            set_if_present("ChemoDurationDays", 180.0)
        elif scenario == "rad_on":
            set_if_present("Post_Rad", 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("RadStartDay", 1.0)
            set_if_present("RadDurationDays", 42.0)
            set_if_present("Radiation_Total_Dose", 60.0)
            set_if_present("Radiation_Fractions", 30.0)
            set_if_present("Rad_Dose_Per_Fraction", 2.0)
        elif scenario == "additional1_on":
            set_if_present("Post_Additional1", 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("Additional1_StartDay", 1.0)
            set_if_present("Additional1_CycleLength", 28.0)
            set_if_present("Additional1_CycleNumber", 6.0)
        elif scenario == "additional2_on":
            set_if_present("Post_Additional2", 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("Additional2_StartDay", 1.0)
            set_if_present("Additional2_CycleLength", 28.0)
            set_if_present("Additional2_CycleNumber", 6.0)
        elif scenario == "immuno_on":
            set_if_present("Post_Immuno", 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("ImmunoStartDay", 1.0)
            set_if_present("ImmunoCycleLength", 14.0)
            set_if_present("ImmunoCycleNumber", 12.0)
        elif scenario == "brachy_on":
            set_if_present("Post_Brachy", 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("BrachyStartDay", 1.0)
        elif scenario == "combined":
            for ind in all_indicators:
                set_if_present(ind, 1.0)
            set_if_present("TreatmentExposureFlag", 1.0)
            set_if_present("ChemoStartDay", 1.0)
            set_if_present("RadStartDay", 1.0)
            set_if_present("Radiation_Total_Dose", 60.0)
            set_if_present("Radiation_Fractions", 30.0)
            set_if_present("Rad_Dose_Per_Fraction", 2.0)
            set_if_present("Additional1_StartDay", 1.0)
            set_if_present("Additional2_StartDay", 1.0)
            set_if_present("ImmunoStartDay", 1.0)
            set_if_present("BrachyStartDay", 1.0)

        return c

    return plan


def train_hybrid_forecaster(patient_objects, clinical_df, biological_priors, treatment_columns, df_kinetics, vmax_priors):
    """
    Ingests states, aligns targets across multiple spreadsheets and imaging timepoints, 
    extracts residuals over mechanistic priors, and builds state-transition GP & CNF models.
    """
    visit_rows = []
    for pid, obj in patient_objects.items():
        clean_id = obj.get('PID_Clean', str(pid).split('-')[0])
        tps = sorted(obj.get('timepoints', []), key=tp_index)
        for tp in tps:
            feat = _extract_tp_features(obj, tp)
            if feat is None:
                continue
            feat["PID_Clean"] = clean_id
            feat["Timepoint"] = tp
            feat["TP_Index"] = tp_index(tp)
            
            tp_num = "".join(filter(str.isdigit, str(tp)))
            day_val = np.nan
            try:
                day_keys = [k for k in obj.get('clinical', {}).keys() if f"Timepoint_{tp_num}" in str(k)]
                if len(day_keys) > 0:
                    day_val = float(obj["clinical"][day_keys[0]])
            except Exception:
                pass
            feat["Day"] = day_val
            visit_rows.append(feat)

    df_visits = pd.DataFrame(visit_rows).sort_values(["PID_Clean", "TP_Index"]).reset_index(drop=True)
    state_cols = [c for c in df_visits.columns if c.startswith("Volume_") or c.startswith("IntensityMean_") or c.startswith("IntensityStd_")]

    df_clinical = clinical_df.copy()
    if "PID_Clean" not in df_clinical.columns:
        id_candidates = [col for col in ["Patient ID", "PatientID", "PID", "Subject"] if col in df_clinical.columns]
        if id_candidates:
            df_clinical["PID_Clean"] = df_clinical[id_candidates[0]].astype(str).str.split("-").str[0]
        elif len(df_clinical.columns) > 0:
            df_clinical["PID_Clean"] = df_clinical[df_clinical.columns[0]].astype(str).str.split("-").str[0]
        else:
            df_clinical["PID_Clean"] = ""

    df_clinical["PID_Clean"] = df_clinical["PID_Clean"].astype(str).str.strip()
    for col in df_clinical.columns:
        if df_clinical[col].dtype == "object" and col != "PID_Clean":
            df_clinical[col] = LabelEncoder().fit_transform(df_clinical[col].astype(str).fillna("missing"))

    if "risk_score" not in df_clinical.columns:
        df_clinical["risk_score"] = 1.0

    df_clinical = df_clinical[list(dict.fromkeys(["PID_Clean"] + [c for c in df_clinical.columns if c != "PID_Clean"]))].copy()
    df_visits["PID_Clean"] = df_visits["PID_Clean"].astype(str).str.strip()
    df_visits = df_visits.merge(df_clinical, on="PID_Clean", how="left", suffixes=("", "_clin"))

    forecast_targets = [
        "BurdenChange_Necrotic", "BurdenChange_Edema",
        "BurdenChange_Enhancing", "BurdenChange_Resection"
    ]
    
    if df_kinetics is not None and not df_kinetics.empty:
        df_kinetics_cp = df_kinetics.copy()
        df_kinetics_cp["PID_Clean"] = df_kinetics_cp["PID_Clean"].astype(str)
        df_kinetics_cp["Timepoint"] = df_kinetics_cp["Timepoint"].astype(str)
        df_visits["Timepoint"] = df_visits["Timepoint"].astype(str)
        
        existing_targets = [c for c in forecast_targets if c in df_kinetics_cp.columns]
        if existing_targets:
            df_targets = df_kinetics_cp[["PID_Clean", "Timepoint"] + existing_targets].copy()
            df_visits = df_visits.merge(df_targets, on=["PID_Clean", "Timepoint"], how="left")

    for c in forecast_targets:
        if c not in df_visits.columns:
            df_visits[c] = np.nan

    extra_context_cols = [
        "Post_Chemo", "Post_Rad", "Post_Additional1", "Post_Additional2", "Post_Immuno", "Post_Brachy",
        "ChemoStartDay", "ChemoEndDay", "ChemoDurationDays",
        "RadStartDay", "RadEndDay", "RadDurationDays", "Radiation_Total_Dose", "Radiation_Fractions", "Rad_Dose_Per_Fraction",
        "Additional1_StartDay", "Additional1_EndDay", "Additional1_DurationDays", "Additional1_CycleLength", "Additional1_CycleNumber",
        "Additional2_StartDay", "Additional2_EndDay", "Additional2_DurationDays", "Additional2_CycleLength", "Additional2_CycleNumber",
        "ImmunoStartDay", "ImmunoEndDay", "ImmunoDurationDays", "ImmunoCycleLength", "ImmunoCycleNumber",
        "BrachyStartDay", "TreatmentExposureFlag", "risk_score"
    ]
    context_cols = [c for c in dict.fromkeys(biological_priors + treatment_columns + extra_context_cols) if c in df_visits.columns]

    for col in context_cols:
        if df_visits[col].dtype == "object":
            df_visits[col] = LabelEncoder().fit_transform(df_visits[col].astype(str))

    X_state = df_visits[state_cols].to_numpy(dtype=float) if len(state_cols) else np.zeros((len(df_visits), 0))
    C_context = df_visits[context_cols].to_numpy(dtype=float) if len(context_cols) else np.zeros((len(df_visits), 0))
    y_observed = df_visits[forecast_targets].to_numpy(dtype=float)

    X_gp_raw = np.hstack([
        X_state,
        C_context,
        df_visits[["Day"]].fillna(0.0).to_numpy(dtype=float) if "Day" in df_visits.columns else np.zeros((len(df_visits), 1))
    ])
    X_gp = SimpleImputer(strategy="median").fit_transform(X_gp_raw)

    y_mechanistic = []
    has_rho = "Fisher_KPP_Rho_1_day" in df_visits.columns
    has_D = "Fisher_KPP_D_mm2_day" in df_visits.columns
    cohort_fitted_results = []

    for pid in df_visits["PID_Clean"].astype(str).unique():
        sub = df_visits[df_visits["PID_Clean"].astype(str) == pid].sort_values("TP_Index")
        
        rho_p = float(sub["Fisher_KPP_Rho_1_day"].iloc[0]) if has_rho and len(sub) > 0 and pd.notna(sub["Fisher_KPP_Rho_1_day"].iloc[0]) else 0.01
        D_p = float(sub["Fisher_KPP_D_mm2_day"].iloc[0]) if has_D and len(sub) > 0 and pd.notna(sub["Fisher_KPP_D_mm2_day"].iloc[0]) else 0.02

        cohort_fitted_results.append({
            "Patient_ID": pid,
            "Fisher_KPP_Rho_1_day": rho_p,
            "Fisher_KPP_D_mm2_day": D_p,
            "V_max_Used": vmax_priors.get("vmax_prior", 200000.0),
            "Optimization_Loss": 0.0
        })

        if len(sub) < 2:
            y_mechanistic.extend([[np.nan] * len(forecast_targets)] * len(sub))
            continue

        vols = sub[[c for c in sub.columns if c.startswith("Volume_")]].to_numpy(dtype=float)
        days = sub["Day"].to_numpy(dtype=float)

        for i in range(len(sub)):
            if i == 0 or not np.isfinite(days[i]) or not np.isfinite(days[i - 1]) or (days[i] - days[i - 1]) <= 0:
                y_mechanistic.append([np.nan] * len(forecast_targets))
                continue

            dt = days[i] - days[i - 1]
            latest_vol = np.nansum(vols[i])

            sim_dvdt, _, _, _ = estimate_fisher_kpp_velocity(
                volume=latest_vol,
                rho=rho_p,
                D=D_p,
                vmax_by_patient=vmax_priors,
                pid=pid,
                patient_objects=patient_objects
            )
            
            sim_delta = sim_dvdt * dt if pd.notna(sim_dvdt) else 0.0
            y_mechanistic.append([sim_delta] * len(forecast_targets))

    y_mechanistic = np.asarray(y_mechanistic, dtype=float)
    y_residuals = SimpleImputer(strategy="median").fit_transform(np.nan_to_num(y_observed) - np.nan_to_num(y_mechanistic))

    kernel = C(1.0, constant_value_bounds=(1e-3, 1e9)) * RBF(10.0, length_scale_bounds=(1e-2, 1e5))
    streaming_gp = StreamingResidualGP(kernel=kernel, alpha=0.5, random_state=42, retrain_every=1).fit_initial(X_gp, y_residuals)

    state_pid_to_idx = {}
    for idx, p in enumerate(df_visits["PID_Clean"].astype(str).tolist()):
        state_pid_to_idx.setdefault(p, []).append(idx)

    X_t_list, X_t1_list, C_t_list = [], [], []
    for p, idxs in state_pid_to_idx.items():
        idxs = sorted(idxs, key=lambda i: df_visits.loc[i, "TP_Index"])
        if len(idxs) < 2:
            continue
        for a, b in zip(idxs[:-1], idxs[1:]):
            X_t_list.append(X_state[a])
            X_t1_list.append(X_state[b])
            C_t_list.append(C_context[a])

    cnf_wrapper = CNFWrapper(state_dim=X_state.shape[1], cond_dim=C_context.shape[1])
    if len(X_t_list) >= 2:
        cnf_wrapper.fit_transition_pairs(
            np.nan_to_num(np.asarray(X_t_list, dtype=float)),
            np.nan_to_num(np.asarray(X_t1_list, dtype=float)),
            C_t=np.nan_to_num(np.asarray(C_t_list, dtype=float)) if len(C_t_list) > 0 else None,
            epochs=50, batch_size=32, lr=1e-3
        )

    return df_visits, cohort_fitted_results, streaming_gp, cnf_wrapper


# ======================================================================================
# 3. METRICS EVALUATION & COUNTERFACTUAL ANALYSIS LAYER
# ======================================================================================

def extract_vht_trajectory_results(patient_forecasts, state_columns, state_labels=(1, 2, 3, 4)):
    """
    Parses multi-scenario generative paths, computes spatial-component tracking,
    and calculates treatment suppression metrics across the cohort.
    """
    if not patient_forecasts:
        print("Warning: No forecast data found to parse.")
        return pd.DataFrame(), {}

    volume_name_map = {
        1: "Volume_Necrotic",
        2: "Volume_Edema",
        3: "Volume_Enhancing",
        4: "Volume_Resection"
    }
    volume_indices = {lvl: state_columns.index(col) for lvl, col in volume_name_map.items() if col in state_columns}

    print("Detected volume indices:", volume_indices)

    summary_rows = []
    patient_trajectories = {}

    for pid, scenarios in patient_forecasts.items():
        patient_trajectories[pid] = {}

        for scn, samples in scenarios.items():
            if samples is None:
                continue

            samples_arr = np.asarray(samples)
            if samples_arr.ndim != 3:
                print(f"Skipping {pid} / {scn}: ndim={samples_arr.ndim}")
                continue

            n_samples, n_steps, n_feats = samples_arr.shape
            if n_samples < 1 or n_steps < 1 or n_feats < 1:
                continue

            mean_path = np.mean(samples_arr, axis=0)
            patient_trajectories[pid][scn] = mean_path

            v0_total = 0.0
            vEnd_total = 0.0
            lvl_vEnd = {}

            for lvl in state_labels:
                v_idx = volume_indices.get(lvl, None)
                if v_idx is not None and v_idx < n_feats:
                    v0 = float(mean_path[0, v_idx])
                    vEnd = float(mean_path[-1, v_idx])
                else:
                    v0, vEnd = 0.0, 0.0

                v0_total += v0
                vEnd_total += vEnd
                lvl_vEnd[lvl] = vEnd

            summary_rows.append({
                "Patient_ID": str(pid),
                "Scenario": scn,
                "Initial_Volume_Total": round(v0_total, 2),
                "Final_Volume_Total": round(vEnd_total, 2),
                "Net_Volume_Change": round(vEnd_total - v0_total, 2),
                "Growth_Ratio": round(vEnd_total / max(v0_total, 1.0), 5),
                "Necrotic_End": round(lvl_vEnd.get(1, 0.0), 2),
                "Edema_End": round(lvl_vEnd.get(2, 0.0), 2),
                "Enhancing_End": round(lvl_vEnd.get(3, 0.0), 2),
                "Resection_End": round(lvl_vEnd.get(4, 0.0), 2),
            })

    df_results = pd.DataFrame(summary_rows)
    print("summary_rows:", len(summary_rows))
    print("df_results shape:", df_results.shape)

    cohort_insights = {}
    if not df_results.empty:
        pivot_df = df_results.pivot(index="Patient_ID", columns="Scenario", values="Final_Volume_Total")
        if "baseline" in pivot_df.columns:
            for treatment in ["chemo_on", "rad_on", "immuno_on", "brachy_on", "combined"]:
                if treatment in pivot_df.columns:
                    suppression = pivot_df["baseline"] - pivot_df[treatment]
                    pct_reduction = (suppression / pivot_df["baseline"].replace(0, np.nan)) * 100.0
                    cohort_insights[f"Avg_Suppression_CC_{treatment}"] = float(suppression.mean())
                    cohort_insights[f"Avg_Pct_Reduction_{treatment}"] = float(pct_reduction.mean())

    return df_results, cohort_insights