# main.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# Import modular layers
from feature_building import (
    tp_index, estimate_vmax_prior_from_masks, 
    build_seg_lookup, build_cohort_kinetic_dataframe
)
from forecasting import (
    train_hybrid_forecaster, build_scenario_plan, extract_vht_trajectory_results
)

def run_patient_pipeline():
    image_path = Path("/Users/kasunachinthaperera/Documents/Final Year Project:Thesis/Data/PKG - MU-Glioma-Post/MU-Glioma-Post")
    seg_vol_path = Path("/Users/kasunachinthaperera/Documents/Final Year Project:Thesis/Data/PKG - MU-Glioma-Post/MU-Glioma-Post_Segmentation_Volumes.xlsx")
    clinical_path = Path("/Users/kasunachinthaperera/Documents/Final Year Project:Thesis/Data/PKG - MU-Glioma-Post/MU-Glioma-Post_ClinicalData-July2025.xlsx")

    label_map = {}
    active_labels = ['Necrotic', 'Edema', 'Enhancing', 'Resection']

    biological_priors = [
        'IDH1 mutation', 'IDH2 mutation', 'MGMT methylation', '1p/19q',
        'ATRX mutation', 'BRAF V600E mutation', 'TERT promoter mutation',
        'Chromosome 7 gain and Chromosome 10 loss', 'H3-3A mutation',
        'EGFR amplification', 'PTEN mutation', 'CDKN2A/B deletion',
        'TP53 alteration', 'Grade of Primary Brain Tumor'
    ]

    treatment_columns = [
        'Initial Chemo Therapy', 'Radiation Therapy', 'Dose', 'Number of Fractions',
        'Additional Therapy', '2nd_Additional Therapy', 'Immuno therapy', 'Brachy therapy',
        'Cycle length of Additional Therapy (q days)',
        'Number of Days from Diagnosis to Starting Additional Therapy',
        'Number of Days from Diagnosis to Starting 2nd_Additional Therapy',
        'Number of Days from Diagnosis to Start Immunotherapy',
        'Number of days from Diagnosis to Initial Chemo Therapy Start date',
        'Number of days from Diagnosis to Radiation Therapy Start date'
    ]

    seg_lookup = build_seg_lookup(seg_vol_path, label_map)

    clinical_df = pd.read_excel(clinical_path, sheet_name='MU Glioma Post')
    clinical_df.columns = clinical_df.columns.str.strip()
    clinical_df['PID_Clean'] = clinical_df['Patient ID'].astype(str).str.split('-').str[0]
    clinical_data = clinical_df.set_index('PID_Clean').to_dict(orient='index')

    patient_ids = sorted([
        d for d in os.listdir(image_path)
        if (image_path / d).is_dir() and not d.startswith('.')
    ])

    patient_objects = {}

    for pid in patient_ids:
        clean_id = pid.split('-')[0]
        p_folder = image_path / pid
        tps = [t.name for t in p_folder.iterdir() if t.is_dir() and not t.name.startswith('.')]
        tps = sorted(tps, key=tp_index)
        if len(tps) < 2:
            continue

        obj = {
            'Patient_ID': pid, 'PID_Clean': clean_id, 'clinical': clinical_data.get(clean_id, {}),
            'timepoints': tps, 'images': {}, 'volumes': {}, 'voxels': {}, 't1c_mean': {},
            't1c_stdevs': {}, 't1n_mean': {}, 't1n_stdevs': {}, 't2f_mean': {}, 't2f_stdevs': {},
            't2w_mean': {}, 't2w_stdevs': {}
        }

        for tp in tps:
            obj['images'][tp] = [str(f) for f in (p_folder / tp).glob("*.nii*")]

        for label_key, per_patient in seg_lookup.items():
            rows = per_patient.get(clean_id, [])
            rows_sorted = sorted(rows, key=lambda r: tp_index(str(r.get('Patient ID', ''))))

            for i, tp in enumerate(tps):
                if i >= len(rows_sorted):
                    continue
                row = rows_sorted[i]
                obj['volumes'].setdefault(tp, {})[label_key] = float(row.iloc[4]) if pd.notna(row.iloc[4]) else np.nan
                obj['voxels'].setdefault(tp, {})[label_key] = int(row.iloc[3]) if pd.notna(row.iloc[3]) else 0
                obj['t1c_mean'].setdefault(tp, {})[label_key] = float(row.iloc[5]) if pd.notna(row.iloc[5]) else np.nan
                obj['t1c_stdevs'].setdefault(tp, {})[label_key] = float(row.iloc[6]) if pd.notna(row.iloc[6]) else np.nan
                obj['t1n_mean'].setdefault(tp, {})[label_key] = float(row.iloc[7]) if pd.notna(row.iloc[7]) else np.nan
                obj['t1n_stdevs'].setdefault(tp, {})[label_key] = float(row.iloc[8]) if pd.notna(row.iloc[8]) else np.nan
                obj['t2f_mean'].setdefault(tp, {})[label_key] = float(row.iloc[9]) if pd.notna(row.iloc[9]) else np.nan
                obj['t2f_stdevs'].setdefault(tp, {})[label_key] = float(row.iloc[10]) if pd.notna(row.iloc[10]) else np.nan
                obj['t2w_mean'].setdefault(tp, {})[label_key] = float(row.iloc[11]) if pd.notna(row.iloc[11]) else np.nan
                obj['t2w_stdevs'].setdefault(tp, {})[label_key] = float(row.iloc[12]) if pd.notna(row.iloc[12]) else np.nan

        patient_objects[pid] = obj

    print(f"Built {len(patient_objects)} longitudinal patient objects.")

    # Call modular feature execution step
    df_kinetics = build_cohort_kinetic_dataframe(patient_objects, active_labels)
    
    return patient_objects, clinical_df, biological_priors, treatment_columns, df_kinetics


if __name__ == "__main__":
    # 1. Run Pipeline Processing Core
    patient_objects, clinical_df, biological_priors, treatment_columns, df_kinetics = run_patient_pipeline()

    # 2. Formulate Boundaries
    vmax_priors = estimate_vmax_prior_from_masks(patient_objects, percentile=95, alpha=1.0, min_factor=1.0)

    # 3. Call Modular Hybrid Forecaster Engine
    print("\nTriggering Training Execution Layer inside forecasting.py...")
    df_visits, cohort_fitted_results, streaming_gp, cnf_wrapper = train_hybrid_forecaster(
        patient_objects=patient_objects,
        clinical_df=clinical_df,
        biological_priors=biological_priors,
        treatment_columns=treatment_columns,
        df_kinetics=df_kinetics,
        vmax_priors=vmax_priors
    )

    print(f"\nCalibrating Hybrid VHT... Profiles synchronized.")
    print(f"Streaming GP trained successfully.")
    print(f"CNF wrapper fitted: {cnf_wrapper.is_fitted}")

    # ======================================================================
    # 4. CONDITIONAL GENERATIVE PATHWAY SAMPLING (Multi-Scenario Forecast Engine)
    # ======================================================================
    state_cols = [c for c in df_visits.columns if c.startswith("Volume_") or c.startswith("IntensityMean_") or c.startswith("IntensityStd_")]
    
    # Mirroring the exact context features extracted inside train_hybrid_forecaster
    extra_context_cols = [
        "Post_Chemo", "Post_Rad", "Post_Additional1", "Post_Additional2", "Post_Immuno", "Post_Brachy",
        "ChemoStartDay", "ChemoEndDay", "ChemoDurationDays",
        "RadStartDay", "RadEndDay", "RadDurationDays", "Radiation_Total_Dose", "Radiation_Fractions", "Rad_Dose_Per_Fraction",
        "Additional1_StartDay", "Additional1_EndDay", "Additional1_DurationDays", "Additional1_CycleLength", "Additional1_CycleNumber",
        "Additional2_StartDay", "Additional2_EndDay", "Additional2_DurationDays", "Additional2_CycleLength", "Additional2_CycleNumber",
        "ImmunoStartDay", "ImmunoEndDay", "ImmunoDurationDays", "ImmunoCycleLength", "ImmunoCycleNumber",
        "BrachyStartDay", "TreatmentExposureFlag", "risk_score"
    ]
    all_context_candidates = list(dict.fromkeys(biological_priors + treatment_columns + extra_context_cols))
    context_cols = [c for c in all_context_candidates if c in df_visits.columns]

    # Safe extraction of baseline matrices
    X_state = df_visits[state_cols].to_numpy(dtype=float)
    C_context = df_visits[context_cols].to_numpy(dtype=float) if len(context_cols) else np.zeros((len(df_visits), 0))

    state_pid_to_idx = {}
    for idx, p in enumerate(df_visits["PID_Clean"].astype(str).tolist()):
        state_pid_to_idx.setdefault(p, []).append(idx)

    def get_last_visit_idx(pid):
        idxs = state_pid_to_idx.get(pid, [])
        if not idxs:
            return None
        return max(idxs, key=lambda i: df_visits.loc[i, "TP_Index"])

    all_patient_forecasts = {}
    if cnf_wrapper.is_fitted:
        scenarios = ["baseline", "chemo_on", "rad_on", "additional1_on", "additional2_on", "immuno_on", "brachy_on", "combined"]

        for pid in state_pid_to_idx.keys():
            last_idx = get_last_visit_idx(pid)
            if last_idx is None:
                continue

            x0 = np.nan_to_num(X_state[last_idx], nan=0.0)
            c0 = np.nan_to_num(C_context[last_idx], nan=0.0) if C_context.shape[1] > 0 else None

            all_patient_forecasts[pid] = {}
            for scenario in scenarios:
                plan = build_scenario_plan(context_cols, scenario)
                all_patient_forecasts[pid][scenario] = cnf_wrapper.sample_counterfactual_path(
                    x0,
                    c0=c0,
                    steps=10,
                    n_samples=20,
                    treatment_plan=plan,
                    step_scale=1.5,
                    noise_scale=0.02
                )
    else:
        print("Skipping trajectory sampling because CNF is not fitted.")

    print(f"Generated forecasts for {len(all_patient_forecasts)} patients.")

    # Parse and extract trajectory metrics summary
    df_forecast_metrics, therapeutic_insights = extract_vht_trajectory_results(
        all_patient_forecasts,
        state_cols
    )

    if df_forecast_metrics.empty:
        print("Still empty: check that state_cols order matches forecast feature order.")
    else:
        # Pretty printing trajectory profiles using tabulate (No file saves)
        print("\n" + "="*80)
        print("VHT FORECAST METRICS SUMMARY (FIRST 20 ROWS):")
        print("="*80)
        print(tabulate(df_forecast_metrics.head(20), headers='keys', tablefmt='psql', showindex=False))
        
        print("\nTherapeutic insights:")
        for k, v in therapeutic_insights.items():
            print(f"{k}: {v:.4f}")

    # ======================================================================
    # 5. RENDER GRAPHICS AND COHORT ARTIFACTS
    # ======================================================================
    df_final = pd.DataFrame(cohort_fitted_results)
    if not df_final.empty:
        print("\nGenerating batch-processed trace structures and cohort summary statistics...")
        
        # Explicit time column check and fallbacks to map timeline tracking back to 'Day'
        if "Day" not in df_final.columns:
            if "Timepoint" in df_final.columns:
                df_final["Day"] = df_final["Timepoint"]
            elif "TP_Index" in df_final.columns:
                df_final["Day"] = df_final["TP_Index"]
            elif "Days" in df_final.columns:
                df_final["Day"] = df_final["Days"]

        try:
            # Inline imports to prevent potential structural signature conflicts
            from plotting_validation import generate_chunked_patient_plots, generate_cohort_distribution_plots
            generate_chunked_patient_plots(df_final, patient_objects, chunk_size=10)
            generate_cohort_distribution_plots(df_final)
        except ImportError as e:
            print(f"Skipping visualization rendering layers: {e}")