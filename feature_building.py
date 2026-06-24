# feature_building.py
import os
import numpy as np
import pandas as pd
from pathlib import Path

def tp_index(tp):
    digits = ''.join(filter(str.isdigit, str(tp)))
    return int(digits) if digits else 10**9

def volume_to_radius(obj, volume):
    vol = float(volume) if pd.notna(volume) else np.nan
    if pd.isna(vol) or vol <= 0:
        return np.nan
    return float((3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0))

def safe_delta(v2, v1):
    val2 = pd.to_numeric(v2, errors="coerce")
    val1 = pd.to_numeric(v1, errors="coerce")
    if pd.isna(val2) or pd.isna(val1):
        return np.nan
    return float(val2 - val1)

def get_numeric(d, k):
    if not isinstance(d, dict):
        return np.nan
    v = d.get(k, np.nan)
    v_num = pd.to_numeric(v, errors="coerce")
    return float(v_num) if pd.notna(v_num) else np.nan

def get_clinical_day(obj, tp):
    tp_num = "".join(filter(str.isdigit, str(tp)))
    try:
        day_keys = [k for k in obj.get('clinical', {}).keys() if f"Timepoint_{tp_num}" in str(k)]
        if day_keys:
            val = pd.to_numeric(obj["clinical"][day_keys[0]], errors="coerce")
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return np.nan

def build_seg_lookup(xlsx_path, label_map=None):
    """
    Parses spreadsheet structures containing spatial labels and maps row series 
    to explicit patient keys.
    """
    xl = pd.ExcelFile(xlsx_path)
    lookup = {}
    for sheet in xl.sheet_names:
        sheet_clean = sheet.strip()
        df = xl.parse(sheet)
        df.columns = df.columns.str.strip()
        
        if 'Patient ID' not in df.columns:
            continue
            
        for _, row in df.iterrows():
            pid_raw = str(row['Patient ID'])
            clean_id = pid_raw.split('-')[0].strip()
            lookup.setdefault(sheet_clean, {}).setdefault(clean_id, []).append(row)
    return lookup

def build_sub_df_from_patient(obj, active_labels):
    """
    Extracts dynamic longitudinal values into structural trajectories.
    """
    rows = []
    tps = sorted(obj.get('timepoints', []), key=tp_index)
    for tp in tps:
        v_total = np.nansum([obj['volumes'].get(tp, {}).get(l, 0.0) for l in active_labels])
        day_val = get_clinical_day(obj, tp)
        rows.append({"Day": day_val, "Volume_Total": v_total, "Timepoint": tp})
    df = pd.DataFrame(rows).sort_values("Day").reset_index(drop=True)
    return df

def mask_volume_from_patient_obj(obj, tp):
    tp_vols = obj.get("volumes", {}).get(tp, {})
    vol_nec = float(tp_vols.get("Necrotic", 0.0))
    vol_ede = float(tp_vols.get("Edema", 0.0))
    vol_enh = float(tp_vols.get("Enhancing", 0.0))
    vol_res = float(tp_vols.get("Resection", 0.0))

    vol_nec = vol_nec if (pd.notna(vol_nec) and vol_nec > 0) else 0.0
    vol_ede = vol_ede if (pd.notna(vol_ede) and vol_ede > 0) else 0.0
    vol_enh = vol_enh if (pd.notna(vol_enh) and vol_enh > 0) else 0.0
    vol_res = vol_res if (pd.notna(vol_res) and vol_res > 0) else 0.0

    active_volumes = [v for v in [vol_nec, vol_ede, vol_enh] if v > 0]
    vol_total = sum(active_volumes) - vol_res if vol_res > 0 else sum(active_volumes)

    return {
        "Volume_Necrotic": vol_nec,
        "Volume_Edema": vol_ede,
        "Volume_Enhancing": vol_enh,
        "Volume_Resection": vol_res,
        "Volume_Active": active_volumes if active_volumes else np.nan,
        "Volume_Total": vol_total if vol_total > 0 else np.nan
    }

def get_initial_mask_volume(obj):
    tps = sorted(obj.get("timepoints", []), key=tp_index)
    if not tps:
        return {
            "Volume_Total": np.nan, "Volume_Necrotic": np.nan, "Volume_Edema": np.nan,
            "Volume_Enhancing": np.nan, "Volume_Resection": np.nan, "Volume_Active": np.nan
        }
    return mask_volume_from_patient_obj(obj, tps[0])

def estimate_vmax_prior_from_masks(patient_objects, percentile=95, alpha=1.0, min_factor=1.0):
    vmax_by_patient = {}
    for pid, obj in patient_objects.items():
        kin = obj.get("kinetic_features", None)
        vols = []
        factor = min_factor

        if kin is not None and len(kin) > 1 and "Day" in kin.columns and "Volume_Total" in kin.columns:
            sub = kin[["Day", "Volume_Total"]].copy()
            sub["Day"] = pd.to_numeric(sub["Day"], errors="coerce")
            sub["Volume_Total"] = pd.to_numeric(sub["Volume_Total"], errors="coerce")
            sub = sub.sort_values("Day").dropna()

            volume = sub["Volume_Total"].to_numpy(dtype=float)
            day = sub["Day"].to_numpy(dtype=float)

            if len(volume) > 1:
                dt = np.diff(day)
                dv = np.diff(volume)
                valid = (dt > 0) & np.isfinite(dt) & np.isfinite(dv) & np.isfinite(volume[:-1]) & (volume[:-1] > 0)
                if np.any(valid):
                    rel_rates = dv[valid] / (volume[:-1][valid] * dt[valid])
                    rel_rates = rel_rates[np.isfinite(rel_rates)]
                    if len(rel_rates) > 0:
                        patient_growth = max(0.0, float(np.median(rel_rates)))
                        factor = max(min_factor, 1.0 + alpha * patient_growth)

            vols.extend(volume[volume > 0].tolist())
        else:
            vols_dict = obj.get("volumes", {})
            for _, sub_regions in vols_dict.items():
                if isinstance(sub_regions, dict):
                    total_biological = np.nansum([
                        sub_regions.get("Necrotic", np.nan),
                        sub_regions.get("Edema", np.nan),
                        sub_regions.get("Enhancing", np.nan)
                    ])
                    if pd.notna(total_biological) and total_biological > 0:
                        vols.append(float(total_biological))

        if len(vols) > 0:
            p = np.percentile(vols, percentile)
            vmax_by_patient[pid] = float(max(factor * p, 1.0))
        else:
            vmax_by_patient[pid] = np.nan

    return vmax_by_patient

def get_mask_volume_at_tp(obj, tp):
    images = obj.get("images", {}).get(tp, [])
    mask_path = next((p for p in images if "mask" in Path(p).name.lower()), None)
    if mask_path is None:
        return {
            "Volume_Necrotic": 0.0, "Volume_Edema": 0.0, "Volume_Enhancing": 0.0, "Volume_Resection": 0.0,
            "Volume_Active": np.nan, "Volume_Total": np.nan
        }
    return mask_volume_from_patient_obj(obj, tp)

def get_baseline_tp(obj):
    tps = obj.get("timepoints", [])
    if not tps:
        return None
    try:
        return sorted(tps, key=tp_index)[0]
    except Exception:
        return sorted(tps)[0]

def get_t1c_flair_ratio(obj):
    baseline_tp = get_baseline_tp(obj)
    if baseline_tp is None:
        return np.nan

    lbl = "Total"
    t1c_mean_val = obj.get("t1c_mean", {}).get(baseline_tp, {}).get(lbl, np.nan)
    t2f_mean_val = obj.get("t2f_mean", {}).get(baseline_tp, {}).get(lbl, np.nan)

    t1c_mean_val = pd.to_numeric(t1c_mean_val, errors="coerce")
    t2f_mean_val = pd.to_numeric(t2f_mean_val, errors="coerce")

    if pd.isna(t1c_mean_val) or t1c_mean_val <= 0:
        return np.nan
    if pd.notna(t2f_mean_val) and t2f_mean_val > 0:
        return float(t1c_mean_val / t2f_mean_val)
    return float(t1c_mean_val)

def get_adc_proxy(obj):
    baseline_tp = get_baseline_tp(obj)
    if baseline_tp is None:
        return np.nan

    lbl = "Total"
    t1cs_val = obj.get("t1c_stdevs", {}).get(baseline_tp, {}).get(lbl, np.nan)
    t2fs_val = obj.get("t2f_stdevs", {}).get(baseline_tp, {}).get(lbl, np.nan)
    t2ws_val = obj.get("t2w_stdevs", {}).get(baseline_tp, {}).get(lbl, np.nan)

    vals = pd.to_numeric(pd.Series([t1cs_val, t2fs_val, t2ws_val]), errors="coerce")
    vals = vals.dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())

def build_cohort_kinetic_dataframe(patient_objects, active_labels):
    """
    Extracts multi-layered longitudinal feature spaces into matching 
    structural vectors across the cohort.
    """
    kinetic_features = []
    for pid, obj in patient_objects.items():
        clean_id = obj.get('PID_Clean', str(pid).split('-')[0])
        clinical = obj.get('clinical', {})

        chemo_startday = get_numeric(clinical, 'Number of days from Diagnosis to Initial Chemo Therapy Start date')
        chemo_endday = get_numeric(clinical, 'Number of days from Diagnosis to Initial Chemo Therapy end date')
        rad_startday = get_numeric(clinical, 'Number of days from Diagnosis to Radiation Therapy Start date')
        rad_dose = get_numeric(clinical, 'Dose')
        rad_fractions = get_numeric(clinical, 'Number of Fractions')
        rad_endday = get_numeric(clinical, 'Number of days from Diagnosis to Radiation Therapy end date')

        additional1_cycle_length = get_numeric(clinical, 'Cycle length of Additional Therapy (q days)')
        additional1_startday = get_numeric(clinical, 'Number of Days from Diagnosis to Starting Additional Therapy')
        additional1_endday = get_numeric(clinical, 'Number of Days from Diagnosis to Complete Additional Therapy')
        additional1_cycle_number = get_numeric(clinical, 'Number of Cycles of Additional Therapy')

        additional2_cycle_length = get_numeric(clinical, 'Cycle length of 2nd_Additional Therapy (q days)')
        additional2_startday = get_numeric(clinical, 'Number of Days from Diagnosis to Starting 2nd_Additional Therapy')
        additional2_endday = get_numeric(clinical, 'Number of Days from Diagnosis to Complete 2nd_Additional Therapy')
        additional2_cycle_number = get_numeric(clinical, 'Number of Cycles of 2nd_Additional Therapy')

        tps = sorted(obj.get('timepoints', []), key=tp_index)
        for tp in tps:
            v_total = np.nansum([obj['volumes'].get(tp, {}).get(l, 0.0) for l in active_labels])
            day_val = get_clinical_day(obj, tp)
            kinetic_features.append({
                "PID_Clean": clean_id, "Timepoint": tp, "Day": day_val, "Volume_Total": v_total,
                "ChemoStartDay": chemo_startday, "ChemoEndDay": chemo_endday, "RadStartDay": rad_startday,
                "RadEndDay": rad_endday, "Radiation_Total_Dose": rad_dose, "Radiation_Fractions": rad_fractions,
                "Additional1_StartDay": additional1_startday, "Additional1_EndDay": additional1_endday,
                "Additional1_CycleLength": additional1_cycle_length, "Additional1_CycleNumber": additional1_cycle_number,
                "Additional2_StartDay": additional2_startday, "Additional2_EndDay": additional2_endday,
                "Additional2_CycleLength": additional2_cycle_length, "Additional2_CycleNumber": additional2_cycle_number
            })
            
    return pd.DataFrame(kinetic_features)