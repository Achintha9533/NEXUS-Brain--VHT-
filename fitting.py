import os
import contextlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from feature_building import volume_to_radius

def estimate_fisher_kpp_velocity(volume, rho, D, vmax_by_patient, pid, patient_objects):
    volume = float(volume) if pd.notna(volume) else np.nan
    rho = float(rho) if pd.notna(rho) else np.nan
    D = float(D) if pd.notna(D) else np.nan

    patient_obj = patient_objects.get(pid, {})
    V_max = max(float(vmax_by_patient.get(pid, 1.0)), 1.0)

    if pd.isna(volume) or pd.isna(rho) or pd.isna(D) or volume <= 0.0 or volume >= V_max:
        return np.nan, rho, D, V_max

    r_now = volume_to_radius(patient_obj, volume)
    if pd.isna(r_now) or r_now <= 0:
        return np.nan, rho, D, V_max

    volume_ratio = volume / V_max
    v_front = 2.0 * np.sqrt(max(D, 1e-12) * max(rho, 1e-12))
    retardation = 1.0 - volume_ratio

    drdt = v_front * retardation
    dvdt = 4.0 * np.pi * (r_now ** 2) * drdt
    return float(dvdt), rho, D, float(V_max)

def log_reparameterized_loss(search_params, sub_df, V_max_ref, pid, patient_objects,
                             prior_log_rho=np.log(0.01), prior_log_D=np.log(0.02), prior_log_ratio=np.log(100.0),
                             lambda_rho=0.05, lambda_D=0.05, lambda_ratio=0.02,
                             lambda_t1c_flair=0.0, lambda_adc=0.0, t1c_flair_target=np.nan, adc_target=np.nan):
    log_v, log_ratio = search_params
    v = np.exp(log_v)
    ratio = np.exp(log_ratio)

    rho_trial = v / (2.0 * np.sqrt(max(ratio, 1e-12)))
    D_trial = ratio * rho_trial

    errors = []
    for i in range(len(sub_df) - 1):
        row_now = sub_df.iloc[i]
        row_next = sub_df.iloc[i + 1]
        dt = row_next["Day"] - row_now["Day"]
        if dt <= 0:
            continue

        target_dvdt = (row_next["Volume_Total"] - row_now["Volume_Total"]) / dt
        vol_current = row_now["Volume_Total"]
        sim_dvdt, _, _, _ = estimate_fisher_kpp_velocity(
            vol_current, rho_trial, D_trial, {pid: V_max_ref}, pid, patient_objects
        )

        if pd.notna(sim_dvdt) and pd.notna(target_dvdt):
            norm_scale = max(abs(target_dvdt), 1.0)
            errors.append(((sim_dvdt - target_dvdt) / norm_scale) ** 2)

    data_loss = 1e9 if len(errors) == 0 else float(np.mean(errors))

    log_rho = np.log(max(rho_trial, 1e-12))
    log_D = np.log(max(D_trial, 1e-12))
    log_ratio = np.log(max(ratio, 1e-12))

    reg = (
        lambda_rho * (log_rho - prior_log_rho) ** 2 +
        lambda_D * (log_D - prior_log_D) ** 2 +
        lambda_ratio * (log_ratio - prior_log_ratio) ** 2
    )

    imaging_reg = 0.0
    if pd.notna(t1c_flair_target):
        imaging_reg += lambda_t1c_flair * (log_ratio - np.log(max(t1c_flair_target, 1e-12))) ** 2
    if pd.notna(adc_target):
        imaging_reg += lambda_adc * (log_D - np.log(max(adc_target, 1e-12))) ** 2

    return data_loss + reg + imaging_reg

def estimate_patient_rho_D(sub_df, Vmax_ref=500000.0, pid=None, patient_objects=None,
                           n_starts=6, prior_log_rho=np.log(0.01), prior_log_D=np.log(0.02), prior_log_ratio=np.log(100.0),
                           t1c_flair_target=np.nan, adc_target=np.nan):
    if patient_objects is None:
        patient_objects = {}

    sub = sub_df.sort_values("Day").reset_index(drop=True)
    if len(sub) < 2:
        return 0.01, 0.02, False, np.nan, np.nan, np.nan, []

    starts = [
        [np.log(0.01), np.log(1.0)], [np.log(0.05), np.log(10.0)], [np.log(0.1), np.log(100.0)],
        [np.log(0.2), np.log(500.0)], [np.log(0.01), np.log(2000.0)], [np.log(0.001), np.log(5000.0)],
    ]
    bounds_log = [(-8.0, 2.0), (-6.0, 10.0)]

    best_res = None
    all_solutions = []

    for x0 in starts[:max(1, n_starts)]:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            res = minimize(
                log_reparameterized_loss, x0=x0,
                args=(sub, Vmax_ref, pid, patient_objects, prior_log_rho, prior_log_D, prior_log_ratio,
                      0.05, 0.05, 0.02, 0.01, 0.01, t1c_flair_target, adc_target),
                method="L-BFGS-B", bounds=bounds_log
            )

        if res is not None and np.all(np.isfinite(res.x)):
            log_v, log_ratio = res.x
            v = np.exp(log_v)
            ratio = np.exp(log_ratio)
            rho_hat = v / (2.0 * np.sqrt(max(ratio, 1e-12)))
            D_hat = ratio * rho_hat
            all_solutions.append((rho_hat, D_hat, ratio, float(res.fun), bool(res.success), res.x))
            if best_res is None or float(res.fun) < float(best_res.fun):
                best_res = res

    if best_res is None:
        return 0.01, 0.02, False, np.nan, np.nan, np.nan, []

    log_v, log_ratio = best_res.x
    v = np.exp(log_v)
    ratio = np.exp(log_ratio)
    rho_hat = v / (2.0 * np.sqrt(max(ratio, 1e-12)))
    D_hat = ratio * rho_hat

    ratio_vals = np.array([s[2] for s in all_solutions], dtype=float) if all_solutions else np.array([np.nan])
    ident_score = float(np.std(np.log(ratio_vals + 1e-12))) if np.all(np.isfinite(ratio_vals)) and len(ratio_vals) > 1 else np.nan

    return float(rho_hat), float(D_hat), bool(best_res.success), float(best_res.fun), float(v), ident_score, all_solutions

def bootstrap_patient_rho_D(sub_df, Vmax_ref, pid, patient_objects, n_boot=100, noise_sd=0.05,
                            drop_fraction=0.25, t1c_flair_target=np.nan, adc_target=np.nan):
    sub = sub_df.sort_values("Day").reset_index(drop=True)
    if len(sub) < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    rows = []

    for b in range(n_boot):
        boot = sub.copy()
        if len(boot) > 2 and drop_fraction > 0:
            keep_n = max(2, int(np.ceil(len(boot) * (1.0 - drop_fraction))))
            keep_idx = np.sort(rng.choice(np.arange(len(boot)), size=keep_n, replace=False))
            boot = boot.iloc[keep_idx].reset_index(drop=True)

        if noise_sd and noise_sd > 0:
            scale = max(boot["Volume_Total"].median(), 1.0)
            boot["Volume_Total"] = boot["Volume_Total"] + rng.normal(0.0, noise_sd * scale, size=len(boot))
            boot["Volume_Total"] = boot["Volume_Total"].clip(lower=1.0)

        rho_hat, D_hat, success, loss, v_hat, ident_score, _ = estimate_patient_rho_D(
            boot, Vmax_ref=Vmax_ref, pid=pid, patient_objects=patient_objects,
            n_starts=6, t1c_flair_target=t1c_flair_target, adc_target=adc_target
        )

        rows.append({
            "bootstrap_id": b, "rho": rho_hat, "D": D_hat, "v": v_hat,
            "loss": loss, "success": success, "ident_score": ident_score, "n_tp": len(boot)
        })
    return pd.DataFrame(rows)

def summarize_samples(df_samples, prefix=""):
    out = {}
    if df_samples is None or df_samples.empty:
        return out

    for col in ["rho", "D", "v"]:
        if col in df_samples.columns:
            s = pd.to_numeric(df_samples[col], errors="coerce").dropna()
            if len(s) > 0:
                out[f"{prefix}{col}_mean"] = float(s.mean())
                out[f"{prefix}{col}_sd"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
                out[f"{prefix}{col}_cv"] = float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() != 0 else np.nan
                out[f"{prefix}{col}_ci_low"] = float(np.percentile(s, 2.5))
                out[f"{prefix}{col}_ci_high"] = float(np.percentile(s, 97.5))
    return out