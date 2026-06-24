This project provides a computational framework for modeling and forecasting glioma progression using longitudinal MRI and clinical data. It combines mechanistic reaction-diffusion modeling (Fisher-KPP) with modern hybrid machine learning techniques—including Neural ODEs (Continuous Normalizing Flows) and online Gaussian Process regression—to simulate tumor trajectories, evaluate counterfactual treatment scenarios, and generate digital twin insights.

---

# Glioma Digital Twin Forecasting Framework

## Overview

This repository contains a modular Python pipeline designed to process patient-specific clinical and imaging data, estimate tumor growth parameters, and perform longitudinal forecasting. Key capabilities include:

* **Mechanistic Modeling:** Parameter estimation for tumor proliferation ($\rho$) and diffusion ($D$) based on the Fisher-KPP model.
* **Hybrid Forecasting:** Simulation of patient-specific tumor volume trajectories using both mechanistic ODE solvers and neural-enhanced transition models.
* **Dynamic Uncertainty Quantification:** Support for online residual learning using `StreamingResidualGP` to adapt predictions as new longitudinal data arrives.
* **Counterfactual Analysis:** Tools for simulating the effects of varying treatment plans (e.g., radiation, chemotherapy) on projected growth.
* **Validation & Visualization:** Automated batch-generation of diagnostic plots comparing observed MRI volumes to digital twin predictions, complete with uncertainty bands and distribution analysis.

## Project Structure

* `main.py`: The entry point for the patient analysis pipeline, orchestrating data loading and processing.
* `feature_building.py`: Utilities for parsing segmentation volumes, mapping timepoints, and preparing longitudinal feature sets.
* `fitting.py`: Contains the logic for estimating Fisher-KPP parameters ($\rho, D$) via optimization.
* `forecasting.py`: Core logic for trajectory simulation, bootstrap analysis, and counterfactual scenario modeling.
* `neural_ode_cnf.py`: Implementation of Continuous Normalizing Flow (CNF) layers using PyTorch for modeling complex state transitions.
* `online_gp.py`: An online Gaussian Process regressor for residual learning and multi-output forecasting.
* `plotting_validation.py`: Modules for generating cohort-wide distribution statistics and patient-specific visualization traces.

## Getting Started

The pipeline is designed to ingest patient data from structured Excel files and directories containing NIfTI imaging segments.

1. Ensure your data environment is configured with the expected paths in `main.py`.
2. Run the pipeline to process cohorts and compute mechanistic parameters.
3. Use the `plotting_validation` module to inspect longitudinal trends and cohort distributions.

## Requirements

The project relies on standard data science and deep learning libraries:

* `numpy`, `pandas`, `scipy`
* `torch`, `torchdiffeq`
* `scikit-learn`
* `plotly` (for visualizations)
* `nibabel` (for NIfTI image processing)