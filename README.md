# Stockout-Aware Demand Forecasting on FreshRetailNet-50K

An end-to-end research notebook for forecasting perishable-product demand when observed sales can be cut short by stockouts. It builds past-only features, simulates three stockout histories, compares forecasting and censoring-aware approaches, calibrates demand quantiles, and converts forecasts into inventory orders. The target is the next day's **06:00–21:00** demand, forecast at the end of the previous day.

**Notebook:** [`FreshRetailNet_Phases_0_to_9.ipynb`](FreshRetailNet_Phases_0_to_9.ipynb)  
**Saved run artifacts:** [`FreshRetailNet_Phases_0_to_9_results/`](FreshRetailNet_Phases_0_to_9_results/)

## Results from the included run

| Item | Recorded result |
| --- | --- |
| Dataset audit | Approximately 4.5 million training rows; the modeling run sampled 5,000 product–store series |
| Final evaluation | 20,641 complete-day donor observations in a held-out seven-day window |
| Main comparison | Pooled-conformal demand-quantile LightGBM had **12.2%–21.4% lower mean simulated newsvendor cost** than seasonal naive plus residual quantile across nine matched stockout-mechanism / cost-ratio settings |
| Tested scenarios | Three simulated stockout mechanisms (A/B/C) × three underage/overage cost ratios |
| Compute | Approximately 3 hours on CPU; no GPU used in the recorded run |

The percentage is a **relative reduction in normalized simulated ordering cost**, not an observed reduction in real-world stockouts, waste, or money spent. The comparison is specifically against the seasonal naive baseline; other baselines, including raw-sales LightGBM, are competitive in some settings. Results are for the sampled run (`FAST_MODE=1`, 5,000 series), not a full 50,000-series benchmark.

The underlying figures are in [`final_order_metrics.csv`](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/phase9_evaluation/final_order_metrics.csv). The percentage compares rows for `demand-quantile LightGBM + pooled conformal` and `seasonal_naive + residual quantile` where the training and test stockout mechanisms match, separately for each cost scenario.

## What the notebook does

1. Audits the data, creates a reproducible series sample and split manifest, and checks for time leakage.
2. Constructs three demand-dependent stockout mechanisms with different observed-sales histories.
3. Trains seasonal naive, raw-sales LightGBM, and demand-quantile LightGBM baselines.
4. Explores TimesNet recovery, TFT forecasts, and a Kaplan–Meier censored-inventory baseline.
5. Applies pooled conformal calibration and converts forecast quantiles into newsvendor orders under three cost ratios.
6. Locks the protocol before scoring the final seven-day window, then reports transfer results, bootstrap summaries, subgroups, and ablations.

The notebook records a **partial** reproduction of the upstream deep-learning baseline. Its task-aligned experiments should not be presented as a direct reproduction of published benchmark numbers.

## Data

Download `train.parquet` and `eval.parquet` from the [FreshRetailNet-50K dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) and place both in `data/` beside the notebook:

```text
your-repository/
├── README.md
├── FreshRetailNet_Phases_0_to_9.ipynb
├── FreshRetailNet_Phases_0_to_9_results/
└── data/
    ├── train.parquet
    └── eval.parquet
```

The notebook also accepts a separate data folder through `FRN_DATA_DIR`. Dataset credit: Dingdong / FreshRetailNet-50K; consult its dataset card for the license and source details. Keep the large Parquet files outside the Git repository unless you have a specific distribution reason.

## Run locally

Use a Python environment with Jupyter, NumPy, pandas, PyArrow, LightGBM, DuckDB, SciPy, Matplotlib, and `psutil`. The later phases use PyTorch and related forecasting packages. The notebook checks for several dependencies and attempts to install missing ones when run; an internet connection may be needed on the first run.

From the repository directory on macOS or Linux:

```bash
python -m pip install jupyter numpy pandas pyarrow lightgbm duckdb scipy matplotlib psutil
FRN_DATA_DIR=./data FRN_FAST_MODE=1 FRN_SAMPLE_N=5000 jupyter lab
```

On Windows PowerShell, set the variables before starting Jupyter:

```powershell
python -m pip install jupyter numpy pandas pyarrow lightgbm duckdb scipy matplotlib psutil
$env:FRN_DATA_DIR = ".\data"
$env:FRN_FAST_MODE = "1"
$env:FRN_SAMPLE_N = "5000"
jupyter lab
```

Open the notebook and run its cells **in order, top to bottom, in one kernel**. Allow several hours for the sampled run. The notebook's `FRN_FAST_MODE` default is `0`; setting it to `1` matches the recorded faster experiment. `FRN_FULL_RUN=1` switches to all series and will take substantially more resources.

New outputs go to `frn_outputs/<run tag>/` for phases 0–4 and `frn_phase5_9_outputs/` for phases 5–9. The main comparison table in a new run is `frn_phase5_9_outputs/phase9_evaluation/final_order_metrics.csv`. The output roots can be changed with `FRN_OUTPUT_DIR` and `FRN_PHASE59_OUTPUT_DIR` respectively.

## Results already in this repository

- [Final order metrics](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/phase9_evaluation/final_order_metrics.csv) — policy cost, fill rate, leftover share, and service by scenario and stockout mechanism.
- [Final forecast metrics](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/phase9_evaluation/final_forecast_metrics.csv) — forecasting evaluation.
- [Paired bootstrap differences](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/phase9_evaluation/bootstrap_paired_differences.csv) — uncertainty for policy comparisons.
- [Final run summary](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/manifests/final_summary.json) and [protocol lock](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/manifests/final_test_lock.json) — run status and evaluation record.
- [Cost heatmaps](FreshRetailNet_Phases_0_to_9_results/results_bundle/frn_phase5_9_outputs/figures/p9_cost_heatmaps.png) — mean cost of the calibrated demand-quantile policy across train/test mechanisms.

## Interpretation and limits

- The evaluation uses sales on complete, non-stockout days as a demand proxy. These donors may differ systematically from stockout days.
- Stockout mechanisms A/B/C simulate censoring. They do not measure outcomes from an actual replenishment intervention.
- The final evaluation spans seven dates, so observations are correlated; deep models were trained on a smaller subsample with fewer epochs.
- Normalized newsvendor costs encode hypothetical shortage and excess-stock preferences, not currency values.
- Re-running with a different sample, dependency versions, or compute budget may change the results. Inspect the notebook's provenance and generated metrics before quoting a number elsewhere.

## Source

Dataset: [Dingdong-Inc/FreshRetailNet-50K on Hugging Face](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K). The notebook also references the [upstream baseline repository](https://github.com/Dingdong-Inc/frn-50k-baseline) for its partial reproduction attempt.
