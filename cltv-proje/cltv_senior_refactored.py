"""
Production-grade CLTV analysis for the FLO customer summary dataset.

Methodology
-----------
- Purchase incidence: BG/NBD (Fader, Hardie, and Lee, 2005).
- Spend per transaction: Gamma-Gamma model.
- CLTV horizon: configurable discounted future expected monetary value.

Important statistical definitions
---------------------------------
The ``lifetimes`` package defines frequency as the number of repeat purchases.
Therefore, this pipeline uses ``frequency = total_orders - 1``.
Recency and customer age are expressed in weeks; accordingly, the CLTV call
uses ``freq='W'``.

Dataset limitation
------------------
FLO is supplied as a customer-level summary dataset, not as a transaction log.
It does not expose a genuine future holdout period or repeat-only monetary
value. Consequently:
- no predictive R-squared, MAE, RMSE, MAPE, accuracy, or similar future
  validation metrics are reported;
- monetary_value is approximated using average value across all observed
  transactions and this limitation is reported explicitly;
- bootstrap confidence intervals describe estimation uncertainty conditional
  on observed customer summaries, not future-realisation prediction intervals.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from scipy import stats
from sklearn.preprocessing import StandardScaler


CONFIG: Dict[str, Any] = {
    "project_name": "FLO CLTV Analytics - BG/NBD and Gamma-Gamma",
    "data_path": "flo_data_20k.csv",
    "output_dir": "outputs_cltv_senior",
    "figures_subdir": "figures",
    "log_file": "pipeline.log",
    "analysis_date_offset_days": 2,
    "time_unit": "W",
    "weeks_per_month": 52.1429 / 12,
    "forecast_horizon_months": 6,
    "discount_rate": 0.01,
    "discount_rates_sensitivity": [0.005, 0.01, 0.02, 0.05],
    "bgnbd_penalizer_coef": 0.001,
    "gamma_gamma_penalizer_coef": 0.001,
    "iqr_multiplier": 3.0,
    "outlier_strategy": "diagnose_only",
    "bootstrap_iterations": 1000,
    "bootstrap_mode": "nonparametric_scored_customer_resampling",
    "bootstrap_ci": 0.95,
    "random_state": 42,
    "kmeans_min_k": 2,
    "kmeans_max_k": 6,
    "silhouette_sample_size": 1000,
    "quantile_segments": ["D", "C", "B", "A"],
    "segment_order": ["A", "B", "C", "D"],
    "palette": {"A": "#1B6B3A", "B": "#1976D2", "C": "#F57C00", "D": "#C62828"},
    "matrix_palette": {
        "High CLTV + High Alive": "#1B6B3A",
        "High CLTV + Low Alive": "#F57C00",
        "Low CLTV + High Alive": "#1976D2",
        "Low CLTV + Low Alive": "#C62828",
    },
    "required_columns": [
        "master_id",
        "order_channel",
        "last_order_channel",
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline",
        "order_num_total_ever_online",
        "order_num_total_ever_offline",
        "customer_value_total_ever_offline",
        "customer_value_total_ever_online",
        "interested_in_categories_12",
    ],
    "date_columns": [
        "first_order_date",
        "last_order_date",
        "last_order_date_online",
        "last_order_date_offline",
    ],
    "numeric_source_columns": [
        "order_num_total_ever_online",
        "order_num_total_ever_offline",
        "customer_value_total_ever_offline",
        "customer_value_total_ever_online",
    ],
}

LOGGER = logging.getLogger("cltv_pipeline")


class CLTVPipelineError(RuntimeError):
    """Raised when the pipeline cannot produce methodologically valid results."""


def configure_logging(output_dir: Path) -> None:
    """Configure file and console logging with INFO, WARNING, and ERROR levels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / CONFIG["log_file"], encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)


def configure_plot_style() -> None:
    """Apply consistent publication-ready figure formatting."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "font.family": "DejaVu Sans",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    """Save a figure as report-ready PNG and close its matplotlib object."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    LOGGER.info("Figure exported: %s", path)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the FLO dataset and validate the availability of required columns."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise CLTVPipelineError(f"Dataset could not be read: {exc}") from exc

    missing_columns = sorted(set(CONFIG["required_columns"]) - set(df.columns))
    if missing_columns:
        raise CLTVPipelineError(f"Required columns are missing: {missing_columns}")
    LOGGER.info("Dataset loaded: %s rows, %s columns.", f"{len(df):,}", df.shape[1])
    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date fields to datetime and fail fast on unparseable observations."""
    result = df.copy()
    for column in CONFIG["date_columns"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    invalid_date_rows = result[CONFIG["date_columns"]].isna().any(axis=1).sum()
    if invalid_date_rows:
        raise CLTVPipelineError(
            f"Date parsing produced missing dates in {invalid_date_rows} observations."
        )
    return result


def create_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a column-level data quality summary table for auditability."""
    records = []
    for column in df.columns:
        numeric = pd.api.types.is_numeric_dtype(df[column])
        records.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": float(df[column].isna().mean() * 100),
                "unique_count": int(df[column].nunique(dropna=True)),
                "negative_count": int((df[column] < 0).sum()) if numeric else np.nan,
                "zero_count": int((df[column] == 0).sum()) if numeric else np.nan,
                "minimum": float(df[column].min()) if numeric else np.nan,
                "maximum": float(df[column].max()) if numeric else np.nan,
            }
        )
    summary = pd.DataFrame(records)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["master_id"].duplicated().sum())
    LOGGER.info("Data quality check: duplicate rows=%s, duplicate customer IDs=%s.", duplicate_rows, duplicate_ids)
    if summary["missing_count"].sum() > 0 or duplicate_rows > 0 or duplicate_ids > 0:
        LOGGER.warning("Data quality exceptions detected. Review data_quality_summary.csv.")
    return summary


def create_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate extreme observations using conservative 3-IQR diagnostic fences.

    High-spend customers are economically meaningful in CLTV analysis. Therefore,
    the default strategy records extreme observations but does not automatically
    truncate them. This avoids systematically suppressing the most valuable tail.
    """
    prepared = df.copy()
    prepared["total_orders"] = (
        prepared["order_num_total_ever_online"] + prepared["order_num_total_ever_offline"]
    )
    prepared["total_value"] = (
        prepared["customer_value_total_ever_online"] + prepared["customer_value_total_ever_offline"]
    )
    prepared["monetary_value"] = prepared["total_value"] / prepared["total_orders"]

    columns = CONFIG["numeric_source_columns"] + ["total_orders", "total_value", "monetary_value"]
    multiplier = CONFIG["iqr_multiplier"]
    rows = []
    for column in columns:
        q1 = prepared[column].quantile(0.25)
        q3 = prepared[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        low_count = int((prepared[column] < lower).sum())
        high_count = int((prepared[column] > upper).sum())
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_fence_3iqr": lower,
                "upper_fence_3iqr": upper,
                "below_fence_count": low_count,
                "above_fence_count": high_count,
                "flagged_pct": (low_count + high_count) / len(prepared) * 100,
                "treatment": "retained_for_model_diagnostic_only",
            }
        )
    report = pd.DataFrame(rows)
    LOGGER.info(
        "Outlier handling strategy: %s. Extreme observations are reported, not winsorized.",
        CONFIG["outlier_strategy"],
    )
    return report


def build_preprocessing_report(df: pd.DataFrame, outlier_report: pd.DataFrame) -> pd.DataFrame:
    """Create an explicit report of preprocessing choices and their rationale."""
    total_orders = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    total_value = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    invalid_orders = int((total_orders < 1).sum())
    invalid_value = int((total_value <= 0).sum())
    records = [
        {
            "step": "Date conversion",
            "rule": "Parse four transaction date columns as datetime",
            "affected_records": 0,
            "decision": "Retained all successfully parsed rows",
            "rationale": "Recency and T require valid temporal fields.",
        },
        {
            "step": "Order validity",
            "rule": "total_orders >= 1",
            "affected_records": invalid_orders,
            "decision": "Exclude invalid rows if present",
            "rationale": "BG/NBD customer histories require an observed initial purchase.",
        },
        {
            "step": "Monetary validity",
            "rule": "total_value > 0 and monetary_value > 0",
            "affected_records": invalid_value,
            "decision": "Exclude invalid rows if present",
            "rationale": "Gamma-Gamma requires positive observed monetary values.",
        },
        {
            "step": "Extreme-value review",
            "rule": f"Diagnostic fences: Q1/Q3 +/- {CONFIG['iqr_multiplier']} * IQR",
            "affected_records": int(outlier_report["above_fence_count"].sum()),
            "decision": "Do not automatically cap observed values",
            "rationale": "High-value customers may be legitimate and central to CLTV concentration.",
        },
        {
            "step": "Repeat purchase definition",
            "rule": "frequency = total_orders - 1",
            "affected_records": len(df),
            "decision": "Methodological correction applied",
            "rationale": "BG/NBD/lifetimes frequency denotes repeat purchases after the first transaction.",
        },
        {
            "step": "Monetary approximation",
            "rule": "monetary_value = total_value / total_orders",
            "affected_records": len(df),
            "decision": "Use with stated limitation",
            "rationale": "The summary dataset does not expose repeat-purchase-only spend.",
        },
    ]
    return pd.DataFrame(records)


def engineer_cltv_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Construct BG/NBD and Gamma-Gamma inputs using correct repeat frequency."""
    prepared = df.copy()
    analysis_date = prepared["last_order_date"].max() + pd.Timedelta(
        days=CONFIG["analysis_date_offset_days"]
    )
    prepared["total_orders"] = (
        prepared["order_num_total_ever_online"] + prepared["order_num_total_ever_offline"]
    )
    prepared["total_value"] = (
        prepared["customer_value_total_ever_online"] + prepared["customer_value_total_ever_offline"]
    )
    prepared["frequency"] = prepared["total_orders"] - 1
    prepared["recency_weekly"] = (
        (prepared["last_order_date"] - prepared["first_order_date"]).dt.days / 7.0
    )
    prepared["T_weekly"] = (analysis_date - prepared["first_order_date"]).dt.days / 7.0
    prepared["monetary_value"] = prepared["total_value"] / prepared["total_orders"]

    valid_mask = (
        (prepared["total_orders"] >= 1)
        & (prepared["frequency"] >= 0)
        & (prepared["monetary_value"] > 0)
        & (prepared["recency_weekly"] >= 0)
        & (prepared["T_weekly"] > 0)
        & (prepared["T_weekly"] >= prepared["recency_weekly"])
    )
    excluded = int((~valid_mask).sum())
    if excluded:
        LOGGER.warning("Excluded %s invalid customer records before model fitting.", excluded)
    model_df = prepared.loc[valid_mask].copy().set_index("master_id")
    if model_df.empty:
        raise CLTVPipelineError("No valid customers remained after preprocessing.")
    LOGGER.info("Analysis date: %s. Modelling population: %s customers.", analysis_date.date(), f"{len(model_df):,}")
    return model_df, analysis_date


def assess_bgnbd_inputs(cltv_df: pd.DataFrame) -> pd.DataFrame:
    """Document observable BG/NBD input checks and non-testable assumptions."""
    checks = [
        {
            "assumption_or_check": "Non-contractual setting",
            "assessment": "Applicable",
            "evidence": "Customer purchase summaries contain no contractual churn event.",
            "status": "Supported by business context",
        },
        {
            "assumption_or_check": "frequency is repeat transactions",
            "assessment": "frequency = total_orders - 1",
            "evidence": f"Minimum frequency = {cltv_df['frequency'].min():.0f}",
            "status": "Verified",
        },
        {
            "assumption_or_check": "recency <= T",
            "assessment": "Temporal consistency",
            "evidence": f"Violations = {(cltv_df['recency_weekly'] > cltv_df['T_weekly']).sum()}",
            "status": "Verified",
        },
        {
            "assumption_or_check": "Poisson purchasing while alive",
            "assessment": "Latent behavioural assumption",
            "evidence": "Cannot be directly proven from aggregated customer summaries.",
            "status": "Model assumption; interpret with caution",
        },
        {
            "assumption_or_check": "Transaction-rate/dropout independence",
            "assessment": "Latent behavioural assumption",
            "evidence": "Requires richer transaction history or alternative model comparison.",
            "status": "Not directly testable in supplied data",
        },
    ]
    return pd.DataFrame(checks)


def fit_bgnbd(cltv_df: pd.DataFrame, quiet: bool = False) -> BetaGeoFitter:
    """Fit a BG/NBD model to valid customer summary data."""
    try:
        model = BetaGeoFitter(penalizer_coef=CONFIG["bgnbd_penalizer_coef"])
        model.fit(
            cltv_df["frequency"],
            cltv_df["recency_weekly"],
            cltv_df["T_weekly"],
            verbose=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise CLTVPipelineError(f"BG/NBD fitting failed: {exc}") from exc
    if not quiet:
        LOGGER.info("BG/NBD fitted. Parameters: %s", model.params_.round(6).to_dict())
    return model


def generate_bgnbd_outputs(cltv_df: pd.DataFrame, bgf: BetaGeoFitter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add expected transactions and probability alive, plus diagnostic summaries."""
    scored = cltv_df.copy()
    weeks_3m = 3 * CONFIG["weeks_per_month"]
    weeks_6m = CONFIG["forecast_horizon_months"] * CONFIG["weeks_per_month"]
    scored["expected_purchases_3m"] = bgf.predict(
        weeks_3m, scored["frequency"], scored["recency_weekly"], scored["T_weekly"]
    )
    scored["expected_purchases_6m"] = bgf.predict(
        weeks_6m, scored["frequency"], scored["recency_weekly"], scored["T_weekly"]
    )
    scored["probability_alive"] = bgf.conditional_probability_alive(
        scored["frequency"], scored["recency_weekly"], scored["T_weekly"]
    )
    diagnostics = (
        scored.assign(
            frequency_band=pd.cut(
                scored["frequency"],
                bins=[-0.1, 1, 2, 4, 9, 19, np.inf],
                labels=["0-1", "2", "3-4", "5-9", "10-19", "20+"],
            )
        )
        .groupby("frequency_band", observed=True)
        .agg(
            customers=("frequency", "size"),
            observed_repeat_frequency_mean=("frequency", "mean"),
            predicted_purchases_3m_mean=("expected_purchases_3m", "mean"),
            predicted_purchases_6m_mean=("expected_purchases_6m", "mean"),
            probability_alive_mean=("probability_alive", "mean"),
        )
        .reset_index()
    )
    return scored, diagnostics


def assess_gamma_gamma_assumption(cltv_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assess frequency-monetary independence on repeat purchasers only."""
    repeat = cltv_df.loc[cltv_df["frequency"] > 0].copy()
    if len(repeat) < 3:
        raise CLTVPipelineError("Gamma-Gamma requires sufficient repeat purchasers with frequency > 0.")
    pearson_r, pearson_p = stats.pearsonr(repeat["frequency"], repeat["monetary_value"])
    spearman_r, spearman_p = stats.spearmanr(repeat["frequency"], repeat["monetary_value"])
    max_abs_corr = max(abs(pearson_r), abs(spearman_r))
    if max_abs_corr < 0.30:
        verdict = "Weak association; Gamma-Gamma use is reasonable with the monetary approximation limitation."
    elif max_abs_corr < 0.50:
        verdict = "Moderate association; apply Gamma-Gamma cautiously and disclose potential dependence."
    else:
        verdict = "Strong association; Gamma-Gamma independence assumption is materially questionable."
    assumption = pd.DataFrame(
        [
            {"measure": "Pearson correlation", "coefficient": pearson_r, "p_value": pearson_p, "interpretation": verdict},
            {"measure": "Spearman correlation", "coefficient": spearman_r, "p_value": spearman_p, "interpretation": verdict},
        ]
    )
    LOGGER.info("Gamma-Gamma assumption correlations: Pearson=%.4f, Spearman=%.4f.", pearson_r, spearman_r)
    return repeat, assumption


def fit_gamma_gamma(repeat_df: pd.DataFrame, quiet: bool = False) -> GammaGammaFitter:
    """Fit the Gamma-Gamma monetary-value model to repeat purchasers."""
    try:
        model = GammaGammaFitter(penalizer_coef=CONFIG["gamma_gamma_penalizer_coef"])
        model.fit(repeat_df["frequency"], repeat_df["monetary_value"], verbose=False)
    except (ValueError, RuntimeError) as exc:
        raise CLTVPipelineError(f"Gamma-Gamma fitting failed: {exc}") from exc
    if not quiet:
        LOGGER.info("Gamma-Gamma fitted. Parameters: %s", model.params_.round(6).to_dict())
    return model


def score_cltv(
    cltv_df: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    discount_rate: float,
) -> pd.DataFrame:
    """Score customers with expected monetary value and discounted future CLTV."""
    scored = cltv_df.copy()
    scored["expected_average_value"] = ggf.conditional_expected_average_profit(
        scored["frequency"], scored["monetary_value"]
    )
    scored["cltv"] = ggf.customer_lifetime_value(
        bgf,
        scored["frequency"],
        scored["recency_weekly"],
        scored["T_weekly"],
        scored["monetary_value"],
        time=CONFIG["forecast_horizon_months"],
        discount_rate=discount_rate,
        freq=CONFIG["time_unit"],
    )
    if scored["cltv"].isna().any() or (scored["cltv"] < 0).any():
        raise CLTVPipelineError("Invalid CLTV predictions were generated.")
    return scored


def create_quantile_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """Assign A-D CLTV quartile segments where A contains the highest values."""
    result = scored.copy()
    result["cltv_segment"] = pd.qcut(
        result["cltv"], q=4, labels=CONFIG["quantile_segments"], duplicates="raise"
    )
    return result


def create_value_alive_matrix(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine CLTV and probability alive into an actionable four-cell matrix."""
    result = scored.copy()
    cltv_cut = result["cltv"].median()
    alive_cut = result["probability_alive"].median()
    high_cltv = result["cltv"] >= cltv_cut
    high_alive = result["probability_alive"] >= alive_cut
    result["value_alive_segment"] = np.select(
        [high_cltv & high_alive, high_cltv & ~high_alive, ~high_cltv & high_alive],
        ["High CLTV + High Alive", "High CLTV + Low Alive", "Low CLTV + High Alive"],
        default="Low CLTV + Low Alive",
    )
    interpretations = {
        "High CLTV + High Alive": "Protect and expand: VIP retention, premium cross-sell and service priority.",
        "High CLTV + Low Alive": "Urgent win-back: high economic value but elevated inactivity risk.",
        "Low CLTV + High Alive": "Develop efficiently: automated cross-sell and basket-building journeys.",
        "Low CLTV + Low Alive": "Low-cost treatment: suppress expensive incentives; use scalable reactivation only.",
    }
    summary = (
        result.groupby("value_alive_segment", observed=True)
        .agg(
            customers=("cltv", "size"),
            average_cltv=("cltv", "mean"),
            total_cltv=("cltv", "sum"),
            average_probability_alive=("probability_alive", "mean"),
            average_frequency=("frequency", "mean"),
        )
        .reset_index()
    )
    summary["cltv_share_pct"] = summary["total_cltv"] / result["cltv"].sum() * 100
    summary["managerial_interpretation"] = summary["value_alive_segment"].map(interpretations)
    return result, summary


def summarize_quantile_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """Produce a business-ready profile of CLTV quantile segments."""
    summary = (
        scored.groupby("cltv_segment", observed=True)
        .agg(
            customers=("cltv", "size"),
            average_cltv=("cltv", "mean"),
            median_cltv=("cltv", "median"),
            total_cltv=("cltv", "sum"),
            average_probability_alive=("probability_alive", "mean"),
            average_expected_purchases_6m=("expected_purchases_6m", "mean"),
            average_expected_value=("expected_average_value", "mean"),
            average_frequency=("frequency", "mean"),
        )
        .reindex(CONFIG["segment_order"])
        .reset_index()
    )
    summary["customer_share_pct"] = summary["customers"] / len(scored) * 100
    summary["cltv_share_pct"] = summary["total_cltv"] / scored["cltv"].sum() * 100
    return summary


def _numpy_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int = 10,
    max_iter: int = 100,
    tolerance: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """Fit deterministic multistart KMeans using vectorised NumPy operations."""
    best_labels: Optional[np.ndarray] = None
    best_inertia = np.inf
    for restart in range(n_init):
        rng = np.random.default_rng(random_state + restart)
        centroids = features[rng.choice(len(features), size=n_clusters, replace=False)].copy()
        for _ in range(max_iter):
            distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centroids = centroids.copy()
            for cluster in range(n_clusters):
                members = features[labels == cluster]
                if len(members):
                    new_centroids[cluster] = members.mean(axis=0)
                else:
                    new_centroids[cluster] = features[rng.integers(0, len(features))]
            if np.max(np.abs(new_centroids - centroids)) < tolerance:
                centroids = new_centroids
                break
            centroids = new_centroids
        final_distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = final_distances.argmin(axis=1)
        inertia = float(np.sum(final_distances[np.arange(len(features)), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    if best_labels is None:
        raise CLTVPipelineError("KMeans failed to allocate customer clusters.")
    return best_labels, best_inertia


def _sampled_silhouette(features: np.ndarray, labels: np.ndarray, sample_limit: int) -> float:
    """Compute a stratified approximate silhouette score for scalable reporting."""
    rng = np.random.default_rng(CONFIG["random_state"])
    clusters = np.unique(labels)
    per_cluster = max(2, sample_limit // len(clusters))
    sampled = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == cluster), size=min(per_cluster, int((labels == cluster).sum())), replace=False)
            for cluster in clusters
        ]
    )
    x_sample = features[sampled]
    y_sample = labels[sampled]
    distances = np.sqrt(((x_sample[:, None, :] - x_sample[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for index, cluster in enumerate(y_sample):
        same_mask = y_sample == cluster
        same_mask[index] = False
        a_value = distances[index, same_mask].mean() if same_mask.any() else 0.0
        other_means = [distances[index, y_sample == other].mean() for other in clusters if other != cluster]
        b_value = min(other_means)
        denominator = max(a_value, b_value)
        scores.append((b_value - a_value) / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def run_kmeans_comparison(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run exploratory KMeans grouping on estimated value features.

    KMeans is a post-estimation segmentation comparison, not a predictive CLTV
    model and not a validation test of BG/NBD or Gamma-Gamma predictions.
    """
    features = pd.DataFrame(
        {
            "log_cltv": np.log1p(scored["cltv"]),
            "probability_alive": scored["probability_alive"],
            "log_frequency": np.log1p(scored["frequency"]),
        },
        index=scored.index,
    )
    scaled = StandardScaler().fit_transform(features)
    search_rows = []
    model_labels: Dict[int, np.ndarray] = {}
    for k in range(CONFIG["kmeans_min_k"], CONFIG["kmeans_max_k"] + 1):
        labels, inertia = _numpy_kmeans(
            scaled, n_clusters=k, random_state=CONFIG["random_state"], n_init=10
        )
        search_rows.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette_score": _sampled_silhouette(
                    scaled, labels, min(CONFIG["silhouette_sample_size"], len(scored))
                ),
                "silhouette_method": "Stratified approximate score on scaled feature sample",
            }
        )
        model_labels[k] = labels
    selection = pd.DataFrame(search_rows)
    selected_k = int(selection.loc[selection["silhouette_score"].idxmax(), "k"])
    result = scored.copy()
    result["kmeans_cluster"] = model_labels[selected_k].astype(str)
    profile = (
        result.groupby("kmeans_cluster")
        .agg(
            customers=("cltv", "size"),
            average_cltv=("cltv", "mean"),
            median_cltv=("cltv", "median"),
            total_cltv=("cltv", "sum"),
            average_probability_alive=("probability_alive", "mean"),
            average_frequency=("frequency", "mean"),
        )
        .reset_index()
        .sort_values("average_cltv", ascending=False)
    )
    profile["selected_k"] = selected_k
    profile["cltv_share_pct"] = profile["total_cltv"] / result["cltv"].sum() * 100
    LOGGER.info("KMeans exploratory comparison selected k=%s by approximate silhouette score.", selected_k)
    return result, selection, profile


def kmeans_segment_crosstab(scored: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate KMeans clusters against CLTV quartile segments."""
    cross = pd.crosstab(scored["cltv_segment"], scored["kmeans_cluster"], margins=True)
    return cross.reset_index()


def gini_coefficient(values: Iterable[float]) -> float:
    """Calculate the Gini concentration coefficient for nonnegative CLTV values."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0 or array.sum() <= 0 or (array < 0).any():
        raise CLTVPipelineError("Gini requires nonnegative, non-empty CLTV values with a positive sum.")
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * array) / (n * array.sum())) - ((n + 1) / n))


def concentration_analysis(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Compute Lorenz/Pareto data, Gini coefficient and top-customer contributions."""
    ordered_low = scored.sort_values("cltv").copy()
    ordered_low["customer_cumulative_pct"] = np.arange(1, len(ordered_low) + 1) / len(ordered_low) * 100
    ordered_low["cltv_cumulative_pct"] = ordered_low["cltv"].cumsum() / ordered_low["cltv"].sum() * 100
    lorenz = ordered_low[["customer_cumulative_pct", "cltv_cumulative_pct"]].reset_index(drop=True)

    ordered_high = scored.sort_values("cltv", ascending=False).copy()
    ordered_high["customer_cumulative_pct"] = np.arange(1, len(ordered_high) + 1) / len(ordered_high) * 100
    ordered_high["cltv_cumulative_pct"] = ordered_high["cltv"].cumsum() / ordered_high["cltv"].sum() * 100
    top10_n = max(1, int(np.ceil(len(ordered_high) * 0.10)))
    top20_n = max(1, int(np.ceil(len(ordered_high) * 0.20)))
    summary = pd.DataFrame(
        [
            {"customer_group": "Top 10%", "customer_count": top10_n, "cltv_contribution_pct": ordered_high.iloc[top10_n - 1]["cltv_cumulative_pct"]},
            {"customer_group": "Top 20%", "customer_count": top20_n, "cltv_contribution_pct": ordered_high.iloc[top20_n - 1]["cltv_cumulative_pct"]},
        ]
    )
    gini = gini_coefficient(scored["cltv"])
    return lorenz, summary, gini


def sensitivity_analysis(scored: pd.DataFrame, bgf: BetaGeoFitter, ggf: GammaGammaFitter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assess CLTV stability under alternative monthly discount-rate assumptions."""
    summary_rows = []
    segment_rows = []
    for rate in CONFIG["discount_rates_sensitivity"]:
        rate_scores = score_cltv(scored, bgf, ggf, rate)
        summary_rows.append(
            {
                "discount_rate": rate,
                "average_cltv": rate_scores["cltv"].mean(),
                "median_cltv": rate_scores["cltv"].median(),
                "total_cltv": rate_scores["cltv"].sum(),
            }
        )
        temp = rate_scores.assign(cltv_segment=scored["cltv_segment"])
        grouped = temp.groupby("cltv_segment", observed=True)["cltv"].agg(["mean", "median", "sum"]).reset_index()
        grouped["discount_rate"] = rate
        segment_rows.append(grouped.rename(columns={"mean": "average_cltv", "median": "median_cltv", "sum": "total_cltv"}))
    return pd.DataFrame(summary_rows), pd.concat(segment_rows, ignore_index=True)


def bootstrap_cltv_confidence_intervals(
    scored: pd.DataFrame,
    iterations: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate percentile confidence intervals through customer resampling.

    The probabilistic CLTV model is fitted once and treated as the scoring rule.
    Bootstrap samples are drawn from the scored customer portfolio, overall and
    within each fixed CLTV segment. Thus, intervals quantify customer-sampling
    uncertainty in current-portfolio mean CLTV conditional on the fitted model.
    They do not quantify model-parameter uncertainty or future realised revenue.
    """
    rng = np.random.default_rng(CONFIG["random_state"])
    segment_order = CONFIG["segment_order"]
    cltv_values = scored["cltv"].to_numpy()
    segment_values = {
        segment: scored.loc[scored["cltv_segment"] == segment, "cltv"].to_numpy()
        for segment in segment_order
    }
    replicate_rows = []
    for iteration in range(1, iterations + 1):
        row: Dict[str, float] = {
            "iteration": iteration,
            "overall_mean": float(rng.choice(cltv_values, size=len(cltv_values), replace=True).mean()),
        }
        for segment, values in segment_values.items():
            row[f"segment_{segment}_mean"] = float(
                rng.choice(values, size=len(values), replace=True).mean()
            )
        replicate_rows.append(row)
    replicates = pd.DataFrame(replicate_rows)
    alpha = (1 - CONFIG["bootstrap_ci"]) / 2
    base_metric_values: Dict[str, float] = {"overall_mean": float(scored["cltv"].mean())}
    for segment in segment_order:
        base_metric_values[f"segment_{segment}_mean"] = float(segment_values[segment].mean())
    interval_rows = []
    for metric in ["overall_mean"] + [f"segment_{segment}_mean" for segment in segment_order]:
        interval_rows.append(
            {
                "metric": metric,
                "mean_cltv": base_metric_values[metric],
                "lower_bound_95": float(replicates[metric].quantile(alpha)),
                "upper_bound_95": float(replicates[metric].quantile(1 - alpha)),
                "successful_iterations": len(replicates),
                "bootstrap_mode": CONFIG["bootstrap_mode"],
                "interpretation": "Portfolio sampling uncertainty conditional on fitted CLTV model; not a future prediction interval.",
            }
        )
    LOGGER.info("Bootstrap completed: %s portfolio resamples.", len(replicates))
    return pd.DataFrame(interval_rows), replicates


def build_executive_summary(
    scored: pd.DataFrame,
    segment_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    gini: float,
    analysis_date: pd.Timestamp,
) -> pd.DataFrame:
    """Produce headline portfolio measures for executive reporting."""
    top_segment = segment_summary.sort_values("total_cltv", ascending=False).iloc[0]
    top20 = concentration.loc[concentration["customer_group"] == "Top 20%", "cltv_contribution_pct"].iloc[0]
    return pd.DataFrame(
        [
            {"metric": "Analysis date", "value": str(analysis_date.date())},
            {"metric": "Customer count", "value": int(len(scored))},
            {"metric": "Average 6-month CLTV (TL)", "value": float(scored["cltv"].mean())},
            {"metric": "Median 6-month CLTV (TL)", "value": float(scored["cltv"].median())},
            {"metric": "Gini coefficient", "value": float(gini)},
            {"metric": "Highest-value quantile segment", "value": str(top_segment["cltv_segment"])},
            {"metric": "Highest-value segment CLTV share (%)", "value": float(top_segment["cltv_share_pct"])},
            {"metric": "Top 20% CLTV contribution (%)", "value": float(top20)},
            {"metric": "Average probability alive", "value": float(scored["probability_alive"].mean())},
            {"metric": "Median probability alive", "value": float(scored["probability_alive"].median())},
        ]
    )


def create_visualizations(
    scored: pd.DataFrame,
    segment_summary: pd.DataFrame,
    lorenz: pd.DataFrame,
    concentration: pd.DataFrame,
    sensitivity: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
    kmeans_selection: pd.DataFrame,
    bgf: BetaGeoFitter,
    figures_dir: Path,
) -> None:
    """Create the ten required publication-quality statistical and managerial figures."""
    palette = CONFIG["palette"]
    segment_order = CONFIG["segment_order"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(scored["cltv"], bins=50, kde=True, ax=ax, color="#1B6B3A")
    ax.axvline(scored["cltv"].mean(), linestyle="--", label=f"Mean: {scored['cltv'].mean():.2f} TL")
    ax.axvline(scored["cltv"].median(), linestyle=":", label=f"Median: {scored['cltv'].median():.2f} TL")
    ax.set(title="Distribution of 6-Month Discounted CLTV", xlabel="CLTV (TL)", ylabel="Customers")
    ax.legend()
    save_figure(fig, "01_cltv_distribution", figures_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(scored["probability_alive"], bins=40, kde=True, ax=ax, color="#1976D2")
    ax.axvline(scored["probability_alive"].median(), linestyle="--", label=f"Median: {scored['probability_alive'].median():.3f}")
    ax.set(title="Distribution of BG/NBD Probability Alive", xlabel="Probability Alive", ylabel="Customers")
    ax.legend()
    save_figure(fig, "02_probability_alive_distribution", figures_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = scored["cltv_segment"].value_counts().reindex(segment_order)
    ax.bar(counts.index, counts.values, color=[palette[x] for x in segment_order])
    ax.set(title="Customer Count by CLTV Quantile Segment", xlabel="CLTV Segment", ylabel="Customers")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom")
    save_figure(fig, "03_segment_distribution", figures_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = segment_summary.set_index("cltv_segment").reindex(segment_order)
    ax.bar(ordered.index, ordered["average_cltv"], color=[palette[x] for x in segment_order])
    ax.set(title="Average CLTV by Segment", xlabel="CLTV Segment", ylabel="Average 6-Month CLTV (TL)")
    for idx, value in enumerate(ordered["average_cltv"]):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    save_figure(fig, "04_segment_average_cltv", figures_dir)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(lorenz["customer_cumulative_pct"], lorenz["cltv_cumulative_pct"], label="CLTV Lorenz curve")
    ax.plot([0, 100], [0, 100], linestyle="--", label="Line of equality")
    ax.fill_between(lorenz["customer_cumulative_pct"], lorenz["cltv_cumulative_pct"], lorenz["customer_cumulative_pct"], alpha=0.15)
    ax.set(title="Lorenz Curve of CLTV Concentration", xlabel="Cumulative Customers (%)", ylabel="Cumulative CLTV (%)")
    ax.legend()
    save_figure(fig, "05_lorenz_curve", figures_dir)

    ordered_desc = scored.sort_values("cltv", ascending=False).copy()
    ordered_desc["customer_pct"] = np.arange(1, len(ordered_desc) + 1) / len(ordered_desc) * 100
    ordered_desc["cltv_pct"] = ordered_desc["cltv"].cumsum() / ordered_desc["cltv"].sum() * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ordered_desc["customer_pct"], ordered_desc["cltv_pct"], label="Cumulative CLTV contribution")
    for group, linestyle in [(10, ":"), (20, "--")]:
        contribution = concentration.loc[concentration["customer_group"] == f"Top {group}%", "cltv_contribution_pct"].iloc[0]
        ax.axvline(group, linestyle=linestyle, label=f"Top {group}%: {contribution:.1f}% of CLTV")
    ax.set(title="Pareto Analysis of Estimated CLTV", xlabel="Top Customers Included (%)", ylabel="Cumulative CLTV Contribution (%)")
    ax.legend()
    save_figure(fig, "06_pareto_curve", figures_dir)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=scored, x="probability_alive", y="cltv", hue="kmeans_cluster", palette="tab10", alpha=0.55, ax=ax)
    ax.set(title="Exploratory KMeans Segmentation", xlabel="Probability Alive", ylabel="CLTV (TL)")
    save_figure(fig, "07_kmeans_visualization", figures_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sensitivity["discount_rate"] * 100, sensitivity["average_cltv"], marker="o", label="Average CLTV")
    ax.plot(sensitivity["discount_rate"] * 100, sensitivity["median_cltv"], marker="s", label="Median CLTV")
    ax.set(title="CLTV Sensitivity to Monthly Discount Rate", xlabel="Monthly Discount Rate (%)", ylabel="6-Month CLTV (TL)")
    ax.legend()
    save_figure(fig, "08_sensitivity_analysis", figures_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(bootstrap_intervals))
    means = bootstrap_intervals["mean_cltv"].to_numpy()
    lower = means - bootstrap_intervals["lower_bound_95"].to_numpy()
    upper = bootstrap_intervals["upper_bound_95"].to_numpy() - means
    ax.errorbar(positions, means, yerr=[lower, upper], fmt="o", capsize=5)
    ax.set_xticks(positions, bootstrap_intervals["metric"], rotation=30, ha="right")
    ax.set(title="Bootstrap 95% Confidence Intervals for Mean CLTV", xlabel="Portfolio Metric", ylabel="Mean CLTV (TL)")
    save_figure(fig, "09_bootstrap_confidence_intervals", figures_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.sca(axes[0])
    plot_period_transactions(bgf)
    axes[0].set_title("BG/NBD Calibration-Period Fit Diagnostic")
    axes[1].plot(kmeans_selection["k"], kmeans_selection["silhouette_score"], marker="o", label="Silhouette")
    axes[1].set(title="KMeans Selection Diagnostic", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[1].legend()
    fig.suptitle("Model and Segmentation Diagnostics")
    fig.tight_layout()
    save_figure(fig, "10_bgnbd_diagnostics", figures_dir)


def export_outputs(outputs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Export all result, audit, diagnostic, and uncertainty tables as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, table in outputs.items():
        LOGGER.info("Exporting CSV: %s with shape %s", key, table.shape)
        table.to_csv(output_dir / f"{key}.csv", index=False, encoding="utf-8-sig")
    LOGGER.info("CSV audit package exported to %s.", output_dir)


def export_required_excel_files(output_dir: Path) -> None:
    """Serialize the five required CSV deliverables to separate Excel workbooks.

    This function is deliberately executed in a clean process after numerical
    model fitting and figure rendering, preventing native-thread contention
    between numerical libraries and the Excel serialization engine.
    """
    required_excel_exports = [
        "cltv_results",
        "segment_summary",
        "kmeans_summary",
        "sensitivity_analysis",
        "bootstrap_results",
    ]
    for name in required_excel_exports:
        csv_path = output_dir / f"{name}.csv"
        if not csv_path.exists():
            raise CLTVPipelineError(f"Excel export source does not exist: {csv_path}")
        safe_table = pd.read_csv(csv_path)
        workbook_path = output_dir / f"{name}.xlsx"
        LOGGER.info("Exporting Excel: %s with shape %s", name, safe_table.shape)
        with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
            safe_table.to_excel(writer, sheet_name=name[:31], index=False)
    LOGGER.info("Required Excel outputs exported to %s.", output_dir)


def export_executive_summary_text(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write the executive summary in a plain-language markdown report."""
    values = dict(zip(summary["metric"], summary["value"]))
    text = "# Executive Summary - FLO CLTV Analysis\n\n"
    text += f"- Analysed customer count: {int(values['Customer count']):,}\n"
    text += f"- Average estimated 6-month CLTV: {float(values['Average 6-month CLTV (TL)']):,.2f} TL\n"
    text += f"- Median estimated 6-month CLTV: {float(values['Median 6-month CLTV (TL)']):,.2f} TL\n"
    text += f"- Gini concentration coefficient: {float(values['Gini coefficient']):.4f}\n"
    text += f"- Highest-value quartile segment: {values['Highest-value quantile segment']} with {float(values['Highest-value segment CLTV share (%)']):.2f}% of total estimated CLTV\n"
    text += f"- Top 20% of customers contribute {float(values['Top 20% CLTV contribution (%)']):.2f}% of total estimated CLTV\n"
    text += f"- Average probability alive: {float(values['Average probability alive']):.4f}\n"
    text += f"- Median probability alive: {float(values['Median probability alive']):.4f}\n\n"
    text += "## Methodological Note\n\n"
    text += "This analysis uses BG/NBD for expected repeat transactions and Gamma-Gamma for expected spend per transaction. "
    text += "The dataset is customer-level summary data without a genuine future holdout period. Bootstrap intervals quantify current-portfolio sampling uncertainty conditional on the fitted model; they are not predictive validation metrics.\n"
    (output_dir / "executive_summary.md").write_text(text, encoding="utf-8")


def run_pipeline(data_path: Path, output_dir: Path, bootstrap_iterations: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Execute the complete CLTV modelling, diagnostic, segmentation, and export pipeline."""
    configure_logging(output_dir)
    configure_plot_style()
    figures_dir = output_dir / CONFIG["figures_subdir"]
    iterations = bootstrap_iterations or CONFIG["bootstrap_iterations"]
    LOGGER.info("Starting %s.", CONFIG["project_name"])
    if iterations < 30:
        LOGGER.warning("Bootstrap iterations below 30 are suitable for testing only, not final reporting.")

    raw_df = convert_dates(load_data(data_path))
    data_quality = create_data_quality_summary(raw_df)
    outlier_report = create_outlier_report(raw_df)
    preprocessing_report = build_preprocessing_report(raw_df, outlier_report)
    cltv_base, analysis_date = engineer_cltv_features(raw_df)
    bgnbd_assumptions = assess_bgnbd_inputs(cltv_base)

    bgf = fit_bgnbd(cltv_base)
    scored, bgnbd_diagnostics = generate_bgnbd_outputs(cltv_base, bgf)
    repeat_df, gamma_assumptions = assess_gamma_gamma_assumption(scored)
    ggf = fit_gamma_gamma(repeat_df)
    scored = score_cltv(scored, bgf, ggf, discount_rate=CONFIG["discount_rate"])
    scored = create_quantile_segments(scored)
    scored, value_alive_summary = create_value_alive_matrix(scored)
    segment_summary = summarize_quantile_segments(scored)

    scored, kmeans_selection, kmeans_summary = run_kmeans_comparison(scored)
    kmeans_comparison = kmeans_segment_crosstab(scored)
    lorenz, concentration_summary, gini = concentration_analysis(scored)
    sensitivity_summary, sensitivity_by_segment = sensitivity_analysis(scored, bgf, ggf)
    bootstrap_results, bootstrap_replicates = bootstrap_cltv_confidence_intervals(
        scored, iterations=iterations
    )
    executive_summary = build_executive_summary(
        scored, segment_summary, concentration_summary, gini, analysis_date
    )

    create_visualizations(
        scored,
        segment_summary,
        lorenz,
        concentration_summary,
        sensitivity_summary,
        bootstrap_results,
        kmeans_selection,
        bgf,
        figures_dir,
    )

    outputs = {
        "cltv_results": scored.reset_index(),
        "segment_summary": segment_summary,
        "kmeans_summary": kmeans_summary,
        "sensitivity_analysis": sensitivity_summary,
        "bootstrap_results": bootstrap_results,
        "executive_summary": executive_summary,
        "data_quality_summary": data_quality,
        "outlier_report": outlier_report,
        "preprocessing_report": preprocessing_report,
        "bgnbd_assumptions": bgnbd_assumptions,
        "bgnbd_diagnostics": bgnbd_diagnostics,
        "gamma_gamma_assumptions": gamma_assumptions,
        "value_alive_summary": value_alive_summary,
        "kmeans_selection": kmeans_selection,
        "kmeans_quantile_comparison": kmeans_comparison,
        "cltv_concentration": concentration_summary,
        "lorenz_curve_data": lorenz,
        "sensitivity_by_segment": sensitivity_by_segment,
        "bootstrap_replicates": bootstrap_replicates,
    }
    export_outputs(outputs, output_dir)
    export_executive_summary_text(executive_summary, output_dir)
    LOGGER.info("Pipeline completed successfully. Outputs: %s", output_dir)
    return outputs


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for reproducible execution."""
    parser = argparse.ArgumentParser(description="FLO senior-level CLTV analytics pipeline")
    parser.add_argument("--data", type=Path, default=Path(CONFIG["data_path"]), help="Path to FLO CSV dataset")
    parser.add_argument("--output", type=Path, default=Path(CONFIG["output_dir"]), help="Directory for exported reports")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=CONFIG["bootstrap_iterations"],
        help="Number of customer bootstrap re-fits for confidence intervals",
    )
    parser.add_argument(
        "--excel-only",
        action="store_true",
        help="Export required Excel files from previously created CSV outputs only.",
    )
    return parser.parse_args()


def main() -> None:
    """Run modelling and then replace the process for stable Excel serialization."""
    args = parse_arguments()
    try:
        if args.excel_only:
            configure_logging(args.output)
            export_required_excel_files(args.output)
            return
        run_pipeline(args.data, args.output, args.bootstrap_iterations)
        LOGGER.info(
            "CSV and figure exports are complete. Run with --excel-only in a clean process to serialize required Excel workbooks."
        )
    except (FileNotFoundError, CLTVPipelineError, ImportError, OSError) as exc:
        LOGGER.error("Pipeline terminated: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
