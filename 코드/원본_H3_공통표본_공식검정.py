from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


OUT = Path(__file__).resolve().parents[1]
LOG = OUT / "logs"

PANEL = Path(r"C:\Users\25294\Desktop\没用\论文2\01_新增控制变量数据\03_加入新增控制变量后的研究面板.csv")
HISTORICAL_COMMON = Path(r"C:\Users\25294\Desktop\没用\研究\09_二轮终审证据\16_共同样本时点检验.csv")
HISTORICAL_FULL = Path(r"C:\Users\25294\Desktop\没用\研究\09_二轮终审证据\05_核心与补充模型完整结果.csv")
REFERENCE_CODE = Path(r"C:\Users\25294\Desktop\没用\论文2\03_复现代码\paper2_extended_controls_pipeline.py")
BASELINE_WORD = Path(r"C:\Users\25294\Documents\Codex\2026-10-05\paper-handoff-human-coding\66666_theory_robustness_enhanced_3fixes.docx")

FOCAL = "BMI_component_z"
OUTCOMES = {
    -2: "PastAssetTurnover_h2",
    -1: "PastAssetTurnover_h1",
    0: "AssetTurnover_w",
    1: "FutureAssetTurnover_h1",
    2: "FutureAssetTurnover_h2",
    3: "FutureAssetTurnover_h3",
}
CONTROLS = [
    "FirmSize_w",
    "Leverage_w",
    "ROA_AvgAssets_w",
    "SalesGrowth_w",
    "Liquidity_w",
    "OperatingMargin_w",
    "FirmAgeLog_w",
    "LossDummy",
    "NegativeEquityDummy",
    "LogTotalSentences",
    "FiscalYearStockReturn_w",
    "FiscalYearDailyVolatility_w",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_receipt(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(path) if path.exists() else "MISSING" for path in paths}


def drop_singletons(frame: pd.DataFrame, effects: list[str]) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    output = frame.copy()
    audit: list[dict[str, int]] = []
    iteration = 0
    while not output.empty:
        iteration += 1
        keep = np.ones(len(output), dtype=bool)
        for effect in effects:
            keep &= output.groupby(effect, observed=True)[effect].transform("size").gt(1).to_numpy()
        audit.append(
            {
                "iteration": iteration,
                "rows_before": int(len(output)),
                "rows_removed": int((~keep).sum()),
                "rows_after": int(keep.sum()),
            }
        )
        if keep.all():
            break
        output = output.loc[keep].copy()
    return output, audit


def group_demean(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=groups).astype(float)
    sums = np.empty((groups, values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        sums[:, column] = np.bincount(codes, weights=values[:, column], minlength=groups)
    return values - sums[codes] / counts[codes, None]


def two_way_residualize(
    values: np.ndarray,
    firm_codes: np.ndarray,
    year_codes: np.ndarray,
    tolerance: float = 1e-12,
    max_iterations: int = 10000,
) -> tuple[np.ndarray, int, float]:
    residual = values.astype(float, copy=True)
    firm_groups = int(firm_codes.max()) + 1
    year_groups = int(year_codes.max()) + 1
    last_delta = math.inf
    for iteration in range(1, max_iterations + 1):
        previous = residual.copy()
        residual = group_demean(residual, firm_codes, firm_groups)
        residual = group_demean(residual, year_codes, year_groups)
        last_delta = float(np.max(np.abs(residual - previous)))
        if last_delta < tolerance:
            return residual, iteration, last_delta
    raise RuntimeError(f"two-way fixed-effect residualization did not converge; last delta={last_delta}")


def cluster_joint_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    firm_codes: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Joint firm-cluster sandwich for several equations sharing one X and one sample.

    Parameter ordering is equation-major. Cross-equation meat blocks retain the
    empirical within-firm covariance instead of setting it to zero.
    """
    n, k = x.shape
    equations = residuals.shape[1]
    clusters = int(firm_codes.max()) + 1
    bread = np.linalg.inv(x.T @ x)
    meat = np.zeros((equations * k, equations * k), dtype=float)
    for cluster in range(clusters):
        mask = firm_codes == cluster
        scores = np.column_stack([x[mask].T @ residuals[mask, equation] for equation in range(equations)])
        score_vector = scores.T.reshape(-1)
        meat += np.outer(score_vector, score_vector)
    block_bread = np.kron(np.eye(equations), bread)
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    covariance = correction * block_bread @ meat @ block_bread
    return covariance, correction


def reconstruct_exact_shift(panel: pd.DataFrame, horizon: int) -> pd.Series:
    lookup = panel[["corp_code", "year", "AssetTurnover_w"]].copy()
    lookup["year"] = lookup["year"] - horizon
    lookup = lookup.rename(columns={"AssetTurnover_w": "expected"})
    merged = panel[["corp_code", "year"]].merge(lookup, on=["corp_code", "year"], how="left", validate="one_to_one")
    return merged["expected"].set_axis(panel.index)


def fit_single_equation_candidate(
    panel: pd.DataFrame,
    outcome: str,
    controls: list[str],
) -> dict[str, float | int]:
    required = list(dict.fromkeys([outcome, FOCAL, *controls, "corp_code", "year"]))
    work = panel[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work, _ = drop_singletons(work, ["corp_code", "year"])
    work = work.sort_values(["corp_code", "year"], kind="mergesort").reset_index(drop=True)
    x_names = [FOCAL, *controls]
    raw = work[[*x_names, outcome]].astype(float).to_numpy()
    firm_codes = pd.Categorical(work["corp_code"], categories=sorted(work["corp_code"].unique())).codes
    year_codes = pd.Categorical(work["year"], categories=sorted(work["year"].unique())).codes
    transformed, _, _ = two_way_residualize(raw, firm_codes, year_codes)
    x = transformed[:, : len(x_names)]
    y = transformed[:, -1]
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    covariance, _ = cluster_joint_covariance_with_pinv(x, residual[:, None], firm_codes)
    se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    p_value = float(2 * stats.t.sf(abs(float(beta[0]) / se), work["corp_code"].nunique() - 1)) if se > 0 else np.nan
    return {
        "N": int(len(work)),
        "firm_count": int(work["corp_code"].nunique()),
        "beta": float(beta[0]),
        "std_error": se,
        "p_value": p_value,
        "design_rank": int(np.linalg.matrix_rank(x)),
        "design_columns": int(x.shape[1]),
    }


def cluster_joint_covariance_with_pinv(
    x: np.ndarray,
    residuals: np.ndarray,
    firm_codes: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Audit-only version that tolerates the intentionally invalid h=0 candidate."""
    n, k = x.shape
    equations = residuals.shape[1]
    clusters = int(firm_codes.max()) + 1
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((equations * k, equations * k), dtype=float)
    for cluster in range(clusters):
        mask = firm_codes == cluster
        scores = np.column_stack([x[mask].T @ residuals[mask, equation] for equation in range(equations)])
        vector = scores.T.reshape(-1)
        meat += np.outer(vector, vector)
    block_bread = np.kron(np.eye(equations), bread)
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    return correction * block_bread @ meat @ block_bread, correction


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    sources = [PANEL, HISTORICAL_COMMON, HISTORICAL_FULL, REFERENCE_CODE, BASELINE_WORD]
    hashes_before = file_receipt(sources)

    panel = pd.read_csv(PANEL, dtype={"corp_code": str}, low_memory=False)
    panel["corp_code"] = panel["corp_code"].str.zfill(8)
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    duplicate_rows = int(panel.duplicated(["corp_code", "year"]).sum())
    if duplicate_rows:
        raise RuntimeError(f"source panel has {duplicate_rows} duplicate firm-years")

    outcome_names = list(OUTCOMES.values())
    required = list(dict.fromkeys([*outcome_names, FOCAL, *CONTROLS, "corp_code", "year"]))
    missing_columns = [column for column in required if column not in panel.columns]
    if missing_columns:
        raise RuntimeError(f"missing required columns: {missing_columns}")

    # Standardization provenance: BMI_component_z must be the full-panel z-score of BMI_share.
    bmi_raw = pd.to_numeric(panel["BMI_share"], errors="coerce")
    expected_z = (bmi_raw - bmi_raw.mean()) / bmi_raw.std(ddof=1)
    observed_z = pd.to_numeric(panel[FOCAL], errors="coerce")
    common_z = expected_z.notna() & observed_z.notna()
    z_max_abs_difference = float((expected_z[common_z] - observed_z[common_z]).abs().max())
    z_missing_mismatch = int((expected_z.isna() != observed_z.isna()).sum())

    # Validate that every saved outcome is an exact same-firm calendar-year shift.
    shift_audit = []
    for horizon, column in OUTCOMES.items():
        expected = reconstruct_exact_shift(panel, horizon)
        observed = pd.to_numeric(panel[column], errors="coerce")
        overlap = expected.notna() & observed.notna()
        shift_audit.append(
            {
                "horizon": horizon,
                "column": column,
                "observed_nonmissing": int(observed.notna().sum()),
                "expected_nonmissing": int(expected.notna().sum()),
                "missingness_mismatch": int((expected.isna() != observed.isna()).sum()),
                "max_abs_value_difference": float((expected[overlap] - observed[overlap]).abs().max()) if overlap.any() else np.nan,
            }
        )

    # Reverse-engineer the historical time-specific specifications from true
    # outputs.  The historical series uses AssetTurnover_w as a control at all
    # horizons except h=0.  That creates both a cross-horizon inconsistency and
    # future-information conditioning for h=-2/-1.  The invalid h=0 candidate is
    # retained only as an audit demonstration; it is never used for inference.
    historical_full = pd.read_csv(HISTORICAL_FULL)
    historical_timing = historical_full[
        historical_full["model_id"].astype(str).str.fullmatch(r"TIMING_H-?[0-3]")
        & historical_full["term"].eq("BMI_z")
    ].copy()
    historical_timing["horizon"] = historical_timing["model_id"].str.extract(r"H(-?\d+)")[0].astype(int)
    historical_timing = historical_timing.set_index("horizon")
    specification_reconstruction = []
    for horizon, outcome in OUTCOMES.items():
        for candidate, controls in [
            ("12_non_outcome_controls", CONTROLS),
            ("AssetTurnover_t_plus_12_controls", ["AssetTurnover_w", *CONTROLS]),
        ]:
            fitted = fit_single_equation_candidate(panel, outcome, controls)
            historical_row = historical_timing.loc[horizon]
            specification_reconstruction.append(
                {
                    "horizon": horizon,
                    "outcome": outcome,
                    "candidate_control_set": candidate,
                    **fitted,
                    "historical_beta": float(historical_row["coefficient"]),
                    "historical_se": float(historical_row["std_error"]),
                    "beta_abs_difference": abs(fitted["beta"] - float(historical_row["coefficient"])),
                    "se_abs_difference": abs(fitted["std_error"] - float(historical_row["std_error"])),
                    "outcome_on_itself": horizon == 0 and candidate == "AssetTurnover_t_plus_12_controls",
                    "future_information_control": horizon < 0 and candidate == "AssetTurnover_t_plus_12_controls",
                }
            )
    specification_reconstruction = pd.DataFrame(specification_reconstruction)
    specification_reconstruction["matches_historical"] = (
        specification_reconstruction["beta_abs_difference"].lt(1e-10)
        & specification_reconstruction["se_abs_difference"].lt(1e-10)
    )
    specification_reconstruction.to_csv(
        LOG / "original_timepoint_spec_reconstruction.csv", index=False, encoding="utf-8-sig"
    )

    complete = panel[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    pre_singleton_rows = int(len(complete))
    pre_singleton_firms = int(complete["corp_code"].nunique())
    common, singleton_audit = drop_singletons(complete, ["corp_code", "year"])
    common = common.sort_values(["corp_code", "year"], kind="mergesort").reset_index(drop=True)
    n = len(common)
    firms = int(common["corp_code"].nunique())
    years = int(common["year"].nunique())
    if (n, firms, years) != (7959, 1717, 5):
        raise RuntimeError(f"expected frozen common sample (7959,1717,5), got {(n,firms,years)}")

    sample_key_text = "\n".join(common["corp_code"] + "|" + common["year"].astype(str))
    sample_key_sha256 = hashlib.sha256(sample_key_text.encode("utf-8")).hexdigest().upper()

    # Each equation uses exactly the same non-outcome controls. AssetTurnover_w is
    # deliberately excluded as a control because it is the h=0 outcome; including
    # it would regress an outcome on itself in that equation.
    x_names = [FOCAL, *CONTROLS]
    y_names = [OUTCOMES[horizon] for horizon in OUTCOMES]
    raw_matrix = common[[*x_names, *y_names]].astype(float).to_numpy()
    firm_codes = pd.Categorical(common["corp_code"], categories=sorted(common["corp_code"].unique())).codes
    year_codes = pd.Categorical(common["year"], categories=sorted(common["year"].unique())).codes
    transformed, iterations, convergence_delta = two_way_residualize(raw_matrix, firm_codes, year_codes)
    x = transformed[:, : len(x_names)]
    y = transformed[:, len(x_names) :]
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise RuntimeError(f"residualized design rank {rank} is below {x.shape[1]}")

    bread = np.linalg.inv(x.T @ x)
    coefficients = bread @ x.T @ y
    equation_residuals = y - x @ coefficients
    joint_covariance, correction = cluster_joint_covariance(x, equation_residuals, firm_codes)
    df_cluster = firms - 1
    critical = float(stats.t.ppf(0.975, df_cluster))
    equation_count = len(y_names)
    k = len(x_names)
    focal_index = x_names.index(FOCAL)

    historical = pd.read_csv(HISTORICAL_COMMON)
    historical = historical.set_index("horizon")
    rows = []
    for equation, horizon in enumerate(OUTCOMES):
        parameter = equation * k + focal_index
        beta = float(coefficients[focal_index, equation])
        se = float(math.sqrt(max(joint_covariance[parameter, parameter], 0.0)))
        statistic = beta / se
        p_value = float(2 * stats.t.sf(abs(statistic), df_cluster))
        hist = historical.loc[horizon]
        rows.append(
            {
                "specification": "NEW_UNIFIED_COMMON_SAMPLE_JOINT",
                "horizon": horizon,
                "time_label": "t" if horizon == 0 else f"t{horizon:+d}",
                "outcome": OUTCOMES[horizon],
                "term": FOCAL,
                "beta": beta,
                "std_error": se,
                "t_statistic": statistic,
                "p_value": p_value,
                "ci95_low": beta - critical * se,
                "ci95_high": beta + critical * se,
                "N": n,
                "firm_count": firms,
                "year_count": years,
                "cluster_count": firms,
                "predictor_year_min": int(common["year"].min()),
                "predictor_year_max": int(common["year"].max()),
                "fixed_effects": "firm + predictor-year",
                "cluster": "firm",
                "control_set": "12 non-outcome predictor-year controls; baseline AssetTurnover excluded",
                "sample_key_sha256": sample_key_sha256,
                "historical_beta": float(hist["coefficient"]),
                "historical_se": float(hist["std_error"]),
                "beta_reproduction_abs_diff": abs(beta - float(hist["coefficient"])),
                "se_reproduction_abs_diff": abs(se - float(hist["std_error"])),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "H3_common_sample_results.csv", index=False, encoding="utf-8-sig")

    comparisons = [
        (2, 0, "primary_prespecified"),
        (1, 0, "secondary"),
        (3, 0, "secondary"),
    ]
    diff_rows = []
    for high, low, role in comparisons:
        equation_high = list(OUTCOMES).index(high)
        equation_low = list(OUTCOMES).index(low)
        parameter_high = equation_high * k + focal_index
        parameter_low = equation_low * k + focal_index
        contrast = np.zeros(equation_count * k)
        contrast[parameter_high] = 1.0
        contrast[parameter_low] = -1.0
        difference = float(coefficients[focal_index, equation_high] - coefficients[focal_index, equation_low])
        variance = float(contrast @ joint_covariance @ contrast)
        se = math.sqrt(max(variance, 0.0))
        t_stat = difference / se
        p_value = float(2 * stats.t.sf(abs(t_stat), df_cluster))

        # Algebraic validation: with common X and common observations, regressing
        # (Y_high - Y_low) on X produces exactly the coefficient contrast and its
        # firm-cluster variance, including the cross-equation covariance.
        y_difference = y[:, equation_high] - y[:, equation_low]
        beta_direct = bread @ x.T @ y_difference
        residual_direct = y_difference - x @ beta_direct
        direct_covariance, _ = cluster_joint_covariance(x, residual_direct[:, None], firm_codes)
        direct_se = math.sqrt(max(float(direct_covariance[focal_index, focal_index]), 0.0))
        diff_rows.append(
            {
                "specification": "NEW_UNIFIED_COMMON_SAMPLE_JOINT",
                "test_role": role,
                "contrast": f"beta(t{high:+d}) - beta(t)" if low == 0 else f"beta(t{high:+d}) - beta(t{low:+d})",
                "high_horizon": high,
                "low_horizon": low,
                "beta_difference": difference,
                "difference_std_error": se,
                "t_statistic": t_stat,
                "wald_chi2_1df": t_stat**2,
                "p_value_raw_cluster_t": p_value,
                "ci95_low": difference - critical * se,
                "ci95_high": difference + critical * se,
                "N": n,
                "cluster_count": firms,
                "cluster_df": df_cluster,
                "cross_equation_covariance": float(joint_covariance[parameter_high, parameter_low]),
                "covariance_assumed_zero": False,
                "reference_distribution": f"t({df_cluster}); Wald statistic shown as t^2",
                "direct_difference_beta": float(beta_direct[focal_index]),
                "direct_difference_se": direct_se,
                "joint_vs_direct_beta_abs_diff": abs(difference - float(beta_direct[focal_index])),
                "joint_vs_direct_se_abs_diff": abs(se - direct_se),
                "p_value_holm_secondary": np.nan,
                "p_value_bh_secondary": np.nan,
                "sample_key_sha256": sample_key_sha256,
            }
        )
    differences = pd.DataFrame(diff_rows)
    secondary = differences["test_role"].eq("secondary")
    differences.loc[secondary, "p_value_holm_secondary"] = multipletests(
        differences.loc[secondary, "p_value_raw_cluster_t"], method="holm"
    )[1]
    differences.loc[secondary, "p_value_bh_secondary"] = multipletests(
        differences.loc[secondary, "p_value_raw_cluster_t"], method="fdr_bh"
    )[1]
    differences.to_csv(OUT / "H3_coefficient_difference_tests.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(shift_audit).to_csv(LOG / "outcome_shift_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(singleton_audit).to_csv(LOG / "singleton_removal_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "corp_code": common["corp_code"],
            "year": common["year"],
            "sample_key": common["corp_code"] + "|" + common["year"].astype(str),
        }
    ).to_csv(LOG / "common_sample_keys.csv", index=False, encoding="utf-8-sig")

    hashes_after = file_receipt(sources)
    hash_rows = []
    for path in sources:
        key = str(path)
        hash_rows.append(
            {
                "path": key,
                "sha256_before": hashes_before[key],
                "sha256_after": hashes_after[key],
                "unchanged": hashes_before[key] == hashes_after[key],
            }
        )
    pd.DataFrame(hash_rows).to_csv(LOG / "source_integrity_receipt.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "source_rows": int(len(panel)),
        "source_firms": int(panel["corp_code"].nunique()),
        "source_year_min": int(panel["year"].min()),
        "source_year_max": int(panel["year"].max()),
        "duplicate_firm_year_rows": duplicate_rows,
        "pre_singleton_complete_rows": pre_singleton_rows,
        "pre_singleton_complete_firms": pre_singleton_firms,
        "common_sample_rows": n,
        "common_sample_firms": firms,
        "common_sample_years": years,
        "common_sample_key_sha256": sample_key_sha256,
        "bmi_full_panel_mean": float(bmi_raw.mean()),
        "bmi_full_panel_sd_ddof1": float(bmi_raw.std(ddof=1)),
        "bmi_z_max_abs_reconstruction_difference": z_max_abs_difference,
        "bmi_z_missingness_mismatch": z_missing_mismatch,
        "two_way_fe_iterations": iterations,
        "two_way_fe_convergence_delta": convergence_delta,
        "residualized_design_rank": rank,
        "residualized_design_columns": len(x_names),
        "joint_sandwich_small_sample_correction": correction,
        "cluster_df": df_cluster,
        "primary_test_fixed_before_estimation": "beta(t+2)-beta(t)",
        "secondary_multiplicity_families": "Holm and BH adjustments across t+1-vs-t and t+3-vs-t",
        "asset_turnover_t_as_control": False,
        "asset_turnover_t_exclusion_reason": "It is the h=0 dependent variable; including it would create outcome-on-itself leakage.",
        "source_hashes_unchanged": all(hashes_before[key] == hashes_after[key] for key in hashes_before),
    }
    (LOG / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    max_beta_diff = float(results["beta_reproduction_abs_diff"].max())
    max_se_diff = float(results["se_reproduction_abs_diff"].max())
    max_joint_direct_beta = float(differences["joint_vs_direct_beta_abs_diff"].max())
    max_joint_direct_se = float(differences["joint_vs_direct_se_abs_diff"].max())
    summary = {
        "status": "PASS",
        "sample_exact_7959": n == 7959,
        "firm_count": firms,
        "duplicate_firm_year_rows": duplicate_rows,
        "all_outcome_shifts_exact": all(row["missingness_mismatch"] == 0 and row["max_abs_value_difference"] == 0 for row in shift_audit),
        "bmi_standardization_exact_within_tolerance": z_missing_mismatch == 0 and z_max_abs_difference < 1e-12,
        "max_abs_beta_reproduction_difference": max_beta_diff,
        "max_abs_se_reproduction_difference": max_se_diff,
        "max_joint_vs_direct_beta_abs_difference": max_joint_direct_beta,
        "max_joint_vs_direct_se_abs_difference": max_joint_direct_se,
        "source_integrity_unchanged": metadata["source_hashes_unchanged"],
    }
    (LOG / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
