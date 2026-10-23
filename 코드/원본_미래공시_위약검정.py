from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy import sparse, stats
from scipy.sparse.linalg import lsmr
from statsmodels.stats.sandwich_covariance import cov_cluster


ROOT = Path(__file__).resolve().parents[2]
IDENT = ROOT / "01_identification"
HET = ROOT / "02_heterogeneity"
AUDIT = ROOT / "04_audit"
SOURCE = Path(
    r"C:\Users\25294\Desktop\没用\论文2\01_新增控制变量数据"
    r"\03_加入新增控制变量后的研究面板.csv"
)
THESIS_SOURCE = Path(r"C:\Users\25294\Desktop\66666.docx")
THESIS_REVISED = Path(
    r"C:\Users\25294\Documents\Codex\2026-10-05\paper-handoff-human-coding"
    r"\outputs\66666_MINIMAL_STRUCTURE_AUDITED.docx"
)

BASE_CONTROLS = [
    "AssetTurnover_w",
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
NON_OUTCOME_CONTROLS = BASE_CONTROLS[1:]
LOCKED_BETA = 0.0302431770


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def exact_shift(panel: pd.DataFrame, column: str, horizon: int) -> pd.Series:
    """Return the exact calendar-year value at t+horizon for each firm-year row."""
    lookup = panel[["corp_code", "year", column]].copy()
    lookup["year"] = lookup["year"] - horizon
    lookup = lookup.rename(columns={column: "_matched"})
    matched = panel[["corp_code", "year"]].merge(
        lookup,
        on=["corp_code", "year"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    return matched["_matched"].set_axis(panel.index)


def drop_singletons(frame: pd.DataFrame, effects: list[str]) -> pd.DataFrame:
    result = frame.copy()
    while not result.empty:
        keep = np.ones(len(result), dtype=bool)
        for effect in effects:
            keep &= result.groupby(effect)[effect].transform("size").gt(1).to_numpy()
        if keep.all():
            break
        result = result.loc[keep].copy()
    return result


def fit_fwl(
    panel: pd.DataFrame,
    model_id: str,
    outcome: str,
    focals: list[str],
    controls: list[str],
    sample: pd.Series | None = None,
    forced_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    controls = list(dict.fromkeys(x for x in controls if x not in focals))
    effects = ["corp_code", "year"]
    required = list(dict.fromkeys([outcome, *focals, *controls, *effects]))
    if forced_index is not None:
        work = panel.loc[forced_index, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(work) != len(forced_index):
            raise RuntimeError(f"{model_id}: forced sample lost rows")
    else:
        mask = sample if sample is not None else pd.Series(True, index=panel.index)
        work = panel.loc[mask, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        work = drop_singletons(work, effects)

    names = [*focals, *controls]
    matrix = work[[outcome, *names]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    codes = np.column_stack([pd.Categorical(work[e].astype(str)).codes for e in effects])
    absorber = pyhdfe.create(
        codes,
        drop_singletons=False,
        compute_degrees=False,
        residualize_method="map",
        options={"transform": "symmetric", "acceleration": "cg", "tol": 1e-10},
    )
    transformed = absorber.residualize(matrix)
    y = transformed[:, 0]
    x_raw = transformed[:, 1:]
    selected_names: list[str] = []
    selected_arrays: list[np.ndarray] = []
    for name, values in zip(names, x_raw.T):
        if np.sqrt(np.mean(values**2)) <= 1e-10:
            continue
        trial = np.column_stack([*selected_arrays, values]) if selected_arrays else values.reshape(-1, 1)
        if np.linalg.matrix_rank(trial) > len(selected_arrays):
            selected_names.append(name)
            selected_arrays.append(values)
    x = np.column_stack(selected_arrays)
    fitted = sm.OLS(y, x).fit()
    firm_codes = pd.Categorical(work["corp_code"].astype(str)).codes
    covariance = cov_cluster(fitted, firm_codes, use_correction=True)
    dof = max(1, work["corp_code"].nunique() - 1)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    critical = stats.t.ppf(0.975, dof)
    pvals = 2 * stats.t.sf(np.abs(fitted.params / se), df=dof)
    tss = float(np.sum(y**2))
    r2 = 1 - float(np.sum(fitted.resid**2)) / tss if tss > 0 else np.nan
    rows = []
    for term in focals:
        if term not in selected_names:
            continue
        j = selected_names.index(term)
        rows.append(
            {
                "model_id": model_id,
                "outcome": outcome,
                "term": term,
                "coefficient": float(fitted.params[j]),
                "std_error": float(se[j]),
                "p_value": float(pvals[j]),
                "ci95_low": float(fitted.params[j] - critical * se[j]),
                "ci95_high": float(fitted.params[j] + critical * se[j]),
                "N": int(len(work)),
                "firms": int(work["corp_code"].nunique()),
                "year_min": int(work["year"].min()),
                "year_max": int(work["year"].max()),
                "within_R2": r2,
                "fixed_effects": "firm + year",
                "cluster": "firm",
            }
        )
    details = {
        "work": work,
        "names": selected_names,
        "params": np.asarray(fitted.params),
        "cov": np.asarray(covariance),
        "dof": dof,
        "se": se,
        "pvals": pvals,
        "residualized": transformed,
        "outcome": outcome,
    }
    return pd.DataFrame(rows), details


def linear_combination(details: dict[str, object], weights: dict[str, float]) -> dict[str, float]:
    names = list(details["names"])
    gradient = np.zeros(len(names))
    for term, value in weights.items():
        if term not in names:
            raise ValueError(f"Term dropped from model: {term}")
        gradient[names.index(term)] = value
    params = np.asarray(details["params"])
    covariance = np.asarray(details["cov"])
    estimate = float(gradient @ params)
    se = float(np.sqrt(max(gradient @ covariance @ gradient, 0)))
    dof = int(details["dof"])
    p_value = float(2 * stats.t.sf(abs(estimate / se), dof)) if se > 0 else np.nan
    critical = float(stats.t.ppf(0.975, dof))
    return {
        "estimate": estimate,
        "std_error": se,
        "p_value": p_value,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
    }


def absorb_firm_trends(matrix: np.ndarray, work: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    firms = pd.Categorical(work["corp_code"].astype(str))
    years = pd.Categorical(work["year"].astype(str))
    firm_codes = firms.codes
    year_codes = years.codes
    n = len(work)
    n_firms = len(firms.categories)
    n_years = len(years.categories)

    firm_dummy = sparse.csr_matrix((np.ones(n), (np.arange(n), firm_codes)), shape=(n, n_firms))
    year_dummy_full = sparse.csr_matrix((np.ones(n), (np.arange(n), year_codes)), shape=(n, n_years))
    year_dummy = year_dummy_full[:, 1:]
    year_values = numeric(work["year"]).to_numpy()
    firm_mean_year = work.groupby("corp_code")["year"].transform("mean").to_numpy(float)
    centered_time = year_values - firm_mean_year
    firm_slope = sparse.csr_matrix((centered_time, (np.arange(n), firm_codes)), shape=(n, n_firms))
    design = sparse.hstack([firm_dummy, year_dummy, firm_slope], format="csr")

    residualized = np.empty_like(matrix, dtype=float)
    solver_log = []
    for column in range(matrix.shape[1]):
        solved = lsmr(design, matrix[:, column], atol=1e-12, btol=1e-12, maxiter=20000)
        residualized[:, column] = matrix[:, column] - design @ solved[0]
        solver_log.append(
            {
                "column": column,
                "istop": int(solved[1]),
                "iterations": int(solved[2]),
                "residual_norm": float(solved[3]),
                "normal_equation_norm": float(solved[4]),
            }
        )
    max_orthogonality_error = float(np.max(np.abs(design.T @ residualized)))
    counts = work.groupby("corp_code")["year"].nunique()
    diagnostics = {
        "absorbed_columns": int(design.shape[1]),
        "firm_count": n_firms,
        "year_count": n_years,
        "firms_with_two_years": int(counts.eq(2).sum()),
        "firms_with_three_or_more_years": int(counts.ge(3).sum()),
        "max_abs_Zt_residual": max_orthogonality_error,
        "solver": solver_log,
    }
    return residualized, diagnostics


def fit_trend_on_locked_sample(panel: pd.DataFrame, baseline_details: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    work = baseline_details["work"].copy()
    names = ["BMI_component_z", *BASE_CONTROLS]
    matrix = work[["FutureAssetTurnover_h2", *names]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    transformed, diagnostics = absorb_firm_trends(matrix, work)
    y = transformed[:, 0]
    x_raw = transformed[:, 1:]
    selected_names: list[str] = []
    selected_arrays: list[np.ndarray] = []
    for name, values in zip(names, x_raw.T):
        if np.sqrt(np.mean(values**2)) <= 1e-10:
            continue
        trial = np.column_stack([*selected_arrays, values]) if selected_arrays else values.reshape(-1, 1)
        if np.linalg.matrix_rank(trial) > len(selected_arrays):
            selected_names.append(name)
            selected_arrays.append(values)
    x = np.column_stack(selected_arrays)
    fitted = sm.OLS(y, x).fit()
    firm_codes = pd.Categorical(work["corp_code"].astype(str)).codes
    covariance = cov_cluster(fitted, firm_codes, use_correction=True)
    dof = max(1, work["corp_code"].nunique() - 1)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    pvals = 2 * stats.t.sf(np.abs(fitted.params / se), df=dof)
    critical = stats.t.ppf(0.975, dof)
    j = selected_names.index("BMI_component_z")
    result = pd.DataFrame(
        [
            {
                "model_id": "FIRM_LINEAR_TRENDS",
                "outcome": "FutureAssetTurnover_h2",
                "term": "BMI_component_z",
                "coefficient": float(fitted.params[j]),
                "std_error": float(se[j]),
                "p_value": float(pvals[j]),
                "ci95_low": float(fitted.params[j] - critical * se[j]),
                "ci95_high": float(fitted.params[j] + critical * se[j]),
                "N": int(len(work)),
                "firms": int(work["corp_code"].nunique()),
                "year_min": int(work["year"].min()),
                "year_max": int(work["year"].max()),
                "fixed_effects": "firm + year + firm-specific linear trend",
                "cluster": "firm",
            }
        ]
    )
    diagnostics["selected_regressors"] = selected_names
    return result, diagnostics


def main() -> int:
    for folder in [IDENT, IDENT / "logs", HET, HET / "logs", AUDIT]:
        folder.mkdir(parents=True, exist_ok=True)

    before_hashes = {
        "source_panel": sha256(SOURCE),
        "thesis_source": sha256(THESIS_SOURCE),
        "thesis_revised": sha256(THESIS_REVISED),
    }
    panel = pd.read_csv(SOURCE, dtype={"corp_code": str, "stock_code": str}, low_memory=False)
    panel["corp_code"] = panel["corp_code"].str.zfill(8)
    panel["year"] = numeric(panel["year"]).astype(int)
    panel = panel.sort_values(["corp_code", "year"]).reset_index(drop=True)

    duplicate_keys = int(panel.duplicated(["corp_code", "year"]).sum())
    if duplicate_keys:
        raise RuntimeError(f"Source panel has {duplicate_keys} duplicate firm-year keys")
    profile = {
        "source": str(SOURCE),
        "source_sha256": before_hashes["source_panel"],
        "rows": int(len(panel)),
        "columns": int(panel.shape[1]),
        "firms": int(panel["corp_code"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "duplicate_firm_year_keys": duplicate_keys,
        "missing_rates": {
            name: float(panel[name].isna().mean())
            for name in ["FutureAssetTurnover_h2", "BMI_component_z", *BASE_CONTROLS, "PPE_Ratio_w"]
        },
    }
    (IDENT / "logs" / "data_quality_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline_rows, baseline_details = fit_fwl(
        panel,
        "BASELINE_REPLICATION",
        "FutureAssetTurnover_h2",
        ["BMI_component_z"],
        BASE_CONTROLS,
    )
    baseline_beta = float(baseline_rows.loc[0, "coefficient"])
    if not np.isclose(baseline_beta, LOCKED_BETA, atol=1e-6, rtol=0):
        raise RuntimeError(f"Baseline did not reproduce locked coefficient: {baseline_beta}")

    # Exact-year placebo variables. All controls occur no later than the outcome year.
    panel["FutureBMI_h1_exact"] = exact_shift(panel, "BMI_component_z", 1)
    panel["FutureBMI_h2_exact"] = exact_shift(panel, "BMI_component_z", 2)
    panel["PastAssetTurnover_h1_exact"] = exact_shift(panel, "AssetTurnover_w", -1)
    placebo_rows = []
    placebo_specs = [
        ("PLACEBO_FUTURE_BMI_H1", "FutureBMI_h1_exact", "BMI_(t+1) -> AssetTurnover_t"),
        ("PLACEBO_FUTURE_BMI_H2", "FutureBMI_h2_exact", "BMI_(t+2) -> AssetTurnover_t"),
    ]
    for model_id, focal, label in placebo_specs:
        rows, _ = fit_fwl(
            panel,
            model_id,
            "AssetTurnover_w",
            [focal],
            ["PastAssetTurnover_h1_exact", *NON_OUTCOME_CONTROLS],
        )
        rows["temporal_order"] = label
        rows["sample_construction"] = (
            "Exact firm-year merges; complete cases for outcome AT_t, future BMI lead, "
            "AT_(t-1), X_t, firm and outcome-year FE; iterative singleton removal."
        )
        rows["no_lookahead_controls"] = True
        placebo_rows.append(rows)
    placebo = pd.concat(placebo_rows, ignore_index=True)
    placebo.to_csv(IDENT / "future_placebo_results.csv", index=False, encoding="utf-8-sig")

    trend, trend_diagnostics = fit_trend_on_locked_sample(panel, baseline_details)
    baseline_for_output = baseline_rows.copy()
    baseline_for_output["fixed_effects"] = "firm + year"
    trend_results = pd.concat([baseline_for_output, trend], ignore_index=True, sort=False)
    trend_beta = float(trend.loc[0, "coefficient"])
    baseline_se = float(baseline_rows.loc[0, "std_error"])
    trend_se = float(trend.loc[0, "std_error"])
    trend_results["coefficient_change_pct_vs_baseline"] = np.nan
    trend_results.loc[trend_results["model_id"].eq("FIRM_LINEAR_TRENDS"), "coefficient_change_pct_vs_baseline"] = (
        (trend_beta - baseline_beta) / abs(baseline_beta) * 100
    )
    trend_results["se_inflation_ratio_vs_baseline"] = np.nan
    trend_results.loc[trend_results["model_id"].eq("FIRM_LINEAR_TRENDS"), "se_inflation_ratio_vs_baseline"] = trend_se / baseline_se
    trend_results["sample_locked_to_baseline"] = True
    trend_results.to_csv(IDENT / "firm_trend_results.csv", index=False, encoding="utf-8-sig")
    (IDENT / "logs" / "firm_trend_solver.json").write_text(
        json.dumps(trend_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Asset-intensity rules were frozen in classification_sources.md before this block ran.
    ppe = numeric(panel["PPE_Ratio_w"])
    ppe_mean = float(ppe.dropna().mean())
    ppe_sd = float(ppe.dropna().std(ddof=1))
    ppe_median = float(ppe.dropna().median())
    panel["PPE_Ratio_z_exante"] = (ppe - ppe_mean) / ppe_sd
    panel["BMI_x_PPE_z"] = numeric(panel["BMI_component_z"]) * panel["PPE_Ratio_z_exante"]
    panel["HighAssetIntensity_exante"] = np.where(
        ppe.notna(), (ppe >= ppe_median).astype(float), np.nan
    )
    panel["BMI_x_HighAssetIntensity"] = (
        numeric(panel["BMI_component_z"]) * panel["HighAssetIntensity_exante"]
    )

    continuous, continuous_details = fit_fwl(
        panel,
        "ASSET_INTENSITY_CONTINUOUS_INTERACTION",
        "FutureAssetTurnover_h2",
        ["BMI_component_z", "PPE_Ratio_z_exante", "BMI_x_PPE_z"],
        BASE_CONTROLS,
    )
    continuous_index = continuous_details["work"].index
    same_sample_base, _ = fit_fwl(
        panel,
        "ASSET_INTENSITY_SAME_SAMPLE_BASELINE",
        "FutureAssetTurnover_h2",
        ["BMI_component_z"],
        BASE_CONTROLS,
        forced_index=continuous_index,
    )
    p25, p75 = continuous_details["work"]["PPE_Ratio_z_exante"].quantile([0.25, 0.75])
    marginal_rows = []
    for label, quantile in [("BMI slope at PPE P25", float(p25)), ("BMI slope at PPE P75", float(p75))]:
        estimate = linear_combination(
            continuous_details,
            {"BMI_component_z": 1.0, "BMI_x_PPE_z": quantile},
        )
        marginal_rows.append(
            {
                "model_id": "ASSET_INTENSITY_CONTINUOUS_INTERACTION",
                "row_type": "marginal_effect",
                "term": label,
                "asset_intensity_value": quantile,
                **estimate,
            }
        )
    difference = linear_combination(
        continuous_details,
        {"BMI_x_PPE_z": float(p75 - p25)},
    )
    marginal_rows.append(
        {
            "model_id": "ASSET_INTENSITY_CONTINUOUS_INTERACTION",
            "row_type": "difference_test",
            "term": "BMI slope at PPE P75 minus P25",
            "asset_intensity_value": float(p75 - p25),
            **difference,
        }
    )

    binary, binary_details = fit_fwl(
        panel,
        "ASSET_INTENSITY_MEDIAN_INTERACTION",
        "FutureAssetTurnover_h2",
        ["BMI_component_z", "HighAssetIntensity_exante", "BMI_x_HighAssetIntensity"],
        BASE_CONTROLS,
    )
    low_effect = linear_combination(binary_details, {"BMI_component_z": 1.0})
    high_effect = linear_combination(
        binary_details,
        {"BMI_component_z": 1.0, "BMI_x_HighAssetIntensity": 1.0},
    )
    binary_difference = linear_combination(binary_details, {"BMI_x_HighAssetIntensity": 1.0})
    for label, row_type, value, estimate in [
        ("BMI slope: below global PPE median", "marginal_effect", 0.0, low_effect),
        ("BMI slope: at/above global PPE median", "marginal_effect", 1.0, high_effect),
        ("High minus low BMI slope", "difference_test", 1.0, binary_difference),
    ]:
        marginal_rows.append(
            {
                "model_id": "ASSET_INTENSITY_MEDIAN_INTERACTION",
                "row_type": row_type,
                "term": label,
                "asset_intensity_value": value,
                **estimate,
            }
        )

    coefficient_rows = pd.concat([same_sample_base, continuous, binary], ignore_index=True, sort=False)
    coefficient_rows["row_type"] = "coefficient"
    coefficient_rows = coefficient_rows.rename(columns={"coefficient": "estimate"})
    marginal = pd.DataFrame(marginal_rows)
    for frame, details in [(marginal, continuous_details)]:
        pass
    marginal["N"] = marginal["model_id"].map(
        {
            "ASSET_INTENSITY_CONTINUOUS_INTERACTION": int(len(continuous_details["work"])),
            "ASSET_INTENSITY_MEDIAN_INTERACTION": int(len(binary_details["work"])),
        }
    )
    marginal["firms"] = marginal["model_id"].map(
        {
            "ASSET_INTENSITY_CONTINUOUS_INTERACTION": int(continuous_details["work"]["corp_code"].nunique()),
            "ASSET_INTENSITY_MEDIAN_INTERACTION": int(binary_details["work"]["corp_code"].nunique()),
        }
    )
    main_n = int(baseline_rows.loc[0, "N"])
    coefficient_rows["main_sample_N"] = main_n
    marginal["main_sample_N"] = main_n
    coefficient_rows["sample_loss_N"] = main_n - coefficient_rows["N"]
    marginal["sample_loss_N"] = main_n - marginal["N"]
    coefficient_rows["sample_loss_pct"] = coefficient_rows["sample_loss_N"] / main_n * 100
    marginal["sample_loss_pct"] = marginal["sample_loss_N"] / main_n * 100
    coefficient_rows["PPE_global_mean"] = ppe_mean
    coefficient_rows["PPE_global_sd"] = ppe_sd
    coefficient_rows["PPE_global_median"] = ppe_median
    marginal["PPE_global_mean"] = ppe_mean
    marginal["PPE_global_sd"] = ppe_sd
    marginal["PPE_global_median"] = ppe_median
    asset_results = pd.concat([coefficient_rows, marginal], ignore_index=True, sort=False)
    asset_results.to_csv(HET / "asset_intensity_results.csv", index=False, encoding="utf-8-sig")

    customer = pd.DataFrame(
        [
            {
                "status": "NOT_RUN",
                "reason_cn": "没有找到足够可靠、可重复的外部分类依据，因此未进行该异质性回归。",
                "interaction_estimated": False,
                "coefficient": np.nan,
                "std_error": np.nan,
                "p_value": np.nan,
                "N": np.nan,
                "firms": np.nan,
            }
        ]
    )
    customer.to_csv(HET / "customer_facing_results.csv", index=False, encoding="utf-8-sig")

    sample_rows = [
        {
            "analysis": "main baseline replication",
            "N": int(baseline_rows.loc[0, "N"]),
            "firms": int(baseline_rows.loc[0, "firms"]),
            "change_from_main_N": 0,
            "reason": "locked main specification",
        },
        {
            "analysis": "firm-specific linear trends",
            "N": int(trend.loc[0, "N"]),
            "firms": int(trend.loc[0, "firms"]),
            "change_from_main_N": int(trend.loc[0, "N"] - main_n),
            "reason": "exactly locked to baseline estimation rows",
        },
    ]
    for _, row in placebo.iterrows():
        sample_rows.append(
            {
                "analysis": row["model_id"],
                "N": int(row["N"]),
                "firms": int(row["firms"]),
                "change_from_main_N": int(row["N"] - main_n),
                "reason": "different temporal availability required by the pre-specified future-exposure placebo",
            }
        )
    sample_rows.append(
        {
            "analysis": "asset-intensity continuous interaction",
            "N": int(len(continuous_details["work"])),
            "firms": int(continuous_details["work"]["corp_code"].nunique()),
            "change_from_main_N": int(len(continuous_details["work"]) - main_n),
            "reason": "baseline PPE is missing for some main-sample firm-years",
        }
    )
    pd.DataFrame(sample_rows).to_csv(AUDIT / "sample_change_check.csv", index=False, encoding="utf-8-sig")

    after_hashes = {
        "source_panel": sha256(SOURCE),
        "thesis_source": sha256(THESIS_SOURCE),
        "thesis_revised": sha256(THESIS_REVISED),
    }
    integrity = {
        "before": before_hashes,
        "after": after_hashes,
        "all_unchanged": before_hashes == after_hashes,
        "source_panel_path": str(SOURCE),
        "thesis_paths": [str(THESIS_SOURCE), str(THESIS_REVISED)],
    }
    (AUDIT / "integrity_hashes.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_metadata = {
        "python": sys.version,
        "source_sha256": before_hashes["source_panel"],
        "baseline_locked_beta": LOCKED_BETA,
        "baseline_reproduced_beta": baseline_beta,
        "baseline_abs_difference": abs(baseline_beta - LOCKED_BETA),
        "ppe_rule_frozen_before_estimation": True,
        "placebo_rules_frozen_before_estimation": True,
        "outputs": [
            str(IDENT / "future_placebo_results.csv"),
            str(IDENT / "firm_trend_results.csv"),
            str(HET / "asset_intensity_results.csv"),
            str(HET / "customer_facing_results.csv"),
            str(AUDIT / "sample_change_check.csv"),
        ],
    }
    (IDENT / "logs" / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
