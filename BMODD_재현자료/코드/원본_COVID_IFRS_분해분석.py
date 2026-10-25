from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyhdfe
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "06_CODE_AND_REPRODUCIBILITY"
STAGING = CODE_DIR / "staging"
LOG_DIR = CODE_DIR / "logs"

SOURCE_PANEL = Path(
    r"C:\Users\25294\Desktop\没用\论文2\01_新增控制变量数据"
    r"\03_加入新增控制变量后的研究面板.csv"
)
FROZEN_PANEL = Path(
    r"C:\Users\25294\Documents\Codex\2026-08-09\ni-g\outputs"
    r"\ssci_empirical_completion_20260816\03_frozen_augmented_research_panel.csv"
)
PRIMARY_CODE = Path(
    r"C:\Users\25294\Desktop\没用\论文2\03_复现代码"
    r"\paper2_extended_controls_pipeline.py"
)
FINANCIAL_CODE = Path(
    r"C:\Users\25294\Documents\Codex\2026-08-09\ni-g\work"
    r"\kosdaq_pipeline\kosdaq_financial_batch_pipeline.py"
)
PANEL_BUILD_CODE = Path(
    r"C:\Users\25294\Documents\Codex\2026-08-09\ni-g\work"
    r"\ssci_empirical_completion\01_audit_and_extract_controls.py"
)
RAW_FINANCIAL_LONG = Path(
    r"C:\Users\25294\Documents\Codex\2026-08-09\ni-g\outputs"
    r"\16_opendart_financial_raw_long.csv"
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
COMMON_CONTROLS = BASE_CONTROLS[1:]
EFFECTS = ["corp_code", "year"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_shift(panel: pd.DataFrame, column: str, horizon: int) -> pd.Series:
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


def drop_singletons(frame: pd.DataFrame, effects: list[str] = EFFECTS) -> pd.DataFrame:
    result = frame.copy()
    while not result.empty:
        keep = np.ones(len(result), dtype=bool)
        for effect in effects:
            keep &= result.groupby(effect)[effect].transform("size").gt(1).to_numpy()
        if keep.all():
            break
        result = result.loc[keep].copy()
    return result


def prepare_work(
    panel: pd.DataFrame,
    outcome: str,
    focals: list[str],
    controls: list[str],
    sample: pd.Series | None = None,
    forced_index: pd.Index | None = None,
) -> pd.DataFrame:
    controls = list(dict.fromkeys(x for x in controls if x not in focals))
    required = list(dict.fromkeys([outcome, *focals, *controls, *EFFECTS]))
    if forced_index is not None:
        work = panel.loc[forced_index, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(work) != len(forced_index):
            raise RuntimeError(f"Forced common sample lost rows for {outcome}")
        return work
    mask = sample if sample is not None else pd.Series(True, index=panel.index)
    work = panel.loc[mask, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    return drop_singletons(work)


def fit_fwl(
    panel: pd.DataFrame,
    model_id: str,
    family: str,
    outcome: str,
    controls: list[str],
    sample: pd.Series | None = None,
    forced_index: pd.Index | None = None,
    baseline_beta: float | None = None,
    sample_rule: str = "",
) -> tuple[dict[str, object], pd.DataFrame]:
    focals = ["BMI_component_z"]
    controls = list(dict.fromkeys(x for x in controls if x not in focals))
    work = prepare_work(panel, outcome, focals, controls, sample, forced_index)
    if work.empty:
        return (
            {
                "family": family,
                "model_id": model_id,
                "outcome": outcome,
                "status": "not_estimable",
                "sample_rule": sample_rule,
                "coefficient": np.nan,
                "std_error": np.nan,
                "p_value": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "N": 0,
                "firms": 0,
                "years": 0,
                "year_min": np.nan,
                "year_max": np.nan,
                "within_R2": np.nan,
                "coefficient_change_pct": np.nan,
                "fixed_effects": "firm + fiscal year",
                "cluster": "firm",
                "controls": "; ".join(controls),
                "bmi_standardization": "frozen full-panel z score; not restandardized",
            },
            work,
        )

    names = [*focals, *controls]
    matrix = work[[outcome, *names]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    codes = np.column_stack([pd.Categorical(work[e].astype(str)).codes for e in EFFECTS])
    absorber = pyhdfe.create(
        codes,
        drop_singletons=False,
        compute_degrees=False,
        residualize_method="map",
        options={"transform": "symmetric", "acceleration": "cg", "tol": 1e-10},
    )
    transformed = absorber.residualize(matrix)
    y = transformed[:, 0]
    selected_names: list[str] = []
    selected_arrays: list[np.ndarray] = []
    for name, values in zip(names, transformed[:, 1:].T):
        if np.sqrt(np.mean(values**2)) <= 1e-10:
            continue
        trial = np.column_stack([*selected_arrays, values]) if selected_arrays else values.reshape(-1, 1)
        if np.linalg.matrix_rank(trial) > len(selected_arrays):
            selected_names.append(name)
            selected_arrays.append(values)
    if "BMI_component_z" not in selected_names:
        return fit_fwl(
            panel.iloc[0:0], model_id, family, outcome, controls,
            forced_index=panel.iloc[0:0].index, baseline_beta=baseline_beta,
            sample_rule=sample_rule,
        )
    x = np.column_stack(selected_arrays)
    fitted = sm.OLS(y, x).fit()
    firm_codes = pd.Categorical(work["corp_code"].astype(str)).codes
    covariance = cov_cluster(fitted, firm_codes, use_correction=True)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    dof = max(1, work["corp_code"].nunique() - 1)
    j = selected_names.index("BMI_component_z")
    coefficient = float(fitted.params[j])
    std_error = float(se[j])
    t_value = coefficient / std_error
    p_value = float(2 * stats.t.sf(abs(t_value), df=dof))
    critical = float(stats.t.ppf(0.975, dof))
    tss = float(np.sum(y**2))
    within_r2 = 1 - float(np.sum(fitted.resid**2)) / tss if tss > 0 else np.nan
    coefficient_change = (
        100 * (coefficient - baseline_beta) / abs(baseline_beta)
        if baseline_beta is not None and baseline_beta != 0
        else np.nan
    )
    return (
        {
            "family": family,
            "model_id": model_id,
            "outcome": outcome,
            "status": "estimated",
            "sample_rule": sample_rule,
            "coefficient": coefficient,
            "std_error": std_error,
            "t_value": t_value,
            "p_value": p_value,
            "ci95_low": coefficient - critical * std_error,
            "ci95_high": coefficient + critical * std_error,
            "ci95_width": 2 * critical * std_error,
            "N": int(len(work)),
            "firms": int(work["corp_code"].nunique()),
            "years": int(work["year"].nunique()),
            "year_min": int(work["year"].min()),
            "year_max": int(work["year"].max()),
            "within_R2": within_r2,
            "coefficient_change_pct": coefficient_change,
            "fixed_effects": "firm + fiscal year",
            "cluster": "firm",
            "controls": "; ".join(controls),
            "bmi_standardization": "frozen full-panel z score; not restandardized",
        },
        work,
    )


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["fdr_scope"] = "all newly estimated supplementary BMI tests in this audit; baseline excluded"
    results["q_value_bh"] = np.nan
    mask = results["status"].eq("estimated") & ~results["model_id"].str.endswith("BASELINE")
    if mask.any():
        results.loc[mask, "q_value_bh"] = multipletests(
            results.loc[mask, "p_value"].to_numpy(float), method="fdr_bh"
        )[1]
    return results


def build_model_results(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    baseline, baseline_work = fit_fwl(
        panel,
        "A1_BASELINE",
        "Formal H1 baseline",
        "FutureAssetTurnover_h2",
        BASE_CONTROLS,
        sample_rule="No additional year exclusion",
    )
    baseline_beta = float(baseline["coefficient"])
    baseline["coefficient_change_pct"] = 0.0
    baseline_n = int(baseline["N"])
    baseline_firms = int(baseline["firms"])
    rows = [baseline]

    covid_specs = [
        ("A2_COVID_EXCLUDE_RESULT_2020_2021", ~panel["year"].add(2).isin([2020, 2021]), "Exclude t+2 in 2020–2021"),
        ("A3_COVID_EXCLUDE_DISCLOSURE_2020_2021", ~panel["year"].isin([2020, 2021]), "Exclude t in 2020–2021"),
        (
            "A4_COVID_EXCLUDE_UNION_2020_2021",
            ~(panel["year"].isin([2020, 2021]) | panel["year"].add(2).isin([2020, 2021])),
            "Exclude t or t+2 in 2020–2021",
        ),
        ("A5X_COVID_EXCLUDE_RESULT_2020_2022", ~panel["year"].add(2).isin([2020, 2021, 2022]), "Extended: exclude t+2 in 2020–2022"),
        ("A5X_COVID_EXCLUDE_DISCLOSURE_2020_2022", ~panel["year"].isin([2020, 2021, 2022]), "Extended: exclude t in 2020–2022"),
        (
            "A5X_COVID_EXCLUDE_UNION_2020_2022",
            ~(panel["year"].isin([2020, 2021, 2022]) | panel["year"].add(2).isin([2020, 2021, 2022])),
            "Extended: exclude t or t+2 in 2020–2022",
        ),
    ]
    for model_id, mask, rule in covid_specs:
        row, _ = fit_fwl(
            panel, model_id, "COVID robustness", "FutureAssetTurnover_h2", BASE_CONTROLS,
            sample=mask, baseline_beta=baseline_beta, sample_rule=rule,
        )
        row["N_change"] = int(row["N"]) - baseline_n
        row["firms_change"] = int(row["firms"]) - baseline_firms
        rows.append(row)

    crossing = panel["year"].lt(2019) & panel["year"].add(2).ge(2019)
    ifrs_specs = [
        ("B2_IFRS_EXCLUDE_RESULT_2019", ~panel["year"].add(2).eq(2019), "Exclude t+2=2019"),
        ("B3_IFRS_EXCLUDE_DISCLOSURE_2019", ~panel["year"].eq(2019), "Exclude t=2019"),
        ("B4_IFRS_EXCLUDE_CROSSING_2019", ~crossing, "Exclude t<2019 and t+2>=2019"),
        ("B5_IFRS_PRE_RESULT_2019", panel["year"].add(2).lt(2019), "Only t+2<2019"),
        ("B5_IFRS_POST_RESULT_2019", panel["year"].add(2).ge(2019), "Only t+2>=2019"),
    ]
    for model_id, mask, rule in ifrs_specs:
        row, _ = fit_fwl(
            panel, model_id, "K-IFRS 1116 robustness", "FutureAssetTurnover_h2", BASE_CONTROLS,
            sample=mask, baseline_beta=baseline_beta, sample_rule=rule,
        )
        row["N_change"] = int(row["N"]) - baseline_n
        row["firms_change"] = int(row["firms"]) - baseline_firms
        rows.append(row)

    results = pd.DataFrame(rows)
    results.loc[results["model_id"].eq("A1_BASELINE"), "N_change"] = 0
    results.loc[results["model_id"].eq("A1_BASELINE"), "firms_change"] = 0
    return results, baseline_work.index


def merge_raw_financials(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_columns = [
        "corp_code", "year", "TotalAssets", "AverageAssets", "Sales", "AssetTurnover", "financial_fs_div"
    ]
    raw = pd.read_csv(
        FROZEN_PANEL,
        usecols=raw_columns,
        dtype={"corp_code": str},
        low_memory=False,
    )
    raw["corp_code"] = raw["corp_code"].str.zfill(8)
    raw["year"] = pd.to_numeric(raw["year"], errors="raise").astype(int)
    raw = raw.sort_values(["corp_code", "year"]).reset_index(drop=True)
    prior_assets = raw.groupby("corp_code")["TotalAssets"].shift(1)
    prior_year = raw.groupby("corp_code")["year"].shift(1)
    expected_average = ((raw["TotalAssets"] + prior_assets) / 2).where(raw["year"].sub(prior_year).eq(1))
    expected_turnover = raw["Sales"] / raw["AverageAssets"]
    average_error = (expected_average - raw["AverageAssets"]).abs().dropna()
    turnover_error = (expected_turnover - raw["AssetTurnover"]).abs().dropna()

    merged = panel.merge(
        raw,
        on=["corp_code", "year"],
        how="left",
        suffixes=("", "_raw"),
        validate="one_to_one",
    )
    for column in ["Sales_raw", "AverageAssets", "TotalAssets_raw", "AssetTurnover_raw", "financial_fs_div_raw"]:
        if column not in merged.columns:
            # Sales, TotalAssets and financial_fs_div have same names in both source panels.
            base_name = column.replace("_raw", "")
            if base_name in merged.columns:
                merged[column] = merged[base_name]
    merged["Sales_t2"] = exact_shift(merged, "Sales_raw", 2)
    merged["AverageAssets_t2"] = exact_shift(merged, "AverageAssets", 2)
    merged["AssetTurnover_raw_t2"] = exact_shift(merged, "AssetTurnover_raw", 2)
    merged["AssetTurnover_w_t2_check"] = exact_shift(merged, "AssetTurnover_w", 2)
    merged["financial_fs_div_t2"] = exact_shift(merged, "financial_fs_div_raw", 2)
    checks = {
        "frozen_panel_rows": int(len(raw)),
        "frozen_panel_firms": int(raw["corp_code"].nunique()),
        "average_assets_formula_max_abs_error": float(average_error.max()) if len(average_error) else None,
        "average_assets_formula_mismatch_gt_1e_8": int(average_error.gt(1e-8).sum()),
        "asset_turnover_formula_max_abs_error": float(turnover_error.max()) if len(turnover_error) else None,
        "asset_turnover_formula_mismatch_gt_1e_12": int(turnover_error.gt(1e-12).sum()),
    }
    return merged, checks


def build_decomposition(
    panel: pd.DataFrame,
    baseline_index: pd.Index,
    baseline_beta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    base = panel.loc[baseline_index].copy()
    flow: list[dict[str, object]] = []

    def record(stage: str, frame: pd.DataFrame, excluded: int, reason: str) -> None:
        flow.append(
            {
                "stage": stage,
                "N": int(len(frame)),
                "firms": int(frame["corp_code"].nunique()) if len(frame) else 0,
                "excluded_at_stage": int(excluded),
                "reason": reason,
            }
        )

    work = base.copy()
    record("Main-regression sample", work, 0, "Reproduced 13,980-observation H1 sample")
    before = len(work)
    work = work[work[["Sales_raw", "Sales_t2"]].notna().all(axis=1)].copy()
    record("Sales observed at t and t+2", work, before - len(work), "Missing raw sales")
    before = len(work)
    work = work[work["Sales_raw"].gt(0) & work["Sales_t2"].gt(0)].copy()
    record("Sales positive at t and t+2", work, before - len(work), "Non-positive raw sales")
    before = len(work)
    work = work[work[["AverageAssets", "AverageAssets_t2"]].notna().all(axis=1)].copy()
    record("Average assets observed at t and t+2", work, before - len(work), "Missing average assets")
    before = len(work)
    work = work[work["AverageAssets"].gt(0) & work["AverageAssets_t2"].gt(0)].copy()
    record("Average assets positive at t and t+2", work, before - len(work), "Non-positive average assets")

    work["DeltaLogSales_t2"] = np.log(work["Sales_t2"]) - np.log(work["Sales_raw"])
    work["DeltaLogAvgAssets_t2"] = np.log(work["AverageAssets_t2"]) - np.log(work["AverageAssets"])
    work["DeltaLogTurnover_t2"] = (
        np.log(work["Sales_t2"] / work["AverageAssets_t2"])
        - np.log(work["Sales_raw"] / work["AverageAssets"])
    )
    before = len(work)
    work = drop_singletons(work)
    record("Final strict common sample", work, before - len(work), "Iterative firm/year FE singleton removal")

    identity_error = work["DeltaLogTurnover_t2"] - (
        work["DeltaLogSales_t2"] - work["DeltaLogAvgAssets_t2"]
    )
    ratio_error = work["AssetTurnover_raw_t2"] - work["Sales_t2"] / work["AverageAssets_t2"]
    lead_w_error = work["FutureAssetTurnover_h2"] - work["AssetTurnover_w_t2_check"]

    results: list[dict[str, object]] = []
    for spec_name, controls in [
        ("MAIN_BASE_CONTROLS", BASE_CONTROLS),
        ("CHANGE_MODEL_CONTROLS_SENSITIVITY", COMMON_CONTROLS),
    ]:
        for outcome in ["DeltaLogSales_t2", "DeltaLogAvgAssets_t2", "DeltaLogTurnover_t2"]:
            row, _ = fit_fwl(
                work,
                f"C_{spec_name}_{outcome}",
                "Turnover decomposition" if spec_name == "MAIN_BASE_CONTROLS" else "Turnover decomposition sensitivity",
                outcome,
                controls,
                forced_index=work.index,
                baseline_beta=baseline_beta,
                sample_rule="Strict common positive-log sample; identical rows and FE design across three outcomes",
            )
            row["control_protocol"] = spec_name
            row["N_change"] = int(row["N"]) - 13980
            row["firms_change"] = int(row["firms"]) - 2054
            results.append(row)

    stable_statement = work[
        work["financial_fs_div_raw"].notna()
        & work["financial_fs_div_t2"].notna()
        & work["financial_fs_div_raw"].eq(work["financial_fs_div_t2"])
    ].copy()
    record(
        "Stable financial-statement-basis sensitivity sample",
        stable_statement,
        len(work) - len(stable_statement),
        "Exclude CFS/OFS basis changes between t and t+2; not the primary decomposition sample",
    )
    for outcome in ["DeltaLogSales_t2", "DeltaLogAvgAssets_t2", "DeltaLogTurnover_t2"]:
        row, _ = fit_fwl(
            stable_statement,
            f"C_STABLE_FS_DIV_SENSITIVITY_{outcome}",
            "Turnover decomposition sensitivity",
            outcome,
            BASE_CONTROLS,
            forced_index=stable_statement.index,
            baseline_beta=baseline_beta,
            sample_rule="Sensitivity: strict common sample with identical CFS/OFS basis at t and t+2",
        )
        row["control_protocol"] = "STABLE_FS_DIV_SENSITIVITY"
        row["N_change"] = int(row["N"]) - 13980
        row["firms_change"] = int(row["firms"]) - 2054
        results.append(row)
    results_df = pd.DataFrame(results)

    coefficient_checks = []
    for protocol, group in results_df.groupby("control_protocol", sort=False):
        beta = group.set_index("outcome")["coefficient"]
        difference = float(beta["DeltaLogSales_t2"] - beta["DeltaLogAvgAssets_t2"])
        turnover = float(beta["DeltaLogTurnover_t2"])
        coefficient_checks.append(
            {
                "control_protocol": protocol,
                "beta_sales": float(beta["DeltaLogSales_t2"]),
                "beta_avg_assets": float(beta["DeltaLogAvgAssets_t2"]),
                "beta_sales_minus_assets": difference,
                "beta_turnover": turnover,
                "coefficient_identity_error": turnover - difference,
            }
        )

    year_counts = work.groupby("year", as_index=False).agg(N=("corp_code", "size"), firms=("corp_code", "nunique"))
    industry_counts = (
        work.groupby("industry_group", dropna=False, as_index=False)
        .agg(N=("corp_code", "size"), firms=("corp_code", "nunique"))
        .sort_values("N", ascending=False)
    )
    industry_counts["sample_share"] = industry_counts["N"] / len(work)

    statement_transition = pd.crosstab(
        work["financial_fs_div_raw"].fillna("Missing"),
        work["financial_fs_div_t2"].fillna("Missing"),
        dropna=False,
    ).stack(future_stack=True).reset_index(name="N")
    statement_transition.columns = ["financial_fs_div_t", "financial_fs_div_t2", "N"]

    common_columns = [
        "corp_code", "year", "market", "industry_group", "financial_fs_div_raw", "financial_fs_div_t2",
        "BMI_component_z", "Sales_raw", "Sales_t2", "AverageAssets", "AverageAssets_t2",
        "DeltaLogSales_t2", "DeltaLogAvgAssets_t2", "DeltaLogTurnover_t2",
    ]
    common_sample = work[common_columns].copy()
    checks = {
        "identity_rows": int(len(work)),
        "delta_log_identity_max_abs_error": float(identity_error.abs().max()),
        "delta_log_identity_mismatch_gt_1e_12": int(identity_error.abs().gt(1e-12).sum()),
        "future_raw_turnover_ratio_max_abs_error": float(ratio_error.abs().max()),
        "future_winsorized_outcome_exact_lead_max_abs_error": float(lead_w_error.abs().max()),
        "common_sample_rows": int(len(work)),
        "common_sample_firms": int(work["corp_code"].nunique()),
        "common_sample_year_min": int(work["year"].min()),
        "common_sample_year_max": int(work["year"].max()),
        "largest_industry_share": float(industry_counts["sample_share"].max()),
        "largest_industry": str(industry_counts.iloc[0]["industry_group"]),
    }
    diagnostics = pd.concat(
        [
            year_counts.assign(table="year_counts"),
            industry_counts.assign(table="industry_counts"),
            statement_transition.assign(table="financial_statement_transition"),
        ],
        ignore_index=True,
        sort=False,
    )
    return results_df, pd.DataFrame(flow), pd.DataFrame(coefficient_checks), common_sample, diagnostics, checks


def audit_lease_variables(panel: pd.DataFrame) -> pd.DataFrame:
    terms = ["lease", "rightofuse", "right_of_use", "사용권", "리스", "임차"]
    rows: list[dict[str, object]] = []
    for source_name, columns in [
        ("extended research panel", panel.columns.tolist()),
        ("frozen augmented panel", pd.read_csv(FROZEN_PANEL, nrows=0).columns.tolist()),
        ("selected raw financial long file", pd.read_csv(RAW_FINANCIAL_LONG, nrows=0).columns.tolist()),
    ]:
        matches = [column for column in columns if any(term in column.lower() for term in terms)]
        rows.append(
            {
                "source": source_name,
                "lease_specific_columns": "; ".join(matches),
                "count": len(matches),
                "assessment": "No integrated lease-dependence variable" if not matches else "Requires substantive validation",
            }
        )
    raw_variables = pd.read_csv(RAW_FINANCIAL_LONG, usecols=["variable"], low_memory=False)["variable"].dropna().astype(str).unique()
    raw_matches = [value for value in raw_variables if any(term in value.lower() for term in terms)]
    rows.append(
        {
            "source": "selected raw financial long variable values",
            "lease_specific_columns": "; ".join(raw_matches),
            "count": len(raw_matches),
            "assessment": "The stored long file contains selected core variables, not lease-specific accounts",
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    required_files = [SOURCE_PANEL, FROZEN_PANEL, PRIMARY_CODE, FINANCIAL_CODE, PANEL_BUILD_CODE, RAW_FINANCIAL_LONG]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))

    panel = pd.read_csv(SOURCE_PANEL, dtype={"corp_code": str, "stock_code": str}, low_memory=False)
    panel["corp_code"] = panel["corp_code"].str.zfill(8)
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel = panel.sort_values(["corp_code", "year"]).reset_index(drop=True)
    if panel.duplicated(["corp_code", "year"]).any():
        raise RuntimeError("Source panel contains duplicate firm-year keys")

    model_results, baseline_index = build_model_results(panel)
    baseline = model_results.loc[model_results["model_id"].eq("A1_BASELINE")].iloc[0]
    if not (
        int(baseline["N"]) == 13980
        and int(baseline["firms"]) == 2054
        and abs(float(baseline["coefficient"]) - 0.030243177005377098) < 1e-10
        and abs(float(baseline["std_error"]) - 0.006895) < 1e-4
    ):
        raise RuntimeError("Frozen main model did not reproduce; supplementary analysis halted")

    merged, raw_checks = merge_raw_financials(panel)
    decomp, sample_flow, coefficient_checks, common_sample, diagnostics, decomp_checks = build_decomposition(
        merged,
        baseline_index,
        float(baseline["coefficient"]),
    )
    all_results = add_fdr(pd.concat([model_results, decomp], ignore_index=True, sort=False))
    covid = all_results[all_results["family"].isin(["Formal H1 baseline", "COVID robustness"])].copy()
    ifrs = all_results[all_results["family"].isin(["Formal H1 baseline", "K-IFRS 1116 robustness"])].copy()
    decomp = all_results[all_results["family"].str.startswith("Turnover decomposition")].copy()
    lease = audit_lease_variables(panel)

    covid.to_csv(STAGING / "covid_models.csv", index=False, encoding="utf-8-sig")
    ifrs.to_csv(STAGING / "ifrs1116_models.csv", index=False, encoding="utf-8-sig")
    decomp.to_csv(STAGING / "turnover_decomposition_models.csv", index=False, encoding="utf-8-sig")
    sample_flow.to_csv(STAGING / "decomposition_sample_flow.csv", index=False, encoding="utf-8-sig")
    coefficient_checks.to_csv(STAGING / "coefficient_identity_checks.csv", index=False, encoding="utf-8-sig")
    common_sample.to_csv(STAGING / "decomposition_common_sample.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(STAGING / "decomposition_sample_diagnostics.csv", index=False, encoding="utf-8-sig")
    lease.to_csv(STAGING / "lease_variable_audit.csv", index=False, encoding="utf-8-sig")
    workbook_payload = {
        "covid": json.loads(covid.to_json(orient="records")),
        "ifrs": json.loads(ifrs.to_json(orient="records")),
        "decomposition": json.loads(decomp.to_json(orient="records")),
        "sample_flow": json.loads(sample_flow.to_json(orient="records")),
        "coefficient_checks": json.loads(coefficient_checks.to_json(orient="records")),
        "common_sample": json.loads(common_sample.to_json(orient="records")),
        "diagnostics": json.loads(diagnostics.to_json(orient="records")),
        "lease": json.loads(lease.to_json(orient="records")),
    }
    (STAGING / "workbook_payload.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False), encoding="utf-8"
    )

    hashes = []
    for path in required_files:
        hashes.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "purpose": {
                    SOURCE_PANEL: "Frozen 22,654-row analysis panel and main regression source",
                    FROZEN_PANEL: "Pre-regression financial panel with raw Sales, TotalAssets and AverageAssets",
                    PRIMARY_CODE: "Original extended-controls reproduction and main-model code",
                    FINANCIAL_CODE: "Original financial statement selection and AverageAssets construction code",
                    PANEL_BUILD_CODE: "Frozen augmented panel construction and AssetTurnover formula",
                    RAW_FINANCIAL_LONG: "Stored selected-account long financial data",
                }[path],
            }
        )
    pd.DataFrame(hashes).to_csv(CODE_DIR / "SOURCE_FILE_HASHES.csv", index=False, encoding="utf-8-sig")

    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "pyhdfe": getattr(pyhdfe, "__version__", "unknown"),
        "working_directory": str(Path.cwd()),
        "output_root": str(ROOT),
    }
    (CODE_DIR / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "main_model_reproduced": True,
        "baseline": baseline.to_dict(),
        "source_panel": {
            "rows": int(len(panel)),
            "firms": int(panel["corp_code"].nunique()),
            "year_min": int(panel["year"].min()),
            "year_max": int(panel["year"].max()),
            "duplicate_firm_year_keys": int(panel.duplicated(["corp_code", "year"]).sum()),
            "BMI_share_mean": float(panel["BMI_share"].mean()),
            "BMI_share_sd_ddof1": float(panel["BMI_share"].std(ddof=1)),
            "BMI_z_mean": float(panel["BMI_component_z"].mean()),
            "BMI_z_sd_ddof1": float(panel["BMI_component_z"].std(ddof=1)),
        },
        "raw_formula_checks": raw_checks,
        "decomposition_checks": decomp_checks,
        "ifrs_crossing_candidate_rows_full_panel": int((panel["year"].lt(2019) & panel["year"].add(2).ge(2019)).sum()),
        "ifrs_crossing_rows_baseline_sample": int((panel.loc[baseline_index, "year"].lt(2019) & panel.loc[baseline_index, "year"].add(2).ge(2019)).sum()),
        "interpretation_boundaries": [
            "Year exclusions are sensitivity checks, not causal identification of COVID-19 or K-IFRS 1116.",
            "The decomposition outcomes are log changes from t to t+2, not an exact decomposition of the level-model coefficient.",
            "No lease-specific variable is integrated into the frozen panel.",
        ],
    }
    (LOG_DIR / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    log_lines = [
        "Audit execution completed.",
        f"Main model beta={baseline['coefficient']:.12f}; SE={baseline['std_error']:.12f}; N={int(baseline['N'])}; firms={int(baseline['firms'])}",
        f"Common decomposition sample N={decomp_checks['common_sample_rows']}; firms={decomp_checks['common_sample_firms']}",
        f"Delta-log identity max error={decomp_checks['delta_log_identity_max_abs_error']:.3e}",
        "No paper, classifier, frozen input, or stored regression output was modified.",
    ]
    (LOG_DIR / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
