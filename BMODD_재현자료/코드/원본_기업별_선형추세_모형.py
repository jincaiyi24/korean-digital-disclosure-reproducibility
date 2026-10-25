from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy import sparse, stats
from scipy.sparse.linalg import lsmr
from statsmodels.stats.sandwich_covariance import cov_cluster

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SOURCE = Path(
    r"C:\Users\25294\Desktop\没用\论文2\01_新增控制变量数据"
    r"\03_加入新增控制变量后的研究面板.csv"
)
PRIOR_RESULTS = Path(
    r"C:\Users\25294\Documents\Codex\2026-10-05\paper-handoff-human-coding"
    r"\enhancement_experiments\01_identification\firm_trend_results.csv"
)
DOCX_ROOT = Path(
    r"C:\Users\25294\Documents\Codex\2026-10-05\paper-handoff-human-coding"
)

OUTCOME = "FutureAssetTurnover_h2"
FOCAL = "BMI_component_z"
CONTROLS = [
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
REGRESSORS = [FOCAL, *CONTROLS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(index: pd.Index) -> str:
    payload = "\n".join(str(int(x)) for x in sorted(index)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def drop_singletons(frame: pd.DataFrame, effects: list[str]) -> pd.DataFrame:
    result = frame.copy()
    while not result.empty:
        keep = np.ones(len(result), dtype=bool)
        for effect in effects:
            keep &= result.groupby(effect, sort=False)[effect].transform("size").gt(1).to_numpy()
        if keep.all():
            return result
        result = result.loc[keep].copy()
    return result


def prepare_work(
    panel: pd.DataFrame,
    effects: list[str],
    forced_index: pd.Index | None = None,
) -> pd.DataFrame:
    required = [OUTCOME, *REGRESSORS, *effects, "corp_code", "year"]
    required = list(dict.fromkeys(required))
    if forced_index is None:
        work = panel.loc[:, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        return drop_singletons(work, effects)
    work = panel.loc[forced_index, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) != len(forced_index):
        raise RuntimeError("Forced common sample lost rows to missing values")
    for effect in effects:
        if work.groupby(effect)[effect].transform("size").lt(2).any():
            raise RuntimeError(f"Forced sample contains singleton for {effect}")
    return work


def select_full_rank(x_raw: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    selected_names: list[str] = []
    selected_arrays: list[np.ndarray] = []
    for name, values in zip(names, x_raw.T):
        if np.sqrt(np.mean(values**2)) <= 1e-10:
            continue
        trial = np.column_stack([*selected_arrays, values]) if selected_arrays else values.reshape(-1, 1)
        if np.linalg.matrix_rank(trial) > len(selected_arrays):
            selected_names.append(name)
            selected_arrays.append(values)
    if not selected_arrays:
        raise RuntimeError("All regressors absorbed")
    return np.column_stack(selected_arrays), selected_names


def finish_fit(
    work: pd.DataFrame,
    transformed: np.ndarray,
    model_id: str,
    specification: str,
    absorption_diagnostics: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    y = transformed[:, 0]
    x, names = select_full_rank(transformed[:, 1:], REGRESSORS)
    fitted = sm.OLS(y, x).fit()
    firm_codes = pd.Categorical(work["corp_code"].astype(str)).codes
    covariance = np.asarray(cov_cluster(fitted, firm_codes, use_correction=True))
    se_all = np.sqrt(np.maximum(np.diag(covariance), 0))
    dof = max(1, work["corp_code"].nunique() - 1)
    pvals = 2 * stats.t.sf(np.abs(np.asarray(fitted.params) / se_all), df=dof)
    critical = stats.t.ppf(0.975, dof)
    j = names.index(FOCAL)
    tss = float(y @ y)
    within_r2 = 1 - float(fitted.resid @ fitted.resid) / tss if tss > 0 else np.nan
    result = {
        "model_id": model_id,
        "specification": specification,
        "coefficient": float(fitted.params[j]),
        "std_error": float(se_all[j]),
        "p_value": float(pvals[j]),
        "ci95_low": float(fitted.params[j] - critical * se_all[j]),
        "ci95_high": float(fitted.params[j] + critical * se_all[j]),
        "t_stat": float(fitted.params[j] / se_all[j]),
        "N": int(len(work)),
        "firms": int(work["corp_code"].nunique()),
        "year_min": int(work["year"].min()),
        "year_max": int(work["year"].max()),
        "within_R2": within_r2,
        "cluster": "firm",
        "sample_index_sha256": index_hash(work.index),
    }
    details = {
        "work": work,
        "transformed": transformed,
        "y": y,
        "x": x,
        "names": names,
        "params": np.asarray(fitted.params),
        "covariance": covariance,
        "std_errors": se_all,
        "residuals": np.asarray(fitted.resid),
        "focal_position": j,
        "condition_number_X": float(np.linalg.cond(x)),
        "rank_X": int(np.linalg.matrix_rank(x)),
        "absorption": absorption_diagnostics or {},
    }
    return result, details


def fit_absorbed(
    panel: pd.DataFrame,
    model_id: str,
    effects: list[str],
    specification: str,
    forced_index: pd.Index | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    work = prepare_work(panel, effects, forced_index)
    matrix = work[[OUTCOME, *REGRESSORS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    codes = np.column_stack([pd.Categorical(work[e].astype(str)).codes for e in effects])
    absorber = pyhdfe.create(
        codes,
        drop_singletons=False,
        compute_degrees=False,
        residualize_method="map",
        options={"transform": "symmetric", "acceleration": "cg", "tol": 1e-10},
    )
    transformed = absorber.residualize(matrix)
    max_group_mean = 0.0
    for effect in effects:
        check = pd.DataFrame(transformed, index=work.index).groupby(work[effect]).mean().to_numpy()
        max_group_mean = max(max_group_mean, float(np.max(np.abs(check))))
    absorption = {
        "method": "pyhdfe alternating projections",
        "effects": effects,
        "max_abs_absorbed_group_mean": max_group_mean,
    }
    return finish_fit(work, transformed, model_id, specification, absorption)


def build_trend_design(work: pd.DataFrame, time_mode: str = "firm_centered") -> sparse.csr_matrix:
    firms = pd.Categorical(work["corp_code"].astype(str))
    years = pd.Categorical(work["year"].astype(str))
    n = len(work)
    firm_dummy = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), firms.codes)), shape=(n, len(firms.categories))
    )
    year_dummy = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), years.codes)), shape=(n, len(years.categories))
    )[:, 1:]
    year_value = work["year"].to_numpy(float)
    if time_mode == "firm_centered":
        time_value = year_value - work.groupby("corp_code")["year"].transform("mean").to_numpy(float)
    elif time_mode == "global_centered":
        time_value = year_value - float(year_value.mean())
    elif time_mode == "raw_year_scaled":
        # Multiplying a slope column by a nonzero constant preserves its span.
        # Scaling avoids the avoidable numerical ill-conditioning of raw years near 2,020.
        time_value = year_value / 2020.0
    else:
        raise ValueError(time_mode)
    firm_slope = sparse.csr_matrix(
        (time_value, (np.arange(n), firms.codes)), shape=(n, len(firms.categories))
    )
    return sparse.hstack([firm_dummy, year_dummy, firm_slope], format="csr")


def residualize_sparse(design: sparse.csr_matrix, matrix: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    transformed = np.empty_like(matrix, dtype=float)
    solver_log: list[dict[str, object]] = []
    for col in range(matrix.shape[1]):
        fit = lsmr(design, matrix[:, col], atol=1e-12, btol=1e-12, maxiter=30000)
        transformed[:, col] = matrix[:, col] - design @ fit[0]
        solver_log.append(
            {
                "column": col,
                "istop": int(fit[1]),
                "iterations": int(fit[2]),
                "residual_norm": float(fit[3]),
                "normal_equation_norm": float(fit[4]),
            }
        )
    return transformed, solver_log


def fit_trend(
    panel: pd.DataFrame,
    model_id: str,
    forced_index: pd.Index,
    time_mode: str = "firm_centered",
) -> tuple[dict[str, object], dict[str, object]]:
    work = prepare_work(panel, ["corp_code", "year"], forced_index)
    matrix = work[[OUTCOME, *REGRESSORS]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    design = build_trend_design(work, time_mode)
    transformed, solver_log = residualize_sparse(design, matrix)
    max_orthogonality = float(np.max(np.abs(design.T @ transformed)))
    counts = work.groupby("corp_code")["year"].nunique()
    absorption = {
        "method": "sparse least-squares projection",
        "time_mode": time_mode,
        "design_rows": int(design.shape[0]),
        "design_columns": int(design.shape[1]),
        "firm_columns": int(work["corp_code"].nunique()),
        "year_dummy_columns": int(work["year"].nunique() - 1),
        "firm_slope_columns": int(work["corp_code"].nunique()),
        "firms_with_two_years": int(counts.eq(2).sum()),
        "firms_with_three_or_more_years": int(counts.ge(3).sum()),
        "max_abs_Zt_residual": max_orthogonality,
        "solver": solver_log,
    }
    return finish_fit(
        work,
        transformed,
        model_id,
        "Firm FE + Year FE + firm-specific linear trend",
        absorption,
    )


def enforce_common_sample(panel: pd.DataFrame, starting_index: pd.Index) -> pd.Index:
    idx = pd.Index(starting_index)
    effect_sets = [
        ["corp_code", "year"],
        ["corp_code", "industry_year"],
        ["corp_code", "market_year"],
    ]
    while True:
        old_n = len(idx)
        work = panel.loc[idx]
        keep = pd.Series(True, index=idx)
        for effects in effect_sets:
            for effect in effects:
                keep &= work.groupby(effect)[effect].transform("size").gt(1)
        idx = idx[keep.to_numpy()]
        if len(idx) == old_n:
            return idx


def parameterization_check(work: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    matrix = work[[OUTCOME, *REGRESSORS]].to_numpy(float)
    ref_x, ref_names = select_full_rank(reference[:, 1:], REGRESSORS)
    ref_beta = float(sm.OLS(reference[:, 0], ref_x).fit().params[ref_names.index(FOCAL)])
    rows: list[dict[str, object]] = []
    for mode in ["firm_centered", "global_centered", "raw_year_scaled"]:
        design = build_trend_design(work, mode)
        transformed, solver = residualize_sparse(design, matrix)
        x, names = select_full_rank(transformed[:, 1:], REGRESSORS)
        beta = float(sm.OLS(transformed[:, 0], x).fit().params[names.index(FOCAL)])
        max_diff = float(np.max(np.abs(transformed - reference)))
        rel_diff = float(np.linalg.norm(transformed - reference) / max(np.linalg.norm(reference), 1e-30))
        rows.append(
            {
                "time_parameterization": mode,
                "BMI_coefficient": beta,
                "coefficient_difference_vs_firm_centered": beta - ref_beta,
                "max_abs_residual_difference_vs_firm_centered": max_diff,
                "relative_frobenius_difference_vs_firm_centered": rel_diff,
                "max_abs_Zt_residual": float(np.max(np.abs(design.T @ transformed))),
                "max_iterations": max(x["iterations"] for x in solver),
                "all_solver_istop": ";".join(str(x["istop"]) for x in solver),
                "column_space_interpretation": "equivalent after firm intercepts; numerical conditioning differs",
            }
        )
    return pd.DataFrame(rows)


def direct_augmented_coefficient(details: dict[str, object]) -> dict[str, object]:
    work = details["work"]
    y = work[OUTCOME].to_numpy(float)
    x = work[REGRESSORS].to_numpy(float)
    absorbed = build_trend_design(work, "firm_centered")
    design = sparse.hstack([sparse.csr_matrix(x), absorbed], format="csr")
    fit = lsmr(design, y, atol=1e-12, btol=1e-12, maxiter=50000)
    coefficient = float(fit[0][0])
    expected = float(details["params"][details["focal_position"]])
    return {
        "method": "direct augmented sparse least squares",
        "coefficient": coefficient,
        "FWL_coefficient": expected,
        "absolute_difference": abs(coefficient - expected),
        "matches_within_1e-8": bool(abs(coefficient - expected) < 1e-8),
        "iterations": int(fit[2]),
        "istop": int(fit[1]),
        "residual_norm": float(fit[3]),
        "normal_equation_norm": float(fit[4]),
    }


def influential_firms(details: dict[str, object]) -> pd.DataFrame:
    work = details["work"]
    x = np.asarray(details["x"])
    y = np.asarray(details["y"])
    beta = np.asarray(details["params"])
    focal_position = int(details["focal_position"])
    focal_se = float(details["std_errors"][focal_position])
    xtx = x.T @ x
    xty = x.T @ y
    inv_xtx = np.linalg.pinv(xtx)
    hat_diag = np.einsum("ij,jk,ik->i", x, inv_xtx, x)
    rows: list[dict[str, object]] = []
    firm_values = work["corp_code"].astype(str).to_numpy()
    for firm in pd.unique(firm_values):
        loc = np.flatnonzero(firm_values == firm)
        xg = x[loc]
        yg = y[loc]
        beta_deleted = np.linalg.pinv(xtx - xg.T @ xg) @ (xty - xg.T @ yg)
        delta = float(beta_deleted[focal_position] - beta[focal_position])
        rows.append(
            {
                "corp_code": firm,
                "observations": int(len(loc)),
                "cluster_leverage_trace": float(hat_diag[loc].sum()),
                "full_sample_coefficient": float(beta[focal_position]),
                "fixed_absorption_deleted_coefficient": float(beta_deleted[focal_position]),
                "deleted_minus_full": delta,
                "absolute_coefficient_change": abs(delta),
                "change_in_full_sample_SE_units": delta / focal_se,
                "diagnostic_scope": (
                    "Cluster deletion on the full-sample partialled-out design; "
                    "firm/year/trend absorption is held fixed and no observation is removed from any reported model."
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("absolute_coefficient_change", ascending=False).reset_index(drop=True)
    result.insert(0, "influence_rank", np.arange(1, len(result) + 1))
    return result


def add_comparison_fields(frame: pd.DataFrame, baseline_n: int, baseline_hash: str) -> pd.DataFrame:
    result = frame.copy()
    result["sample_difference_N_vs_M1"] = result["N"] - baseline_n
    result["same_sample_as_M1"] = result["sample_index_sha256"].eq(baseline_hash)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    HERE.mkdir(parents=True, exist_ok=True)

    input_paths = [SOURCE, PRIOR_RESULTS]
    input_paths += sorted(DOCX_ROOT.rglob("*.docx"))
    before_hashes = {str(path): sha256(path) for path in input_paths if path.exists()}

    panel = pd.read_csv(SOURCE, dtype={"corp_code": str, "stock_code": str}, low_memory=False)
    panel["corp_code"] = panel["corp_code"].str.zfill(8)
    panel["year"] = numeric(panel["year"]).astype(int)
    panel = panel.sort_values(["corp_code", "year"]).reset_index(drop=True)
    if panel.duplicated(["corp_code", "year"]).any():
        raise RuntimeError("Duplicate firm-year keys in source panel")
    panel["industry_year"] = panel["industry_group"].fillna("UNKNOWN").astype(str) + "_" + panel["year"].astype(str)
    panel["market_year"] = panel["market"].fillna("UNKNOWN").astype(str) + "_" + panel["year"].astype(str)

    # Model A / M1 establishes the locked 13,980-row estimation sample.
    m1, d1 = fit_absorbed(
        panel,
        "M1_MODEL_A",
        ["corp_code", "year"],
        "Firm FE + Year FE",
    )
    locked_index = d1["work"].index
    m4, d4 = fit_trend(panel, "M4_MODEL_B", locked_index)

    prior = pd.read_csv(PRIOR_RESULTS)
    prior_base = prior.loc[prior["model_id"].eq("BASELINE_REPLICATION")].iloc[0]
    prior_trend = prior.loc[prior["model_id"].eq("FIRM_LINEAR_TRENDS")].iloc[0]
    replication = {
        "model_A_coefficient_difference_vs_prior": float(m1["coefficient"] - prior_base["coefficient"]),
        "model_A_SE_difference_vs_prior": float(m1["std_error"] - prior_base["std_error"]),
        "model_B_coefficient_difference_vs_prior": float(m4["coefficient"] - prior_trend["coefficient"]),
        "model_B_SE_difference_vs_prior": float(m4["std_error"] - prior_trend["std_error"]),
        "model_A_matches_prior_1e_10": bool(abs(m1["coefficient"] - prior_base["coefficient"]) < 1e-10),
        "model_B_matches_prior_1e_10": bool(abs(m4["coefficient"] - prior_trend["coefficient"]) < 1e-10),
    }

    model_comparison = pd.DataFrame([m1, m4])
    model_comparison = add_comparison_fields(model_comparison, int(m1["N"]), str(m1["sample_index_sha256"]))
    model_comparison["coefficient_change_vs_A_pct"] = np.nan
    model_comparison.loc[model_comparison["model_id"].eq("M4_MODEL_B"), "coefficient_change_vs_A_pct"] = (
        (m4["coefficient"] - m1["coefficient"]) / m1["coefficient"] * 100
    )
    model_comparison["SE_change_vs_A_pct"] = np.nan
    model_comparison.loc[model_comparison["model_id"].eq("M4_MODEL_B"), "SE_change_vs_A_pct"] = (
        (m4["std_error"] - m1["std_error"]) / m1["std_error"] * 100
    )
    model_comparison.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")

    m2, d2 = fit_absorbed(
        panel,
        "M2_INDUSTRY_YEAR",
        ["corp_code", "industry_year"],
        "Firm FE + Industry×Year FE",
    )
    m3, d3 = fit_absorbed(
        panel,
        "M3_MARKET_YEAR",
        ["corp_code", "market_year"],
        "Firm FE + Market×Year FE",
    )
    trend_comparison = add_comparison_fields(
        pd.DataFrame([m1, m2, m3, m4]), int(m1["N"]), str(m1["sample_index_sha256"])
    )
    trend_comparison.to_csv(OUT / "model_trend_comparison.csv", index=False, encoding="utf-8-sig")

    common_initial = locked_index.intersection(d2["work"].index).intersection(d3["work"].index)
    common_index = enforce_common_sample(panel, common_initial)
    cm1, cd1 = fit_absorbed(
        panel, "M1_COMMON", ["corp_code", "year"], "Firm FE + Year FE", common_index
    )
    cm2, cd2 = fit_absorbed(
        panel,
        "M2_COMMON",
        ["corp_code", "industry_year"],
        "Firm FE + Industry×Year FE",
        common_index,
    )
    cm3, cd3 = fit_absorbed(
        panel,
        "M3_COMMON",
        ["corp_code", "market_year"],
        "Firm FE + Market×Year FE",
        common_index,
    )
    cm4, cd4 = fit_trend(panel, "M4_COMMON", common_index)
    common = pd.DataFrame([cm1, cm2, cm3, cm4])
    common["common_sample"] = True
    common["sample_difference_N_vs_full_M1"] = common["N"] - int(m1["N"])
    common["all_sample_hashes_equal"] = common["sample_index_sha256"].nunique() == 1
    common.to_csv(OUT / "common_sample_model_comparison.csv", index=False, encoding="utf-8-sig")

    parameter_checks = parameterization_check(d4["work"], d4["transformed"])
    parameter_checks.to_csv(HERE / "trend_parameterization_checks.csv", index=False, encoding="utf-8-sig")
    direct_check = direct_augmented_coefficient(d4)
    (HERE / "direct_augmented_validation.json").write_text(
        json.dumps(direct_check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (HERE / "trend_absorption_solver.json").write_text(
        json.dumps(d4["absorption"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    influence = influential_firms(d4)
    influence.to_csv(OUT / "influential_firm_diagnostics.csv", index=False, encoding="utf-8-sig")
    influence_summary = {
        "firms_checked": int(len(influence)),
        "minimum_fixed_absorption_deleted_coefficient": float(influence["fixed_absorption_deleted_coefficient"].min()),
        "maximum_fixed_absorption_deleted_coefficient": float(influence["fixed_absorption_deleted_coefficient"].max()),
        "maximum_absolute_change": float(influence["absolute_coefficient_change"].max()),
        "maximum_absolute_change_in_full_sample_SE_units": float(
            influence["change_in_full_sample_SE_units"].abs().max()
        ),
        "firms_above_half_SE_change": int(
            influence["change_in_full_sample_SE_units"].abs().gt(0.5).sum()
        ),
        "top_one_share_of_total_absolute_change": float(
            influence["absolute_coefficient_change"].iloc[:1].sum()
            / influence["absolute_coefficient_change"].sum()
        ),
        "top_ten_share_of_total_absolute_change": float(
            influence["absolute_coefficient_change"].iloc[:10].sum()
            / influence["absolute_coefficient_change"].sum()
        ),
        "interpretation_limit": (
            "Deletion holds the full-sample fixed-effect/trend projection fixed; "
            "it is an influence diagnostic, not an alternative reported regression."
        ),
    }
    (HERE / "influence_summary.json").write_text(
        json.dumps(influence_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    after_hashes = {str(path): sha256(path) for path in input_paths if path.exists()}
    integrity = {
        "before": before_hashes,
        "after": after_hashes,
        "all_read_only_inputs_unchanged": before_hashes == after_hashes,
    }
    (HERE / "input_integrity_receipt.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "python": sys.version,
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "duplicate_firm_year_keys": int(panel.duplicated(["corp_code", "year"]).sum()),
        "variables": {"outcome": OUTCOME, "focal": FOCAL, "controls": CONTROLS},
        "replication": replication,
        "model_A": m1,
        "model_B": m4,
        "model_specific_samples": {x["model_id"]: {"N": x["N"], "firms": x["firms"]} for x in [m1, m2, m3, m4]},
        "common_sample": {
            "N": int(len(common_index)),
            "firms": int(panel.loc[common_index, "corp_code"].nunique()),
            "sample_index_sha256": index_hash(common_index),
        },
        "direct_validation": direct_check,
        "influence_summary": influence_summary,
        "design_diagnostics": {
            "condition_number_residualized_X": d4["condition_number_X"],
            "rank_residualized_X": d4["rank_X"],
            "regressor_count": len(d4["names"]),
        },
        "all_models_run_and_reported": [
            "M1 firm+year FE",
            "M2 firm+industry-year FE",
            "M3 firm+market-year FE",
            "M4 firm+year FE+firm linear trends",
            "M1-M4 on forced common sample",
            "M4 parameterization equivalence checks",
            "M4 direct augmented-design validation",
            "M4 fixed-absorption cluster-deletion diagnostics",
        ],
    }
    (HERE / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
