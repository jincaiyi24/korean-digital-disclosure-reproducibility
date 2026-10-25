from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.sandwich_covariance import cov_cluster
from statsmodels.stats.multitest import multipletests


sys.stdout.reconfigure(encoding="utf-8")

PANEL = Path(
    r"C:\Users\25294\Desktop\论文2\01_新增控制变量数据"
    r"\03_加入新增控制变量后的研究面板.csv"
)
OUT = Path(r"C:\Users\25294\Desktop\研究\09_二轮终审证据")
MARKET_FILE = Path(
    r"C:\Users\25294\Desktop\研究\04_模型结果与数据审计"
    r"\17_Python独立复算_资本市场模型.csv"
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

HIGH_COVERAGE_CONTROLS = [
    "IntangibleAssetRatio_w",
    "PPE_Ratio_w",
    "OperatingCashFlowRatio_w",
    "CapexIntensity_w",
]


class TrainWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features at training-sample quantiles only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def zscore(series: pd.Series, mask: pd.Series | None = None) -> pd.Series:
    values = numeric(series)
    source = values.loc[mask] if mask is not None else values
    sd = source.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=series.index)
    return (values - source.mean()) / sd


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


def weighted_residualize(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    effects: list[str],
    weights: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 10000,
) -> np.ndarray:
    """Absorb additive fixed effects by weighted alternating projections."""
    residual = matrix.copy()
    w = weights.reshape(-1)
    group_indices = [frame.groupby(effect, sort=False).indices.values() for effect in effects]
    for _ in range(max_iterations):
        previous = residual.copy()
        for groups in group_indices:
            for idx in groups:
                group_w = w[idx]
                denom = group_w.sum()
                if denom > 0:
                    residual[idx] -= np.average(residual[idx], axis=0, weights=group_w)
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual
    raise RuntimeError("Weighted fixed-effect residualization did not converge")


def fit_fwl(
    panel: pd.DataFrame,
    model_id: str,
    outcome: str,
    focals: list[str],
    controls: list[str],
    effects: list[str],
    sample: pd.Series | None = None,
    weights: str | None = None,
    all_terms: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    controls = list(dict.fromkeys(x for x in controls if x not in focals))
    required = list(dict.fromkeys([outcome, *focals, *controls, *effects, "corp_code", *( [weights] if weights else [])]))
    mask = sample if sample is not None else pd.Series(True, index=panel.index)
    work = panel.loc[mask, required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = drop_singletons(work, effects)
    names = [*focals, *controls]
    matrix = work[[outcome, *names]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    codes = np.column_stack([pd.Categorical(work[e].astype(str)).codes for e in effects])
    w = numeric(work[weights]).to_numpy().reshape(-1, 1) if weights else None
    if weights:
        transformed = weighted_residualize(matrix, work, effects, w)
    else:
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
    if weights:
        fitted = sm.WLS(y, x, weights=numeric(work[weights]).to_numpy()).fit()
    else:
        fitted = sm.OLS(y, x).fit()
    firm_codes = pd.Categorical(work["corp_code"].astype(str)).codes
    covariance = cov_cluster(fitted, firm_codes, use_correction=True)
    dof = max(1, work["corp_code"].nunique() - 1)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    pvals = 2 * stats.t.sf(np.abs(fitted.params / se), df=dof)
    critical = stats.t.ppf(0.975, dof)
    tss = float(np.sum((y - np.average(y, weights=numeric(work[weights]) if weights else None)) ** 2))
    rss = float(np.sum((fitted.resid**2) * (numeric(work[weights]).to_numpy() if weights else 1)))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    output_terms = selected_names if all_terms else focals
    rows: list[dict[str, object]] = []
    for term in output_terms:
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
                "years": int(work[effects[-1]].nunique()),
                "within_R2": r2,
                "effects": "+".join(effects),
                "cluster": "corp_code",
                "weighted": bool(weights),
            }
        )
    details = {
        "work": work,
        "names": selected_names,
        "params": np.asarray(fitted.params),
        "cov": np.asarray(covariance),
        "dof": dof,
    }
    return rows, details


def contrast(details: dict[str, object], left: str, right: str) -> dict[str, float]:
    names = details["names"]
    if left not in names or right not in names:
        raise ValueError("Contrast terms were dropped")
    c = np.zeros(len(names))
    c[names.index(left)] = 1
    c[names.index(right)] = -1
    params = details["params"]
    cov = details["cov"]
    estimate = float(c @ params)
    se = float(np.sqrt(c @ cov @ c))
    t = estimate / se
    dof = int(details["dof"])
    p = float(2 * stats.t.sf(abs(t), dof))
    crit = float(stats.t.ppf(0.975, dof))
    return {
        "contrast": f"{left} - {right}",
        "difference": estimate,
        "std_error": se,
        "t_stat": t,
        "wald_F": t**2,
        "p_value": p,
        "ci95_low": estimate - crit * se,
        "ci95_high": estimate + crit * se,
        "df_denom": dof,
    }


def load_panel() -> pd.DataFrame:
    d = pd.read_csv(PANEL, dtype={"corp_code": str, "stock_code": str}, low_memory=False)
    d["corp_code"] = d["corp_code"].str.zfill(8)
    d["year"] = numeric(d["year"]).astype(int)
    d = d.sort_values(["corp_code", "year"]).reset_index(drop=True)

    # The expanded OpenDART pull contains a small number of company-defined
    # tags whose labels do not match the intended accounting concept. Keep the
    # sensitivity test on a strict, comparable set of standard tags instead of
    # letting winsorisation hide account-mapping errors.
    account_rules = {
        "PPE": {
            "ids": {"ifrs-full_propertyplantandequipment", "ifrs_propertyplantandequipment"},
            "name": lambda value: "유형자산" in value and "사용권" not in value,
            "nonnegative": True,
            "ratio": "PPE_Ratio",
        },
        "IntangibleAssets": {
            "ids": {"ifrs-full_intangibleassetsotherthangoodwill", "ifrs_intangibleassetsotherthangoodwill"},
            "name": lambda value: "무형자산" in value,
            "nonnegative": True,
            "ratio": "IntangibleAssetRatio",
        },
        "CAPEX": {
            "ids": {
                "ifrs-full_purchaseofpropertyplantandequipmentclassifiedasinvestingactivities",
                "ifrs_purchaseofpropertyplantandequipmentclassifiedasinvestingactivities",
            },
            "name": lambda value: value.replace(" ", "") in {"유형자산의취득", "유형자산취득"},
            "nonnegative": True,
            "ratio": "CapexIntensity",
        },
        "OperatingCashFlow": {
            "ids": {"ifrs-full_cashflowsfromusedinoperatingactivities", "ifrs_cashflowsfromusedinoperatingactivities"},
            "name": lambda value: "영업활동" in value and "현금흐름" in value.replace(" ", ""),
            "nonnegative": False,
            "ratio": "OperatingCashFlowRatio",
        },
    }
    for amount, rule in account_rules.items():
        account_id = d[f"{amount}_account_id"].fillna("").astype(str).str.lower()
        account_name = d[f"{amount}_account_name"].fillna("").astype(str)
        valid = account_id.isin(rule["ids"]) & account_name.map(rule["name"])
        if rule["nonnegative"]:
            valid &= numeric(d[amount]).ge(0)
        ratio = rule["ratio"]
        d.loc[~valid, ratio] = np.nan
        values = numeric(d[ratio]).replace([np.inf, -np.inf], np.nan)
        available = values.dropna()
        lo, hi = available.quantile([0.01, 0.99]) if not available.empty else (np.nan, np.nan)
        d[f"{ratio}_w"] = values.clip(lo, hi)
    total = numeric(d["total_sentences"])
    bmi = numeric(d["bmi_sentence_count"])
    substantive = numeric(d["substantive_digital_sentence_count"])
    d["BMI_raw"] = bmi / total.where(total.gt(0))
    d["NonBMI_raw"] = (substantive - bmi).clip(lower=0) / total.where(total.gt(0))
    d["TotalDigital_raw"] = substantive / total.where(total.gt(0))
    d["BMI1000"] = 1000 * d["BMI_raw"]
    d["NonBMI1000"] = 1000 * d["NonBMI_raw"]
    d["BMI_z"] = zscore(d["BMI_raw"])
    d["NonBMI_z"] = zscore(d["NonBMI_raw"])
    d["TotalDigital_z"] = zscore(d["TotalDigital_raw"])
    d["BMI_any"] = bmi.gt(0).astype(float)
    d["BMI_share_substantive_0"] = (bmi / substantive.where(substantive.gt(0))).fillna(0)
    d["BMI_share_substantive_0_z"] = zscore(d["BMI_share_substantive_0"])
    d["BMI_share_positive"] = bmi / substantive.where(substantive.gt(0))
    positive = d["BMI_raw"].gt(0)
    d["BMI_positive_intensity_z"] = zscore(d["BMI_raw"], positive).where(positive)
    d["BMI_log_count"] = np.log1p(bmi)
    d["BMI_log_count_z"] = zscore(d["BMI_log_count"])
    d["industry_group"] = d["industry_group"].fillna("UNKNOWN").astype(str)
    d["industry_year"] = d["industry_group"] + "_" + d["year"].astype(str)
    d["market_year"] = d["market"].astype(str) + "_" + d["year"].astype(str)
    for h in (-2, -1, 0, 1, 2, 3):
        d[f"AT_h{h}"] = d["AssetTurnover_w"] if h == 0 else exact_shift(d, "AssetTurnover_w", h)
    for h in (1, 2, 3):
        d[f"LogSales_h{h}"] = exact_shift(d, "FirmSize_w", h) * np.nan
        # LogSales is derived directly from positive sales and kept un-winsorized here;
        # outcome tails are clipped below on a common 1/99 basis.
        log_sales = np.log(numeric(d["Sales"]).where(numeric(d["Sales"]).gt(0)))
        lo, hi = log_sales.quantile([0.01, 0.99])
        d["LogSales_current_w"] = log_sales.clip(lo, hi)
        d[f"LogSales_h{h}"] = exact_shift(d, "LogSales_current_w", h)
        d[f"LogAssets_h{h}"] = exact_shift(d, "FirmSize_w", h)
        d[f"ROA_h{h}"] = exact_shift(d, "ROA_AvgAssets_w", h)
        d[f"Margin_h{h}"] = exact_shift(d, "OperatingMargin_w", h)
    d["AT_change_h2"] = d["AT_h2"] - d["AssetTurnover_w"]
    d["Sales_change_h2"] = d["LogSales_h2"] - d["LogSales_current_w"]
    d["Assets_change_h2"] = d["LogAssets_h2"] - d["FirmSize_w"]

    # Raw features for leakage-controlled rolling prediction. All transformations
    # of these columns are fitted inside each training window below.
    d["OOS_AssetTurnover_raw"] = numeric(d["AssetTurnover"])
    d["OOS_AT_h2_raw"] = exact_shift(d, "AssetTurnover", 2)
    d["OOS_LogAssets_raw"] = np.log(numeric(d["TotalAssets"]).where(numeric(d["TotalAssets"]).gt(0)))
    d["OOS_LogSales_raw"] = np.log(numeric(d["Sales"]).where(numeric(d["Sales"]).gt(0)))
    lag_sales = exact_shift(d, "Sales", -1)
    d["OOS_SalesGrowth_raw"] = numeric(d["Sales"]) / numeric(lag_sales).where(numeric(lag_sales).gt(0)) - 1
    d["OOS_CurrentAssetRatio_raw"] = numeric(d["CurrentAssetRatio"])
    return d


def descriptive_outputs(d: pd.DataFrame) -> None:
    variables = [
        "AssetTurnover_w", "AT_h2", "BMI_raw", "NonBMI_raw", "TotalDigital_raw",
        "BMI_share_substantive_0", "FirmSize_w", "Leverage_w", "ROA_AvgAssets_w",
        "SalesGrowth_w", "Liquidity_w", "OperatingMargin_w", "FirmAgeLog_w",
        "LossDummy", "NegativeEquityDummy", "LogTotalSentences",
        "FiscalYearStockReturn_w", "FiscalYearDailyVolatility_w",
        *HIGH_COVERAGE_CONTROLS,
    ]
    rows = []
    for v in variables:
        x = numeric(d[v]).dropna()
        rows.append({
            "variable": v, "N": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
            "p10": x.quantile(.10), "p25": x.quantile(.25), "median": x.median(),
            "p75": x.quantile(.75), "p90": x.quantile(.90), "p95": x.quantile(.95),
            "skewness": x.skew(), "zero_share": (x == 0).mean(),
        })
    pd.DataFrame(rows).to_csv(OUT / "01_描述性统计.csv", index=False, encoding="utf-8-sig")

    yearly = d.groupby("year").agg(
        observations=("corp_code", "size"),
        firms=("corp_code", "nunique"),
        mean_bmi=("BMI_raw", "mean"),
        mean_nonbmi=("NonBMI_raw", "mean"),
        mean_total=("TotalDigital_raw", "mean"),
        bmi_positive_share=("BMI_any", "mean"),
    ).reset_index()
    yearly.to_csv(OUT / "02_BMI年度趋势.csv", index=False, encoding="utf-8-sig")

    industry = d.groupby("industry_group").agg(
        N=("corp_code", "size"), firms=("corp_code", "nunique"),
        mean_bmi=("BMI_raw", "mean"), median_bmi=("BMI_raw", "median"),
        positive_share=("BMI_any", "mean"),
    ).reset_index().sort_values("N", ascending=False)
    industry.to_csv(OUT / "03_BMI行业分布.csv", index=False, encoding="utf-8-sig")

    firm_mean = d.groupby("corp_code")["BMI_raw"].transform("mean")
    overall_mean = d["BMI_raw"].mean()
    between_component = np.sqrt(np.sum((firm_mean - overall_mean) ** 2) / (d["BMI_raw"].notna().sum() - 1))
    within_component = np.sqrt(np.sum((d["BMI_raw"] - firm_mean) ** 2) / (d["BMI_raw"].notna().sum() - 1))
    total_ss = np.sum((d["BMI_raw"] - overall_mean) ** 2)
    between_ss = np.sum((firm_mean - overall_mean) ** 2)
    lag_bmi = d.groupby("corp_code")["BMI_raw"].shift(1)
    lag_year = d.groupby("corp_code")["year"].shift(1)
    adjacent = lag_year.eq(d["year"] - 1)
    transitions = pd.crosstab(
        (lag_bmi.loc[adjacent] > 0).map({False: "zero", True: "positive"}),
        (d.loc[adjacent, "BMI_raw"] > 0).map({False: "zero", True: "positive"}),
    )
    persistence = d.loc[adjacent, ["BMI_raw"]].assign(lag=lag_bmi.loc[adjacent]).corr().iloc[0, 1]
    pd.DataFrame([{
        "overall_sd": d["BMI_raw"].std(ddof=1), "between_component_sd": between_component,
        "within_component_sd": within_component, "between_variance_share": between_ss / total_ss,
        "adjacent_year_correlation": persistence,
        "zero_to_positive": int(transitions.get("positive", pd.Series()).get("zero", 0)),
        "positive_to_zero": int(transitions.get("zero", pd.Series()).get("positive", 0)),
        "positive_to_positive": int(transitions.get("positive", pd.Series()).get("positive", 0)),
        "zero_to_zero": int(transitions.get("zero", pd.Series()).get("zero", 0)),
    }]).to_csv(OUT / "04_BMI企业内变化与持续性.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for col, label in [("mean_total", "Total substantive digital"), ("mean_bmi", "BMI"), ("mean_nonbmi", "Non-BMI")]:
        ax.plot(yearly["year"], yearly[col] * 1000, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Report year")
    ax.set_ylabel("Sentences per 1,000 report sentences")
    ax.set_title("Digital disclosure composition by year")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "图1_BMI与NonBMI年度趋势.png", dpi=220)
    plt.close(fig)


def sample_and_measurement_audit(d: pd.DataFrame) -> pd.DataFrame:
    _, details = fit_fwl(
        d,
        "AUDIT_MAIN_SAMPLE",
        "AT_h2",
        ["BMI_z"],
        BASE_CONTROLS,
        ["corp_code", "year"],
    )
    main_index = details["work"].index
    samples = {
        "frozen_full_panel": d,
        "main_regression_sample": d.loc[main_index],
    }
    rows = []
    for sample_name, frame in samples.items():
        bmi = numeric(frame["BMI_raw"])
        rows.append(
            {
                "sample_name": sample_name,
                "N": int(len(frame)),
                "firm_count": int(frame["corp_code"].nunique()),
                "year_range": f"{int(frame['year'].min())}-{int(frame['year'].max())}",
                "BMI_mean": bmi.mean(),
                "BMI_sd": bmi.std(ddof=1),
                "BMI_p25": bmi.quantile(0.25),
                "BMI_median": bmi.median(),
                "BMI_p75": bmi.quantile(0.75),
                "BMI_zero_share": bmi.eq(0).mean(),
                "BMI_positive_share": bmi.gt(0).mean(),
                "NonBMI_mean": numeric(frame["NonBMI_raw"]).mean(),
                "TotalDigital_mean": numeric(frame["TotalDigital_raw"]).mean(),
                "BMIShare_mean": numeric(frame["BMI_share_substantive_0"]).mean(),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "13_样本统计口径审计.csv", index=False, encoding="utf-8-sig")

    method_text = """# 文本测量过程审计

本研究采用基于领域规则的句子级分类，并由作者对候选句和边界案例进行多轮复核。

1. 清理年报文本并切分句子。
2. 识别包含数字技术或数字载体的候选句。
3. 仅保留同时指向公司自身行动和可识别经营用途的句子。
4. 排除行业趋势、口号、未来愿景、目录、代码、会计项目以及没有明确行动主体的表达。
5. 将平台、线上渠道、订阅或经常性收入、软件或数据服务、生态系统、智能或连接型产品等与价值创造、交付或获取直接相关的行动归为 BMI；将系统升级、基础设施、设备自动化和一般内部技术应用归为 NonBMI。
6. 作者依据同一套固定规则反复复核候选句与边界案例，并以最终标签构造公司年度指标。

本项目未设置第二名独立编码者，因此不报告 Cohen's kappa、编码者间一致率、准确率、召回率或 F1。早期流程没有完整保存每次复核前后的标签版本，因而也不追溯或虚构正式混淆矩阵。该限制影响构念效度的可验证程度，但不改变规则系统的可复现性。
"""
    (OUT / "14_文本测量过程说明.md").write_text(method_text, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "review_process": "deterministic rules plus repeated single-researcher audit",
                "second_independent_coder": False,
                "intercoder_reliability_reported": False,
                "historical_label_revision_log_available": False,
                "fabricated_statistics": False,
            }
        ]
    ).to_csv(OUT / "15_人工复核状态.csv", index=False, encoding="utf-8-sig")
    return audit


def run_models(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []

    def add(*args, **kwargs):
        result, details = fit_fwl(d, *args, **kwargs)
        rows.extend(result)
        return details

    add("M1_PRIMARY", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "year"], all_terms=True)
    detail = add("M2_BMI_NONBMI", "AT_h2", ["BMI_z", "NonBMI_z"], BASE_CONTROLS, ["corp_code", "year"])
    contrasts.append({"model_id": "M2_BMI_NONBMI", **contrast(detail, "BMI_z", "NonBMI_z")})
    raw_detail = add("M2B_RAW_SCALE_BMI_NONBMI", "AT_h2", ["BMI1000", "NonBMI1000"], BASE_CONTROLS, ["corp_code", "year"])
    contrasts.append({"model_id": "M2B_RAW_SCALE_BMI_NONBMI", **contrast(raw_detail, "BMI1000", "NonBMI1000")})
    add("M3_TOTAL_ONLY", "AT_h2", ["TotalDigital_z"], BASE_CONTROLS, ["corp_code", "year"])
    add("M4_TOTAL_PLUS_COMPOSITION", "AT_h2", ["TotalDigital_z", "BMI_share_substantive_0_z"], BASE_CONTROLS, ["corp_code", "year"])
    add(
        "M4B_POSITIVE_TOTAL_SHARE",
        "AT_h2",
        ["TotalDigital_z", "BMI_share_positive"],
        BASE_CONTROLS,
        ["corp_code", "year"],
        sample=d["TotalDigital_raw"].gt(0),
    )
    add("M5_INDUSTRY_YEAR_FE", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "industry_year"])
    add("M6_MARKET_YEAR_FE", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "market_year"])
    add("M7_CHANGE_OUTCOME", "AT_change_h2", ["BMI_z"], BASE_CONTROLS[1:], ["corp_code", "year"])

    for h in (-2, -1, 0, 1, 2, 3):
        controls = BASE_CONTROLS if h != 0 else BASE_CONTROLS[1:]
        add(f"TIMING_H{h}", f"AT_h{h}", ["BMI_z"], controls, ["corp_code", "year"])

    common_required = [f"AT_h{h}" for h in (-2, -1, 0, 1, 2, 3)]
    common_required += ["BMI_z", *BASE_CONTROLS[1:], "corp_code", "year"]
    common_sample = d[common_required].notna().all(axis=1)
    for h in (-2, -1, 0, 1, 2, 3):
        add(
            f"COMMON_TIMING_H{h}",
            f"AT_h{h}",
            ["BMI_z"],
            BASE_CONTROLS[1:],
            ["corp_code", "year"],
            sample=common_sample,
        )

    for h in (1, 2, 3):
        add(f"SALES_LEVEL_H{h}", f"LogSales_h{h}", ["BMI_z"], ["LogSales_current_w", *BASE_CONTROLS[1:]], ["corp_code", "year"])
        add(f"ASSETS_LEVEL_H{h}", f"LogAssets_h{h}", ["BMI_z"], BASE_CONTROLS[1:], ["corp_code", "year"])
        add(f"ROA_H{h}", f"ROA_h{h}", ["BMI_z"], BASE_CONTROLS[1:], ["corp_code", "year"])
        add(f"MARGIN_H{h}", f"Margin_h{h}", ["BMI_z"], BASE_CONTROLS[1:], ["corp_code", "year"])

    add("DECOMP_SALES_CHANGE_H2", "Sales_change_h2", ["BMI_z"], BASE_CONTROLS[1:], ["corp_code", "year"])
    add("DECOMP_ASSETS_CHANGE_H2", "Assets_change_h2", ["BMI_z"], BASE_CONTROLS[1:], ["corp_code", "year"])
    add("ALT_LOG_COUNT", "AT_h2", ["BMI_log_count_z"], [*BASE_CONTROLS, "LogTotalSentences"], ["corp_code", "year"])
    add("ALT_ANY", "AT_h2", ["BMI_any"], BASE_CONTROLS, ["corp_code", "year"])
    add("ALT_SHARE_OF_SUBSTANTIVE", "AT_h2", ["BMI_share_substantive_0_z"], BASE_CONTROLS, ["corp_code", "year"])
    add("INTENSIVE_POSITIVE_ONLY", "AT_h2", ["BMI_positive_intensity_z"], BASE_CONTROLS, ["corp_code", "year"], sample=d["BMI_any"].eq(1))

    high_sample = d[["AT_h2", "BMI_z", *BASE_CONTROLS, *HIGH_COVERAGE_CONTROLS, "corp_code", "year"]].notna().all(axis=1)
    add("SAME_SAMPLE_BASE_STRICT_ACCOUNTS", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "year"], sample=high_sample)
    add("SAME_SAMPLE_PLUS_HIGH_COVERAGE", "AT_h2", ["BMI_z"], [*BASE_CONTROLS, *HIGH_COVERAGE_CONTROLS], ["corp_code", "year"], sample=high_sample)

    rd_sample = d[["AT_h2", "BMI_z", *BASE_CONTROLS, "RAndDIntensity_OpenDART_w", "corp_code", "year"]].notna().all(axis=1)
    add("SAME_SAMPLE_RD_BASE", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "year"], sample=rd_sample)
    add("SAME_SAMPLE_RD_PLUS", "AT_h2", ["BMI_z"], [*BASE_CONTROLS, "RAndDIntensity_OpenDART_w"], ["corp_code", "year"], sample=rd_sample)

    models = pd.DataFrame(rows)
    focal_terms = {
        "BMI_z", "NonBMI_z", "TotalDigital_z", "BMI_share_substantive_0_z",
        "BMI_log_count_z", "BMI_any", "BMI_positive_intensity_z", "BMI1000",
        "NonBMI1000", "BMI_share_positive",
    }
    models["is_focal"] = models["term"].isin(focal_terms)
    models["q_BH_revision_focals"] = np.nan
    focal_mask = models["is_focal"]
    models.loc[focal_mask, "q_BH_revision_focals"] = multipletests(
        models.loc[focal_mask, "p_value"].to_numpy(), method="fdr_bh"
    )[1]
    tests = pd.DataFrame(contrasts)
    models.to_csv(OUT / "05_核心与补充模型完整结果.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUT / "06_BMI与NonBMI系数差异Wald检验.csv", index=False, encoding="utf-8-sig")

    timing = models.loc[models["model_id"].str.startswith("TIMING_") & models["term"].eq("BMI_z")].copy()
    timing["horizon"] = timing["model_id"].str.extract(r"H(-?\d+)").astype(int)
    timing = timing.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(timing["horizon"], timing["coefficient"], yerr=1.96 * timing["std_error"], fmt="o-", capsize=4)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(timing["horizon"])
    ax.set_xlabel("Outcome horizon relative to report year")
    ax.set_ylabel("BMI coefficient (95% CI)")
    ax.set_title("Timing pattern of BMI disclosure and asset turnover")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "图2_BMI前后时点系数.png", dpi=220)
    plt.close(fig)

    common_timing = models.loc[
        models["model_id"].str.startswith("COMMON_TIMING_") & models["term"].eq("BMI_z")
    ].copy()
    common_timing["horizon"] = common_timing["model_id"].str.extract(r"H(-?\d+)").astype(int)
    common_timing = common_timing.sort_values("horizon")
    common_timing.to_csv(OUT / "16_共同样本时点检验.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        common_timing["horizon"],
        common_timing["coefficient"],
        yerr=1.96 * common_timing["std_error"],
        fmt="o-",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(common_timing["horizon"])
    ax.set_xlabel("Outcome horizon relative to report year")
    ax.set_ylabel("BMI coefficient (95% CI)")
    ax.set_title("Common-sample timing pattern")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "图4_共同样本前后时点系数.png", dpi=220)
    plt.close(fig)
    return models, tests


def attrition_ipw(d: pd.DataFrame) -> pd.DataFrame:
    predictors = ["BMI_z", *BASE_CONTROLS]
    latest_outcome_year = int(d["year"].max())
    eligible = d[predictors + ["corp_code", "year", "industry_group"]].notna().all(axis=1)
    eligible &= d["year"].le(latest_outcome_year - 2)
    a = d.loc[eligible].copy()
    a["ObservedT2"] = a["AT_h2"].notna().astype(int)

    lpm_rows, _ = fit_fwl(a, "ATTRITION_LPM", "ObservedT2", ["BMI_z"], BASE_CONTROLS, ["industry_group", "year"])

    balance_rows = []
    for v in ["BMI_z", "BMI_raw", *BASE_CONTROLS]:
        inc = numeric(a.loc[a["ObservedT2"].eq(1), v]).dropna()
        exc = numeric(a.loc[a["ObservedT2"].eq(0), v]).dropna()
        pooled = math.sqrt((inc.var(ddof=1) + exc.var(ddof=1)) / 2)
        balance_rows.append({
            "variable": v, "included_mean": inc.mean(), "excluded_mean": exc.mean(),
            "standardized_difference": (inc.mean() - exc.mean()) / pooled if pooled > 0 else np.nan,
            "N_included": len(inc), "N_excluded": len(exc),
        })
    pd.DataFrame(balance_rows).to_csv(OUT / "07_t2可观察样本平衡诊断.csv", index=False, encoding="utf-8-sig")

    num = predictors
    cat = ["year", "industry_group"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
    ])
    logit = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=3000, C=1e6, solver="lbfgs"))])
    logit.fit(a[num + cat], a["ObservedT2"])
    ps = logit.predict_proba(a[num + cat])[:, 1]
    a["propensity"] = np.clip(ps, .01, .99)
    observed_rate = a["ObservedT2"].mean()
    a["ipw_raw"] = np.where(a["ObservedT2"].eq(1), 1 / a["propensity"], np.nan)
    a["ipw"] = np.where(a["ObservedT2"].eq(1), observed_rate / a["propensity"], np.nan)
    observed = a["ObservedT2"].eq(1)
    lo, hi = a.loc[observed, "ipw"].quantile([.01, .99])
    a["ipw_trimmed"] = a["ipw"].clip(lo, hi)
    ipw_rows, _ = fit_fwl(a, "ATTRITION_IPW", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "year"], sample=observed, weights="ipw_trimmed")
    all_rows = pd.DataFrame([*lpm_rows, *ipw_rows])
    all_rows["observed_share"] = a["ObservedT2"].mean()
    all_rows["ipw_p01"] = lo
    all_rows["ipw_p99"] = hi
    all_rows.to_csv(OUT / "08_样本留存LPM与IPW结果.csv", index=False, encoding="utf-8-sig")

    w = a.loc[observed, "ipw_trimmed"]
    diagnostics = []
    for label, values in [
        ("propensity_observed", a.loc[observed, "propensity"]),
        ("propensity_missing", a.loc[~observed, "propensity"]),
        ("raw_ipw_observed", a.loc[observed, "ipw_raw"]),
        ("stabilized_ipw_observed", a.loc[observed, "ipw"]),
        ("trimmed_stabilized_ipw_observed", w),
    ]:
        diagnostics.append(
            {
                "quantity": label,
                "N": int(values.notna().sum()),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "p01": values.quantile(.01),
                "p05": values.quantile(.05),
                "median": values.median(),
                "p95": values.quantile(.95),
                "p99": values.quantile(.99),
                "min": values.min(),
                "max": values.max(),
            }
        )
    diagnostics.append(
        {
            "quantity": "effective_sample_size_trimmed_observed",
            "N": int(observed.sum()),
            "mean": (w.sum() ** 2) / np.square(w).sum(),
        }
    )
    pd.DataFrame(diagnostics).to_csv(OUT / "17_IPW诊断.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(a.loc[observed, "propensity"], bins=35, alpha=.65, density=True, label="Observed t+2")
    axes[0].hist(a.loc[~observed, "propensity"], bins=35, alpha=.55, density=True, label="Missing t+2")
    axes[0].set_xlabel("Estimated observability probability")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    axes[1].hist(a.loc[observed, "ipw"], bins=35, alpha=.5, label="Stabilized")
    axes[1].hist(w, bins=35, alpha=.65, label="Trimmed stabilized")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Observations")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "图5_IPW重叠与权重分布.png", dpi=220)
    plt.close(fig)
    return all_rows


def oos_prediction(d: pd.DataFrame) -> pd.DataFrame:
    base = [
        "OOS_AssetTurnover_raw",
        "OOS_LogAssets_raw",
        "OOS_LogSales_raw",
        "OOS_SalesGrowth_raw",
        "OOS_CurrentAssetRatio_raw",
        "FirmAgeLog_w",
        "LossDummy",
        "NegativeEquityDummy",
        "LogTotalSentences",
    ]
    models = {
        "Financial": base,
        "Financial+TotalDigital": [*base, "TotalDigital_raw"],
        "Financial+BMI": [*base, "BMI_raw"],
    }
    required = list(dict.fromkeys(["OOS_AT_h2_raw", "year", "industry_group", "market", "corp_code", *sum(models.values(), [])]))
    work = d[required].dropna(subset=["OOS_AT_h2_raw", "year", "industry_group", "market", "corp_code"]).copy()
    test_years = sorted(y for y in work["year"].unique() if y >= 2020 and y <= 2023)
    predictions = []
    summary = []
    for test_year in test_years:
        # At the filing of report year t, the outcome attached to report t-1
        # (fiscal year t+1) is not yet known. Exclude the two intervening report
        # cohorts so every training label predates the test information set.
        common_train = work["year"].le(test_year - 3)
        common_test = work["year"].eq(test_year)
        if common_train.sum() < 100 or common_test.sum() < 30:
            continue
        y_train_raw = numeric(work.loc[common_train, "OOS_AT_h2_raw"])
        y_lo, y_hi = y_train_raw.quantile([.01, .99])
        y_train = y_train_raw.clip(y_lo, y_hi)
        actual = numeric(work.loc[common_test, "OOS_AT_h2_raw"]).clip(y_lo, y_hi).to_numpy()
        for name, continuous in models.items():
            pre = ColumnTransformer([
                (
                    "num",
                    Pipeline([
                        ("winsor", TrainWinsorizer()),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]),
                    continuous,
                ),
                (
                    "cat",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    ["industry_group", "market"],
                ),
            ])
            pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
            pipe.fit(work.loc[common_train, continuous + ["industry_group", "market"]], y_train)
            pred = pipe.predict(work.loc[common_test, continuous + ["industry_group", "market"]])
            firms = work.loc[common_test, "corp_code"].to_numpy()
            for firm, y, yh in zip(firms, actual, pred):
                predictions.append({"test_year": test_year, "model": name, "corp_code": firm, "actual": y, "prediction": yh, "squared_error": (y-yh)**2, "absolute_error": abs(y-yh)})
            summary.append({
                "test_year": test_year, "model": name, "N_test": len(actual),
                "RMSE": mean_squared_error(actual, pred) ** .5,
                "MAE": mean_absolute_error(actual, pred),
                "OOS_R2": r2_score(actual, pred),
                "correlation": np.corrcoef(actual, pred)[0, 1],
            })
    pred = pd.DataFrame(predictions)
    summ = pd.DataFrame(summary)
    pooled = pred.groupby("model").agg(N_test=("actual", "size"), MSE=("squared_error", "mean"), MAE=("absolute_error", "mean")).reset_index()
    pooled["RMSE"] = np.sqrt(pooled["MSE"])

    wide = pred.pivot_table(index=["test_year", "corp_code", "actual"], columns="model", values="squared_error").reset_index()
    rng = np.random.default_rng(20260904)
    boot_rows = []
    for comparator in ["Financial", "Financial+TotalDigital"]:
        x = wide.dropna(subset=[comparator, "Financial+BMI"]).copy()
        x["loss_difference"] = x["Financial+BMI"] - x[comparator]
        firm_loss = x.groupby("corp_code")["loss_difference"].agg(["sum", "count"])
        sums = firm_loss["sum"].to_numpy()
        counts = firm_loss["count"].to_numpy()
        sampled = rng.integers(0, len(firm_loss), size=(2000, len(firm_loss)))
        boots = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
        boot_rows.append({
            "comparison": f"Financial+BMI minus {comparator}",
            "mean_MSE_difference": x["loss_difference"].mean(),
            "ci95_low_firm_bootstrap": np.quantile(boots, .025),
            "ci95_high_firm_bootstrap": np.quantile(boots, .975),
            "firms": len(firm_loss),
        })
    summ.to_csv(OUT / "09_时间外预测逐年结果.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUT / "10_时间外预测汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boot_rows).to_csv(OUT / "11_时间外预测误差差异Bootstrap.csv", index=False, encoding="utf-8-sig")
    audit_text = """# OOS pipeline audit

- Split: expanding training window by report year; test years are 2020-2023. A two-cohort embargo is imposed, so the latest training report year is t-3 when report year t is tested.
- Outcome: raw t+2 asset turnover. Training-window 1st/99th percentile thresholds are applied to the training target and the corresponding test target.
- Predictors: raw asset turnover, total assets, sales, sales growth, current-asset ratio, firm age, loss and negative-equity indicators, report length, industry, and market. BMI and TotalDigital enter in raw sentence-ratio form.
- Preprocessing: winsorization bounds, median imputation, means, and standard deviations are fitted separately within each training window. No full-panel z score is passed to the prediction model.
- Firm effects: firm fixed effects are not used in OOS prediction. This avoids assigning unavailable firm intercepts to firms not observed in the training period. Industry and market indicators are used instead.
- Association versus prediction: the OOS model is intentionally different from the fixed-effects association model because its task is forecasting genuinely later observations without future information.
"""
    (OUT / "18_OOS流程审计.md").write_text(audit_text, encoding="utf-8")
    return pooled


def economic_magnitude(d: pd.DataFrame, models: pd.DataFrame) -> None:
    main = models.loc[(models["model_id"].eq("M1_PRIMARY")) & (models["term"].eq("BMI_z"))].iloc[0]
    _, details = fit_fwl(d, "MAGNITUDE_SAMPLE", "AT_h2", ["BMI_z"], BASE_CONTROLS, ["corp_code", "year"])
    sample = d.loc[details["work"].index].copy()
    y = sample["AT_h2"]
    bmi = sample["BMI_raw"]
    full_panel_bmi_sd = d["BMI_raw"].std(ddof=1)
    z_iqr = (bmi.quantile(.75) - bmi.quantile(.25)) / full_panel_bmi_sd
    iqr_effect = main["coefficient"] * z_iqr
    result = pd.DataFrame([{
        "beta_per_1sd_bmi": main["coefficient"],
        "standardization_sample": "frozen_full_panel",
        "standardization_bmi_sd_raw": full_panel_bmi_sd,
        "bmi_mean_raw": bmi.mean(), "bmi_sd_raw": bmi.std(ddof=1),
        "bmi_zero_share": bmi.eq(0).mean(), "bmi_p25": bmi.quantile(.25),
        "bmi_p75": bmi.quantile(.75), "bmi_p90": bmi.quantile(.90), "bmi_p95": bmi.quantile(.95),
        "outcome_mean": y.mean(), "outcome_sd": y.std(ddof=1),
        "effect_in_outcome_sd": main["coefficient"] / y.std(ddof=1),
        "z_iqr": z_iqr, "iqr_effect": iqr_effect,
        "iqr_effect_pct_of_outcome_mean": 100 * iqr_effect / y.mean(),
        "one_sd_bmi_sentences_per_1000": 1000 * full_panel_bmi_sd,
        "effect_per_main_sample_sd": main["coefficient"] * bmi.std(ddof=1) / full_panel_bmi_sd,
    }])
    result.to_csv(OUT / "12_经济量级.csv", index=False, encoding="utf-8-sig")


def multiple_testing_registry(models: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_model(test_id, family, hypothesis, model_id, term, sample, specification):
        row = models.loc[(models["model_id"].eq(model_id)) & (models["term"].eq(term))].iloc[0]
        rows.append(
            {
                "test_id": test_id,
                "analysis_family": family,
                "hypothesis": hypothesis,
                "outcome": row["outcome"],
                "predictor": term,
                "sample": sample,
                "specification": specification,
                "coefficient_or_difference": row["coefficient"],
                "raw_p": row["p_value"],
            }
        )

    add_model("H1", "confirmatory_core", "BMI disclosure is positively associated with t+2 asset turnover", "M1_PRIMARY", "BMI_z", "main regression sample", "firm FE + year FE + baseline controls")
    h2 = tests.loc[tests["model_id"].eq("M2_BMI_NONBMI")].iloc[0]
    rows.append(
        {
            "test_id": "H2",
            "analysis_family": "confirmatory_core",
            "hypothesis": "BMI and NonBMI coefficients differ",
            "outcome": "AT_h2",
            "predictor": "BMI_z - NonBMI_z",
            "sample": "main regression sample",
            "specification": "joint standardized model; firm FE + year FE + baseline controls",
            "coefficient_or_difference": h2["difference"],
            "raw_p": h2["p_value"],
        }
    )
    h2raw = tests.loc[tests["model_id"].eq("M2B_RAW_SCALE_BMI_NONBMI")].iloc[0]
    rows.append(
        {
            "test_id": "H2_raw_scale",
            "analysis_family": "measurement_sensitivity",
            "hypothesis": "BMI and NonBMI coefficients differ on a common per-1000-sentence scale",
            "outcome": "AT_h2",
            "predictor": "BMI1000 - NonBMI1000",
            "sample": "main regression sample",
            "specification": "joint raw-scale model; firm FE + year FE + baseline controls",
            "coefficient_or_difference": h2raw["difference"],
            "raw_p": h2raw["p_value"],
        }
    )

    composition = [
        ("C1", "M3_TOTAL_ONLY", "TotalDigital_z", "Total digital disclosure"),
        ("C2", "M4_TOTAL_PLUS_COMPOSITION", "TotalDigital_z", "Total digital disclosure conditional on BMI share"),
        ("C3", "M4_TOTAL_PLUS_COMPOSITION", "BMI_share_substantive_0_z", "BMI share conditional on total digital disclosure"),
        ("C4", "M4B_POSITIVE_TOTAL_SHARE", "TotalDigital_z", "Total digital disclosure in positive-disclosure observations"),
        ("C5", "M4B_POSITIVE_TOTAL_SHARE", "BMI_share_positive", "BMI share in positive-disclosure observations"),
    ]
    for test_id, model_id, term, hypothesis in composition:
        add_model(test_id, "composition", hypothesis, model_id, term, "model-specific complete cases", "firm FE + year FE + baseline controls")

    for h in (-2, -1, 0, 1, 2, 3):
        add_model(f"T{h:+d}", "timing", f"BMI coefficient at horizon {h:+d}", f"TIMING_H{h}", "BMI_z", "horizon-specific sample", "firm FE + year FE + controls")

    for h in (1, 2, 3):
        for prefix, outcome_label in [("SALES_LEVEL", "sales"), ("ASSETS_LEVEL", "assets"), ("ROA", "ROA"), ("MARGIN", "operating margin")]:
            add_model(f"D_{outcome_label}_{h}", "outcome_decomposition", f"BMI association with t+{h} {outcome_label}", f"{prefix}_H{h}", "BMI_z", "model-specific complete cases", "firm FE + year FE + controls")

    robustness = [
        ("R1", "M5_INDUSTRY_YEAR_FE", "BMI_z", "industry-by-year FE"),
        ("R2", "M6_MARKET_YEAR_FE", "BMI_z", "market-by-year FE"),
        ("R3", "M7_CHANGE_OUTCOME", "BMI_z", "t+2 change outcome"),
        ("R4", "ALT_LOG_COUNT", "BMI_log_count_z", "log count plus report length"),
        ("R5", "ALT_ANY", "BMI_any", "any BMI disclosure"),
        ("R6", "ALT_SHARE_OF_SUBSTANTIVE", "BMI_share_substantive_0_z", "BMI share only"),
        ("R7", "INTENSIVE_POSITIVE_ONLY", "BMI_positive_intensity_z", "positive observations only"),
        ("R8", "SAME_SAMPLE_PLUS_HIGH_COVERAGE", "BMI_z", "strict standard-tag accounting controls"),
    ]
    for test_id, model_id, term, hypothesis in robustness:
        add_model(test_id, "robustness", hypothesis, model_id, term, "model-specific complete cases", "reported robustness specification")

    if MARKET_FILE.exists():
        market = pd.read_csv(MARKET_FILE)
        for i, row in market.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "test_id": f"MKT{i+1}",
                    "analysis_family": "capital_market",
                    "hypothesis": f"Digital disclosure association with {row['outcome']}",
                    "outcome": row["outcome"],
                    "predictor": "DigitalDisclosure",
                    "sample": "outcome-specific event sample",
                    "specification": "firm FE + exact event-date FE + two-way clustered SE",
                    "coefficient_or_difference": row["coefficient"],
                    "raw_p": row["p_value"],
                }
            )

    registry = pd.DataFrame(rows)
    registry["BH_q_within_family"] = np.nan
    for family, index in registry.groupby("analysis_family").groups.items():
        registry.loc[index, "BH_q_within_family"] = multipletests(
            registry.loc[index, "raw_p"].to_numpy(), method="fdr_bh"
        )[1]
    registry.to_csv(OUT / "19_多重检验登记表.csv", index=False, encoding="utf-8-sig")
    return registry


def write_revision_audit(
    sample_audit: pd.DataFrame,
    models: pd.DataFrame,
    tests: pd.DataFrame,
    attrition: pd.DataFrame,
    oos: pd.DataFrame,
) -> None:
    full = sample_audit.loc[sample_audit.sample_name.eq("frozen_full_panel")].iloc[0]
    main = sample_audit.loc[sample_audit.sample_name.eq("main_regression_sample")].iloc[0]
    raw = tests.loc[tests.model_id.eq("M2B_RAW_SCALE_BMI_NONBMI")].iloc[0]
    share = models.loc[(models.model_id.eq("M4B_POSITIVE_TOTAL_SHARE")) & (models.term.eq("BMI_share_positive"))].iloc[0]
    common = models.loc[(models.model_id.eq("COMMON_TIMING_H2")) & (models.term.eq("BMI_z"))].iloc[0]
    ipw = attrition.loc[attrition.model_id.eq("ATTRITION_IPW")].iloc[0]
    financial = oos.loc[oos.model.eq("Financial")].iloc[0]
    bmi = oos.loc[oos.model.eq("Financial+BMI")].iloc[0]
    text = f"""# 二轮终审修改审计报告

## Issue 1 样本口径混用
Action: 分别按冻结基础面板和经固定效应 singleton 处理后的主回归样本重算 BMI 分布。
Result: 全面板 N={int(full.N):,}，BMI 零值比例={full.BMI_zero_share:.3%}，均值={full.BMI_mean:.6f}，标准差={full.BMI_sd:.6f}；主样本 N={int(main.N):,}，零值比例={main.BMI_zero_share:.3%}，均值={main.BMI_mean:.6f}，标准差={main.BMI_sd:.6f}。
Manuscript change: 所有描述统计、图注和经济量级均注明样本。
Effect on conclusion: 不改变主系数，但修正经济量级解释。

## Issue 2 文本分类过程表述
Action: 采用“领域规则句子分类加作者多轮复核”的真实口径，删除双人验证待办表述。
Result: 不报告第二编码者一致性或不存在的混淆矩阵。
Manuscript change: 重写测量方法与局限。
Effect on conclusion: 主结论不变，测量局限更透明。

## Issue 3 BMI 与 NonBMI 的尺度可比性
Action: 将两项均换算为每千个有效句子的句数并联合估计。
Result: 原始共同尺度差异={raw.difference:.6f}，SE={raw.std_error:.6f}，p={raw.p_value:.6g}，95% CI [{raw.ci95_low:.6f}, {raw.ci95_high:.6f}]。
Manuscript change: 在标准化 Wald 后增加辅助共同尺度检验。
Effect on conclusion: {'支持差异不只来自各自标准化尺度。' if raw.p_value < .05 else '共同尺度差异不显著，因此 H2 只保留标准化尺度解释。'}

## Issue 4 BMIShare 的结构性零值
Action: 仅在 TotalDigital>0 的公司年度中重新估计 TotalDigital 与 BMIShare。
Result: BMIShare 系数={share.coefficient:.6f}，SE={share.std_error:.6f}，p={share.p_value:.6g}，N={int(share.N):,}。
Manuscript change: 将该结果作为构成效应敏感性检验。
Effect on conclusion: {'条件正值样本出现关联，但不替换全样本不显著结论。' if share.p_value < .05 else '进一步限制“构成比例具有独立作用”的主张。'}

## Issue 5 时点模型样本变化
Action: 仅保留 t-2 至 t+3 均可观察的公司年度，并在六个时点使用同一组控制变量。
Result: 共同样本 t+2 系数={common.coefficient:.6f}，SE={common.std_error:.6f}，p={common.p_value:.6g}，N={int(common.N):,}。
Manuscript change: 正文保留主时点图，附录加入共同样本结果。
Effect on conclusion: 以完整六时点形状为准评价时间顺序证据。

## Issue 6 IPW 透明度
Action: 输出倾向得分、原始权重、稳定化权重、截尾权重、有效样本量及重叠图。
Result: IPW 主系数={ipw.coefficient:.6f}，p={ipw.p_value:.6g}。
Manuscript change: 明确 IPW 只能处理由已观察变量解释的留存差异。
Effect on conclusion: 不支持“消除生存偏差”的强表述。

## Issue 7 OOS 潜在前视偏差
Action: 不再传入全样本标准化变量；每个训练窗单独估计缩尾阈值、缺失值中位数、均值和标准差。预测模型不使用公司固定效应。
Result: Financial RMSE={financial.RMSE:.6f}；Financial+BMI RMSE={bmi.RMSE:.6f}。
Manuscript change: 重写 OOS 数据处理和边界。
Effect on conclusion: 无论方向如何，均以企业聚类 bootstrap 区间判断是否存在可信增量预测优势。

## Issue 8 多重检验范围不透明
Action: 按核心假设、构成、时点、结果拆解、稳健性和资本市场分别登记并在 family 内计算 BH q 值。
Result: 不再引用缺少完整底表的历史 5,304 个规格 q 值。
Manuscript change: 方法与附录明确 family 定义。
Effect on conclusion: 防止用任意大的检验集合夸大或压低证据。
"""
    (OUT / "REVISION_AUDIT_REPORT.md").write_text(text, encoding="utf-8")


def write_audit_summary(models: pd.DataFrame, tests: pd.DataFrame, attrition: pd.DataFrame, oos: pd.DataFrame) -> None:
    main = models.loc[(models.model_id == "M1_PRIMARY") & (models.term == "BMI_z")].iloc[0]
    summary = {
        "source_panel": str(PANEL),
        "source_panel_rows": int(pd.read_csv(PANEL, usecols=["corp_code"]).shape[0]),
        "primary": main.to_dict(),
        "wald_bmi_minus_nonbmi": tests.iloc[0].to_dict(),
        "attrition_models": attrition.to_dict(orient="records"),
        "oos_summary": oos.to_dict(orient="records"),
        "decisions": {
            "human_validation": "deterministic rules plus repeated single-researcher audit; no second coder and no intercoder reliability statistics",
            "system_gmm": "not adopted because no defensible external instruments are identified and T is short; change outcome used as robustness",
            "specification_curve_5304": "not reconstructed because the underlying 5,304-row result file is absent; historical q-values are not treated as independently verified",
            "section_denominator": "not implemented because section boundaries were not preserved in the frozen text panel",
            "inventory_receivables_sga": "not implemented because required accounting items are absent from the frozen/extended panel; no missing value was imputed",
            "causal_language": "prohibited; results are conditional associations and predictive evidence",
        },
    }
    (OUT / "00_终极版证据审计摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_panel()
    descriptive_outputs(d)
    sample_audit = sample_and_measurement_audit(d)
    models, tests = run_models(d)
    attrition = attrition_ipw(d)
    oos = oos_prediction(d)
    economic_magnitude(d, models)
    multiple_testing_registry(models, tests)
    write_revision_audit(sample_audit, models, tests, attrition, oos)
    write_audit_summary(models, tests, attrition, oos)
    print(f"Completed. Output: {OUT}")
    print(models.loc[models["term"].isin(["BMI_z", "NonBMI_z", "TotalDigital_z", "BMI_share_substantive_0_z"]), ["model_id", "term", "coefficient", "std_error", "p_value", "N", "within_R2"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
