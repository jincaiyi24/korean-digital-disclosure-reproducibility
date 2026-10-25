"""Run only statistical specifications retained in the pinned current paper.

Historical functions are selected as unedited AST nodes; historical main(),
document writes, capital-market regressions and discarded models never execute.
All data/source/output paths are explicit. Baseline and rebuilt runs are separate.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy import sparse, stats
from scipy.sparse.linalg import lsmr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster

PINS = {
    "caller": "96b0d7c90f3e2c90765a7e4e7a08ee9bcad606547bf5ebd26a383af44215bd64",
    "original_program": "937cd7f3a3d3a730bda21d01b8cefa68eb6555d6846d6703a8cba321e606b2a5",
    "trend_program": "8ebd869b1fec9ec45c748e92a760049c9695c1eb173f3a038de5658150177202",
    "placebo_program": "822f5054fdb6fae259d5811656c9161f30ea4c152b0c3305c9205f4e51c2df6a",
    "formal_program": "583dd4fd61413097c80e68c73ba065e5ec4f58b506f61c782bd5b41c06fe9e21",
    "supplement_program": "a4142cdbe7c786e396bdbf5f02f7237250ce36967fb5172980d16c3141a533aa",
    "frozen_financial_panel": "bdeceb72a4eb0c903d4038d2e7748be4c6826358aa9e4a4ae81bd63020fff2f6",
    "manuscript": "17862a4f3c3aa9df2095f78afa1077fb59e7495ee95089405b7a305c547dfc40",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_selected(path, expected, functions, constants, extra=None):
    if sha(path) != expected:
        raise ValueError("Source pin failed: " + str(path))
    source = Path(path).read_text(encoding="utf-8-sig")
    nodes = []
    for n in ast.parse(source).body:
        if isinstance(n, ast.FunctionDef) and n.name in functions:
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets):
            nodes.append(n)
    names = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if names != set(functions):
        raise ValueError("Missing requested source functions")
    ns = {"np": np, "pd": pd, "pyhdfe": pyhdfe, "sm": sm, "stats": stats, "sparse": sparse,
          "lsmr": lsmr, "math": math, "hashlib": hashlib, "cov_cluster": cov_cluster,
          "multipletests": multipletests, "Path": Path}
    ns.update(extra or {})
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), ns)
    return ns, [{"name": n.name if isinstance(n, ast.FunctionDef) else ast.unparse(n.targets[0]),
                 "line": n.lineno, "end_line": n.end_lineno,
                 "ast_sha256": hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()} for n in nodes]


def core_models(d, original, output):
    rows = []
    samples = []
    c = original["BASE_CONTROLS"]
    def fit(mid, y, x, controls, effects=None, sample=None):
        result, detail = original["fit_fwl"](d, mid, y, x, controls, effects or ["corp_code", "year"], sample=sample, all_terms=True)
        rows.extend(result)
        key = d.loc[detail["work"].index, ["corp_code", "year"]].copy()
        key["model_id"] = mid
        samples.append(key)
        pd.DataFrame(rows).to_csv(output / "core_coefficients.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({"completed_model": mid, "N": len(key)}, ensure_ascii=False), flush=True)
    fit("M3_TOTAL_ONLY", "AT_h2", ["TotalDigital_z"], c)
    fit("M4_TOTAL_PLUS_COMPOSITION", "AT_h2", ["TotalDigital_z", "BMI_share_substantive_0_z"], c)
    for h in (-2, -1, 0, 1, 2, 3, 4, 5):
        if h not in (-2, -1, 0, 1, 2, 3):
            d[f"AT_h{h}"] = original["exact_shift"](d, "AssetTurnover_w", h)
        fit(f"TIMING_H{h}", f"AT_h{h}", ["BMI_z"], c if h != 0 else c[1:])
    required = [f"AT_h{h}" for h in (-2, -1, 0, 1, 2, 3)] + ["BMI_z", *c[1:], "corp_code", "year"]
    common = d[required].notna().all(axis=1)
    for h in (-2, -1, 0, 1, 2, 3):
        fit(f"COMMON_TIMING_H{h}", f"AT_h{h}", ["BMI_z"], c[1:], sample=common)
    fit("M5_INDUSTRY_YEAR_FE", "AT_h2", ["BMI_z"], c, ["corp_code", "industry_year"])
    fit("M6_MARKET_YEAR_FE", "AT_h2", ["BMI_z"], c, ["corp_code", "market_year"])
    fit("M7_CHANGE_OUTCOME", "AT_change_h2", ["BMI_z"], c[1:])
    fit("ALT_LOG_COUNT", "AT_h2", ["BMI_log_count_z"], [*c, "LogTotalSentences"])
    fit("ALT_ANY", "AT_h2", ["BMI_any"], c)
    fit("ALT_SHARE_OF_SUBSTANTIVE", "AT_h2", ["BMI_share_substantive_0_z"], c)
    fit("INTENSIVE_POSITIVE_ONLY", "AT_h2", ["BMI_positive_intensity_z"], c, sample=d["BMI_any"].eq(1))
    high = ["IntangibleAssetRatio_w", "PPE_Ratio_w", "OperatingCashFlowRatio_w", "CapexIntensity_w"]
    high_sample = d[["AT_h2", "BMI_z", *c, *high, "corp_code", "year"]].notna().all(axis=1)
    fit("SAME_SAMPLE_BASE_STRICT_ACCOUNTS", "AT_h2", ["BMI_z"], c, sample=high_sample)
    fit("SAME_SAMPLE_PLUS_HIGH_COVERAGE", "AT_h2", ["BMI_z"], [*c, *high], sample=high_sample)
    pd.concat(samples, ignore_index=True).to_csv(output / "core_sample_keys.csv", index=False, encoding="utf-8-sig")


def h3_joint(d, source, output):
    ns, provenance = load_selected(source, PINS["formal_program"],
        {"drop_singletons", "group_demean", "two_way_residualize", "cluster_joint_covariance"}, {"FOCAL", "OUTCOMES", "CONTROLS"})
    write(output / "FUNCTION_AST_PROVENANCE.json", provenance)
    controls, outcomes, focal = ns["CONTROLS"], ns["OUTCOMES"], ns["FOCAL"]
    required = list(dict.fromkeys([*outcomes.values(), focal, *controls, "corp_code", "year"]))
    complete = d[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    common, audit = ns["drop_singletons"](complete, ["corp_code", "year"])
    common = common.sort_values(["corp_code", "year"], kind="mergesort").reset_index(drop=True)
    names = [focal, *controls]
    raw = common[[*names, *outcomes.values()]].astype(float).to_numpy()
    fc = pd.Categorical(common["corp_code"], categories=sorted(common["corp_code"].unique())).codes
    yc = pd.Categorical(common["year"], categories=sorted(common["year"].unique())).codes
    transformed, iterations, delta = ns["two_way_residualize"](raw, fc, yc)
    k = len(names)
    x, y = transformed[:, :k], transformed[:, k:]
    if np.linalg.matrix_rank(x) != k:
        raise RuntimeError("H3 joint design rank deficiency")
    bread = np.linalg.inv(x.T @ x)
    b = bread @ x.T @ y
    residuals = y - x @ b
    covariance, correction = ns["cluster_joint_covariance"](x, residuals, fc)
    n, firms, df = len(common), int(common["corp_code"].nunique()), int(common["corp_code"].nunique()) - 1
    critical = float(stats.t.ppf(.975, df))
    rows, differences = [], []
    hs = list(outcomes)
    for j, h in enumerate(hs):
        beta = float(b[0, j]); se = math.sqrt(max(float(covariance[j*k, j*k]), 0))
        rows.append({"horizon": h, "coefficient": beta, "std_error": se, "p_value": float(2*stats.t.sf(abs(beta/se), df)),
                     "ci95_low": beta-critical*se, "ci95_high": beta+critical*se, "N": n, "firms": firms})
    for high, low, role in [(2, 0, "primary_prespecified"), (1, 0, "secondary"), (3, 0, "secondary")]:
        hi, lo = hs.index(high), hs.index(low)
        contrast = np.zeros(len(hs)*k); contrast[hi*k] = 1; contrast[lo*k] = -1
        diff = float(b[0, hi]-b[0, lo]); se = math.sqrt(max(float(contrast@covariance@contrast), 0))
        yd = y[:, hi]-y[:, lo]; bd = bread@x.T@yd; ed = yd-x@bd
        direct, _ = ns["cluster_joint_covariance"](x, ed[:, None], fc)
        direct_se = math.sqrt(max(float(direct[0, 0]), 0))
        if abs(diff-float(bd[0])) > 1e-10 or abs(se-direct_se) > 1e-10:
            raise RuntimeError("H3 cross-equation/direct difference mismatch")
        differences.append({"high_horizon": high, "low_horizon": low, "test_role": role, "difference": diff, "std_error": se,
            "p_value": float(2*stats.t.sf(abs(diff/se), df)), "ci95_low": diff-critical*se, "ci95_high": diff+critical*se,
            "N": n, "firms": firms, "df": df, "cross_equation_covariance": float(covariance[hi*k, lo*k]),
            "direct_difference_beta": float(bd[0]), "direct_difference_se": direct_se})
    differences = pd.DataFrame(differences)
    mask = differences["test_role"].eq("secondary")
    for method in ("holm", "fdr_bh"):
        differences.loc[mask, method] = multipletests(differences.loc[mask, "p_value"], method=method)[1]
    pd.DataFrame(rows).to_csv(output / "h3_joint_coefficients.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(output / "h3_joint_differences.csv", index=False, encoding="utf-8-sig")
    common[["corp_code", "year"]].to_csv(output / "h3_joint_sample_keys.csv", index=False, encoding="utf-8-sig")
    write(output / "H3_ABSORPTION_RECEIPT.json", {"iterations": iterations, "delta": delta, "correction": correction, "singleton_audit": audit})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, choices=["core", "h3_joint", "trend", "placebo", "ipw", "supplement"])
    for name in ("caller", "original-program", "panel", "manuscript", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("trend-program", "placebo-program", "formal-program", "supplement-program", "frozen-financial-panel"):
        p.add_argument("--"+name, type=Path)
    p.add_argument("--panel-sha256", required=True)
    a = p.parse_args()
    all_paths = {name: getattr(a, name) for name in PINS if getattr(a, name, None) is not None}
    all_paths["panel"] = a.panel
    for name, path in all_paths.items():
        expected = a.panel_sha256 if name == "panel" else PINS[name]
        if sha(path) != expected:
            raise ValueError("Input pin mismatch: " + name)
    a.output.mkdir(parents=True, exist_ok=False)
    before = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in all_paths.items()}
    start = time.perf_counter()
    write(a.output / "START_RECEIPT.json", {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "command": sys.argv, "inputs": before})
    try:
        spec = importlib.util.spec_from_file_location("audited_h1h2_core", a.caller)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        write(a.output / "CORE_AST_PROVENANCE.json", helper.core_snapshot(a.original_program, a.output / "core.py"))
        original = helper.load_core(a.output / "core.py", a.panel)
        d = original["load_panel"]()
        if len(d) != 22654 or d.duplicated(["corp_code", "year"]).any():
            raise ValueError("Panel grain invalid")
        if not np.allclose(d["LogTotalSentences"], np.log1p(d["total_sentences"]), atol=1e-14, rtol=0):
            raise ValueError("Stale report length")
        if a.group == "core":
            core_models(d, original, a.output)
        elif a.group == "h3_joint":
            h3_joint(d, a.formal_program, a.output)
        elif a.group == "trend":
            ns, prov = load_selected(a.trend_program, PINS["trend_program"], {"index_hash", "numeric", "drop_singletons", "prepare_work", "select_full_rank", "finish_fit", "fit_absorbed", "build_trend_design", "residualize_sparse", "fit_trend"}, {"OUTCOME", "FOCAL", "CONTROLS", "REGRESSORS"})
            write(a.output / "FUNCTION_AST_PROVENANCE.json", prov)
            base, detail = ns["fit_absorbed"](d, "M1_MODEL_A", ["corp_code", "year"], "Firm FE + Year FE")
            trend, td = ns["fit_trend"](d, "M4_MODEL_B", detail["work"].index)
            pd.DataFrame([base, trend]).to_csv(a.output / "trend_models.csv", index=False, encoding="utf-8-sig")
            write(a.output / "TREND_ABSORPTION.json", td["absorption"])
        elif a.group == "placebo":
            ns, prov = load_selected(a.placebo_program, PINS["placebo_program"], {"numeric", "exact_shift", "drop_singletons", "fit_fwl"}, {"BASE_CONTROLS", "NON_OUTCOME_CONTROLS"})
            write(a.output / "FUNCTION_AST_PROVENANCE.json", prov)
            d["PastAssetTurnover_h1_exact"] = ns["exact_shift"](d, "AssetTurnover_w", -1)
            rows = []
            for h in (1, 2):
                focal = f"FutureBMI_h{h}_exact"; d[focal] = ns["exact_shift"](d, "BMI_component_z", h)
                result, _ = ns["fit_fwl"](d, f"PLACEBO_FUTURE_BMI_H{h}", "AssetTurnover_w", [focal], ["PastAssetTurnover_h1_exact", *ns["NON_OUTCOME_CONTROLS"]])
                rows.append(result)
            pd.concat(rows, ignore_index=True).to_csv(a.output / "placebo_models.csv", index=False, encoding="utf-8-sig")
        elif a.group == "ipw":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            extra = {**original, "OUT": a.output, "plt": plt, "ColumnTransformer": ColumnTransformer, "StandardScaler": StandardScaler,
                     "OneHotEncoder": OneHotEncoder, "Pipeline": Pipeline, "LogisticRegression": LogisticRegression}
            ns, prov = load_selected(a.original_program, PINS["original_program"], {"attrition_ipw"}, set(), extra)
            write(a.output / "FUNCTION_AST_PROVENANCE.json", prov)
            ns["attrition_ipw"](d)
        elif a.group == "supplement":
            names = {"exact_shift", "drop_singletons", "prepare_work", "fit_fwl", "add_fdr", "build_model_results", "merge_raw_financials", "build_decomposition"}
            ns, prov = load_selected(a.supplement_program, PINS["supplement_program"], names, {"BASE_CONTROLS", "COMMON_CONTROLS", "EFFECTS"}, {"FROZEN_PANEL": a.frozen_financial_panel})
            write(a.output / "FUNCTION_AST_PROVENANCE.json", prov)
            models, index = ns["build_model_results"](d)
            merged, raw_checks = ns["merge_raw_financials"](d)
            baseline = float(models.loc[models["model_id"].eq("A1_BASELINE"), "coefficient"].iloc[0])
            decomp, flow, checks, common, diagnostics, decomp_checks = ns["build_decomposition"](merged, index, baseline)
            all_models = ns["add_fdr"](pd.concat([models, decomp], ignore_index=True, sort=False))
            all_models.to_csv(a.output / "supplement_models_with_BH.csv", index=False, encoding="utf-8-sig")
            for name, frame in [("decomposition_sample_flow", flow), ("coefficient_identity_checks", checks), ("decomposition_common_sample", common), ("decomposition_diagnostics", diagnostics)]:
                frame.to_csv(a.output / (name+".csv"), index=False, encoding="utf-8-sig")
            write(a.output / "RAW_FINANCIAL_AND_DECOMPOSITION_CHECKS.json", {"raw": raw_checks, "decomp": decomp_checks,
                "bh_family_estimated_nonbaseline_count": int((all_models["status"].eq("estimated") & ~all_models["model_id"].str.endswith("BASELINE")).sum())})
        unchanged = all(sha(path) == before[name]["sha256"] for name, path in all_paths.items())
        if not unchanged:
            raise ValueError("Protected source/input changed")
        write(a.output / "RUN_RECEIPT.json", {"status": "COMPLETED_ORIGINAL_SPECIFICATION_MEASUREMENT_SENSITIVITY", "group": a.group,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter()-start, "pid": os.getpid(),
              "inputs": before, "protected_inputs_unchanged": unchanged, "no_main_no_paper_writes": True,
              "outputs": [{"name": x.name, "sha256": sha(x)} for x in a.output.iterdir() if x.is_file()]})
        print(json.dumps({"group": a.group, "status": "COMPLETED", "seconds": time.perf_counter()-start}), flush=True)
        return 0
    except Exception as e:
        write(a.output / "FAILURE_RECEIPT.json", {"status": "FAILED", "group": a.group, "type": type(e).__name__, "message": str(e),
              "protected_inputs_unchanged": all(sha(path) == before[name]["sha256"] for name, path in all_paths.items())})
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
