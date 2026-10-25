"""Audit H1/H2 downstream of the frozen firm-year panel; never run legacy main().

This is NOT a download-to-classification reproduction. All paths are explicit
arguments. The mechanically extracted core preserves the legacy function ASTs,
while removing unrelated top-level writes and private path defaults.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

LEGACY_SHA256 = "937cd7f3a3d3a730bda21d01b8cefa68eb6555d6846d6703a8cba321e606b2a5"
PANEL_SHA256 = "df2cfbb716dd091f4604379cb6d66c9ff1d4f296b4b92a51c18a51c26031f16d"
RESULT_SHA256 = "69c1cba392ec96f8a9016435e9bf0a22a0bc8a38f97fe8f8e05274f3048c6a1f"
FUNCTIONS = {
    "numeric", "zscore", "exact_shift", "drop_singletons",
    "weighted_residualize", "fit_fwl", "contrast", "load_panel",
}
SCOPE = "frozen-panel downstream H1/H2 re-estimation only; not end-to-end classifier reproduction"
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def select_nodes(source: str) -> list[ast.AST]:
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BASE_CONTROLS" for target in node.targets
        ):
            selected.append(node)
    found = {node.name for node in selected if isinstance(node, ast.FunctionDef)}
    if found != FUNCTIONS:
        raise ValueError(f"Missing or unexpected selected functions: {sorted(found ^ FUNCTIONS)}")
    if len(selected) != len(FUNCTIONS) + 1:
        raise ValueError("BASE_CONTROLS must have exactly one assignment")
    return selected


def core_snapshot(original: Path, destination: Path) -> dict:
    source = original.read_text(encoding="utf-8-sig")
    digest = sha256(original)
    if digest != LEGACY_SHA256:
        raise ValueError("Legacy estimator source hash differs from the audited production script")
    nodes = select_nodes(source)
    header = (
        '"""Mechanically extracted legacy estimator functions. No legacy main or private paths.\n'
        f'Original source SHA-256: {digest}\n'
        'PANEL must be assigned by the explicit-path runner before load_panel().\n"""\n'
        'from __future__ import annotations\nfrom pathlib import Path\n'
        'import numpy as np\nimport pandas as pd\nimport pyhdfe\n'
        'import statsmodels.api as sm\nfrom scipy import stats\n'
        'from statsmodels.stats.sandwich_covariance import cov_cluster\n\n'
    )
    content = header + "\n\n".join(ast.get_source_segment(source, node) for node in nodes) + "\n"
    destination.write_text(content, encoding="utf-8")
    extracted = select_nodes(content)
    receipts = []
    for old, new in zip(nodes, extracted):
        old_ast = ast.dump(old, include_attributes=False)
        new_ast = ast.dump(new, include_attributes=False)
        if old_ast != new_ast:
            raise AssertionError("Extracted function/constant AST differs")
        name = old.name if isinstance(old, ast.FunctionDef) else "BASE_CONTROLS"
        receipts.append({"name": name, "original_start_line": old.lineno,
                         "original_end_line": old.end_lineno,
                         "ast_sha256": hashlib.sha256(old_ast.encode()).hexdigest(),
                         "ast_identical": True})
    return {"original_name": original.name, "original_sha256": digest,
            "snapshot_name": destination.name, "snapshot_sha256": sha256(destination),
            "nodes": receipts}


def load_core(path: Path, panel: Path) -> dict:
    # Only explicitly selected functions and BASE_CONTROLS are evaluated; no
    # top-level import side effects, legacy main(), or document writes execute.
    import pyhdfe
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.sandwich_covariance import cov_cluster
    source = path.read_text(encoding="utf-8-sig")
    namespace = {"np": np, "pd": pd, "pyhdfe": pyhdfe, "sm": sm,
                 "stats": stats, "cov_cluster": cov_cluster, "PANEL": panel}
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *select_nodes(source)]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, path.name, "exec"), namespace)
    return namespace


def receipt(path: Path, purpose: str) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "purpose": purpose}


def worker(args: argparse.Namespace) -> int:
    started = utc_now()
    clock = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original, panel, stored, wald_reference = [
        Path(value).resolve() for value in
        (args.original_program, args.panel, args.reference_results, args.reference_wald)
    ]
    if sha256(panel) != PANEL_SHA256 or sha256(stored) != RESULT_SHA256:
        raise ValueError("Frozen panel or stored result hash differs; this is not the audited target")
    inputs = [receipt(original, "legacy estimator"), receipt(panel, "frozen expanded firm-year panel"),
              receipt(stored, "stored H1/H2 reference"), receipt(wald_reference, "stored H2 Wald reference")]
    if args.manuscript:
        inputs.append(receipt(Path(args.manuscript), "protected manuscript, hash only; never modified"))
    core = output / "frozen_empirical_core.py"
    ast_receipt = core_snapshot(original, core)
    write_json(output / "function_ast_provenance.json", ast_receipt)
    runtime = load_core(core, panel)
    frame = runtime["load_panel"]()
    if len(frame) != 22654 or frame.duplicated(["corp_code", "year"]).any():
        raise ValueError("Frozen panel grain/size mismatch")
    if (frame["bmi_sentence_count"] > frame["substantive_digital_sentence_count"]).any():
        raise ValueError("BMI count exceeds substantive digital count")
    rows = []
    one, _ = runtime["fit_fwl"](frame, "M1_PRIMARY", "AT_h2", ["BMI_z"],
                                runtime["BASE_CONTROLS"], ["corp_code", "year"], all_terms=True)
    rows.extend(one)
    two, details = runtime["fit_fwl"](frame, "M2_BMI_NONBMI", "AT_h2", ["BMI_z", "NonBMI_z"],
                                      runtime["BASE_CONTROLS"], ["corp_code", "year"], all_terms=True)
    rows.extend(two)
    actual = pd.DataFrame(rows)
    actual.to_csv(output / "estimated_coefficients.csv", index=False, encoding="utf-8-sig")
    contrast = {"model_id": "M2_BMI_NONBMI", **runtime["contrast"](details, "BMI_z", "NonBMI_z")}
    pd.DataFrame([contrast]).to_csv(output / "estimated_wald.csv", index=False, encoding="utf-8-sig")
    expected = pd.read_csv(stored)
    expected = expected.loc[expected["model_id"].isin(["M1_PRIMARY", "M2_BMI_NONBMI"])]
    compare_columns = ["coefficient", "std_error", "p_value", "ci95_low", "ci95_high", "N", "firms", "within_R2"]
    joined = expected.merge(actual, on=["model_id", "term"], how="outer", suffixes=("_reference", "_actual"), indicator=True)
    checks = []
    for row in joined.to_dict("records"):
        # Legacy M2 stores only its two focal terms. Its additional controls are
        # recomputed and emitted, but have no stored M2 reference to verify.
        if row["_merge"] == "right_only":
            continue
        if row["_merge"] != "both":
            checks.append({"model_id": row["model_id"], "term": row["term"], "field": "row_presence", "pass": False})
            continue
        for column in compare_columns:
            left, right = float(row[column + "_reference"]), float(row[column + "_actual"])
            difference = right - left
            exact_field = column in {"N", "firms"}
            passed = difference == 0 if exact_field else abs(difference) <= TOLERANCE
            checks.append({"model_id": row["model_id"], "term": row["term"], "field": column,
                           "reference": left, "actual": right, "difference": difference, "pass": bool(passed)})
    expected_wald = pd.read_csv(wald_reference)
    expected_wald = expected_wald.loc[expected_wald["model_id"].eq("M2_BMI_NONBMI")]
    if len(expected_wald) != 1:
        raise ValueError("Expected exactly one stored H2 Wald contrast")
    for column in ("difference", "std_error", "p_value", "ci95_low", "ci95_high", "df_denom"):
        left, right = float(expected_wald.iloc[0][column]), float(contrast[column])
        difference = right - left
        checks.append({"model_id": "M2_BMI_NONBMI_WALD", "term": "BMI_z - NonBMI_z", "field": column,
                       "reference": left, "actual": right, "difference": difference,
                       "pass": bool(abs(difference) <= (0 if column == "df_denom" else TOLERANCE))})
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "reference_diff.csv", index=False, encoding="utf-8-sig")
    # Recheck protected files at the end, including the paper if supplied.
    paths = [original, panel, stored, wald_reference] + ([Path(args.manuscript)] if args.manuscript else [])
    unchanged = all(sha256(path) == before["sha256"] for path, before in zip(paths, inputs))
    if not unchanged:
        raise AssertionError("A source/protected manuscript changed during the run")
    passed = bool(checks_frame["pass"].all())
    metrics = {"scope": SCOPE, "started_utc": started, "finished_utc": utc_now(),
               "elapsed_seconds": time.perf_counter() - clock, "pid": os.getpid(),
               "python_version": platform.python_version(),
               "versions": {name: importlib.metadata.version(name) for name in
                            ("numpy", "pandas", "scipy", "statsmodels", "pyhdfe")},
               "inputs": inputs, "panel_rows": len(frame), "firm_count": int(frame["corp_code"].nunique()),
               "checks": len(checks), "all_reference_checks_pass": passed,
               "max_abs_difference": float(checks_frame.get("difference", pd.Series([0])).abs().max()),
               "absolute_tolerance": TOLERANCE, "protected_inputs_unchanged": unchanged,
               "note": "No classifier, download, financial-data regeneration, H3, or supplemental model executed."}
    write_json(output / "run_receipt.json", metrics)
    print(json.dumps({"scope": SCOPE, "run": output.name, "pid": os.getpid(), "checks": len(checks),
                      "pass": passed, "max_abs_difference": metrics["max_abs_difference"]}, ensure_ascii=False))
    return 0 if passed else 1


def repeat(args: argparse.Namespace) -> int:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(1, args.repeats + 1):
        destination = root / f"run_{index:02d}"
        command = [sys.executable, str(Path(__file__).resolve()), "--panel", args.panel,
                   "--original-program", args.original_program, "--reference-results", args.reference_results,
                   "--reference-wald", args.reference_wald, "--output", str(destination), "--worker"]
        if args.manuscript:
            command += ["--manuscript", args.manuscript]
        environment = os.environ.copy()
        environment.update({"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
        (root / f"run_{index:02d}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (root / f"run_{index:02d}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        record = {"run": index, "exit_code": completed.returncode, "success": completed.returncode == 0}
        for filename in ("estimated_coefficients.csv", "estimated_wald.csv", "reference_diff.csv"):
            path = destination / filename
            record[filename + "_sha256"] = sha256(path) if path.exists() else None
        records.append(record)
        print(completed.stdout.strip(), flush=True)
        if completed.returncode:
            print(f"Run {index} failed; inspect its stderr and receipts.", file=sys.stderr, flush=True)
            break
    results = pd.DataFrame(records)
    results.to_csv(root / "repeat_execution_summary.csv", index=False, encoding="utf-8-sig")
    artifact_columns = [column for column in results if column.endswith("_sha256")]
    deterministic = len(records) == args.repeats and all(results[column].nunique(dropna=False) == 1 for column in artifact_columns)
    passed = len(records) == args.repeats and bool(results["success"].all()) and deterministic
    write_json(root / "repeat_receipt.json", {"scope": SCOPE, "requested_repetitions": args.repeats,
               "completed_repetitions": len(records), "new_process_per_run": True,
               "fresh_output_directory_per_run": True, "numeric_outputs_hash_identical": deterministic,
               "frozen_downstream_pass": passed, "end_to_end_reproduction_pass": False,
               "end_to_end_status": "NOT EXECUTED: raw download/classifier/aggregation/financial rebuild are outside this runner"})
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, help="Frozen expanded firm-year CSV; SHA-256 checked")
    parser.add_argument("--original-program", required=True, help="Original audited estimator .py; AST extraction only")
    parser.add_argument("--reference-results", required=True, help="Stored full model result CSV")
    parser.add_argument("--reference-wald", required=True, help="Stored H2 Wald result CSV")
    parser.add_argument("--manuscript", help="Optional protected DOCX; read for hash only")
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent output directory")
    parser.add_argument("--repeats", type=int, choices=(3, 4), default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    options = parse_args()
    raise SystemExit(worker(options) if options.worker else repeat(options))
