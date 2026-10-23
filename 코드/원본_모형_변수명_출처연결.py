"""Build a new measurement-sensitivity panel, without changing original files.

The rebuilt classification is a recorded full actual run plus fifteen
independently source-selected, actually rescored viewer reports. It is not
claimed to be exact historical reproduction or a fresh second full run.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PANEL_SHA = "df2cfbb716dd091f4604379cb6d66c9ff1d4f296b4b92a51c18a51c26031f16d"
ORIGINAL07_SHA = "774942fd173d28df48a79a766c2b9ed89282fb11cb316fd5e3f978ad184c2a60"
ACTUAL_SHA = "c0d362044833810fca63bb986e86e03cefa3eae8bbe8ec03bee1ee8aa645060e"
HANDOFF_SHA = "468091af98decc8eec0c1cea95873ae6e87b4ffd3dc1c88fc287863b3ee25777"
FIELDS = ("total_sentences", "digital_sentence_count", "substantive_digital_sentence_count", "bmi_sentence_count")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def number(s):
    n = Decimal(str(s))
    if not n.is_finite() or n != n.to_integral_value() or n < 0:
        raise ValueError("invalid nonnegative integer")
    return int(n)


def firm_year(r):
    return (str(number(r["corp_code"])).zfill(8), str(number(r["year"])))


def report_key(r):
    return (*firm_year(r), str(number(r["rcept_no"])).zfill(14))


def rows(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def unique(data, getkey):
    result = {}
    for r in data:
        k = getkey(r)
        if k in result:
            raise ValueError("duplicate stable input key")
        result[k] = r
    return result


def counts(r):
    d, c, s, b = [number(r[k]) for k in FIELDS]
    if not 0 <= b <= s <= c <= d:
        raise ValueError("classification count nesting invalid")
    return (d, c, s, b)


def aliases(raw_rows, count_map):
    keys = [firm_year(r) for r in raw_rows]
    d = pd.Series([count_map[k][0] for k in keys], dtype=float)
    s = pd.Series([count_map[k][2] for k in keys], dtype=float)
    b = pd.Series([count_map[k][3] for k in keys], dtype=float)
    positive_d = d.where(d.gt(0))
    br, nr, dr = b / positive_d, (s - b) / positive_d, s / positive_d
    def z(v):
        return (v - v.mean()) / v.std(ddof=1)
    bz, nz, dz = z(br), z(nr), z(dr)
    bz_lookup = {k: float(value) for k, value in zip(keys, bz)}
    result = {
        "BMI_share": br, "NonBMI_share": nr,
        "BMI_component_z": bz, "NonBMI_share_z": nz,
        "Digital_share": dr, "Digital_share_z": dz,
        "BMIDisclosureAny": b.gt(0).astype(float),
        "LogTotalSentences": np.log1p(d.clip(lower=0)),
        "BMI_x_KOSDAQ": bz * pd.Series([float(r["KOSDAQ"]) for r in raw_rows]),
        "FutureBMI_h2": pd.Series([bz_lookup.get((k[0], str(int(k[1]) + 2)), np.nan) for k in keys]),
    }
    return result, {"BMODD_raw_mean": float(br.mean()), "BMODD_raw_sd_ddof1": float(br.std(ddof=1)),
                    "NonBMODD_raw_mean": float(nr.mean()), "NonBMODD_raw_sd_ddof1": float(nr.std(ddof=1)),
                    "TotalDigital_raw_mean": float(dr.mean()), "TotalDigital_raw_sd_ddof1": float(dr.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-root", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--original07", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    task = args.task_root.resolve()
    actual_path = task / "runs/full_rebuild_source_aware_01/report_results.jsonl"
    handoff_path = task / "evidence_classifier/paired_text_source15_handoff_01/ACTUAL15_COUNT_SOURCE_HANDOFF.json"
    pins = {str(args.panel.resolve()): PANEL_SHA, str(args.original07.resolve()): ORIGINAL07_SHA,
            str(actual_path): ACTUAL_SHA, str(handoff_path): HANDOFF_SHA,
            str(Path(__file__).resolve()): sha(__file__)}
    for p, expected in pins.items():
        if sha(p) != expected:
            raise ValueError("original input digest mismatch: " + p)
    if args.output.exists():
        raise FileExistsError("use a fresh sensitivity-input output directory")
    original07 = unique(rows(args.original07), firm_year)
    raw_rows = list(rows(args.panel))
    raw_by_key = unique(raw_rows, firm_year)
    with actual_path.open(encoding="utf-8") as f:
        actual = unique((json.loads(line) for line in f if line.strip()), firm_year)
    if len(raw_rows) != 22654 or set(raw_by_key) != set(original07) or set(actual) != set(original07):
        raise ValueError("firm-year universe/receipt bridge differs")
    original_counts, rebuilt_counts, sources = {}, {}, {}
    for k, r in original07.items():
        v = counts(r)
        for name in ("total_sentences", "substantive_digital_sentence_count", "bmi_sentence_count"):
            if number(raw_by_key[k][name]) != number(r[name]):
                raise ValueError("expanded financial panel count differs from original07")
        if report_key(actual[k]) != report_key(r) or actual[k]["status"] not in {"EXACT_COUNTS", "COUNT_MISMATCH"}:
            raise ValueError("actual result receipt/status not bound to original panel")
        original_counts[k] = v
        rebuilt_counts[k] = counts(actual[k])
        sources[k] = "FULL_ACTUAL_SOURCE_AWARE_RUN"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff["independently_source_selected_reports"] != 15:
        raise ValueError("source-selected viewer scope differs")
    viewer_keys = set()
    for h in handoff["rows"]:
        k = tuple(h["report_key"][:2])
        if k in viewer_keys or k not in original07 or tuple(h["report_key"]) != report_key(original07[k]):
            raise ValueError("viewer source identity differs or duplicated")
        viewer_keys.add(k)
        if h["target_count_fallback_used"] or not h["source_selected_without_classification_target"]:
            raise ValueError("viewer source cannot depend on frozen target counts")
        for field, hash_field in (("actual_score_file", "actual_score_file_sha256"), ("canonical_file", "canonical_sha256"),
                                  ("source_text_file", "source_text_sha256"), ("source_receipt_file", "source_receipt_sha256")):
            p, expected = h[field], h[hash_field]
            if sha(p) != expected:
                raise ValueError("viewer actual score/source/canonical digest mismatch")
            pins[p] = expected
        scores = json.loads(Path(h["actual_score_file"]).read_text(encoding="utf-8"))["scores"]
        if report_key(scores) != report_key(original07[k]) or scores["score_status"] != "ok":
            raise ValueError("actual viewer score context mismatch")
        rebuilt_counts[k] = counts(scores)  # actual scores only, never old_/target_ values
        sources[k] = "SOURCE_SELECTED_ACTUAL_VIEWER_RESCORING"
    if len(viewer_keys) != 15:
        raise ValueError("missing source-selected viewer score")
    old_aliases, old_scale = aliases(raw_rows, original_counts)
    new_aliases, new_scale = aliases(raw_rows, rebuilt_counts)
    alias_checks = {}
    for name, rebuilt in old_aliases.items():
        old_values = pd.to_numeric(pd.Series([r[name] for r in raw_rows]), errors="coerce")
        if not np.array_equal(old_values.isna(), rebuilt.isna()):
            raise ValueError("old text alias missingness not reproduced: " + name)
        maximum = float((old_values - rebuilt).abs().max())
        if maximum > 2e-12:
            raise ValueError("old text alias formula not reproduced: " + name)
        alias_checks[name] = maximum
    allowed = {"total_sentences", "substantive_digital_sentence_count", "bmi_sentence_count", *new_aliases}
    updated_rows, comparison = [], []
    for i, original in enumerate(raw_rows):
        k = firm_year(original)
        d, c, s, b = rebuilt_counts[k]
        out = dict(original)
        out.update(total_sentences=str(d), substantive_digital_sentence_count=str(s), bmi_sentence_count=str(b))
        for name, values in new_aliases.items():
            value = float(values.iloc[i])
            out[name] = "" if not np.isfinite(value) else repr(value)
        if any(out[name] != original[name] for name in original if name not in allowed):
            raise ValueError("non-text financial/outcome/control cell changed")
        updated_rows.append(out)
        old = original_counts[k]
        comparison.append({"corp_code": k[0], "year": k[1], "rcept_no": report_key(original07[k])[2],
            "count_source": sources[k],
            **dict(zip(("old_D", "old_candidate", "old_S", "old_BMODD"), old)),
            **dict(zip(("D", "candidate", "S", "BMODD"), (d, c, s, b))),
            "old_NonBMODD": old[2] - old[3], "NonBMODD": s - b,
            "changed_count_vector": old != (d, c, s, b),
            "BMODD_raw": float(new_aliases["BMI_share"].iloc[i]),
            "NonBMODD_raw": float(new_aliases["NonBMI_share"].iloc[i]),
            "TotalDigital_raw": float(new_aliases["Digital_share"].iloc[i]),
            "BMODD_z_fullbase": float(new_aliases["BMI_component_z"].iloc[i]),
            "NonBMODD_z_fullbase": float(new_aliases["NonBMI_share_z"].iloc[i])})
    args.output.mkdir(parents=True, exist_ok=False)
    def save_csv(name, data):
        with (args.output / name).open("x", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    save_csv("REBUILT_MEASUREMENT_PANEL.csv", updated_rows)
    save_csv("BMODD_FIRM_YEAR_CLASSIFICATION_AND_OLD_NEW_COMPARISON.csv", comparison)
    after = {p: sha(p) for p in pins}
    if after != pins:
        raise RuntimeError("protected original input changed")
    receipt = {"status": "PAIRED_MEASUREMENT_INPUTS_CREATED_NOT_REGRESSION_RESULTS", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "firm_years": len(raw_rows), "source_selected_actual_viewer_reports": 15,
        "changed_report_count_vectors": sum(r["changed_count_vector"] for r in comparison),
        "old_totals_D_candidate_S_BMODD": np.array(list(original_counts.values()), dtype=np.int64).sum(axis=0).tolist(),
        "rebuilt_totals_D_candidate_S_BMODD": np.array(list(rebuilt_counts.values()), dtype=np.int64).sum(axis=0).tolist(),
        "text_columns_recomputed": sorted(allowed), "non_text_cells_changed": 0, "original_alias_formula_max_abs_errors": alias_checks,
        "standardization": "full22654 base, sample SD ddof1, before model filters; formulas unchanged",
        "old_scale": old_scale, "rebuilt_scale": new_scale,
        "original_input_hashes_before": pins, "original_input_hashes_after": after,
        "output_hashes": {p.name: sha(p) for p in args.output.iterdir()},
        "classification_rules_changed": False, "regressions_run": False,
        "historical_exact_reproduction_claimed": False, "publication_gate": "NOT_PASSED"}
    (args.output / "PAIRED_INPUT_EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "firm_years", "changed_report_count_vectors", "old_totals_D_candidate_S_BMODD", "rebuilt_totals_D_candidate_S_BMODD", "non_text_cells_changed")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
