"""Regenerate the text panel and all CURRENT-paper statistical specifications.

Called only with freshly sealed report predictions. Prior result tables are
opened only after all model outputs are sealed, for posthoc validation. The
three no-baseline-AT decomposition rows remain inside the original 19-test BH
family but are not designated for display in the revised paper.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

CORE_IDS = {"M3_TOTAL_ONLY", "M4_TOTAL_PLUS_COMPOSITION", "M5_INDUSTRY_YEAR_FE", "M6_MARKET_YEAR_FE", "M7_CHANGE_OUTCOME",
    "ALT_LOG_COUNT", "ALT_ANY", "ALT_SHARE_OF_SUBSTANTIVE", "INTENSIVE_POSITIVE_ONLY", "SAME_SAMPLE_BASE_STRICT_ACCOUNTS", "SAME_SAMPLE_PLUS_HIGH_COVERAGE",
    *[f"TIMING_H{h}" for h in (-2,-1,0,1,2,3,4,5)], *[f"COMMON_TIMING_H{h}" for h in (-2,-1,0,1,2,3)]}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def dump(p,value):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def resolve(base,p):
    p=Path(p);return p.resolve() if p.is_absolute() else (base/p).resolve()


def save_csv(p,records):
    with Path(p).open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def generate_panel(cfg,base,method,results,out):
    """Reuse the already-audited pure alias formulas, not its checkpoint main."""
    validate_prediction_keys(results)
    import numpy as np
    aliases=module(method/"model_code/alias_provenance.py","current_measurement_pure_alias_formulas")
    source=resolve(base,cfg["financial_panel"]["path"])
    if sha(source)!=cfg["financial_panel"]["sha256"]:raise ValueError("Financial input pin failed")
    with source.open(encoding="utf-8-sig",newline="") as f:raw=list(csv.DictReader(f))
    counts={tuple(r["key"][:2]):tuple(int(r[n]) for n in ("total_sentences","digital_sentence_count","substantive_digital_sentence_count","bmi_sentence_count")) for r in results}
    keys=[aliases.firm_year(r) for r in raw]
    if len(raw)!=22654 or len(set(keys))!=22654 or set(counts)!=set(keys):raise ValueError("Text/financial firm-year bridge is not one-to-one")
    new,scales=aliases.aliases(raw,counts)
    allowed={"total_sentences","substantive_digital_sentence_count","bmi_sentence_count",*new}
    updated=[];public=[]
    prediction={tuple(r["key"][:2]):r for r in results}
    for i,original in enumerate(raw):
        k=keys[i];d,c,s,b=counts[k]
        row=dict(original);row.update(total_sentences=str(d),substantive_digital_sentence_count=str(s),bmi_sentence_count=str(b))
        for name,values in new.items():
            value=float(values.iloc[i]);row[name]="" if not np.isfinite(value) else repr(value)
        if any(row[n]!=original[n] for n in original if n not in allowed):raise ValueError("Non-text financial/outcome/control cell changed")
        updated.append(row)
        public.append({"corp_code":k[0],"year":k[1],"rcept_no":prediction[k]["key"][2],"source_kind":prediction[k]["source_kind"],
            "D":d,"candidate":c,"S":s,"BMODD":b,"NonBMODD":s-b,
            "BMODD_raw":float(new["BMI_share"].iloc[i]),"NonBMODD_raw":float(new["NonBMI_share"].iloc[i]),
            "TotalDigital_raw":float(new["Digital_share"].iloc[i]),"BMODD_z_fullbase":float(new["BMI_component_z"].iloc[i]),"NonBMODD_z_fullbase":float(new["NonBMI_share_z"].iloc[i])})
    panel=out/"CURRENT_MEASUREMENT_PANEL_INTERNAL.csv";save_csv(panel,updated)
    save_csv(out/"BMODD_FIRM_YEAR_CLASSIFICATION.csv",public)
    if sha(panel)!=cfg["expected_new_panel_sha256_after_generation"]:raise ValueError("Fresh panel differs from the frozen current-version transformation target")
    dump(out/"PANEL_GENERATION_RECEIPT.json",{"rows":len(raw),"counts_taken_only_from_fresh_predictions":True,"text_columns_recomputed":sorted(allowed),"non_text_cells_changed":0,"panel_sha256":sha(panel),"scale":scales,
        "public_alias_mapping":{"BMI_share":"BMODD_raw","NonBMI_share":"NonBMODD_raw","BMI_component_z":"BMODD_z_fullbase","NonBMI_share_z":"NonBMODD_z_fullbase"}})
    return panel


def validate_prediction_keys(results):
    keys=[tuple(r["key"]) for r in results]
    if len(results)!=22654 or any(len(k)!=3 for k in keys) or len(set(keys))!=22654 or len({k[:2] for k in keys})!=22654:
        raise ValueError("Panel generation requires exactly 22654 unique complete report and firm-year identities")
    for r in results:
        v=[r[n] for n in ("total_sentences","digital_sentence_count","substantive_digital_sentence_count","bmi_sentence_count")]
        if r["status"]!="FRESH_PREDICTION_COMPLETED" or any(type(x) is not int for x in v) or not 0<=v[3]<=v[2]<=v[1]<=v[0]:raise ValueError("Panel generation rejected incomplete/noninteger prediction")


def compare_frame(actual,expected,keys,label,checks):
    import numpy as np
    import pandas as pd
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():raise ValueError("Duplicate model-result identity: "+label)
    numeric=[c for c in ("coefficient","difference","std_error","p_value","ci95_low","ci95_high","N","firms","within_R2","q_value_bh","holm","fdr_bh","cross_equation_covariance","direct_difference_beta","direct_difference_se","observed_share","ipw_p01","ipw_p99","t_stat","t_value","wald_F","df_denom","df","years","year_min","year_max","ci95_width","coefficient_change_pct","N_change","firms_change") if c in expected]
    if not numeric or any(c not in actual for c in numeric):raise ValueError("A required reference numeric column is missing in actual model output: "+label)
    descriptors=[c for c in ("status","outcome","test_role","fixed_effects","effects","cluster","controls","control_protocol","weighted","sample_rule","fdr_scope") if c in expected]
    if any(c not in actual for c in descriptors):raise ValueError("A required model descriptor is missing: "+label)
    for c in descriptors:
        joined_text=expected[keys+[c]].merge(actual[keys+[c]],on=keys,how="outer",validate="one_to_one",suffixes=("_expected","_fresh"),indicator=True)
        if not joined_text["_merge"].eq("both").all() or not joined_text[c+"_expected"].fillna("").astype(str).eq(joined_text[c+"_fresh"].fillna("").astype(str)).all():raise ValueError("Model specification/status descriptor changed: "+label+" "+c)
    joined=expected[keys+numeric].merge(actual[keys+numeric],on=keys,how="outer",validate="one_to_one",suffixes=("_expected","_fresh"),indicator=True)
    if not joined["_merge"].eq("both").all():raise ValueError("Model-result identity set differs: "+label)
    for col in numeric:
        l=pd.to_numeric(joined[col+"_expected"],errors="raise");r=pd.to_numeric(joined[col+"_fresh"],errors="raise")
        good=np.isclose(l,r,atol=0 if col in ("N","firms") else 1e-10,rtol=0,equal_nan=True)
        checks.append({"group":label,"field":col,"rows":len(good),"failed":int((~good).sum()),"max_abs_difference":float((r-l).abs().max()) if (r-l).notna().any() else None})
        if not good.all():raise ValueError("Current-version model target differs: "+label+" "+col)


def regenerate_panel_and_models(cfg,base,method,results,out):
    out.mkdir(exist_ok=False);os.environ["MPLCONFIGDIR"]=str(out/"_matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    helper=module(method/"model_code/empirical_adapter.py","audited_empirical_adapter")
    comp=module(method/"model_code/current_components.py","audited_current_paper_components")
    panel=generate_panel(cfg,base,method,results,out)
    original_source=method/"model_sources/original_estimator.py"
    dump(out/"CORE_AST_PROVENANCE.json",helper.core_snapshot(original_source,out/"core.py"))
    ns=helper.load_core(out/"core.py",panel);d=ns["load_panel"]();c=ns["BASE_CONTROLS"]
    primary,detail=ns["fit_fwl"](d,"M1_PRIMARY","AT_h2",["BMI_z"],c,["corp_code","year"],all_terms=True)
    joint,jd=ns["fit_fwl"](d,"M2_BMI_NONBMI","AT_h2",["BMI_z","NonBMI_z"],c,["corp_code","year"],all_terms=True)
    primary=pd.DataFrame([*primary,*joint]);primary.to_csv(out/"primary_h1_h2.csv",index=False,encoding="utf-8-sig")
    wald=pd.DataFrame([{"model_id":"M2_BMI_NONBMI",**ns["contrast"](jd,"BMI_z","NonBMI_z")}]);wald.to_csv(out/"h2_wald.csv",index=False,encoding="utf-8-sig")
    h1p=float(primary.loc[primary["model_id"].eq("M1_PRIMARY")&primary["term"].eq("BMI_z"),"p_value"].iloc[0]);h2p=float(wald["p_value"].iloc[0])
    pd.DataFrame({"hypothesis":["H1","H2"],"raw_p":[h1p,h2p],"BH_q_within_confirmatory_core":multipletests([h1p,h2p],method="fdr_bh")[1]}).to_csv(out/"confirmatory_two_test_BH.csv",index=False,encoding="utf-8-sig")
    core=out/"current_core";core.mkdir();comp.core_models(d,ns,core)
    h3=out/"h3_joint";h3.mkdir();comp.h3_joint(d,method/"model_sources/formal_h3.py",h3)
    trend,prov=comp.load_selected(method/"model_sources/trend.py",comp.PINS["trend_program"],{"index_hash","numeric","drop_singletons","prepare_work","select_full_rank","finish_fit","fit_absorbed","build_trend_design","residualize_sparse","fit_trend"},{"OUTCOME","FOCAL","CONTROLS","REGRESSORS"})
    tr,td=trend["fit_trend"](d,"M4_MODEL_B",detail["work"].index)
    pd.DataFrame([tr]).to_csv(out/"trend.csv",index=False,encoding="utf-8-sig");dump(out/"TREND_AST_PROVENANCE.json",prov);dump(out/"TREND_ABSORPTION.json",td["absorption"])
    placebo,prov=comp.load_selected(method/"model_sources/placebo.py",comp.PINS["placebo_program"],{"numeric","exact_shift","drop_singletons","fit_fwl"},{"BASE_CONTROLS","NON_OUTCOME_CONTROLS"})
    d["PastAssetTurnover_h1_exact"]=placebo["exact_shift"](d,"AssetTurnover_w",-1);pr=[]
    for h in (1,2):
        focal=f"FutureBMI_h{h}_exact";d[focal]=placebo["exact_shift"](d,"BMI_component_z",h)
        row,_=placebo["fit_fwl"](d,f"PLACEBO_FUTURE_BMI_H{h}","AssetTurnover_w",[focal],["PastAssetTurnover_h1_exact",*placebo["NON_OUTCOME_CONTROLS"]]);pr.append(row)
    pd.concat(pr,ignore_index=True).to_csv(out/"placebo.csv",index=False,encoding="utf-8-sig");dump(out/"PLACEBO_AST_PROVENANCE.json",prov)
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    ipw=out/"ipw";ipw.mkdir()
    ins,prov=comp.load_selected(original_source,comp.PINS["original_program"],{"attrition_ipw"},set(),{**ns,"OUT":ipw,"plt":plt,"ColumnTransformer":ColumnTransformer,"StandardScaler":StandardScaler,"OneHotEncoder":OneHotEncoder,"Pipeline":Pipeline,"LogisticRegression":LogisticRegression})
    ins["attrition_ipw"](d);dump(out/"IPW_AST_PROVENANCE.json",prov)
    names={"exact_shift","drop_singletons","prepare_work","fit_fwl","add_fdr","build_model_results","merge_raw_financials","build_decomposition"}
    supp,prov=comp.load_selected(method/"model_sources/supplement.py",comp.PINS["supplement_program"],names,{"BASE_CONTROLS","COMMON_CONTROLS","EFFECTS"},{"FROZEN_PANEL":resolve(base,cfg["frozen_financial_panel"]["path"])})
    models,index=supp["build_model_results"](d);merged,rawchecks=supp["merge_raw_financials"](d)
    decomp,flow,identity,common,diagnostics,decompchecks=supp["build_decomposition"](merged,index,float(models.iloc[0]["coefficient"]))
    all_supp=supp["add_fdr"](pd.concat([models,decomp],ignore_index=True,sort=False))
    all_supp.to_csv(out/"supplement_with_original_BH19.csv",index=False,encoding="utf-8-sig")
    family=all_supp["status"].eq("estimated")&~all_supp["model_id"].str.endswith("BASELINE")
    if int(family.sum())!=19:raise ValueError("Original 19-test BH family changed")
    flow.to_csv(out/"decomposition_sample_flow.csv",index=False,encoding="utf-8-sig");common.to_csv(out/"decomposition_common_sample.csv",index=False,encoding="utf-8-sig")
    identity.to_csv(out/"decomposition_coefficient_identity.csv",index=False,encoding="utf-8-sig");dump(out/"SUPPLEMENT_AST_PROVENANCE.json",prov)
    dump(out/"RAW_FINANCIAL_DECOMPOSITION_CHECKS.json",{"raw":rawchecks,"decomposition":decompchecks,"bh19":19,"no_ATt_three_rows_kept_in_family_but_not_paper_display":True})
    annual=d.groupby("year",as_index=False)[["BMI_raw","NonBMI_raw","TotalDigital_raw"]].mean();annual.rename(columns={"BMI_raw":"BMODD_raw_mean","NonBMI_raw":"NonBMODD_raw_mean"}).to_csv(out/"figure1_annual_mean_source_data.csv",index=False,encoding="utf-8-sig")
    # Seal every fitted model/data output before opening any prior coefficients.
    outputs={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and "_matplotlib_cache" not in p.parts}
    dump(out/"MODEL_OUTPUTS_SEAL.json",{"status":"ALL_CURRENT_MODEL_OUTPUTS_SEALED","files":outputs,"reference_coefficients_opened_before_seal":False})
    refs={}
    for role,info in cfg["posthoc_models"].items():
        p=resolve(base,info["path"])
        if sha(p)!=info["sha256"]:raise ValueError("Posthoc model reference pin failed")
        refs[role]=pd.read_csv(p)
    checks=[]
    compare_frame(primary,refs["primary"],["model_id","term"],"H1_H2",checks)
    compare_frame(wald,refs["wald"],["model_id","contrast"],"H2_WALD",checks)
    ca=pd.read_csv(core/"core_coefficients.csv");expected_core=refs["core"].loc[refs["core"]["model_id"].isin(CORE_IDS)].copy()
    extra=refs["h45"].loc[refs["h45"]["term"].eq("BMI_z")].copy();extra["model_id"]=extra["model_id"].str.replace("EXPLORATORY_horizon_specific_","TIMING_",regex=False)
    expected_core=pd.concat([expected_core,extra],ignore_index=True)
    if set(ca["model_id"])!=CORE_IDS:raise ValueError("Current core specification set changed")
    reference_keys=set(zip(expected_core["model_id"],expected_core["term"]))
    ca=ca.loc[[pair in reference_keys for pair in zip(ca["model_id"],ca["term"])]]
    compare_frame(ca,expected_core,["model_id","term"],"CURRENT_CORE",checks)
    compare_frame(pd.read_csv(h3/"h3_joint_coefficients.csv"),refs["joint"],["horizon"],"H3_COMMON",checks)
    compare_frame(pd.read_csv(h3/"h3_joint_differences.csv"),refs["differences"],["high_horizon","low_horizon"],"H3_DIFFERENCES",checks)
    compare_frame(pd.DataFrame([tr]),refs["trend"].loc[refs["trend"]["model_id"].eq("M4_MODEL_B")],["model_id"],"TREND",checks)
    compare_frame(pd.read_csv(out/"placebo.csv"),refs["placebo"],["model_id","term"],"PLACEBO",checks)
    compare_frame(pd.read_csv(ipw/"08_样本留存LPM与IPW结果.csv"),refs["ipw"],["model_id","term"],"IPW",checks)
    compare_frame(all_supp,refs["supplement"],["model_id"],"SUPPLEMENT_BH19",checks)
    if any(sha(out/p)!=h for p,h in outputs.items()):raise ValueError("Sealed actual model output changed during comparison")
    receipt={"status":"CURRENT_PAPER_MODEL_REPLAY_PASS","source_counts_from_actual_fresh_predictions":True,"non_text_finance_changed":False,"checks":checks,"model_output_files":outputs,"coefficient_tolerance":1e-10,"sample_count_tolerance":0,"BH_family_unchanged":19,"capital_market_models_run":False}
    dump(out/"POSTHOC_MODEL_VALIDATION.json",receipt)
    return {"status":receipt["status"],"panel_sha256":sha(panel),"validation_sha256":sha(out/"POSTHOC_MODEL_VALIDATION.json"),"model_seal_sha256":sha(out/"MODEL_OUTPUTS_SEAL.json")}
