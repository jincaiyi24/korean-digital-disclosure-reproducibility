"""고정된 1,500/300/300 분할과 네 시드의 한국어 BERT 비교를 검증·재실행한다.

verify/evaluate는 저장된 예측을 사용하며 학습이나 모델 추론을 하지 않는다.
train은 별도 인간 참조 파일을 열지 않고 기존 학습 연산을 그대로 호출한다.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata

LABELS=("BMODD","NonBMODD","Exclude")
SEEDS=(20261022,20261023,20261024,20261025)
STRATA=("BMODD","NonBMODD","candidate_Exclude","noncandidate_valid")
SCHEMES=("unweighted","design_weighted")
METRICS=("Accuracy","Macro_F1","BMODD_Precision","BMODD_Recall","BMODD_F1","NonBMODD_Precision","NonBMODD_Recall","NonBMODD_F1","Exclude_Precision","Exclude_Recall","Exclude_F1")
GROUPS=("python_rules","bert_four_seed_mean","finbert_four_seed_mean")
METHODS=("python_rules",)+tuple(f"bert_seed_{s}" for s in SEEDS)+tuple(f"finbert_seed_{s}" for s in SEEDS)
ALIASES={"BMI":"BMODD","NonBMI":"NonBMODD",**{x:x for x in LABELS}}
BASE_SHA="189f135a951e92bd6522947c12b6b348da2889458c9823257f4a731adcd49614"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(s):return hashlib.sha256(s.encode("utf-8")).hexdigest()
def squeeze(s):return re.sub(r"\s+","",unicodedata.normalize("NFKC",s)).casefold()
def read_csv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def load(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def output_csv(p,rows):
    with Path(p).open("x",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def safe(root,p):
    rel=Path(p);v=(root/rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not v.is_relative_to(root):raise ValueError("Repository-relative path required")
    return v
def require(condition,message):
    if not condition:raise ValueError(message)
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def check_probabilities(rows,source):
    require(len(rows)==len(source),"Prediction row count differs")
    for p,r in zip(rows,source,strict=True):
        for key in ("sentence_id","corp_code","year","rcept_no","sentence_sha256","input_text_sha256"):
            require(p[key]==r[key],"Prediction/source alignment differs: "+key)
        require(p["rule_pseudo_label"]==r["label"],"Teacher target changed")
        probs=[float(p["probability_"+x]) for x in LABELS]
        require(all(math.isfinite(x) and 0<=x<=1 for x in probs) and abs(sum(probs)-1)<1e-5,"Invalid probabilities")
        require(p["prediction"]==LABELS[max(range(3),key=lambda j:probs[j])],"Prediction is not probability argmax")

def verify(root,*,read_human=True,check_saved_runs=True):
    cfg=load(root/"설명서/벤치마크_고정설정.json")
    manifest=load(root/"설명서/벤치마크_파일해시.json")
    for name,digest in manifest["files"].items():require(sha(safe(root,name))==digest,"Published file hash changed: "+name)
    require(cfg["seeds"]==list(SEEDS) and cfg["labels"]==list(LABELS),"Seed or label mapping changed")
    splits={k:read_csv(safe(root,p)) for k,p in cfg["splits"].items()}
    corp={};norm={};ids=set()
    for split,n in (("train",1500),("dev",300),("test",300)):
        rows=splits[split];require(len(rows)==n,"Fixed split count changed")
        corp[split]=set();norm[split]=set()
        for r in rows:
            require(r["sentence_id"] not in ids and r["split"]==split,"Duplicate ID/wrong split")
            ids.add(r["sentence_id"]);corp[split].add(r["corp_code"]);norm[split].add(th(squeeze(r["sentence"])))
            require(re.fullmatch(r"\d{8}",r["corp_code"]) and re.fullmatch(r"\d{14}",r["rcept_no"]),"Malformed source identity")
            require(th(r["sentence"])==r["sentence_sha256"],"Original sentence changed")
            require(th(squeeze(r["sentence"]))==r["normalized_sentence_sha256"],"Normalized sentence hash changed")
            require(r["input_text"]==cfg["input_template"].format(**r) and th(r["input_text"])==r["input_text_sha256"],"Model input changed")
            require(r["label"] in LABELS and r["current_rule_label"]==r["label"] and not r.get("human_label"),"Training targets must remain rule pseudo-labels")
        require(len(norm[split])==n,"Duplicate normalized text inside split")
    for a,b in (("train","dev"),("train","test"),("dev","test")):
        require(not corp[a]&corp[b] and not norm[a]&norm[b],"Company/text leakage across splits")
    require(len(corp["test"])==248,"Original fixed test corporations changed")
    require([r["sample_id"] for r in splits["test"]]==[f"N{i:03d}" for i in range(1,301)],"Fixed test order changed")
    old={r["normalized_sentence_sha256"] for r in read_csv(safe(root,cfg["previous_sentence_hashes"]))}
    require(not (norm["train"]|norm["dev"])&old,"Previous experiment text entered development")
    result={"config":cfg,"splits":splits,"root":root,"verified_file_count":len(manifest["files"])}
    if check_saved_runs:
        base=module(root/"코드/벤치마크_원본학습연산.py","original_benchmark_operators")
        require(sha(base.__file__)==BASE_SHA,"Original training operators changed")
        for role in ("bert","finbert"):
            for seed in SEEDS:
                run=cfg["runs"][f"{role}_{seed}"];scores=[]
                for epoch,p in enumerate(run["dev_predictions"],1):
                    rows=read_csv(safe(root,p));check_probabilities(rows,splits["dev"])
                    scores.append(base.metrics([r["label"] for r in splits["dev"]],[r["prediction"] for r in rows])["macro_f1"])
                best=max(range(1,4),key=lambda x:scores[x-1]);require(best==run["best_epoch"],"Dev-selected epoch differs")
                require(math.isclose(scores[best-1],run["dev_macro_f1"],abs_tol=1e-12),"Stored development score differs")
                check_probabilities(read_csv(safe(root,run["test_predictions"])),splits["test"])
    if read_human:
        gold=read_csv(safe(root,cfg["human_reference"]));require(len(gold)==300,"Human reference count changed")
        for g,t in zip(gold,splits["test"],strict=True):
            for key in ("sample_id","sentence_sha256","corp_code","year","rcept_no"):require(g[key]==t[key],"Human/test identity mismatch")
            require(g["human_label"] in LABELS,"Invalid final human reference")
            stratum=ALIASES.get(g["validation_stratum"],g["validation_stratum"])
            nh=cfg["strata"][stratum]["n_h"];Nh=cfg["strata"][stratum]["N_h"]
            require(int(g["N_h"])==Nh and int(g["n_h"])==nh==75,"Published stratum population/sample count differs")
            require(math.isclose(float(g["design_weight"]),Nh/nh,rel_tol=1e-14,abs_tol=1e-10),"Design weight is not original N_h/75")
        from collections import Counter
        require(Counter(ALIASES.get(g["validation_stratum"],g["validation_stratum"]) for g in gold)=={s:75 for s in STRATA},"Original strata changed")
        require(Counter(g["human_label"] for g in gold)=={"BMODD":75,"NonBMODD":86,"Exclude":139},"Final human labels changed")
        result["gold_rows"]=gold
    return result

def training_binding(root,cfg):
    return {"published_config_sha256":sha(root/"설명서/벤치마크_고정설정.json"),
            "training_operators_sha256":sha(root/"코드/벤치마크_원본학습연산.py"),
            "fixed_split_sha256":{k:sha(safe(root,p)) for k,p in cfg["splits"].items()},
            "labels":list(LABELS),"input_template":cfg["input_template"],
            "training":cfg["training"],
            "models":{role:{k:cfg["models"][role][k] for k in ("model_id","revision")} for role in ("bert","finbert")}}

def external_run_predictions(root,cfg,splits,runs):
    """Verify a complete external eight-run output before human scoring.

    Development scores are recomputed from that execution's own predictions.
    They are not required to equal the archived experiment's scores.
    """
    complete=load(runs/"전체_학습실행기록.json")
    require(complete.get("status")=="EIGHT_FRESH_MODEL_RUNS_COMPLETE","External eight-run training receipt is missing or incomplete")
    require(complete.get("binding")==training_binding(root,cfg),"External training inputs/config/operators differ")
    require(complete.get("human_reference_loaded_for_training") is False and complete.get("actual_test_rows")==300,"External test/human-label contract differs")
    intended=[(role,seed) for seed in SEEDS for role in ("bert","finbert")]
    records=complete.get("runs",[])
    require([(r.get("role"),r.get("seed")) for r in records]==intended,"External experiment must contain all eight unique role/seed runs")
    if complete.get("receipt_origin")=="READ_ONLY_MIRROR_OF_PREVIOUS_ACTUAL_TRAINING_NOT_NEW_TRAINING":
        source=runs/"SOURCE_ACTUAL_PAIRED_EXECUTION_RECEIPT.json"
        require(sha(source)==complete.get("source_actual_execution_receipt_sha256"),"Mirror lacks original actual execution evidence")
        original=load(source)
        require(original.get("status")=="ALL_EIGHT_HOLDOUT300_PAIRED_MODEL_RUNS_COMPLETE" and original.get("models_completed")==8 and original.get("runs")==records,"Mirror is not the eight original completed runs")
    else:require(complete.get("receipt_origin")=="PUBLIC_PORTABLE_TRAIN_ENTRY","Unknown external execution provenance")
    base=module(root/"코드/벤치마크_원본학습연산.py","external_development_metrics")
    require(sha(base.__file__)==BASE_SHA,"Original training operators changed")
    paths={};checks=[];paired_heads={}
    for stored in records:
        role=stored["role"];seed=stored["seed"];directory=runs/role/f"seed_{seed}"
        require(load(directory/"MODEL_SEED_RECEIPT.json")==stored,"External per-seed receipt differs from complete receipt")
        require(stored.get("status")=="PAIRED_MODEL_SEED_COMPLETE" and stored.get("test_rows")==300 and stored.get("test_evaluation_passes")==1 and stored.get("old_trained_state_reused") is False,"External seed did not complete one fixed test pass")
        require(all(stored.get(k)==cfg["models"][role][k] for k in ("model_id","revision")),"External pretrained model revision differs")
        if seed in paired_heads:require(stored["new_pooler_and_classifier_state_sha256"]==paired_heads[seed],"External same-seed common head initialization differs")
        else:paired_heads[seed]=stored["new_pooler_and_classifier_state_sha256"]
        best_path=directory/"BEST_MODEL_SEAL.json";best=load(best_path)
        require(sha(best_path)==stored["best_model_seal_sha256"] and best.get("status")=="DEV_SELECTED_BEST_CHECKPOINT_SEALED_BEFORE_TEST","External best checkpoint selection was not sealed")
        require(best.get("checkpoint_files")==stored.get("checkpoint_files") and best.get("epochs_completed")==3 and best.get("test_evaluations")==0,"External best-checkpoint receipt differs")
        scores=[]
        for epoch in (1,2,3):
            rows=read_csv(directory/f"dev_predictions_epoch_{epoch:02d}.csv");check_probabilities(rows,splits["dev"])
            score=base.metrics([r["label"] for r in splits["dev"]],[r["prediction"] for r in rows])["macro_f1"]
            require(abs(score-float(load(directory/f"dev_metrics_epoch_{epoch:02d}.json")["macro_f1"]))<1e-12,"External development metric differs from its predictions")
            scores.append(score)
        selected=max(range(1,4),key=lambda e:scores[e-1])
        require(selected==best["best_epoch"]==stored["best_epoch"] and abs(scores[selected-1]-float(stored["dev_macro_f1"]))<1e-12,"External checkpoint was not selected by its own dev Macro F1")
        path=directory/"test_predictions.csv";test_rows=read_csv(path);check_probabilities(test_rows,splits["test"])
        actual_rule_metrics=base.metrics([r["label"] for r in splits["test"]],[r["prediction"] for r in test_rows])
        require(actual_rule_metrics==stored.get("test_metrics"),"External test predictions do not match their recorded rule-reference metrics")
        paths[f"{role}_{seed}"]=path
        checks.append({"role":role,"seed":seed,"best_epoch":selected,"dev_macro_f1":scores[selected-1],"test_predictions_sha256":sha(path)})
    return paths,{"external_runs_receipt_sha256":sha(runs/"전체_학습실행기록.json"),"receipt_origin":complete["receipt_origin"],"eight_runs_and_dev_selection_verified":True,"checks":checks,"checkpoint_tensor_bytes_reverified_during_metric_evaluation":False}

def evaluate(root,out,runs=None):
    import numpy as np
    d=verify(root,check_saved_runs=runs is None);cfg=d["config"];test=d["splits"]["test"];gold=d["gold_rows"]
    require(np.__version__==cfg["versions"]["numpy"],"Original NumPy 2.2.6 required for the fixed bootstrap")
    operators=module(root/"코드/벤치마크_원본평가연산.py","original_benchmark_metrics")
    operators.STRATA=STRATA
    external_paths,external_evidence=external_run_predictions(root,cfg,d["splits"],runs) if runs else ({},None)
    pred=[[LABELS.index(r["label"]) for r in test]]
    for role in ("bert","finbert"):
        for seed in SEEDS:
            path=external_paths[f"{role}_{seed}"] if runs else safe(root,cfg["runs"][f"{role}_{seed}"]["test_predictions"])
            pred.append([LABELS.index(r["prediction"]) for r in read_csv(path)])
    data={"gold":np.asarray([LABELS.index(g["human_label"]) for g in gold]),"predictions":np.asarray(pred),"weights":np.asarray([float(g["design_weight"]) for g in gold]),"strata":np.asarray([ALIASES.get(g["validation_stratum"],g["validation_stratum"]) for g in gold])}
    results=operators.calculate_results(data)
    require(not out.exists(),"Fresh output directory required");out.mkdir(parents=True)
    seed_rows=[];summary=[];conf=[];seed_ci=[];paired=[]
    for si,scheme in enumerate(SCHEMES):
        for mi,method in enumerate(METHODS):
            seed_rows.append({"weighting":scheme,"method":method,**{k:float(v) for k,v in zip(METRICS,results["point_seed"][si,mi],strict=True)}})
            ci_seed=np.quantile(results["bootstrap_seed_metrics"][si,:,mi],(.025,.975),axis=0,method="linear")
            for j,k in enumerate(METRICS):seed_ci.append({"weighting":scheme,"method":method,"metric":k,"point_estimate":float(results["point_seed"][si,mi,j]),"ci_95_low":float(ci_seed[0,j]),"ci_95_high":float(ci_seed[1,j])})
            for a in range(3):
                for b in range(3):conf.append({"weighting":scheme,"method":method,"human_reference_class":LABELS[a],"prediction_class":LABELS[b],"count":float(results["point_matrices"][si,mi,a,b])})
        for gi,group in enumerate(GROUPS):
            ci=np.quantile(results["bootstrap_group_metrics"][si,:,gi],(.025,.975),axis=0,method="linear")
            for j,k in enumerate(METRICS):summary.append({"weighting":scheme,"method":group,"metric":k,"mean":float(results["point_groups"][si,gi,j]),"ci_95_low":float(ci[0,j]),"ci_95_high":float(ci[1,j])})
            if gi:
                difference=results["bootstrap_differences"][si,:,gi-1];ci_diff=np.quantile(difference,(.025,.975),method="linear")
                paired.append({"weighting":scheme,"comparison":group+"_minus_python_rules","metric":"Macro_F1","point_difference":float(results["point_groups"][si,gi,1]-results["point_groups"][si,0,1]),"paired_ci_95_low":float(ci_diff[0]),"paired_ci_95_high":float(ci_diff[1])})
    output_csv(out/"개별시드_인간참조지표.csv",seed_rows);output_csv(out/"세방법_인간참조지표와신뢰구간.csv",summary);output_csv(out/"혼동행렬.csv",conf)
    output_csv(out/"개별시드_지표신뢰구간.csv",seed_ci);output_csv(out/"규칙법대비_MacroF1_쌍체차이.csv",paired)
    comparisons=[];differences=[]
    if cfg.get("reference_outputs"):
        expected=read_csv(safe(root,cfg["reference_outputs"]["seed_metrics"]))
        for a,b in zip(seed_rows,expected,strict=True):
            require(a["weighting"]==b["weighting"] and a["method"]==b["method"],"Reference metric row order differs")
            for metric in METRICS:
                delta=a[metric]-float(b[metric])
                if runs:differences.append({"scope":"seed_point","weighting":a["weighting"],"method":a["method"],"metric":metric,"quantity":"point_estimate","new_value":a[metric],"archived_value":float(b[metric]),"difference":delta})
                else:require(abs(delta)<1e-12,"Saved human-reference metric differs")
        expected_ci=read_csv(safe(root,cfg["reference_outputs"]["group_ci"]))
        for a,b in zip(summary,expected_ci,strict=True):
            require(all(a[k]==b[k] for k in ("weighting","method","metric")),"CI reference order differs")
            for x,y in (("mean","point_estimate"),("ci_95_low","ci_95_low"),("ci_95_high","ci_95_high")):
                delta=a[x]-float(b[y])
                if runs:differences.append({"scope":"group_summary","weighting":a["weighting"],"method":a["method"],"metric":a["metric"],"quantity":x,"new_value":a[x],"archived_value":float(b[y]),"difference":delta})
                else:require(abs(delta)<1e-12,"Published confidence interval differs")
        if not runs:comparisons.append("All 198 seed-level metric values and 66 group confidence intervals match saved human evaluation")
    if differences:output_csv(out/"보관결과대비_새실행차이.csv",differences)
    receipt={"status":"FIXED300_HUMAN_METRICS_RECOMPUTED","saved_prediction_rows":2400,"model_fits":0,"model_forwards":0,"bootstrap_replicates":5000,"bootstrap_seed":20261025,"prediction_source":"external_training_runs" if runs else "archived_predictions","external_run_verification":external_evidence,"reference_outputs_checked":comparisons,"archived_comparison_max_absolute_difference":max((abs(r["difference"]) for r in differences),default=None),"new_run_differences_are_retained_not_repaired":bool(runs),"bootstrap_indices_sha256":hashlib.sha256(results["bootstrap_indices"].astype("<i2",copy=False).tobytes(order="C")).hexdigest(),"scope":"설계가중 평가는 사후 중복제거 검증 표본틀에 한정; 네 시드 지표의 평균이며 앙상블 아님"}
    require(receipt["bootstrap_indices_sha256"]=="fc439eebfb7b9dfd3971ff78b7755dfcae4cd207cff964bc2e5de39f97c62f54","Original bootstrap draw order changed")
    dump(out/"재계산_기록.json",receipt);return receipt

def check_models(root,cfg,model_root):
    for role,info in cfg["models"].items():
        for item in info["verified_files"]:
            p=model_root/role/item["file"]
            require(p.is_file() and p.stat().st_size==item["size"] and sha(p)==item["sha256"],"Pinned model file differs: "+role+"/"+item["file"])

def normalize_safe_metadata(path,revision):
    """Fix only metadata key order; preserve header size and all tensor bytes.

    safetensors 0.6.2 serializes its metadata map in a nondeterministic order.
    The original trained file used format, then source_revision. Explicitly keep
    that order so repeated downloads produce the recorded file SHA as well as
    the identical tensor values. This is a container-format operation only.
    """
    import struct
    with Path(path).open("r+b") as f:
        size_bytes=f.read(8);require(len(size_bytes)==8,"Invalid safetensors prefix")
        size=struct.unpack("<Q",size_bytes)[0];raw=f.read(size)
        require(len(raw)==size,"Truncated safetensors header")
        header=json.loads(raw);metadata=header.get("__metadata__")
        require(metadata=={"format":"pt","source_revision":revision},"Unexpected safe metadata")
        offsets={k:v["data_offsets"] for k,v in header.items() if k!="__metadata__"}
        def payload_hash():
            f.seek(8+size);h=hashlib.sha256()
            for block in iter(lambda:f.read(1<<20),b""):h.update(block)
            return h.hexdigest()
        before_payload=payload_hash()
        header["__metadata__"]={"format":"pt","source_revision":revision}
        canonical=json.dumps(header,ensure_ascii=False,separators=(",",":")).encode("utf-8")
        require(len(canonical)<=size,"Header normalization would change tensor offsets")
        f.seek(8);f.write(canonical+b" "*(size-len(canonical)))
        f.flush();f.seek(8);after_header=json.loads(f.read(size));after_payload=payload_hash()
        require(after_header==header and {k:v["data_offsets"] for k,v in after_header.items() if k!="__metadata__"}==offsets,"Tensor offset/header semantics changed")
        require(before_payload==after_payload,"Tensor payload changed during metadata ordering")
    return {"fixed_metadata_order":["format","source_revision"],"header_size_unchanged":True,"all_data_offsets_unchanged":True,"tensor_payload_sha256_before":before_payload,"tensor_payload_sha256_after":after_payload,"tensor_payload_not_rewritten":True}

def convert_finbert(bin_path,safe_path,revision):
    import torch
    from safetensors.torch import save_file,load_file
    require(tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])>=(2,6),"torch>=2.6 weights_only required")
    state=torch.load(bin_path,weights_only=True,map_location="cpu")
    require(isinstance(state,dict) and state and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()),"Only named tensor state accepted")
    for k,s in {"bert.embeddings.word_embeddings.weight":(20000,768),"bert.embeddings.position_embeddings.weight":(512,768),"bert.encoder.layer.11.output.dense.weight":(768,3072)}.items():require(k in state and tuple(state[k].shape)==s,"Pinned FinBERT structure differs")
    require(not safe_path.exists(),"Converted model exists")
    converted={k:v.detach().cpu().contiguous().clone() for k,v in sorted(state.items())}
    part=safe_path.with_name(safe_path.name+".part");require(not part.exists(),"Partial conversion already exists")
    save_file(converted,str(part),metadata={"format":"pt","source_revision":revision})
    normalization=normalize_safe_metadata(part,revision)
    restored=load_file(str(part),device="cpu");require(set(restored)==set(state),"Converted keys differ")
    for k,v in state.items():require(v.dtype==restored[k].dtype and v.shape==restored[k].shape and torch.equal(v.cpu(),restored[k]),"Converted tensor differs")
    del restored;gc.collect();os.link(part,safe_path,follow_symlinks=False);part.unlink()
    return {"tensor_count":len(state),"all_keys_shapes_dtypes_values_exact":True,"source_bin_sha256":sha(bin_path),"converted_sha256":sha(safe_path),"container_metadata_normalization":normalization}

def download_models(root,model_root):
    from huggingface_hub import hf_hub_download
    cfg=load(root/"설명서/벤치마크_고정설정.json")
    for role,info in cfg["models"].items():
        dest=model_root/role;dest.mkdir(parents=True,exist_ok=False)
        for item in info["verified_files"]:
            if role=="finbert" and item["file"]=="model.safetensors":continue
            hf_hub_download(repo_id=info["model_id"],revision=info["revision"],filename=item["file"],local_dir=dest)
            require(sha(dest/item["file"])==item["sha256"],"Downloaded model file hash differs")
        if role=="finbert":convert_finbert(dest/"pytorch_model.bin",dest/"model.safetensors",info["revision"])
    check_models(root,cfg,model_root)
    return {"status":"PINNED_PRETRAINED_FILES_VERIFIED","model_root":str(model_root)}

def train(root,model_root,out):
    # The human-reference CSV is deliberately not parsed in this execution path.
    d=verify(root,read_human=False,check_saved_runs=False);cfg=d["config"]
    require(sys.version_info[:3]==(3,12,14),"Original Python 3.12.14 required")
    for k,v in cfg["versions"].items():require(importlib.metadata.version(k)==v,"Original dependency version required: "+k)
    check_models(root,cfg,model_root)
    require(not out.exists(),"Fresh training output required")
    base_path=root/"코드/벤치마크_원본학습연산.py";require(sha(base_path)==BASE_SHA,"Original training operators changed")
    base=module(base_path,"original_training_operators")
    def loader(dataset,split,requested_n):
        require(Path(dataset).resolve()==root and split in d["splits"],"Unexpected split/path")
        require(requested_n==({"train":1500,"dev":300,"test":900}[split]),"Unexpected original split request")
        return [dict(r) for r in d["splits"][split]]
    base.load_split=loader;base.disable_network();out.mkdir(parents=True)
    class Args:dataset=root
    completed=[];active=None;binding=training_binding(root,cfg)
    try:
        for seed in SEEDS:
            heads=None
            for role in ("bert","finbert"):
                require(training_binding(root,cfg)==binding,"Fixed training inputs changed during execution")
                check_models(root,cfg,model_root)
                active={"role":role,"seed":seed};info={**cfg["models"][role],"model_dir":str(model_root/role)}
                r=base.run_one(role,info,seed,Args(),{"training":cfg["training"]},out/role/f"seed_{seed}",expected_initial_heads=heads)
                completed.append(r);heads=r["new_pooler_and_classifier_state_sha256"]
        require(training_binding(root,cfg)==binding,"Fixed training inputs changed before completion")
        check_models(root,cfg,model_root)
    except BaseException as e:
        dump(out/"FAILED_학습실행기록.json",{"status":"FAILED_NOT_A_COMPLETE_EIGHT_RUN_EXPERIMENT","binding":binding,"active_run":active,"completed_role_seeds":[{"role":r["role"],"seed":r["seed"]} for r in completed],"error_type":type(e).__name__,"human_reference_loaded_for_training":False})
        raise
    receipt={"status":"EIGHT_FRESH_MODEL_RUNS_COMPLETE","receipt_origin":"PUBLIC_PORTABLE_TRAIN_ENTRY","binding":binding,"runs":completed,"human_reference_loaded_for_training":False,"actual_test_rows":300,"historical_base_test_request":900,"explicit_fixed300_loader_adapter":True}
    dump(out/"전체_학습실행기록.json",receipt);return {"status":receipt["status"],"runs":len(completed)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("verify","evaluate","download-models","train"),nargs="?",default="verify")
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--out",type=Path);p.add_argument("--models",type=Path);p.add_argument("--runs",type=Path,help="evaluate: 새 train 실행의 외부 출력 경로; 생략하면 공개된 원래 예측을 재계산")
    a=p.parse_args();root=a.root.resolve()
    if a.command=="verify":
        d=verify(root);r={"status":"FIXED_SPLITS_AND_EIGHT_SAVED_RUNS_VERIFIED","splits":{k:len(v) for k,v in d["splits"].items()},"verified_files":d["verified_file_count"],"training_runs":0,"model_forwards":0}
    elif a.command=="evaluate":
        require(a.out is not None,"--out required");r=evaluate(root,a.out.resolve(),a.runs.resolve() if a.runs else None)
    elif a.command=="download-models":
        require(a.models is not None,"--models required");r=download_models(root,a.models.resolve())
    else:
        require(a.models is not None and a.out is not None,"--models and --out required");r=train(root,a.models.resolve(),a.out.resolve())
    print(json.dumps(r,ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
