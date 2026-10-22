"""Serial fresh-process, fresh-output OFFLINE replays; first failure stops.

Every individual round must already have an exact root approval. This command
never fabricates approvals, shares predictions/checkpoints, or retries a failed
round. It does not create a repository or upload anything.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p,v):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def verify_completed_round(directory):
    p=directory/"FULL_CURRENT_METHOD_RUN_RECEIPT.json"
    r=json.loads(p.read_text(encoding="utf-8"))
    if r["status"]!="CURRENT_METHOD_REPLAY_PASS" or not r["protected_unchanged"] or r["reports"]!=22654:raise ValueError("Round is incomplete")
    seal=json.loads((directory/"classification/PREDICTION_SEAL.json").read_text(encoding="utf-8"))
    if seal["status"]!="FRESH_PREDICTIONS_SEALED" or seal["completed"]!=22654 or seal["classifier_calls"]<=0 or seal["prediction_checkpoint_reused"] or seal["remaining_temp_files"]:raise ValueError("Fresh complete classification was not proved")
    if sha(directory/"classification/fresh_report_predictions.jsonl")!=seal["prediction_file_sha256"]:raise ValueError("Sealed prediction bytes changed")
    ms=json.loads((directory/"empirical/MODEL_OUTPUTS_SEAL.json").read_text(encoding="utf-8"))
    if ms["status"]!="ALL_CURRENT_MODEL_OUTPUTS_SEALED" or ms["reference_coefficients_opened_before_seal"]:raise ValueError("Model seal missing")
    if any(sha(directory/"empirical"/name)!=h for name,h in ms["files"].items()):raise ValueError("Sealed model output changed")
    # Code-provenance JSON contains source/output paths; exclude it, not numerical
    # data. All numeric CSV bytes, panel bytes, plots, and sample keys must agree.
    stable={name:h for name,h in ms["files"].items() if name.endswith((".csv",".png"))}
    return {"round_receipt_sha256":sha(p),"measurement_digest":seal["ordered_measurement_sha256"],"classifier_calls":seal["classifier_calls"],"panel_sha256":r["model_receipt"]["panel_sha256"],"numeric_and_plot_files":stable}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--inputs",required=True,type=Path);p.add_argument("--approval-plan",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();runner=Path(__file__).with_name("rebuild_current_method.py")
    plan=json.loads(a.approval_plan.read_text(encoding="utf-8"))
    required={"coordinator_sha256":sha(__file__),"runner_sha256":sha(runner),"inputs_sha256":sha(a.inputs),"output":str(a.output.resolve()),"rounds":3,"offline":True}
    if any(plan.get(k)!=v for k,v in required.items()) or len(plan.get("approvals",[]))!=3:raise ValueError("Coordinator requires an exact three-round root approval plan")
    if a.output.exists():raise FileExistsError("Three-round output root must not exist")
    for i,item in enumerate(plan["approvals"],1):
        approval=Path(item["path"]).resolve()
        if sha(approval)!=item["sha256"]:raise ValueError("Round approval bytes changed")
        row=json.loads(approval.read_text(encoding="utf-8"))
        target={"runner_sha256":sha(runner),"inputs_sha256":sha(a.inputs),"output":str((a.output/f"round_{i:02d}").resolve()),"allow_full_locked_original_replay":True}
        if any(row.get(k)!=v for k,v in target.items()):raise ValueError("A round approval targets different code/inputs/output")
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    dump(a.output/"THREE_ROUND_START.json",{**required,"utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"approval_plan_sha256":sha(a.approval_plan)})
    env={k:v for k,v in os.environ.items() if k not in {"PYTHONPATH","PYTHONHOME"}}
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    try:
        for i,item in enumerate(plan["approvals"],1):
            directory=a.output/f"round_{i:02d}"
            command=[sys.executable,"-B",str(runner),"--inputs",str(a.inputs.resolve()),"--mode","run","--output",str(directory.resolve()),"--approval",str(Path(item["path"]).resolve()),"--workers","8"]
            with (a.output/f"round_{i:02d}_console.log").open("x",encoding="utf-8",newline="\n") as log:
                cp=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
            if cp.returncode:raise RuntimeError(f"Round {i} exited {cp.returncode}; next rounds NOT STARTED")
            record=verify_completed_round(directory);record["round"]=i;records.append(record)
            dump(a.output/f"ROUND_{i:02d}_VERIFIED.json",record)
            if i>1 and any(record[k]!=records[0][k] for k in ("measurement_digest","classifier_calls","panel_sha256","numeric_and_plot_files")):raise ValueError("Round outputs differ; no majority vote or replacement run allowed")
        dump(a.output/"THREE_ROUND_RECEIPT.json",{"status":"THREE_FRESH_OFFLINE_CURRENT_METHOD_REPLAYS_PASS","rounds":records,"network_requests":0,"fresh_website_or_financial_download":False,"historical_old_measurement_recovered":False,"publication":"REQUIRES_SEPARATE_RIGHTS_PRIVACY_DOCUMENTATION_REVIEW"})
    except BaseException as error:
        dump(a.output/"THREE_ROUND_FAILED.json",{"status":"STOPPED_NO_PUBLISH","error":str(error),"completed_verified_rounds":records});raise


if __name__=="__main__":main()
