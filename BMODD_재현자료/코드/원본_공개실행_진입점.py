"""Portable layer for the SAME hash-locked current-method execution core.

No Desktop/HIST paths or hand-edited approval JSON are needed. Provide locked
raw sources under a data root; optionally provide a finance root. Without
--run this stages and checks only. --run is the user's explicit authorization
for local offline computation. This layer never downloads or publishes data.

Recommended external data layout:
  DATA/raw_dart/*.zip
  DATA/supplemental_raw/*.zip
  DATA/viewer_capture/report_*/http_responses/* (locked raw HTML only)
Path-free published viewer descriptors are supplied by BUNDLE, not manually
reconstructed or copied from private historical capture receipts.
Finance layout (default BUNDLE/finance or explicit --finance-root):
  expanded_panel.csv, frozen_augmented_panel.csv
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PINS={"runner":"db797d939967554599ca822b8206bb0343a86b51c327ebb0c1ff83c1446b381c","models":"ceba9ae9a8a8b982d2bb5457d29c876d0e4b9d66c40c7f89d21a3dc47d6d9dbf","coordinator":"f2793c354fc9be7e91711adb70753ef4280d106d2cd0a4595f4c6c743bea24e9","template":"7435acc785505c1a8af15cc83e5f687ce30d6300993a77f3a6f6e47a256cb58d"}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def dump(p,v):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def resolve(base,value):
    p=Path(value);return p.resolve() if p.is_absolute() else (base/p).resolve()


def bundled_child(root,value):
    p=Path(value)
    if p.is_absolute() or ".." in p.parts:raise ValueError("A package reference must be a relative child")
    result=(root/p).resolve()
    if not result.is_relative_to(root.resolve()):raise ValueError("A package reference escaped its bundle")
    return result


def copied(src,dst,expected):
    if sha(src)!=expected:raise ValueError("A packaged component/input changed: "+src.name)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError("Portable staging never overwrites a file")
    shutil.copy2(src,dst)
    if sha(dst)!=expected:raise ValueError("Copied component bytes changed")


def stage(bundle,data_root,finance_root,out,source_overrides=None):
    """Paths are adapted, while all parser/classifier/model source bytes stay same."""
    template=bundle/"PUBLIC_INPUT_TEMPLATE.json"
    if sha(template)!=PINS["template"]:raise ValueError("Public bundle template is not the reviewed current-method version")
    cfg=json.loads(template.read_text(encoding="utf-8"))
    finance_root=finance_root.resolve()
    inputs={}
    if data_root:
        data_root=data_root.resolve();inputs={"raw_dir":data_root/"raw_dart","supplemental_raw_dir":data_root/"supplemental_raw","viewer_capture_root":data_root/"viewer_capture"}
    inputs.update({k:Path(v).resolve() for k,v in (source_overrides or {}).items() if v is not None})
    if set(inputs)!={"raw_dir","supplemental_raw_dir","viewer_capture_root"}:raise ValueError("Provide --data-root or all three explicit source directory options")
    for p in [*inputs.values(),finance_root]:
        if not p.is_dir():raise FileNotFoundError("Required offline input directory missing: "+str(p))
        if out.resolve().is_relative_to(p.resolve()):raise ValueError("Output cannot be inside an input directory")
    if out.exists():raise FileExistsError("A portable execution must start in a nonexistent output directory")
    required=[("financial_panel","expanded_panel.csv"),("frozen_financial_panel","frozen_augmented_panel.csv")]
    for role,name in required:
        if sha(finance_root/name)!=cfg[role]["sha256"]:raise ValueError("Financial snapshot identity differs; no automatic source substitution")
    # Verify all packaged core bytes before creating the working tree.
    pipeline={Path(relative).name:(resolve(bundle,relative),h) for relative,h in cfg["pipeline_pins"].items()}
    if len(pipeline)!=len(cfg["pipeline_pins"]):raise ValueError("Duplicate component basenames")
    if any(not path.is_relative_to((bundle.parent/"code").resolve()) for path,_ in pipeline.values()):raise ValueError("Pipeline component escaped packaged code directory")
    for name,role in (("rebuild_current_method.py","runner"),("complete_current_models.py","models"),("replay_three_rounds.py","coordinator")):
        if name not in pipeline or pipeline[name][1]!=PINS[role]:raise ValueError("Required current-method core component is absent")
    for path,h in pipeline.values():
        if sha(path)!=h:raise ValueError("Packaged pipeline pin differs")
    out.mkdir(parents=True,exist_ok=False);cfgdir=out/"config";cfgdir.mkdir();opaque=out/"reference";opaque.mkdir()
    method=bundled_child(bundle,cfg["method_root"])
    for relative,h in cfg["code_pins"].items():copied(bundled_child(method,relative),bundled_child(out/"method",relative),h)
    for name,(path,h) in pipeline.items():copied(path,out/"code"/name,h)
    copied(Path(__file__),out/"code"/"public_run.py",sha(__file__))
    cfg["pipeline_pins"]={"../code/"+name:h for name,(_,h) in pipeline.items()}
    cfg["pipeline_pins"]["../code/public_run.py"]=sha(__file__)
    cfg["method_root"]="../method"
    for field,hashfield,name in (("source_plan","source_plan_sha256","SOURCE_PLAN.json"),("viewer_rows","viewer_rows_sha256","VIEWER_REPLAY_ROWS.json")):
        copied(bundled_child(bundle,cfg[field]),opaque/name,cfg[hashfield]);cfg[field]="../reference/"+name
    env=cfg["environment_reference"];copied(bundled_child(bundle,env["path"]),opaque/"ENVIRONMENT.json",env["sha256"]);env["path"]="../reference/ENVIRONMENT.json"
    contract=json.loads((opaque/"ENVIRONMENT.json").read_text(encoding="utf-8"));payload=contract["package_payload_lock"]
    copied(bundled_child(bundle,payload["path"]),opaque/payload["path"],payload["sha256"])
    source_plan=json.loads((opaque/"SOURCE_PLAN.json").read_text(encoding="utf-8"));descriptor_root=bundled_child(bundle,cfg["viewer_descriptor_root"])
    for row in source_plan["rows"]:
        if row["source_kind"]!="locked_public_viewer_http":continue
        v=row["viewer"];relative=str(Path(v["capture_directory"])/v["receipt_filename"])
        copied(bundled_child(descriptor_root,relative),bundled_child(opaque/"viewer_descriptors",relative),v["receipt_sha256"])
    cfg["viewer_descriptor_root"]="../reference/viewer_descriptors"
    # Binary transport/hash checks only; no JSON/CSV target values are parsed.
    posthoc=cfg["posthoc_classification"];copied(bundled_child(bundle,posthoc["path"]),opaque/"classification_expected.json",posthoc["sha256"]);posthoc["path"]="../reference/classification_expected.json"
    for role,info in cfg["posthoc_models"].items():
        copied(bundled_child(bundle,info["path"]),opaque/(role+".csv"),info["sha256"]);info["path"]="../reference/"+role+".csv"
    for role,name in required:cfg[role]["path"]=str((finance_root/name).resolve())
    cfg.update({k:str(v.resolve()) for k,v in inputs.items()})
    cfg["scope"]["portable_configuration_generated_without_manual_edits"]=True
    path=cfgdir/"INPUTS.json";dump(path,cfg)
    dump(out/"PORTABLE_SETUP_RECEIPT.json",{"status":"STAGED_NOT_YET_CLASSIFIED","source_bundle_template_sha256":PINS["template"],"generated_config_sha256":sha(path),"method_bytes_changed":False,"count_targets_parsed_for_source_selection":False,"source_raw_files_copied":False,"fresh_website_or_financial_download":False})
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,default=Path(__file__).resolve().parent.parent/"bundle")
    p.add_argument("--data-root",type=Path);p.add_argument("--finance-root",type=Path)
    p.add_argument("--raw-dir",type=Path);p.add_argument("--supplemental-raw-dir",type=Path);p.add_argument("--viewer-capture-root",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--run",action="store_true")
    p.add_argument("--rounds",type=int,choices=(1,3),default=1)
    a=p.parse_args();bundle=a.bundle.resolve();finance=(a.finance_root or bundle/"finance")
    cfg=stage(bundle,a.data_root,finance,a.output,{"raw_dir":a.raw_dir,"supplemental_raw_dir":a.supplemental_raw_dir,"viewer_capture_root":a.viewer_capture_root})
    runner=a.output/"code/rebuild_current_method.py";coordinator=a.output/"code/replay_three_rounds.py"
    env={k:v for k,v in os.environ.items() if k not in {"PYTHONPATH","PYTHONHOME"}}
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
    subprocess.run([sys.executable,"-B",str(runner),"--inputs",str(cfg),"--mode","preflight"],env=env,check=True)
    if not a.run:
        print(json.dumps({"status":"PORTABLE_CONFIGURATION_AND_PREFLIGHT_PASS_NOT_EXECUTED","config":str(cfg.resolve()),"next_step":"Repeat in a new output directory with --run to authorize offline computation"}));return
    method_version=json.loads(cfg.read_text(encoding="utf-8"))["scope"]["method_version"]
    approvaldir=a.output/"execution_permissions";approvaldir.mkdir();approvals=[]
    runroot=a.output/"runs"
    for i in range(1,a.rounds+1):
        target=runroot/f"round_{i:02d}"
        approval={"runner_sha256":sha(runner),"inputs_sha256":sha(cfg),"source_plan_sha256":json.loads(cfg.read_text(encoding="utf-8"))["source_plan_sha256"],"output":str(target.resolve()),"workers":8,"method_version":method_version,"allow_full_locked_original_replay":True,"authority":"EXPLICIT_USER_COMMAND_LINE_RUN_LOCAL_OFFLINE_COMPUTATION"}
        path=approvaldir/f"round_{i:02d}.json";dump(path,approval);approvals.append({"path":str(path.resolve()),"sha256":sha(path)})
    if a.rounds==1:
        subprocess.run([sys.executable,"-B",str(runner),"--inputs",str(cfg),"--mode","run","--output",str((runroot/"round_01").resolve()),"--approval",approvals[0]["path"],"--workers","8"],env=env,check=True)
    else:
        plan={"coordinator_sha256":sha(coordinator),"runner_sha256":sha(runner),"inputs_sha256":sha(cfg),"output":str(runroot.resolve()),"rounds":3,"offline":True,"approvals":approvals}
        planpath=approvaldir/"three_round_plan.json";dump(planpath,plan)
        subprocess.run([sys.executable,"-B",str(coordinator),"--inputs",str(cfg),"--approval-plan",str(planpath),"--output",str(runroot)],env=env,check=True)


if __name__=="__main__":main()
