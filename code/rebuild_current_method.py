"""Fresh locked-original replay of the CURRENT measurement, never old-count fit.

Default is metadata-only preflight. Full execution needs an exact root approval.
Source tasks contain no classification targets. Expected counts/digests are
opened only AFTER all fresh predictions have been durably sealed. Raw sources
are not copied or modified; temporary text belongs only to the new run.
"""
from __future__ import annotations
import argparse
import ctypes
import ctypes.wintypes
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import site
import struct
import sys
import sysconfig
import time
import traceback

VERSION = "locked_source_aware_plus_metadata_viewer15_20261023_v1"
COUNTS = ("total_sentences", "digital_sentence_count", "substantive_digital_sentence_count", "bmi_sentence_count")
API = None
VIEWER = None
VIEWER_FUNCTIONS = None
TEMP = None
ROOTS = None
for _thread in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread] = "1"
sys.dont_write_bytecode = True


def disable_network():
    def rejected(*args, **kwargs):
        raise RuntimeError("NETWORK_DISABLED: strict locked-original offline replay")
    socket.socket.connect = rejected
    socket.create_connection = rejected


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")


def dump(p, value):
    with Path(p).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def protected_closure(protected):
    records=[]
    for path,expected in protected.items():
        try:
            actual=sha(path);error=None
        except Exception as e:
            actual=None;error=type(e).__name__+": "+str(e)[:500]
        records.append({"path":path,"expected_sha256":expected,"actual_sha256":actual,"unchanged":actual==expected,"read_error":error})
    return {"unchanged":all(r["unchanged"] for r in records),"records":records}


def resolve(root, relative):
    p = Path(relative)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def child(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Source path must be a relative child")
    candidate = (root / p).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Source child escaped root")
    return candidate


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def key(row):
    m = row["metadata"]
    return tuple(str(m[x]) for x in ("corp_code", "year", "rcept_no"))


def environment(reference):
    import lxml.etree
    import bs4
    if platform.system()!="Windows":raise ValueError("Supported and validated runtime is Windows AMD64 CPython3.13.12")
    buf=ctypes.create_unicode_buffer(32768);kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetModuleFileNameW.argtypes=[ctypes.wintypes.HMODULE,ctypes.wintypes.LPWSTR,ctypes.wintypes.DWORD];kernel.GetModuleFileNameW.restype=ctypes.wintypes.DWORD
    if not kernel.GetModuleFileNameW(ctypes.wintypes.HMODULE(sys.dllhandle),buf,len(buf)):raise ctypes.WinError(ctypes.get_last_error())
    packages=reference["required"]["package_versions"]
    purelib=Path(sysconfig.get_path("purelib")).resolve()
    return {"python_version_info":list(sys.version_info[:3]),"python_version_string":sys.version,"implementation":platform.python_implementation(),
        "system":platform.system(),"machine":platform.machine(),"bits":struct.calcsize("P")*8,
        "platform_record_only":platform.platform(),"venv_launcher_sha256":sha(sys.executable),"base_executable_sha256":sha(sys._base_executable),"loaded_python_dll_sha256":sha(buf.value),
        "isolated_venv":sys.prefix!=sys.base_prefix and site.ENABLE_USER_SITE is False,
        "pythonpath_env_present":bool(os.environ.get("PYTHONPATH")),"pythonhome_env_present":bool(os.environ.get("PYTHONHOME")),
        "package_versions":{p:importlib.metadata.version(p) for p in packages},
        "selected_distribution_roots_match_venv":all(Path(importlib.metadata.distribution(p).locate_file("")).resolve()==purelib for p in packages),
        "libxml_runtime_version": list(lxml.etree.LIBXML_VERSION), "libxml_compiled_version": list(lxml.etree.LIBXML_COMPILED_VERSION),
        "libxslt_runtime_version": list(lxml.etree.LIBXSLT_VERSION), "libxslt_compiled_version": list(lxml.etree.LIBXSLT_COMPILED_VERSION),
        "etree_extension_sha256": sha(lxml.etree.__file__), "bs4_module_sha256": sha(bs4.__file__),
        "thread_limits": {k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}}


def validate_environment(observed, reference):
    if any(observed.get(k)!=v for k,v in reference["required"].items()) or not observed["isolated_venv"] or not observed["selected_distribution_roots_match_venv"] or observed["pythonpath_env_present"] or observed["pythonhome_env_present"] or any(v!="1" for v in observed["thread_limits"].values()):
        raise ValueError("Official isolated runtime/parser/package/thread contract differs; no automatic bypass or downgrade")


def validate_package_payload(reference, reference_path):
    info=reference["package_payload_lock"];lock=child(reference_path.parent,info["path"])
    if sha(lock)!=info["sha256"]:raise ValueError("Official package payload lock changed")
    files=json.loads(lock.read_text(encoding="utf-8"))["files"];root=Path(sysconfig.get_path("purelib")).resolve()
    if len(files)!=info["payload_files"] or len({r["relative_path"] for r in files})!=len(files) or sum(r["bytes"] for r in files)!=info["payload_bytes"]:raise ValueError("Official package payload scope changed")
    for row in files:
        p=child(root,row["relative_path"])
        if row["distribution"] not in reference["required"]["package_versions"] or p.stat().st_size!=row["bytes"] or sha(p)!=row["sha256"]:raise ValueError("Installed package payload differs from official wheel: "+row["relative_path"])
    return lock,info["sha256"]


def preflight(inputs_path):
    base = inputs_path.resolve().parent
    cfg = json.loads(inputs_path.read_text(encoding="utf-8"))
    method_root = resolve(base, cfg["method_root"])
    protected = {str(inputs_path.resolve()): sha(inputs_path), str(Path(__file__).resolve()): sha(__file__)}
    for relative, expected in cfg["pipeline_pins"].items():
        p=resolve(base,relative)
        if not p.is_relative_to(base.parent) or sha(p)!=expected:raise ValueError("Pipeline component changed")
        protected[str(p)]=expected
    environment_path=resolve(base,cfg["environment_reference"]["path"])
    if sha(environment_path)!=cfg["environment_reference"]["sha256"]:raise ValueError("Environment reference changed")
    protected[str(environment_path)]=cfg["environment_reference"]["sha256"]
    environment_reference=json.loads(environment_path.read_text(encoding="utf-8"))
    payload_path,payload_sha=validate_package_payload(environment_reference,environment_path)
    protected[str(payload_path)]=payload_sha
    for relative, expected in cfg["code_pins"].items():
        p = child(method_root, relative)
        if sha(p) != expected:
            raise ValueError("Pinned method code changed: "+relative)
        protected[str(p)] = expected
    source_path = resolve(base, cfg["source_plan"])
    viewer_rows_path = resolve(base, cfg["viewer_rows"])
    for p, expected in [(source_path, cfg["source_plan_sha256"]), (viewer_rows_path, cfg["viewer_rows_sha256"])]:
        if sha(p) != expected:
            raise ValueError("Source metadata pin mismatch")
        protected[str(p)] = expected
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source["method_version"] != VERSION or source["selection_uses_classification_counts"]:
        raise ValueError("Unexpected source policy")
    rows = source["rows"]
    if len(rows) != 22654 or len({key(r) for r in rows}) != 22654 or len({key(r)[:2] for r in rows})!=22654:
        raise ValueError("Need the entire unique source universe")
    if any(any(k in r for k in (*COUNTS, "counts", "target", "expected_coefficient")) for r in rows):
        raise ValueError("Classification targets were mixed into source tasks")
    viewer_rows = json.loads(viewer_rows_path.read_text(encoding="utf-8"))
    viewers = {tuple(str(r[x]) for x in ("corp_code", "year", "panel_rcept_no")): r for r in viewer_rows}
    vk = {key(r) for r in rows if r["source_kind"] == "locked_public_viewer_http"}
    if len(vk) != 15 or vk != set(viewers) or sum(r["historical_evidence"] == "DIRECT_VIEWER_DOCUMENT_RECORD" for r in viewers.values()) != 8:
        raise ValueError("Viewer metadata scope or known/unknown split changed")
    for role in ("financial_panel", "frozen_financial_panel"):
        p = resolve(base, cfg[role]["path"])
        if sha(p) != cfg[role]["sha256"]:
            raise ValueError("Pinned financial input changed")
        protected[str(p)] = cfg[role]["sha256"]
    roots = {r: resolve(base, cfg[r]) for r in ("raw_dir", "supplemental_raw_dir", "viewer_capture_root","viewer_descriptor_root")}
    for p in roots.values():
        if not p.is_dir():
            raise FileNotFoundError("Required locked-original source directory missing")
    for row in rows:
        m=row["metadata"]
        if set(m)!={"corp_code","year","rcept_no","stock_code","corp_name","market","download_status","text_source_rcept_no","text_source_rcept_no_raw","download_request_rcept_no","source_resolution_basis"}:
            raise ValueError("Source metadata contains unapproved fields")
        if any(not re.fullmatch(pattern,str(m[name])) for name,pattern in (("corp_code",r"\d{8}"),("year",r"\d{4}"),("rcept_no",r"\d{14}"),("download_request_rcept_no",r"\d{14}"),("stock_code",r"\d{6}"))):
            raise ValueError("Source metadata identity width changed")
        actual=m["download_request_rcept_no"]; recorded=m["text_source_rcept_no"]
        if recorded and recorded!=actual:raise ValueError("Recorded and requested actual source disagree")
        if actual!=m["rcept_no"] and (not recorded or m["source_resolution_basis"]!="explicit_production_manifest_record"):raise ValueError("Alternate source lacks explicit original production evidence")
        if row["source_kind"] == "locked_dart_zip":
            if row["raw_filename"]!=f"{m['corp_code']}_{m['year']}_{actual}.zip":raise ValueError("Raw filename disagrees with immutable source identity")
            paths = [child(roots[r], row["raw_filename"]) for r in ("raw_dir", "supplemental_raw_dir")]
            if not any(p.is_file() for p in paths):
                raise FileNotFoundError("Locked ZIP is missing: "+row["raw_filename"])
            if not re.fullmatch(r"[a-f0-9]{64}",row["expected_raw_sha256"]):
                raise ValueError("Missing exact raw hash")
        elif row["source_kind"] == "locked_public_viewer_http":
            v = row["viewer"]; directory = child(roots["viewer_capture_root"], v["capture_directory"])
            if v["receipt_filename"]!="PUBLIC_VIEWER_SOURCE_DESCRIPTOR.json":raise ValueError("Expected a published metadata projection, not a fabricated new capture receipt")
            receipt=child(roots["viewer_descriptor_root"],str(Path(v["capture_directory"])/v["receipt_filename"]))
            if sha(receipt) != v["receipt_sha256"]:
                raise ValueError("Public viewer source descriptor changed")
            descriptor=json.loads(receipt.read_text(encoding="utf-8"))
            if descriptor["report_key"]!=list(key(row)) or any(descriptor[name]!=v[name] for name in ("current_selected","current_nodes","current_resolution_method","expected_parsed_sha256","original_private_receipt_sha256","requests")) or descriptor["new_acquisition_claimed"]:
                raise ValueError("Public descriptor identity/ordered requests differ from source plan")
            protected[str(receipt)] = v["receipt_sha256"]
            for req in v["requests"]:
                body = directory / "http_responses" / f"{req['request_order']:04d}_response_body.html"
                if sha(body) != req["response_sha256"] or body.stat().st_size != req["response_bytes"]:
                    raise ValueError("Locked viewer HTTP bytes changed")
                protected[str(body)] = req["response_sha256"]
        else:
            raise ValueError("Unknown source kind")
    env = environment(environment_reference)
    validate_environment(env,environment_reference)
    roots["environment_reference"]=environment_path
    return cfg, rows, viewers, roots, method_root, protected, env


class OfflineViewerSession:
    """Replay only ordered, hash-locked raw HTTP bodies, not parsed text."""
    def __init__(self, plan, root, response_class):
        self.plan = plan; self.root = root; self.response_class = response_class; self.position = 0

    def authorize_selected(self, selected, nodes):
        if selected != self.plan["current_selected"] or nodes != self.plan["current_nodes"]:
            raise ValueError("Fresh original HTML parsing changed selected document/nodes")

    def get(self, url, params=None, headers=None, timeout=60, **kwargs):
        if kwargs or self.position >= len(self.plan["requests"]):
            raise ValueError("Unexpected offline viewer request")
        expected = self.plan["requests"][self.position]
        if url != expected["endpoint"] or dict(params or {}) != expected["params"] or dict(headers or {}) != expected["headers"]:
            raise ValueError("Fresh original viewer code changed request order/identity")
        p = self.root / "http_responses" / f"{expected['request_order']:04d}_response_body.html"
        body = p.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected["response_sha256"] or len(body) != expected["response_bytes"] or not expected["response_body_complete"] or expected["transport_error"] or not 200<=expected["status_code"]<300:
            raise ValueError("Offline viewer response bytes/status failed")
        self.position += 1
        return self.response_class(expected["status_code"], body, expected["response_headers"])


def initialize(method_root, temp_root, roots, viewers):
    global API, VIEWER, VIEWER_FUNCTIONS, TEMP, ROOTS
    disable_network()
    method_root = Path(method_root); sys.path.insert(0, str(method_root))
    contract_path=Path(roots["environment_reference"]);contract=json.loads(contract_path.read_text(encoding="utf-8"))
    validate_environment(environment(contract),contract);validate_package_payload(contract,contract_path)
    from production_source_api import SourceAwareClassifier
    API = SourceAwareClassifier()
    VIEWER = module(method_root / "recover_historical_viewer_sources.py", "locked_viewer_recovery_adapter")
    VIEWER_FUNCTIONS, _ = VIEWER.load_original_functions(method_root / "model_sources/original_viewer_recovery.py")
    TEMP = Path(temp_root)
    ROOTS = {k: Path(v) for k, v in roots.items()}
    ROOTS["viewer_rows"] = viewers


def score_report(row):
    started = time.perf_counter(); m = row["metadata"]; k = key(row)
    result = {"index": row["index"], "key": list(k), "metadata": m, "source_kind": row["source_kind"], "pid": os.getpid()}
    result["child_thread_limits"]={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}
    temp = TEMP / f"worker_{os.getpid()}_{row['index']}.txt"
    made = False
    try:
        if row["source_kind"] == "locked_dart_zip":
            available = [child(ROOTS[r], row["raw_filename"]) for r in ("raw_dir", "supplemental_raw_dir")]
            present = [p for p in available if p.is_file()]
            if not present or any(sha(p) != row["expected_raw_sha256"] for p in present):
                raise ValueError("ZIP missing/ambiguous/changed exact bytes")
            raw_path = present[0]; raw = raw_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row["expected_raw_sha256"]:raise ValueError("The ZIP bytes actually supplied to parser differ from source lock")
            text, parser = API.decode_from_production_source(raw, m["download_status"])
            result.update(parser=parser, raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw))
            del raw
            result["raw_source_unchanged"] = sha(raw_path) == row["expected_raw_sha256"]
        else:
            v = row["viewer"]; directory = child(ROOTS["viewer_capture_root"], v["capture_directory"])
            session = OfflineViewerSession(v, directory, VIEWER.RecordedResponse)
            recovered = VIEWER.recover_row(VIEWER_FUNCTIONS, ROOTS["viewer_rows"][k], session, delay=0)
            text = recovered["text"]
            if session.position != len(v["requests"]) or recovered["selected"] != v["current_selected"] or recovered["resolution_method"] != v["current_resolution_method"]:
                raise ValueError("Raw viewer replay did not consume the complete locked source")
            result.update(parser={"parser_name": "original_public_viewer_functions_from_locked_raw_http"},
                replayed_http_requests=session.position, historical_evidence=v["historical_evidence"],
                historical_complete_text_identity_proven=False, raw_source_unchanged=True)
        text_bytes = text.encode("utf-8", errors="replace")
        result["parsed_sha256"] = hashlib.sha256(text_bytes).hexdigest()
        if row["source_kind"] == "locked_public_viewer_http" and result["parsed_sha256"] != row["viewer"]["expected_parsed_sha256"]:
            raise ValueError("Fresh viewer extraction differs from source-bound text hash")
        with temp.open("xb") as f:
            made = True; f.write(text_bytes)
        del text, text_bytes
        original = API.ns["classify_sentence_v4"]
        digest = hashlib.sha256(); calls = 0; categories = Counter()
        def observe(sentence, report_year, corp_name):
            nonlocal calls
            flags = original(sentence, report_year, corp_name); calls += 1
            category = API.category(flags); categories[category] += 1
            digest.update(canonical({"sentence_id": calls, "sentence_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest(), "category": category, "flags": flags}))
            return flags
        API.ns["classify_sentence_v4"] = observe
        try:
            scores, _ = API.score_original_text_file({**m, "year": int(m["year"]), "text_cache_path": str(temp)})
        finally:
            API.ns["classify_sentence_v4"] = original
        result.update({f: int(scores[f]) for f in COUNTS})
        d, c, s, b = [result[f] for f in COUNTS]
        if scores["score_status"] != "ok" or not 0 <= b <= s <= c <= d or calls != d or categories["BMI"] != b or categories["NonBMI"] != s-b or categories["Exclude"] != d-s:
            raise ValueError("Original scorer/observer accounting failed")
        result.update(NonBMODD=s-b, classifier_calls=calls, canonical_sentence_predictions_sha256=digest.hexdigest(), status="FRESH_PREDICTION_COMPLETED")
        if row["source_kind"]=="locked_dart_zip" and sha(raw_path)!=row["expected_raw_sha256"]:raise ValueError("ZIP changed during classification")
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error)[:2000])
    finally:
        if made and temp.is_file():
            if temp.parent.resolve() != TEMP.resolve() or not temp.name.startswith(f"worker_{os.getpid()}_"):
                raise RuntimeError("Temporary text cleanup escaped owned run directory")
            temp.unlink()
        result["seconds"] = time.perf_counter()-started
    return result


def classify_all(rows, method_root, roots, viewers, out, workers):
    out.mkdir(exist_ok=False); temp = out / "_temporary_text"; temp.mkdir()
    results = []; failed = None; started = time.perf_counter();submitted=[];future_failed=[];fatal=None
    try:
        with (out / "fresh_report_predictions.jsonl").open("x", encoding="utf-8", newline="\n") as log:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                  initargs=(str(method_root), str(temp), {k: str(v) for k,v in roots.items()}, viewers)) as pool:
                pending = {}; next_index = 0
                def submit_one():
                    nonlocal next_index
                    row=rows[next_index];f=pool.submit(score_report,row)
                    pending[f]=row;submitted.append(row["index"]);next_index+=1
                while next_index < len(rows) and len(pending) < workers*2:submit_one()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        request=pending.pop(f)
                        try:
                            value=f.result()
                            if value.get("index")!=request["index"] or value.get("key")!=list(key(request)):
                                raise ValueError("Worker result identity differs from submitted source task")
                            results.append(value)
                            log.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n"); log.flush()
                            if value["status"] != "FRESH_PREDICTION_COMPLETED": failed=value
                        except Exception as error:
                            value={"index":request["index"],"key":list(key(request)),"status":"FUTURE_FAILED_NO_COMPLETE_WORKER_RESULT","actual_work_started_unknown":True,"error_type":type(error).__name__,"error":str(error)[:2000]}
                            future_failed.append(value);failed=value
                    observed=len(results)+len(future_failed)
                    if observed % 200 < len(done) or failed:
                        print(json.dumps({"phase":"classification", "complete_worker_results":len(results),"future_failures":len(future_failed), "target":len(rows), "calls_in_observed_worker_results":sum(r.get("classifier_calls",0) for r in results), "seconds":time.perf_counter()-started, "failed":bool(failed)}), flush=True)
                    while not failed and next_index < len(rows) and len(pending) < workers*2:submit_one()
    except BaseException as error:
        fatal={"type":type(error).__name__,"error":str(error)[:2000]};failed=fatal
    observed_indices={r["index"] for r in results}|{r["index"] for r in future_failed}
    submitted_indices=set(submitted);unobserved=submitted_indices-observed_indices
    dump(out/"CLASSIFICATION_EXECUTION_STATE.json",{"submitted_report_indices":submitted,
        "complete_worker_result_indices":[r["index"] for r in results],"future_failed_without_complete_worker_result":future_failed,
        "submitted_but_execution_result_unobserved":[r["index"] for r in rows if r["index"] in unobserved],
        "never_submitted_report_indices":[r["index"] for r in rows if r["index"] not in submitted_indices],"fatal_exception":fatal,
        "unobserved_submitted_does_not_mean_not_executed":True})
    results.sort(key=lambda r:r["index"])
    ordered = hashlib.sha256(b"".join(canonical(r) for r in results)).hexdigest()
    measurement_fields=("index","key","source_kind","raw_sha256","parsed_sha256",*COUNTS,"NonBMODD","canonical_sentence_predictions_sha256")
    measurement_digest=hashlib.sha256(b"".join(canonical({k:r.get(k) for k in measurement_fields}) for r in results)).hexdigest()
    seal={"status":"FRESH_PREDICTIONS_SEALED" if len(results)==len(rows) and not failed and not future_failed else "FAILED_NOT_ALL_REPORTS_PREDICTED",
        "completed":len(results),"required":len(rows),"classifier_calls":sum(r.get("classifier_calls",0) for r in results),
        "prediction_file_sha256":sha(out/"fresh_report_predictions.jsonl"),"ordered_result_sha256":ordered,"ordered_measurement_sha256":measurement_digest,
        "remaining_temp_files":len(list(temp.iterdir())),"failure":failed,"timestamp":now(),
        "expected_counts_read_before_seal":False,"prediction_checkpoint_reused":False}
    dump(out/"PREDICTION_SEAL.json",seal)
    if seal["status"] != "FRESH_PREDICTIONS_SEALED" or seal["remaining_temp_files"]:
        raise RuntimeError("Fresh classification failed; do not aggregate/model or proceed to another round")
    return results


def check_after_seal(cfg, base, results, out):
    ref=cfg["posthoc_classification"]; p=resolve(base,ref["path"])
    if sha(p)!=ref["sha256"]: raise ValueError("Posthoc classification reference changed")
    expected={tuple(r["key"]):r for r in json.loads(p.read_text(encoding="utf-8"))}
    if len(expected)!=len(results) or set(expected)!={tuple(r["key"]) for r in results}: raise ValueError("Posthoc identities differ")
    differences=[]
    for r in results:
        old=expected[tuple(r["key"])]
        for f in COUNTS:
            if r[f]!=old["counts"][f]: differences.append({"key":r["key"],"field":f,"actual":r[f],"expected":old["counts"][f]})
        for f in ("parsed_sha256","canonical_sentence_predictions_sha256"):
            if r[f]!=old[f]: differences.append({"key":r["key"],"field":f,"actual":r[f],"expected":old[f]})
    dump(out/"POSTHOC_CURRENT_MEASUREMENT_CHECK.json",{"reports":len(results),"differences":differences,"reference_sha256":ref["sha256"],"expected_opened_after_prediction_seal":True})
    if differences: raise ValueError("Current-version measurement differs; retain all evidence and stop")


def main():
    disable_network()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--mode",choices=["preflight","run"],default="preflight")
    p.add_argument("--output",type=Path);p.add_argument("--approval",type=Path);p.add_argument("--workers",type=int,choices=range(1,9),default=8)
    a=p.parse_args(); cfg,rows,viewers,roots,method_root,protected,env=preflight(a.inputs)
    if a.mode=="preflight":
        print(json.dumps({"status":"METADATA_ONLY_PREFLIGHT_PASS","reports":len(rows),"viewer":len(viewers),"inputs_sha256":sha(a.inputs),"runner_sha256":sha(__file__),"environment":env,"raw_parses":0,"classifications":0,"model_fits":0},ensure_ascii=False));return 0
    if not a.output or not a.approval: raise ValueError("Full run requires explicit output and root approval")
    expected={"runner_sha256":sha(__file__),"inputs_sha256":sha(a.inputs),"source_plan_sha256":cfg["source_plan_sha256"],"output":str(a.output.resolve()),"workers":a.workers,"method_version":VERSION,"allow_full_locked_original_replay":True}
    approval=json.loads(a.approval.read_text(encoding="utf-8"))
    if any(approval.get(k)!=v for k,v in expected.items()): raise ValueError("Root approval does not bind this exact run")
    if a.output.exists(): raise FileExistsError("Every full replay needs a genuinely nonexistent output directory")
    protected_roots=[method_root,*[v for k,v in roots.items() if k!="environment_reference"],a.inputs.resolve().parent]
    if any(a.output.resolve().is_relative_to(p.resolve()) for p in protected_roots):raise ValueError("Output cannot be inside an input/code directory")
    a.output.mkdir(parents=True,exist_ok=False);started=now()
    dump(a.output/"START_RECEIPT.json",{**expected,"utc":started,"pid":os.getpid(),"protected":protected,"environment":env,"network_calls":0,"scope":"Locked-original offline replay of current measurement; not a fresh website or financial download"})
    try:
        results=classify_all(rows,method_root,roots,viewers,a.output/"classification",a.workers)
        check_after_seal(cfg,a.inputs.resolve().parent,results,a.output)
        # The separate model-stage candidate is hash bound before approval.
        from complete_current_models import regenerate_panel_and_models
        model_receipt=regenerate_panel_and_models(cfg,a.inputs.resolve().parent,method_root,results,a.output/"empirical")
        closure=protected_closure(protected)
        dump(a.output/"PROTECTED_INPUT_CLOSURE.json",closure)
        if not closure["unchanged"]:raise ValueError("Protected source/input changed")
        dump(a.output/"FULL_CURRENT_METHOD_RUN_RECEIPT.json",{"status":"CURRENT_METHOD_REPLAY_PASS","utc_started":started,"utc_finished":now(),"reports":len(results),"classifier_calls":sum(r["classifier_calls"] for r in results),"protected_unchanged":True,"model_receipt":model_receipt,"historical_old_measurement_exact_recovery":False,"fresh_website_and_financial_redownload":False,"publication_gate":"PENDING_THREE_ROUNDS_AND_RIGHTS_REVIEW"});return 0
    except BaseException as error:
        closure=protected_closure(protected)
        dump(a.output/"FAILED_RUN_RECEIPT.json",{"status":"FAILED_STOP_NEXT_ROUND","type":type(error).__name__,"message":str(error)[:3000],"utc":now(),"protected_unchanged":closure["unchanged"],"protected_input_closure":closure})
        (a.output/"FAILURE_TRACEBACK.txt").write_text(traceback.format_exc(),encoding="utf-8");raise


if __name__=="__main__":
    raise SystemExit(main())
