"""Bounded source recovery using original DART-viewer functions, never counts.

Default CLI is metadata-only. Real HTTP requires a matching root approval JSON.
No original run/update/main, raw ZIP, classifier, regression or BERT is invoked.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

ORIGINAL_SOURCE_SHA256 = "eaa04dbd970b5032372c75822f8b7673d4d20a098904ab9ee1191aebee3cad6b"
FUNCTIONS = {"get_html", "parse_selected_document", "parse_nodes", "annual_document_score", "find_base_annual_rcp",
             "resolve_annual_document", "viewer_section_to_text", "download_document_text", "validate_text"}
CONSTANTS = {"MAIN_URL", "VIEWER_URL", "VIEW_DOC_RE", "NODE_BLOCK_RE", "NODE_FIELD_TEMPLATE", "HANGUL_RE", "SPACE_RE", "BLANK_RE"}
MAIN_URL = "https://dart.fss.or.kr/dsaf001/main.do"
VIEWER_URL = "https://dart.fss.or.kr/report/viewer.do"
LIMITS = {"max_requests_per_report": 32, "max_response_bytes": 16 * 2**20, "max_total_response_bytes_per_report": 128 * 2**20}


class SourceGateError(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def canonical_dcm(value):
    """No fixed width for document IDs; do not add or strip leading zeroes."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise SourceGateError("Boolean document identifier")
    if isinstance(value, float):
        if not value.is_integer() or abs(value) > 2**53:
            raise SourceGateError("Document identifier is not losslessly integral")
        value = str(int(value))
    else:
        value = str(value)
        if re.fullmatch(r"\d+\.0", value):
            value = value[:-2]
    if not re.fullmatch(r"\d+", value):
        raise SourceGateError("Malformed document identifier")
    return value


def report_key(row):
    return [row[k] for k in ("corp_code", "year", "panel_rcept_no")]


def load_original_functions(path):
    path = Path(path)
    if sha(path) != ORIGINAL_SOURCE_SHA256:
        raise SourceGateError("Original recovery source SHA mismatch")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected, names, receipt = [], set(), []
    for node in tree.body:
        name = None
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            name = node.name
        elif isinstance(node, ast.Assign):
            name = next((target.id for target in node.targets if isinstance(target, ast.Name) and target.id in CONSTANTS), None)
        if name:
            selected.append(node); names.add(name)
            receipt.append({"name": name, "source_line": node.lineno, "source_end_line": node.end_lineno,
                            "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
    if names != FUNCTIONS | CONSTANTS:
        raise SourceGateError("Original pure-function dependency declaration incomplete")
    namespace = {"re": re, "html": html, "time": time, "Any": Any, "BeautifulSoup": BeautifulSoup, "requests": requests}
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + selected, type_ignores=[]), str(path), "exec"), namespace)
    if namespace["MAIN_URL"] != MAIN_URL or namespace["VIEWER_URL"] != VIEWER_URL:
        raise SourceGateError("Original public endpoint constants disagree")
    return namespace, receipt


def load_manifest(path, expected_sha256):
    path = Path(path)
    if sha(path) != expected_sha256:
        raise SourceGateError("Manifest SHA mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema"] != "JCI_HISTORICAL_VIEWER_SOURCE_MANIFEST_V1":
        raise SourceGateError("Unsupported manifest")
    rows = manifest["rows"]
    if len(rows) != 15 or len({tuple(report_key(r)) for r in rows}) != 15:
        raise SourceGateError("Expected fixed15 unique identities")
    known = 0
    for row in rows:
        for field, pattern in (("corp_code", r"\d{8}"), ("year", r"\d{4}"), ("panel_rcept_no", r"\d{14}"),
                               ("requested_rcept_no", r"\d{14}"), ("source_rcept_no", r"\d{14}")):
            if not isinstance(row[field], str) or not re.fullmatch(pattern, row[field]):
                raise SourceGateError("Malformed manifest identity")
        dcm = canonical_dcm(row["source_dcm_no_raw_original_cell"])
        if dcm != row["source_dcm_no_lossless_text_no_padding"]:
            raise SourceGateError("Document ID is not a lossless original-cell representation")
        if row["source_rcept_no"] != row["panel_rcept_no"]:
            raise SourceGateError("Source/panel receipt conflict")
        if row["historical_evidence"] == "DIRECT_VIEWER_DOCUMENT_RECORD":
            known += 1
            if dcm is None or not isinstance(row["historical_node_count"], int) or row["historical_node_count"] <= 0:
                raise SourceGateError("Known viewer identity lacks document/node evidence")
        elif row["historical_evidence"] == "RESUMED_CACHE_DOCUMENT_ID_UNKNOWN":
            if dcm is not None or row["historical_node_count"] is not None:
                raise SourceGateError("Unknown-cache identity fabricated")
        else:
            raise SourceGateError("Unsupported historical evidence status")
    if known != 8 or manifest["original_recovery_source_sha256"] != ORIGINAL_SOURCE_SHA256:
        raise SourceGateError("Expected8 directly documented and7 unknown-cache records")
    return manifest


def validate_resolved_identity(row, selected, nodes, resolution):
    if selected.get("rcpNo") != row["source_rcept_no"]:
        raise SourceGateError("Resolved source receipt differs from original recovery record")
    if not nodes:
        raise SourceGateError("Resolved document has no nodes")
    for node in nodes:
        if node.get("rcpNo") != selected["rcpNo"] or node.get("dcmNo") != selected["dcmNo"]:
            raise SourceGateError("Node document identity disagrees with selected document")
    offsets = [int(n.get("offset", "0") or 0) for n in nodes]
    if offsets != sorted(offsets):
        raise SourceGateError("Original parse_nodes did not preserve offset ordering")
    known = row["historical_evidence"] == "DIRECT_VIEWER_DOCUMENT_RECORD"
    if known:
        if selected.get("dcmNo") != row["source_dcm_no_lossless_text_no_padding"]:
            raise SourceGateError("Resolved document ID differs from exact unpadded original cell")
        if len(nodes) != row["historical_node_count"]:
            raise SourceGateError("Resolved node count differs from original viewer structure record")
        if resolution != row["historical_resolution_method"]:
            raise SourceGateError("Document-resolution route differs from recorded route")
    return known


def recover_row(api, row, session, delay=0.5):
    _, selected, nodes, resolution = api["resolve_annual_document"](session, row["requested_rcept_no"])
    known = validate_resolved_identity(row, selected, nodes, resolution)
    if hasattr(session, "authorize_selected"):
        session.authorize_selected(selected, nodes)
    text = api["download_document_text"](session, selected, nodes, delay)
    quality = api["validate_text"](text)
    if not quality["quality_pass"]:
        raise SourceGateError("Original Korean-text quality gate failed")
    return {"text": text, "selected": selected, "nodes_in_original_order": nodes, "resolution_method": resolution,
            "quality": quality, "historical_document_id_and_node_count_verified": known,
            "historical_text_byte_identity_verified": False,
            "status": "CURRENT_RETRIEVAL_MATCHES_KNOWN_DOCUMENT_IDENTITY" if known else "CURRENT_RETRIEVAL_HISTORICAL_DOCUMENT_ID_UNKNOWN"}


class RecordedResponse:
    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code; self.content = content; self.headers = headers or {}; self.encoding = "utf-8"

    @property
    def text(self):
        return self.content.decode(self.encoding, errors="replace")

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise SourceGateError("Public DART HTTP status " + str(self.status_code))


class RecordingPublicSession:
    """Only two original public endpoints; bounded bytes; no env/netrc auth."""
    def __init__(self, row, output):
        self.row = row; self.output = Path(output); self.output.mkdir(exist_ok=False)
        self.requests = []; self.total_bytes = 0; self.allowed_document = None
        self.session = requests.Session(); self.session.trust_env = False
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
                                     "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7"})

    def authorize_selected(self, selected, nodes):
        self.allowed_document = (selected["rcpNo"], selected["dcmNo"])
        self.allowed_node_params = {tuple(str(node.get(k) or selected.get(k, "")) for k in ("eleId", "offset", "length", "dtd")) for node in nodes}

    def get(self, url, params=None, headers=None, timeout=60, **kwargs):
        params = dict(params or {}); headers = dict(headers or {})
        if kwargs or url not in {MAIN_URL, VIEWER_URL}:
            raise SourceGateError("Unapproved public request interface")
        if len(self.requests) >= LIMITS["max_requests_per_report"]:
            raise SourceGateError("Public request bound exceeded")
        if url == MAIN_URL:
            if set(params) != {"rcpNo"} or params["rcpNo"] not in {self.row["requested_rcept_no"], self.row["source_rcept_no"]} or headers:
                raise SourceGateError("Main-page receipt not explicitly authorized")
        else:
            if set(params) != {"rcpNo", "dcmNo", "eleId", "offset", "length", "dtd"}:
                raise SourceGateError("Viewer parameter schema differs from original function")
            if (params["rcpNo"], params["dcmNo"]) != self.allowed_document:
                raise SourceGateError("Viewer document not source-validated")
            if tuple(str(params[k]) for k in ("eleId", "offset", "length", "dtd")) not in self.allowed_node_params:
                raise SourceGateError("Viewer node not in source-validated original node list")
            if headers != {"Referer": MAIN_URL + "?rcpNo=" + params["rcpNo"]}:
                raise SourceGateError("Unexpected viewer Referer")
        number = len(self.requests) + 1
        started = now(); content = bytearray(); response = None
        transport_error = None; body_complete = False; status = None; safe_headers = {}
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=min(float(timeout), 90), stream=True, allow_redirects=False)
            status = response.status_code
            safe_headers = {k: v for k, v in response.headers.items() if k.lower() in {"content-type", "content-length", "date", "etag", "last-modified"}}
            for part in response.iter_content(chunk_size=65536):
                remaining = min(LIMITS["max_response_bytes"] - len(content), LIMITS["max_total_response_bytes_per_report"] - self.total_bytes - len(content))
                if len(part) > remaining:
                    content.extend(part[:max(remaining, 0)])
                    raise SourceGateError("Public response byte bound exceeded")
                content.extend(part)
            body_complete = True
        except SourceGateError as error:
            transport_error = error
        except requests.RequestException:
            transport_error = SourceGateError("Public HTTP transport failure; credentials/query secrets not logged")
        finally:
            if response is not None:
                response.close()
        body = bytes(content); self.total_bytes += len(body)
        body_path = self.output / f"{number:04d}_response_body.html"
        with body_path.open("xb") as f:
            f.write(body)
        record = {"request_order": number, "started_utc": started, "finished_utc": now(), "endpoint": url,
                  "params": params, "headers": headers, "status_code": status, "response_headers": safe_headers,
                  "response_bytes": len(body), "response_sha256": sha(body_path), "response_path": str(body_path),
                  "response_body_complete": body_complete,
                  "transport_error": str(transport_error) if transport_error else None}
        self.requests.append(record)
        write_new(self.output / f"{number:04d}_request_response.json", record)
        if transport_error:
            raise transport_error
        recorded = RecordedResponse(status, body, safe_headers)
        recorded.raise_for_status()
        return recorded

    def close(self):
        self.session.close()


def validate_approval(path, manifest_path, original_source, rows):
    approval = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema": "JCI_VIEWER_SOURCE_APPROVAL_V1", "allow_network": True,
                "adapter_sha256": sha(Path(__file__)), "manifest_sha256": sha(manifest_path),
                "original_source_sha256": sha(original_source), "report_keys": [report_key(r) for r in rows]}
    if any(approval.get(k) != value for k, value in required.items()):
        raise SourceGateError("Network approval does not match exact code/manifest/source/fixed15 identities")
    return approval


def run(options):
    manifest_path = options.manifest.resolve(); source_path = options.original_source.resolve()
    manifest = load_manifest(manifest_path, options.manifest_sha256)
    api, ast_receipt = load_original_functions(source_path)
    protected_paths = [Path(__file__), manifest_path, source_path, Path(manifest["source_audit_path"]), Path(manifest["source_update_path"])]
    if sha(protected_paths[-2]) != manifest["source_audit_sha256"] or sha(protected_paths[-1]) != manifest["source_update_sha256"]:
        raise SourceGateError("Original recovery workbooks differ from manifest")
    if not options.execute_approved:
        print(json.dumps({"status": "PREFLIGHT_ONLY_NO_HTTP", "fixed_reports": 15, "known_document_ids": 8, "unknown_cache_document_ids": 7,
                          "manifest_sha256": sha(manifest_path), "adapter_sha256": sha(Path(__file__)), "original_ast": ast_receipt}, ensure_ascii=False))
        return 0
    if options.approval_json is None or options.output is None:
        raise SourceGateError("Network execution requires matching approval JSON and fresh output")
    validate_approval(options.approval_json, manifest_path, source_path, manifest["rows"])
    protected_paths.append(options.approval_json.resolve())
    before = {str(p): sha(p) for p in protected_paths}
    output = options.output.resolve()
    # Never permit the original historical source/cache tree as an output.
    if output.is_relative_to(source_path.parent.parent) or output.is_relative_to(source_path.parent):
        raise SourceGateError("Refusing output in original historical input tree")
    output.mkdir(parents=False, exist_ok=False)
    write_new(output / "SOURCE_PLAN_BEFORE_HTTP.json", {"started_utc": now(), "manifest_sha256": sha(manifest_path),
              "adapter_sha256": sha(Path(__file__)), "original_ast": ast_receipt, "rows": manifest["rows"], "limits": LIMITS,
              "classification_counts_read": False, "protected_hashes_before": before})
    results = []; failure = None
    for index, row in enumerate(manifest["rows"], 1):
        entry = {"report_key": report_key(row), "historical_evidence": row["historical_evidence"]}
        if failure:
            entry.update(status="NOT_EXECUTED_AFTER_SOURCE_OR_TRANSPORT_GATE_FAILURE")
        else:
            report_output = output / f"report_{index:02d}_{row['corp_code']}_{row['year']}"; report_output.mkdir()
            session = RecordingPublicSession(row, report_output / "http_responses")
            try:
                recovered = recover_row(api, row, session, options.delay)
                text = recovered.pop("text")
                target = report_output / "RECOVERED_SOURCE_TEXT.txt"
                with target.open("x", encoding="utf-8", newline="") as f:
                    f.write(text)
                entry.update(recovered, text_path=str(target), text_sha256=sha(target), http_requests=session.requests)
            except Exception as error:
                # Never record arbitrary response HTML/errors in an exception message.
                failure = {"report_key": report_key(row), "error_type": type(error).__name__,
                           "message": str(error) if isinstance(error, SourceGateError) else "Original source function rejected response; inspect retained bounded HTML"}
                entry.update(status="FAILED_SOURCE_GATE_STOPPED", failure=failure, http_requests=session.requests)
            finally:
                session.close()
            write_new(report_output / "REPORT_SOURCE_RECEIPT.json", entry)
        results.append(entry)
    after = {str(p): sha(p) for p in protected_paths}
    unchanged = before == after
    receipt = {"finished_utc": now(), "status": "RETRIEVAL_FINISHED_NOT_HISTORICAL_TEXT_IDENTITY_PROOF" if not failure and unchanged else "STOPPED_OR_INPUT_MUTATION",
               "results": results, "failure": failure, "protected_hashes_before": before, "protected_hashes_after": after,
               "protected_files_unchanged": unchanged, "original_files_overwritten": False, "classification_run": False,
               "historical_text_identity_verified": False, "known_vs_unknown_document_gate": "8 exact original source/dcm/node gates;7 same original resolve but historical dcm/node unknown",
               "publication_gate": "NOT_PASSED"}
    write_new(output / "VIEWER_SOURCE_EXECUTION_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "reports_accounted": len(results), "failure": failure, "input_hashes_unchanged": unchanged}, ensure_ascii=False))
    return 0 if not failure and unchanged else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--approval-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delay", type=float, default=0.5)
    options = parser.parse_args()
    if options.delay < 0:
        parser.error("Nonnegative original inter-section delay required")
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
