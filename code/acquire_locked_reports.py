"""Verify or explicitly fetch only immutable report bytes named by the bundle.

Default is offline --check-only. --fetch never selects another filing, changes a
parser, classifies text, or estimates models. A changed online response is an
error, not permission to update the source plan. Acquisition is not a completed
reproduction run. Historic viewer descriptors are copied, never regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys
import uuid
import zipfile
import xml.etree.ElementTree as ET

EXPECTED_TEMPLATE_SHA256 = "7435acc785505c1a8af15cc83e5f687ce30d6300993a77f3a6f6e47a256cb58d"
EXPECTED_RECORDS = 22654
EXPECTED_VIEWERS = 15
EXPECTED_VIEWER_REQUESTS = 249
METHOD_VERSION = "locked_source_aware_plus_metadata_viewer15_20261023_v1"
DART_DOCUMENT = "https://opendart.fss.or.kr/api/document.xml"
DART_MAIN = "https://dart.fss.or.kr/dsaf001/main.do"
DART_VIEWER = "https://dart.fss.or.kr/report/viewer.do"
MAX_RESPONSE_BYTES = 512 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")
VIEWER_PARAMS = ("rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")
STANDARD_DATA_DIRECTORIES = ("raw_dart", "supplemental_raw", "viewer_capture")


class SafeFailure(Exception):
    """The message is a fixed safe code, never a transport exception string."""
    def __init__(self, code, **safe_context):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", code):
            raise ValueError("Unsafe error-code format")
        super().__init__(code)
        self.code = code
        self.context = safe_context


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(body):
    return hashlib.sha256(body).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SafeFailure("DUPLICATE_JSON_KEY")
        value[key] = item
    return value


def read_locked_json(path, expected):
    if not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
        raise SafeFailure("INVALID_EXPECTED_HASH")
    body = Path(path).read_bytes()
    if sha_bytes(body) != expected:
        raise SafeFailure("LOCKED_METADATA_HASH_MISMATCH", file=Path(path).name)
    return json.loads(body.decode("utf-8-sig"), object_pairs_hook=unique_object)


def child(root, relative):
    if not isinstance(relative, str) or not relative or "\x00" in relative or "\\" in relative:
        raise SafeFailure("INVALID_CHILD_PATH")
    p = Path(relative)
    if p.is_absolute() or PureWindowsPath(relative).is_absolute() or ":" in relative or any(part in {"..", "."} for part in relative.split("/")):
        raise SafeFailure("PATH_ESCAPE_REJECTED")
    root = Path(root).resolve()
    candidate = (root / p).resolve()
    if not candidate.is_relative_to(root):
        raise SafeFailure("PATH_ESCAPE_REJECTED")
    return candidate


def identity(row):
    m = row["metadata"]
    return (m["corp_code"], m["year"], m["rcept_no"])


def validate_viewer(row, public_row):
    m = row["metadata"]
    viewer = row["viewer"]
    directory = viewer["capture_directory"]
    if not re.fullmatch(r"report_\d{2}_" + re.escape(m["corp_code"]) + "_" + re.escape(m["year"]), directory):
        raise SafeFailure("VIEWER_DIRECTORY_IDENTITY_MISMATCH")
    for field in ("receipt_sha256", "expected_parsed_sha256"):
        if not isinstance(viewer[field], str) or not SHA_RE.fullmatch(viewer[field]):
            raise SafeFailure("INVALID_VIEWER_HASH")
    if viewer["receipt_filename"] != "PUBLIC_VIEWER_SOURCE_DESCRIPTOR.json" or not SHA_RE.fullmatch(viewer["original_private_receipt_sha256"]):
        raise SafeFailure("INVALID_PUBLIC_DESCRIPTOR_PROVENANCE")
    if tuple(str(public_row[k]) for k in ("corp_code", "year", "panel_rcept_no")) != identity(row):
        raise SafeFailure("VIEWER_SOURCE_IDENTITY_MISMATCH")
    if public_row["source_rcept_no"] != m["rcept_no"]:
        raise SafeFailure("VIEWER_ACTUAL_RECEIPT_MISMATCH")
    selected = viewer["current_selected"]
    if selected["rcpNo"] != m["rcept_no"] or not re.fullmatch(r"\d+", selected["dcmNo"]):
        raise SafeFailure("VIEWER_SELECTED_DOCUMENT_MISMATCH")
    requests = viewer["requests"]
    nodes = viewer["current_nodes"]
    if not requests or not nodes:
        raise SafeFailure("EMPTY_VIEWER_REQUESTS")
    main_receipts = [public_row["requested_rcept_no"]]
    if main_receipts[0] != public_row["source_rcept_no"]:
        main_receipts.append(public_row["source_rcept_no"])
    if len(requests) != len(main_receipts) + len(nodes):
        raise SafeFailure("VIEWER_REQUEST_COUNT_MISMATCH")
    for index, req in enumerate(requests, 1):
        if req["request_order"] != index or req["status_code"] != 200 or req["transport_error"] is not None or req["response_body_complete"] is not True:
            raise SafeFailure("INVALID_LOCKED_VIEWER_RESPONSE")
        if not isinstance(req["response_bytes"], int) or isinstance(req["response_bytes"], bool) or not 0 < req["response_bytes"] <= MAX_RESPONSE_BYTES or not SHA_RE.fullmatch(req["response_sha256"]):
            raise SafeFailure("INVALID_LOCKED_VIEWER_SIZE_OR_HASH")
        if index <= len(main_receipts):
            if req["endpoint"] != DART_MAIN or list(req["params"]) != ["rcpNo"] or req["params"]["rcpNo"] != main_receipts[index - 1] or req["headers"] != {}:
                raise SafeFailure("VIEWER_MAIN_REQUEST_IDENTITY_MISMATCH")
        else:
            node = nodes[index - len(main_receipts) - 1]
            if req["endpoint"] != DART_VIEWER or tuple(req["params"]) != VIEWER_PARAMS or req["params"] != {k: node[k] for k in VIEWER_PARAMS}:
                raise SafeFailure("VIEWER_NODE_REQUEST_IDENTITY_MISMATCH")
            if node["rcpNo"] != selected["rcpNo"] or node["dcmNo"] != selected["dcmNo"] or req["headers"] != {"Referer": DART_MAIN + "?rcpNo=" + selected["rcpNo"]}:
                raise SafeFailure("VIEWER_NODE_SOURCE_OR_HEADER_MISMATCH")
            for field in ("rcpNo", "dcmNo", "eleId", "offset", "length"):
                if not re.fullmatch(r"\d+", str(req["params"][field])):
                    raise SafeFailure("INVALID_VIEWER_PARAMETER")
            if req["params"]["dtd"] not in {"dart3.xsd", "dart4.xsd"}:
                raise SafeFailure("INVALID_VIEWER_DTD")


def load_bundle(bundle):
    bundle = Path(bundle).resolve()
    template_path = child(bundle, "PUBLIC_INPUT_TEMPLATE.json")
    cfg = read_locked_json(template_path, EXPECTED_TEMPLATE_SHA256)
    plan_path = child(bundle, cfg["source_plan"])
    viewer_rows_path = child(bundle, cfg["viewer_rows"])
    plan = read_locked_json(plan_path, cfg["source_plan_sha256"])
    public_viewers = read_locked_json(viewer_rows_path, cfg["viewer_rows_sha256"])
    if plan.get("schema") != "CURRENT_MEASUREMENT_SOURCE_PLAN_V1" or plan.get("method_version") != METHOD_VERSION or plan.get("selection_uses_classification_counts") is not False:
        raise SafeFailure("INVALID_SOURCE_PLAN_POLICY")
    rows = plan["rows"]
    if len(rows) != EXPECTED_RECORDS or len({identity(r) for r in rows}) != EXPECTED_RECORDS or len({identity(r)[:2] for r in rows}) != EXPECTED_RECORDS:
        raise SafeFailure("SOURCE_UNIVERSE_NOT_COMPLETE_UNIQUE")
    if not isinstance(public_viewers, list):
        raise SafeFailure("INVALID_VIEWER_ROWS")
    viewers = {tuple(str(r[k]) for k in ("corp_code", "year", "panel_rcept_no")): r for r in public_viewers}
    viewer_keys = {identity(r) for r in rows if r["source_kind"] == "locked_public_viewer_http"}
    if len(public_viewers) != EXPECTED_VIEWERS or len(viewers) != EXPECTED_VIEWERS or set(viewers) != viewer_keys:
        raise SafeFailure("VIEWER_UNIVERSE_MISMATCH")
    request_count = 0
    for index, row in enumerate(rows, 1):
        if row["index"] != index:
            raise SafeFailure("SOURCE_ROW_ORDER_MISMATCH")
        m = row["metadata"]
        if set(m) != {"corp_code", "year", "rcept_no", "stock_code", "corp_name", "market", "download_status", "text_source_rcept_no", "text_source_rcept_no_raw", "download_request_rcept_no", "source_resolution_basis"}:
            raise SafeFailure("UNAPPROVED_METADATA_FIELDS")
        for field, width in (("corp_code", 8), ("year", 4), ("rcept_no", 14), ("stock_code", 6), ("download_request_rcept_no", 14)):
            if not isinstance(m[field], str) or not re.fullmatch(r"\d{" + str(width) + "}", m[field]):
                raise SafeFailure("INVALID_SOURCE_IDENTITY_WIDTH")
        actual = m["download_request_rcept_no"]
        recorded = m["text_source_rcept_no"]
        if recorded and recorded != actual:
            raise SafeFailure("RECORDED_ACTUAL_RECEIPT_MISMATCH")
        if actual != m["rcept_no"] and (recorded != actual or m["source_resolution_basis"] != "explicit_production_manifest_record"):
            raise SafeFailure("UNSUPPORTED_ALTERNATIVE_RECEIPT")
        if set(row) & {"counts", "target", "expected_coefficient", "total_sentences", "bmi_sentence_count", "substantive_digital_sentence_count", "digital_sentence_count"}:
            raise SafeFailure("CLASSIFICATION_TARGETS_IN_SOURCE_PLAN")
        if row["source_kind"] == "locked_dart_zip":
            if row["raw_filename"] != f"{m['corp_code']}_{m['year']}_{actual}.zip" or not SHA_RE.fullmatch(row["expected_raw_sha256"]):
                raise SafeFailure("ZIP_SOURCE_FILENAME_OR_HASH_MISMATCH")
        elif row["source_kind"] == "locked_public_viewer_http":
            validate_viewer(row, viewers[identity(row)])
            request_count += len(row["viewer"]["requests"])
        else:
            raise SafeFailure("UNKNOWN_SOURCE_KIND")
    if request_count != EXPECTED_VIEWER_REQUESTS:
        raise SafeFailure("VIEWER_TOTAL_REQUEST_COUNT_MISMATCH")
    pins = {template_path: EXPECTED_TEMPLATE_SHA256, plan_path: cfg["source_plan_sha256"], viewer_rows_path: cfg["viewer_rows_sha256"]}
    descriptor_root = child(bundle, cfg["viewer_descriptor_root"])
    descriptors = {}
    for row in rows:
        if row["source_kind"] == "locked_public_viewer_http":
            viewer = row["viewer"]
            path = child(descriptor_root, viewer["capture_directory"]+"/"+viewer["receipt_filename"])
            if not existing(path, viewer["receipt_sha256"]):
                raise SafeFailure("BUNDLED_PUBLIC_DESCRIPTOR_MISSING")
            pins[path] = viewer["receipt_sha256"]
            descriptors[identity(row)] = path
    return rows, pins, descriptors


def existing(path, expected, expected_size=None, require_zip=False):
    if not path.exists():
        return False
    if not path.is_file() or sha_file(path) != expected or (expected_size is not None and path.stat().st_size != expected_size):
        raise SafeFailure("EXISTING_FILE_IDENTITY_MISMATCH", file=path.name)
    if require_zip and not zipfile.is_zipfile(path):
        raise SafeFailure("EXISTING_LOCKED_FILE_NOT_ZIP", file=path.name)
    return True


def put_verified(root, relative, body, expected, expected_size=None, require_zip=False):
    if not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES or sha_bytes(body) != expected or (expected_size is not None and len(body) != expected_size):
        raise SafeFailure("FETCHED_BODY_IDENTITY_MISMATCH", file=Path(relative).name)
    if require_zip and not zipfile.is_zipfile(io.BytesIO(body)):
        raise SafeFailure("FETCHED_BODY_NOT_ZIP", file=Path(relative).name)
    path = child(root, relative)
    if path.exists():
        if existing(path, expected, expected_size, require_zip) is not True:
            raise SafeFailure("FINAL_PATH_RACE_NOT_VALID", file=path.name)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path = child(root, relative)  # recheck resolved containment after creating parents
    temporary_relative = relative + ".acquire-" + uuid.uuid4().hex + ".tmp"
    temporary = child(root, temporary_relative)
    temporary_created = False
    try:
        with temporary.open("xb") as stream:
            temporary_created = True
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if not existing(temporary, expected, expected_size, require_zip):
            raise SafeFailure("WRITTEN_FILE_VERIFICATION_FAILED", file=path.name)
        # Hard-link creation is atomic and fails if the final path already exists.
        # No rename/replace fallback can overwrite an existing original.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if existing(path, expected, expected_size, require_zip) is not True:
                raise SafeFailure("FINAL_PATH_RACE_NOT_VALID", file=path.name)
            return False
        if not existing(path, expected, expected_size, require_zip):
            raise SafeFailure("WRITTEN_FILE_VERIFICATION_FAILED", file=path.name)
        return True
    finally:
        checked_temporary = child(root, temporary_relative)
        if temporary_created and checked_temporary == temporary and checked_temporary.is_file():
            checked_temporary.unlink()


class RequestsTransport:
    """Instantiated only after explicit --fetch and a complete metadata check."""
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False

    def get(self, endpoint, params, headers):
        if endpoint not in {DART_DOCUMENT, DART_MAIN, DART_VIEWER}:
            raise SafeFailure("ENDPOINT_NOT_ALLOWED")
        try:
            response = self.session.get(endpoint, params=params, headers=headers, timeout=(15, 90), allow_redirects=False, stream=True)
            with response:
                status = response.status_code
                body = bytearray()
                for block in response.iter_content(chunk_size=1024 * 1024):
                    body.extend(block)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise SafeFailure("RESPONSE_SIZE_LIMIT_EXCEEDED")
                return status, bytes(body)
        except SafeFailure:
            raise
        except Exception:
            raise SafeFailure("TRANSPORT_ERROR_NO_EXCEPTION_TEXT") from None

    def close(self):
        self.session.close()


def checked_get(transport, endpoint, params, headers, expected, expected_size=None, require_zip=False):
    # The safe receipt never receives params/headers, key, response text or URL.
    try:
        status, body = transport.get(endpoint, params=params, headers=headers)
    except SafeFailure:
        raise
    except Exception:
        raise SafeFailure("TRANSPORT_ERROR_NO_EXCEPTION_TEXT") from None
    if not isinstance(status, int) or not isinstance(body, bytes):
        raise SafeFailure("INVALID_TRANSPORT_RESPONSE")
    observed_hash = sha_bytes(body)
    if status != 200:
        raise SafeFailure("HTTP_STATUS_FAILURE", status=status, response_sha256=observed_hash)
    if require_zip and not zipfile.is_zipfile(io.BytesIO(body)):
        api_status = None
        try:
            result = ET.fromstring(body)
            candidate = result.findtext("status")
            if candidate is not None and re.fullmatch(r"\d{3}", candidate):
                api_status = candidate
        except (ET.ParseError, ValueError):
            pass
        raise SafeFailure("NON_ZIP_API_RESPONSE", api_status=api_status, response_sha256=observed_hash)
    if len(body) > MAX_RESPONSE_BYTES or observed_hash != expected or (expected_size is not None and len(body) != expected_size):
        raise SafeFailure("FETCHED_BODY_IDENTITY_MISMATCH", response_sha256=observed_hash)
    return body


def ordinary_directory_if_present(path):
    """No creation: reject files, dangling links and Windows reparse points."""
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return False
    if getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024) or stat.S_ISLNK(info.st_mode):
        raise SafeFailure("DATA_LAYOUT_LINK_OR_REPARSE_REJECTED")
    if not stat.S_ISDIR(info.st_mode):
        raise SafeFailure("DATA_LAYOUT_TARGET_NOT_DIRECTORY")
    return True


def inspect_standard_layout(data_root):
    """Check the exact named root/ancestors/children without resolving links."""
    root = Path(os.path.abspath(os.fspath(data_root)))
    for ancestor in reversed((root, *root.parents)):
        ordinary_directory_if_present(ancestor)
    present = {name: ordinary_directory_if_present(root / name) for name in STANDARD_DATA_DIRECTORIES}
    if os.path.normcase(str(root.resolve())) != os.path.normcase(str(root)):
        raise SafeFailure("DATA_LAYOUT_RESOLVED_PATH_DIFFERS")
    return root, present


def complete_standard_layout(data_root, actions):
    """Called only after successful fetch and metadata closure, never offline."""
    root, _ = inspect_standard_layout(data_root)
    if not ordinary_directory_if_present(root):
        raise SafeFailure("VERIFIED_DATA_ROOT_MISSING")
    for name in STANDARD_DATA_DIRECTORIES:
        # Inspect before each bounded mkdir; do not follow a changed parent.
        checked_root, _ = inspect_standard_layout(root)
        if checked_root != root:
            raise SafeFailure("DATA_LAYOUT_ROOT_CHANGED")
        path = root / name
        if ordinary_directory_if_present(path):
            actions.append({"directory": name, "status": "PRESERVED_EXISTING_ORDINARY_DIRECTORY"})
            continue
        try:
            path.mkdir(exist_ok=False)
            created = True
        except FileExistsError:
            created = False
        action = {"directory": name, "status": "CREATED_DIRECTORY_PENDING_RECHECK" if created else "RACE_TARGET_FOUND_PENDING_VERIFICATION"}
        actions.append(action)
        if not ordinary_directory_if_present(path):
            raise SafeFailure("STANDARD_DIRECTORY_CREATION_NOT_VERIFIED")
        inspect_standard_layout(root)
        action["status"] = "CREATED_EMPTY_STANDARD_DIRECTORY" if created else "PRESERVED_ORDINARY_DIRECTORY_AFTER_RACE"
    return sum(action["status"].startswith("CREATED_") for action in actions)


def acquire(bundle, data_root, fetch=False, viewer_receipts_dir=None, transport_factory=None):
    """Return only safe identities/status; failures never expose exception text."""
    result = {"mode":"EXPLICIT_FETCH" if fetch else "OFFLINE_CHECK_ONLY", "status":"STARTED", "records":[], "network_requests":0, "new_files_written":0, "classifier_or_model_executed":False, "historical_capture_receipts_regenerated":False, "public_descriptors_are_projections_not_new_captures":True, "standard_directory_actions":[], "new_standard_directories_created":0, "standard_layout_checked":False, "standard_layout_ready_for_stage":False, "missing_standard_directories":None, "payload_files_verified":False, "public_run_handoff_status":"NOT_READY"}
    transport = None
    pins = {}
    try:
        rows, pins, bundled_descriptors = load_bundle(bundle)
        data_root, standard_present = inspect_standard_layout(data_root)
        result["standard_layout_checked"] = True
        result["standard_layout_ready_for_stage"] = all(standard_present.values())
        result["missing_standard_directories"] = [name for name, present in standard_present.items() if not present]
        data_root = data_root.resolve()
        if data_root.exists() and not data_root.is_dir():
            raise SafeFailure("DATA_ROOT_NOT_DIRECTORY")
        descriptors = {}
        missing_zip = False
        # Validate every existing target and descriptor before any request/write.
        for row in rows:
            if row["source_kind"] == "locked_dart_zip":
                candidates = [child(data_root, root+"/"+row["raw_filename"]) for root in ("raw_dart", "supplemental_raw")]
                present = [existing(p, row["expected_raw_sha256"], require_zip=True) for p in candidates]
                missing_zip = missing_zip or not any(present)
            else:
                viewer = row["viewer"]
                if viewer_receipts_dir is not None:
                    relative = "viewer_descriptors/"+viewer["capture_directory"]+"/"+viewer["receipt_filename"]
                    existing(child(data_root, relative), viewer["receipt_sha256"])
                    source = child(Path(viewer_receipts_dir), viewer["capture_directory"]+"/"+viewer["receipt_filename"])
                    if not existing(source, viewer["receipt_sha256"]):
                        raise SafeFailure("EXPLICIT_PUBLIC_DESCRIPTOR_MISSING")
                    descriptors[identity(row)] = source
                for req in viewer["requests"]:
                    existing(child(data_root, "viewer_capture/"+viewer["capture_directory"]+"/http_responses/"+f"{req['request_order']:04d}_response_body.html"), req["response_sha256"], req["response_bytes"])
        api_key = None
        if fetch and missing_zip:
            api_key = os.environ.get("DART_API_KEY", "")
            if not re.fullmatch(r"[A-Za-z0-9]{40}", api_key):
                raise SafeFailure("DART_API_KEY_ENV_MISSING_OR_INVALID")
        if fetch:
            transport = (transport_factory or RequestsTransport)()
        for row in rows:
            m = row["metadata"]
            record = {"key":list(identity(row)), "source_kind":row["source_kind"], "files":[]}
            result["records"].append(record)
            if row["source_kind"] == "locked_dart_zip":
                filename = row["raw_filename"]
                present = [existing(child(data_root, root+"/"+filename), row["expected_raw_sha256"], require_zip=True) for root in ("raw_dart", "supplemental_raw")]
                if any(present):
                    record["files"].append({"file":filename,"status":"EXISTING_HASH_VERIFIED"})
                elif not fetch:
                    record["files"].append({"file":filename,"status":"MISSING"})
                else:
                    result["network_requests"] += 1
                    body = checked_get(transport, DART_DOCUMENT, [("crtfc_key", api_key), ("rcept_no", m["download_request_rcept_no"])], {}, row["expected_raw_sha256"], require_zip=True)
                    written = put_verified(data_root, "raw_dart/"+filename, body, row["expected_raw_sha256"], require_zip=True)
                    result["new_files_written"] += int(written)
                    record["files"].append({"file":filename,"status":"FETCHED_EXACT_HASH" if written else "EXISTING_HASH_VERIFIED"})
            else:
                viewer = row["viewer"]
                prefix = "viewer_capture/"+viewer["capture_directory"]+"/"
                record["files"].append({"file":viewer["capture_directory"]+"/"+viewer["receipt_filename"],"status":"BUNDLED_IMMUTABLE_PUBLIC_DESCRIPTOR_VERIFIED_NOT_NEW_CAPTURE"})
                if viewer_receipts_dir is not None:
                    receipt_name = "viewer_descriptors/"+viewer["capture_directory"]+"/"+viewer["receipt_filename"]
                    if existing(child(data_root, receipt_name), viewer["receipt_sha256"]):
                        record["files"].append({"file":receipt_name,"status":"EXISTING_PUBLIC_DESCRIPTOR_HASH_VERIFIED"})
                    elif not fetch:
                        record["files"].append({"file":receipt_name,"status":"MISSING_OPTIONAL_PUBLIC_DESCRIPTOR_COPY"})
                    else:
                        body = descriptors[identity(row)].read_bytes()
                        written = put_verified(data_root, receipt_name, body, viewer["receipt_sha256"])
                        result["new_files_written"] += int(written)
                        record["files"].append({"file":receipt_name,"status":"COPIED_IMMUTABLE_PUBLIC_DESCRIPTOR_NOT_NEW_CAPTURE"})
                for req in viewer["requests"]:
                    filename = prefix+"http_responses/"+f"{req['request_order']:04d}_response_body.html"
                    if existing(child(data_root, filename), req["response_sha256"], req["response_bytes"]):
                        record["files"].append({"file":filename,"status":"EXISTING_HASH_VERIFIED"})
                    elif not fetch:
                        record["files"].append({"file":filename,"status":"MISSING"})
                    else:
                        result["network_requests"] += 1
                        body = checked_get(transport, req["endpoint"], list(req["params"].items()), dict(req["headers"]), req["response_sha256"], req["response_bytes"])
                        written = put_verified(data_root, filename, body, req["response_sha256"], req["response_bytes"])
                        result["new_files_written"] += int(written)
                        record["files"].append({"file":filename,"status":"FETCHED_EXACT_HASH" if written else "EXISTING_HASH_VERIFIED"})
        missing = sum(item["status"].startswith("MISSING") for record in result["records"] for item in record["files"])
        result["missing_files"] = missing
        result["payload_files_verified"] = missing == 0
        result["status"] = "ALL_LOCKED_FILES_VERIFIED_NOT_A_REPRODUCTION_RUN" if missing == 0 else "OFFLINE_CHECK_INCOMPLETE_MISSING_FILES"
    except SafeFailure as error:
        result["status"] = "FAILED_CLOSED"
        result["error"] = {"code":error.code, **error.context}
    except Exception as error:
        result["status"] = "FAILED_CLOSED"
        result["error"] = {"code":"LOCAL_FAILURE_NO_EXCEPTION_TEXT", "exception_type":type(error).__name__}
    finally:
        if transport is not None:
            try:
                transport.close()
            except Exception:
                result["transport_close_failed"] = True
        unchanged = True
        unreadable = 0
        for path, expected in pins.items():
            try:
                unchanged = bool(path.is_file() and sha_file(path) == expected) and unchanged
            except Exception:
                unchanged = False
                unreadable += 1
        result["locked_metadata_unchanged"] = unchanged if pins else None
        result["locked_metadata_unreadable_count"] = unreadable
        if pins and result["locked_metadata_unchanged"] is not True:
            result["status"] = "FAILED_CLOSED"
            result["payload_files_verified"] = False
            result["error"] = {"code":"METADATA_CHANGED_DURING_ACQUISITION"}
    # Directory completion is a final handoff step, not a downloaded file. Do
    # not create even empty directories after failed/incomplete fixed inputs.
    if fetch and result["status"] == "ALL_LOCKED_FILES_VERIFIED_NOT_A_REPRODUCTION_RUN" and result["locked_metadata_unchanged"] is True:
        try:
            result["new_standard_directories_created"] = complete_standard_layout(data_root, result["standard_directory_actions"])
            result["standard_layout_ready_for_stage"] = True
            result["missing_standard_directories"] = []
        except SafeFailure as error:
            result["status"] = "FAILED_CLOSED"
            result["error"] = {"code":error.code, **error.context}
        except Exception as error:
            result["status"] = "FAILED_CLOSED"
            result["error"] = {"code":"STANDARD_DIRECTORY_COMPLETION_FAILURE_NO_EXCEPTION_TEXT", "exception_type":type(error).__name__}
        finally:
            result["new_standard_directories_created"] = sum(action["status"].startswith("CREATED_") for action in result["standard_directory_actions"])
    if result["status"] == "ALL_LOCKED_FILES_VERIFIED_NOT_A_REPRODUCTION_RUN" and result["payload_files_verified"] and result["standard_layout_ready_for_stage"]:
        result["public_run_handoff_status"] = "READY_FOR_PUBLIC_RUN_STAGE_NOT_A_REPRODUCTION_RUN"
    elif result["status"] == "FAILED_CLOSED":
        result["public_run_handoff_status"] = "NOT_READY_FAILED_ACQUISITION_OR_LAYOUT"
    elif result["payload_files_verified"]:
        result["public_run_handoff_status"] = "NOT_READY_MISSING_STANDARD_DIRECTORIES"
    else:
        result["public_run_handoff_status"] = "NOT_READY_MISSING_PAYLOAD_FILES"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--viewer-receipts-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    result = acquire(args.bundle, args.data_root, fetch=args.fetch, viewer_receipts_dir=args.viewer_receipts_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result["public_run_handoff_status"] == "READY_FOR_PUBLIC_RUN_STAGE_NOT_A_REPRODUCTION_RUN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
