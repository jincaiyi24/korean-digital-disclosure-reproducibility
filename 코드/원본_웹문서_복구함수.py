"""Recover missing Korean annual-report text from public DART viewer pages.

The OpenDART API key is deliberately not required.  DART correction filings
often point to a tiny corrected attachment instead of the full annual report;
this script follows the filing relationship, records the source document used,
and validates the recovered Korean text before it is admitted to the panel.
"""

from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "korea_digital_strategy_output"
PANEL_PATH = DATA_DIR / "04_digital_strategy_panel_korean_v2.xlsx"
AUDIT_PATH = SCRIPT_DIR / "11_missing_korean_reports_recovery.xlsx"
MAIN_URL = "https://dart.fss.or.kr/dsaf001/main.do"
VIEWER_URL = "https://dart.fss.or.kr/report/viewer.do"

# Two rows inherited the wrong periodic-report metadata from the English search.
RCEPT_OVERRIDES = {
    ("00117230", 2016): "20170331001576",  # TYM, annual report (2016.12)
    ("00141626", 2017): "20171010000798",  # Orient Bio, corrected annual (2017.03)
}

VIEW_DOC_RE = re.compile(
    r'viewDoc\("(\d+)",\s*"(\d+)",\s*"(\d+)",\s*"(\d+)",'
    r'\s*"(\d+)",\s*"([^"]+)"',
    re.S,
)
NODE_BLOCK_RE = re.compile(
    r"var\s+(node\d+)\s*=\s*\{\};(.*?)(?:treeData\.push|$)", re.S
)
NODE_FIELD_TEMPLATE = r"{node}\['([^']+)'\]\s*=\s*\"(.*?)\";"
HANGUL_RE = re.compile(r"[가-힣]")
SPACE_RE = re.compile(r"[ \t\u00a0]+")
BLANK_RE = re.compile(r"\n{3,}")


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            ),
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
        }
    )
    return session


def get_html(session: requests.Session, url: str, **kwargs: Any) -> str:
    response = session.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def parse_selected_document(page_html: str) -> dict[str, str]:
    match = VIEW_DOC_RE.search(page_html)
    if not match:
        raise ValueError("DART page does not contain a selected viewer document")
    rcp_no, dcm_no, ele_id, offset, length, dtd = match.groups()
    return {
        "rcpNo": rcp_no,
        "dcmNo": dcm_no,
        "eleId": ele_id,
        "offset": offset,
        "length": length,
        "dtd": dtd,
    }


def parse_nodes(page_html: str, rcp_no: str, dcm_no: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for node_name, body in NODE_BLOCK_RE.findall(page_html):
        fields = dict(
            re.findall(
                NODE_FIELD_TEMPLATE.format(node=re.escape(node_name)), body, re.S
            )
        )
        if fields.get("rcpNo") != rcp_no or fields.get("dcmNo") != dcm_no:
            continue
        key = (
            fields.get("eleId", ""),
            fields.get("offset", ""),
            fields.get("length", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(fields)
    rows.sort(key=lambda row: int(row.get("offset", "0") or 0))
    return rows


def annual_document_score(nodes: list[dict[str, str]]) -> tuple[int, int]:
    labels = " ".join(html.unescape(row.get("text", "")) for row in nodes)
    total_length = sum(int(row.get("length", "0") or 0) for row in nodes)
    heading_hits = sum(
        token in labels
        for token in ("사업보고서", "회사의 개요", "사업의 내용", "재무에 관한 사항")
    )
    return heading_hits, total_length


def find_base_annual_rcp(page_html: str, current_rcp: str) -> str | None:
    soup = BeautifulSoup(page_html, "lxml")
    candidates: list[str] = []
    for option in soup.find_all("option"):
        value = html.unescape(str(option.get("value", "")))
        label = " ".join(option.get_text(" ", strip=True).split())
        match = re.search(r"rcpNo=(\d{14})", value)
        if not match or "사업보고서" not in label:
            continue
        # No dcmNo means the option represents the filing body, not an attachment.
        if "dcmNo=" not in value:
            candidates.append(match.group(1))
    for candidate in candidates:
        if candidate != current_rcp:
            return candidate
    return candidates[0] if candidates else None


def resolve_annual_document(
    session: requests.Session, requested_rcp: str
) -> tuple[str, dict[str, str], list[dict[str, str]], str]:
    page_html = get_html(session, MAIN_URL, params={"rcpNo": requested_rcp})
    selected = parse_selected_document(page_html)
    nodes = parse_nodes(page_html, selected["rcpNo"], selected["dcmNo"])
    heading_hits, total_length = annual_document_score(nodes)
    if heading_hits >= 3 and total_length >= 100_000:
        return page_html, selected, nodes, "requested_filing_body"

    base_rcp = find_base_annual_rcp(page_html, requested_rcp)
    if not base_rcp:
        raise ValueError(
            f"Selected document is not a full annual report and no base filing was found "
            f"(headings={heading_hits}, bytes={total_length})"
        )
    base_html = get_html(session, MAIN_URL, params={"rcpNo": base_rcp})
    selected = parse_selected_document(base_html)
    nodes = parse_nodes(base_html, selected["rcpNo"], selected["dcmNo"])
    heading_hits, total_length = annual_document_score(nodes)
    if heading_hits < 3 or total_length < 100_000:
        raise ValueError(
            f"Resolved filing still fails annual-report structure checks "
            f"(headings={heading_hits}, bytes={total_length})"
        )
    return base_html, selected, nodes, "followed_base_annual_filing"


def viewer_section_to_text(raw_html: bytes) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def download_document_text(
    session: requests.Session,
    selected: dict[str, str],
    nodes: list[dict[str, str]],
    delay: float,
) -> str:
    sections: list[str] = []
    referer = f"{MAIN_URL}?rcpNo={selected['rcpNo']}"
    for node in nodes:
        params = {
            "rcpNo": selected["rcpNo"],
            "dcmNo": selected["dcmNo"],
            "eleId": node.get("eleId", ""),
            "offset": node.get("offset", ""),
            "length": node.get("length", ""),
            "dtd": node.get("dtd") or selected["dtd"],
        }
        response = session.get(
            VIEWER_URL,
            params=params,
            headers={"Referer": referer},
            timeout=90,
        )
        response.raise_for_status()
        section_text = viewer_section_to_text(response.content)
        if section_text:
            heading = html.unescape(node.get("text", "")).strip()
            sections.append(f"\n{heading}\n{section_text}" if heading else section_text)
        if delay > 0:
            time.sleep(delay)
    return BLANK_RE.sub("\n\n", "\n".join(sections)).strip() + "\n"


def validate_text(text: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", text)
    hangul_count = len(HANGUL_RE.findall(text))
    hangul_share = hangul_count / max(len(compact), 1)
    checks = {
        "char_count": len(text),
        "hangul_count": hangul_count,
        "hangul_share": hangul_share,
        "has_company_overview": "회사의 개요" in text,
        "has_business_section": "사업의 내용" in text,
        "has_financial_section": "재무에 관한 사항" in text,
    }
    checks["quality_pass"] = bool(
        checks["char_count"] >= 20_000
        and hangul_count >= 5_000
        and hangul_share >= 0.05
        and checks["has_company_overview"]
        and checks["has_business_section"]
    )
    return checks


def write_audit(audit_rows: list[dict[str, Any]]) -> None:
    """Persist progress after every firm-year so an interrupted run can resume."""
    audit_df = pd.DataFrame(audit_rows)
    summary = pd.DataFrame(
        [
            {"metric": "reports_processed", "value": len(audit_df)},
            {
                "metric": "reports_recovered",
                "value": int(audit_df.get("status", pd.Series(dtype=str)).eq("ok").sum()),
            },
            {
                "metric": "reports_failed",
                "value": int(audit_df.get("status", pd.Series(dtype=str)).eq("error").sum()),
            },
            {
                "metric": "metadata_overrides",
                "value": int(
                    pd.to_numeric(
                        audit_df.get("metadata_override", pd.Series(dtype=float)),
                        errors="coerce",
                    ).fillna(0).sum()
                ),
            },
        ]
    )
    with pd.ExcelWriter(AUDIT_PATH, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        audit_df.to_excel(writer, sheet_name="recovery_audit", index=False)


def run(delay: float, limit: int | None = None) -> None:
    panel = pd.read_excel(
        PANEL_PATH,
        dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
    )
    missing_mask = panel["v2_rescore_status"].ne("ok") | panel["text_cache_path"].map(
        lambda value: not Path(str(value)).exists()
    )
    missing = panel.loc[missing_mask].copy()
    if limit is not None:
        missing = missing.head(limit)
    print(f"Missing reports to recover: {len(missing):,}", flush=True)
    if missing.empty:
        print("All panel reports already have valid cached text; nothing to recover.")
        return

    session = build_session()
    audit_rows: list[dict[str, Any]] = []
    for sequence, (index, row) in enumerate(missing.iterrows(), start=1):
        corp_code = str(row["corp_code"]).zfill(8)
        year = int(row["year"])
        original_rcp = str(row["rcept_no"])
        requested_rcp = RCEPT_OVERRIDES.get((corp_code, year), original_rcp)
        print(
            f"[{sequence:02d}/{len(missing):02d}] {row['corp_name']} {year}: "
            f"{requested_rcp}",
            flush=True,
        )
        audit: dict[str, Any] = {
            "panel_row_index": int(index),
            "corp_code": corp_code,
            "corp_name": row.get("corp_name"),
            "stock_code": str(row.get("stock_code", "")).zfill(6),
            "year": year,
            "original_panel_rcept_no": original_rcp,
            "requested_rcept_no": requested_rcp,
            "metadata_override": requested_rcp != original_rcp,
            "original_report_name": row.get("report_name"),
            "status": "error",
            "error": "",
        }
        try:
            existing_candidates = sorted(
                (DATA_DIR / "raw_reports").glob(
                    f"{audit['stock_code']}_{year}_*.txt"
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if existing_candidates:
                existing_path = existing_candidates[0]
                existing_text = existing_path.read_text(encoding="utf-8", errors="replace")
                quality = validate_text(existing_text)
                if quality["quality_pass"]:
                    source_match = re.search(r"_(\d{14})\.txt$", existing_path.name)
                    audit.update(
                        {
                            "source_rcept_no": (
                                source_match.group(1) if source_match else requested_rcp
                            ),
                            "source_dcm_no": pd.NA,
                            "resolution_method": "resumed_validated_cache",
                            "node_count": pd.NA,
                            "recovered_text_path": str(existing_path.resolve()),
                            **quality,
                            "status": "ok",
                        }
                    )
                    audit_rows.append(audit)
                    write_audit(audit_rows)
                    print("  reused validated recovered file", flush=True)
                    continue

            _, selected, nodes, resolution = resolve_annual_document(session, requested_rcp)
            report_text = download_document_text(session, selected, nodes, delay)
            quality = validate_text(report_text)
            source_rcp = selected["rcpNo"]
            target_path = DATA_DIR / "raw_reports" / (
                f"{audit['stock_code']}_{year}_{source_rcp}.txt"
            )
            audit.update(
                {
                    "source_rcept_no": source_rcp,
                    "source_dcm_no": selected["dcmNo"],
                    "resolution_method": resolution,
                    "node_count": len(nodes),
                    "recovered_text_path": str(target_path.resolve()),
                    **quality,
                }
            )
            if not quality["quality_pass"]:
                raise ValueError(
                    "Recovered text failed quality validation: "
                    f"chars={quality['char_count']}, Hangul={quality['hangul_count']}, "
                    f"share={quality['hangul_share']:.3f}"
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(report_text, encoding="utf-8")
            audit["status"] = "ok"
        except Exception as exc:  # Keep the complete batch auditable.
            audit["error"] = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        audit_rows.append(audit)
        write_audit(audit_rows)
        time.sleep(max(2.0, delay * 4))

    audit_df = pd.DataFrame(audit_rows)
    write_audit(audit_rows)
    print(f"Recovery audit: {AUDIT_PATH}", flush=True)
    print(
        f"Recovered {audit_df['status'].eq('ok').sum():,}/{len(audit_df):,} reports.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.50)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(delay=max(args.delay, 0.0), limit=args.limit)
