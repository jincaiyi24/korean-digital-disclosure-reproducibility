"""Source-aware parsing without changing the frozen sentence classifier.

Parser selection uses production download provenance only, never target counts.
Explicit reused_legacy text follows its original legacy ZIP/HTML parser.
Other download states use the surviving current production XML parser, with an
explicit limitation: the historical download-batch parser snapshot is missing.
"""
from __future__ import annotations

import html
import io
import re
import warnings
import zipfile

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from lxml import html as lxml_html

from frozen_api import FrozenClassifier, extract

PARSER_POLICY_VERSION = "production_download_provenance_no_count_fitting_v1"
ALLOWED_STATES = {"reused_legacy", "downloaded", "cached_new", "downloaded_fallback_latest_correction"}


class SourceAwareClassifier(FrozenClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        namespace = {"BeautifulSoup": BeautifulSoup, "lxml_html": lxml_html,
                     "html_lib": html, "re": re, "io": io, "zipfile": zipfile}
        self.receipt["legacy_cache_parser"] = extract(
            self.paths["decoder"], {"decode_bytes", "normalize_text", "markup_to_text", "zip_document_to_text"}, namespace)
        self.legacy_namespace = namespace

    @staticmethod
    def parser_policy(download_status: str) -> dict:
        if download_status not in ALLOWED_STATES:
            raise ValueError("Unsupported original production download_status; do not infer a parser")
        if download_status == "reused_legacy":
            return {"parser_policy_version": PARSER_POLICY_VERSION,
                    "parser_name": "legacy_zip_document_to_text_bs4_lxml_html",
                    "parser_selection_basis": "explicit reused_legacy record; v4 download_one lines 591-593 reuses legacy text; legacy download_report_text lines 1456-1458 calls zip_document_to_text lines 693-704",
                    "parser_provenance_status": "legacy_cache_source_code_path_explicit; original text bytes unavailable",
                    "historical_parser_batch_snapshot_available": False}
        return {"parser_policy_version": PARSER_POLICY_VERSION,
                "parser_name": "surviving_v4_document_to_text_lxml_xml",
                "parser_selection_basis": "surviving v4 download_one lines 603/626 calls document_to_text; no count-dependent parser choice",
                "parser_provenance_status": "historical download/cached_new batch parser version not archived; current surviving source used, not proof of original text identity",
                "historical_parser_batch_snapshot_available": False}

    def decode_from_production_source(self, raw: bytes, download_status: str) -> tuple[str, dict]:
        policy = self.parser_policy(download_status)
        if download_status == "reused_legacy":
            # The original base module suppresses this warning globally. Match
            # that display-only policy locally; never change the parser choice.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
                text, members = self.legacy_namespace["zip_document_to_text"](raw)
            policy["legacy_zip_member_count"] = members
            return text, policy
        return self.decode(raw), policy
