"""Hash-pinned adapter; production rules are executed, never rewritten.

No production main, network request, checkpoint-resume logic or automatic pip
installer is executed. Only audited declarations/functions are loaded.
"""
from __future__ import annotations

import ast
import hashlib
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import types
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from lxml import etree

ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract(path: Path, names: set[str], namespace: dict) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = []
    receipt = []
    for node in tree.body:
        found = None
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found = node.name
        elif isinstance(node, ast.Assign):
            found = next((n.id for n in node.targets if isinstance(n, ast.Name) and n.id in names), None)
        if found:
            selected.append(node)
            receipt.append({"name": found, "line_start": node.lineno, "line_end": node.end_lineno,
                            "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
    missing = names - {x["name"] for x in receipt}
    if missing:
        raise RuntimeError(f"Missing pinned declarations: {sorted(missing)}")
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + selected, type_ignores=[]), str(path), "exec"), namespace)
    return receipt


class FrozenClassifier:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.config = json.loads((self.root / "config/frozen_sources.json").read_text(encoding="utf-8"))
        self.paths = {}
        for role, info in self.config["files"].items():
            path = self.root / info["path"]
            if file_sha256(path) != info["sha256"]:
                raise RuntimeError(f"SOURCE HASH MISMATCH: {role}. Do not run an unpinned version.")
            self.paths[role] = path
        self.receipt = {}
        # v2: only rx() constants and pure classification helpers, not evaluator.
        v2tree = ast.parse(self.paths["v2"].read_text(encoding="utf-8-sig"))
        v2names = {n.targets[0].id for n in v2tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Call)
                   and isinstance(n.value.func, ast.Name) and n.value.func.id == "rx"}
        v2names |= {"rx", "stale_historical", "stale_chronology", "platform_is_digital", "normalized_company_tokens", "classify_sentence_v2"}
        v2ns = {"re": re}
        self.receipt["v2"] = extract(self.paths["v2"], v2names, v2ns)

        splitter_names = {"DIGITAL_CORE_TERMS", "TABLE_NOISE_PATTERNS", "HANGUL_RE", "LATIN_RE", "KOREAN_TOKEN_RE", "SENTENCE_SPLIT_RE", "KOREAN_END_RE",
                          "compile_terms", "DIGITAL_CORE_PATTERNS", "TABLE_NOISE_REGEX", "normalize_text", "clean_segment", "match_terms",
                          "is_table_noise", "is_korean_narrative_segment", "split_korean_sentences"}
        splitns = {"html": html, "re": re}
        self.receipt["splitter"] = extract(self.paths["splitter"], splitter_names, splitns)
        self.splitter = types.SimpleNamespace(**splitns)
        decoderns = {"ET": ET, "json": json}
        self.receipt["decoder"] = extract(self.paths["decoder"], {"decode_bytes", "parse_open_dart_error"}, decoderns)
        v4tree = ast.parse(self.paths["v4"].read_text(encoding="utf-8-sig"))
        v4names = {n.targets[0].id for n in v4tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == "re" and n.value.func.attr == "compile"}
        v4names |= {"clean_stale_history", "classify_sentence_v4", "document_to_text", "score_one"}
        self.ns = {"re": re, "io": io, "zipfile": zipfile, "etree": etree, "Path": Path,
                   "classify_sentence_v2": v2ns["classify_sentence_v2"], "korean_base": self.splitter,
                   "dart_base": types.SimpleNamespace(**decoderns)}
        self.receipt["v4"] = extract(self.paths["v4"], v4names, self.ns)

    @property
    def version(self):
        return self.config["classifier_version"]

    def classify(self, sentence: str, year: int, corp_name: str) -> dict:
        # Do not normalize input here: v2 and v4 have their own original order.
        return self.ns["classify_sentence_v4"](sentence, int(year), corp_name)

    @staticmethod
    def category(flags: dict) -> str:
        if int(flags["bmi_sentence_v2"]) == 1:
            return "BMI"
        if int(flags["substantive_digital_sentence_v2"]) == 1:
            return "NonBMI"
        return "Exclude"

    def decode(self, raw: bytes) -> str:
        return self.ns["document_to_text"](raw)

    def score_original_text_file(self, task: dict):
        return self.ns["score_one"](task)

    def unique_sentences(self, text: str):
        seen = set()
        for sentence in self.splitter.split_korean_sentences(text):
            key = "".join(sentence.lower().split())
            if key and key not in seen:
                seen.add(key)
                yield sentence

