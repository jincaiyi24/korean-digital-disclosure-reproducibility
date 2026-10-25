"""Collect and score Korean DART annual reports for KOSPI and KOSDAQ firms.

The pipeline is resumable. API responses, extracted report text, checkpoints,
and research-ready CSV files are stored separately from the earlier KOSPI run.
The API key is read only from DART_API_KEY.
"""

from __future__ import annotations

import argparse
import calendar
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from lxml import etree


WORKSPACE = Path(__file__).resolve().parents[2]
LEGACY_CODE_DIR = WORKSPACE / "outputs"
LEGACY_DATA_DIR = LEGACY_CODE_DIR / "korea_digital_strategy_output"
OUTPUT_DIR = WORKSPACE / "outputs" / "kosdaq_full_market_20260815"
CACHE_DIR = OUTPUT_DIR / "_cache"
LIST_CACHE_DIR = CACHE_DIR / "dart_list"
REPORT_DIR = CACHE_DIR / "raw_reports"
CHECKPOINT_PATH = CACHE_DIR / "report_download_checkpoint.csv"
SECTION_DIR = CACHE_DIR / "strategy_sections"
SECTION_CHECKPOINT_PATH = CACHE_DIR / "strategy_section_checkpoint.csv"
SCORE_CHECKPOINT_PATH = CACHE_DIR / "score_checkpoint_v4_final.csv"
EXAMPLE_CHECKPOINT_PATH = CACHE_DIR / "sentence_example_checkpoint_v4_final.csv"
SOURCE_CORP_PATH = LEGACY_DATA_DIR / "01_listed_corporations_from_dart.xlsx"
LEGACY_REPORT_DIR = LEGACY_DATA_DIR / "raw_reports"

START_YEAR = 2015
END_YEAR = 2025
API_BASE = "https://opendart.fss.or.kr/api"
MARKET_LABELS = {"Y": "KOSPI", "K": "KOSDAQ"}
FINANCIAL_KSIC_PREFIXES = ("64", "65", "66")
KIND_DELIST_URL = "https://kind.krx.co.kr/investwarn/delcompany.do"

sys.path.insert(0, str(LEGACY_CODE_DIR))
import korea_digital_strategy_pipeline as dart_base  # noqa: E402
import korea_digital_strategy_korean_pipeline as korean_base  # noqa: E402
from ssci_korean_classifier_v2 import classify_sentence_v2  # noqa: E402


# Clean-Unicode v4 vetoes target recurrent false positives found in first-filed
# KOSPI/KOSDAQ annual reports. They supplement, rather than replace, the v2
# technology/action/attribution gates.
CLEAN_FOCAL_RE = re.compile(
    r"\ub2f9\uc0ac|\ub2f9\ud68c\uc0ac|\ud68c\uc0ac\ub294|\ubcf8\uc0ac\ub294|\uc6b0\ub9ac\s*\ud68c\uc0ac|"
    r"\ub2f9\s*\uadf8\ub8f9|\uc5f0\uacb0\ud68c\uc0ac|\uc885\uc18d\ud68c\uc0ac"
)
MARKET_TREND_RE = re.compile(
    r"\uc2dc\uc7a5\uc870\uc0ac|\ubcf4\uace0\uc11c\uc5d0\s*\ub530\ub974\uba74|\uac83\uc73c\ub85c\s*(?:\uc804\ub9dd|\uc608\uc0c1)|"
    r"\uc2dc\uc7a5\s*\uaddc\ubaa8|\uc131\uc7a5\ub960|\uc218\uc694.{0,30}\uc99d\uac00|\uc0b0\uc5c5\uc740|\uc2dc\uc7a5\uc740|"
    r"\uae30\uc220\uc774\s*\ubc1c\uc804\ud568\uc5d0\s*\ub530\ub77c|\uc9d1\uc911\ub3c4\uac00\s*\uc804\ud658"
)
EXTERNAL_ACTOR_RE = re.compile(
    r"(?<![A-Za-z])[A-Za-z][A-Za-z0-9.\-]{1,30}\s*\uc5d0\uc11c.{0,100}(?:\uac1c\ubc1c|\uc0dd\uc0b0|\ucd9c\uc2dc|\uc81c\uacf5)"
)
GENERIC_EXTERNAL_ACTOR_RE = re.compile(
    r"(?:\ube45\ud14c\ud06c\s*\uae30\uc5c5\ub4e4|\uae30\uc5c5\ub4e4|\uc5c5\uccb4\ub4e4|IT\uc11c\ube44\uc2a4\s*\uc5c5\uccb4\ub4e4|"
    r"\ud1b5\uc2e0\uc0ac|\ud1b5\uc2e0\uc0ac\uc5c5\uc790|\uace0\uac1d\uc0ac|\ud3ec\ud138\s*\ud50c\ub7ab\ud3fc\ub4e4).{0,80}"
    r"(?:\uac1c\ubc1c|\ub3c4\uc785|\uc6b4\uc601|\uc81c\uacf5|\uad6c\ucd95|\ud22c\uc790|\ucd9c\uc2dc|\uc0c1\uc6a9\ud654)"
)
STATIC_SCOPE_RE = re.compile(
    r"(?:\uac1c\ubc1c|\uc81c\uc870)\s*\ubc0f\s*\ud310\ub9e4(?:\s*\uc0ac\uc5c5)?\s*$|"
    r"\uc0ac\uc5c5\s*\ubaa9\uc801|\uc0ac\uc5c5\uc758\s*\uac1c\uc694|\uc0ac\uc5c5\uc758\s*\ub0b4\uc6a9\s*$|"
    r"\uac1c\uc694\s*(?:\ub2f9\uc0ac|\ud68c\uc0ac)\ub294|\uc9c0\ubc30\ud68c\uc0ac\s*\ubc0f\s*\uc885\uc18d\ud68c\uc0ac\ub294|"
    r"\uacf5\uc2dc\ub300\uc0c1\uc0ac\uc5c5\uad6c\ubd84|\ud310\ub9e4\uacbd\ub85c|"
    r"\uc790\uc6d0\s*\uc870\ub2ec\s*\uc0c1\ud669|\uc8fc\ub825\s*\uc81c\ud488(?:\uc740|\uc73c\ub85c)|"
    r"\uc18c\ud504\ud2b8\uc6e8\uc5b4.{0,60}(?:\uacf5\uae09\uc5c5|\ud310\ub9e4\uc5c5)\s*\d*\.?\s*$"
)
SEGMENT_LIST_RE = re.compile(
    r"\uc0ac\uc5c5\ubd80\ubb38\ubcc4\s*\ud604\ud669|\uc0ac\uc5c5\ubd80\ubb38.{0,100}(?:\uad6c\uc131|\ud604\ud669)|"
    r"\ubd80\ubb38\uc73c\ub85c\s*\uad6c\uc131|\uc8fc\uc694\s*\uc81c\ud488.{0,80}\ud604\ud669|"
    r"\ubd80\ubb38\ubcc4\s*\uc8fc\uc694\s*\uc81c\ud488|\uc8fc\uc694\s*\uc0ac\uc5c5\uc758\s*\ub0b4\uc6a9|"
    r"\ud310\ub9e4\uacbd\ub85c|\ud310\ub9e4\uc804\ub7b5\s*\uc0ac\uc5c5\ubd80\ubb38|(?:\u2192.{0,160}){3}"
)
ACTIVE_CLEAN_RE = re.compile(
    r"\uc6b4\uc601\s*\uc911|\uc6b4\uc601(?:\ud558|\ud574\uc624)|\uc81c\uacf5\s*\uc911|\ud310\ub9e4\s*\uc911|\uc2dc\ud310\s*\uc911|"
    r"(?<!\ucc3d)\ucd9c\uc2dc|\ub860\uce6d|\uc624\ud508|\uac1c\uc7a5|\uc124\ub9bd|\ud569\ubcd1|\uccb4\uacb0|\uc644\ub8cc|\uc778\uc218|\ucde8\ub4dd|"
    r"\uad6c\ucd95(?:\ud558|\ud588|\ub418|\uc911|\uc644\ub8cc)|\uad6c\ucd95.{0,30}(?:\uc9c4\ud589|\uc218\ud589)|"
    r"\ud22c\uc790(?:\ub97c|\ub3c4|\uac00)?\s*(?:\ud558\uc600|\ud588|\ud558\uace0|\ud560|\uc911|\uc9c4\ud589|\uc9d1\ud589|\uacc4\ud68d|\ud655\ub300|\uc2b9\uc778)|\ub3c4\uc785(?:\ud558|\ud588|\ub418|\uc911|\uc644\ub8cc)|"
    r"\uac1c\ubc1c(?:\ud558|\ud588|\ud574\uc624|\uc911|\uc644\ub8cc)|\uac1c\ubc1c.{0,30}(?:\uc9c4\ud589|\uc218\ud589|\uc644\ub8cc)|"
    r"\uc9c4\ud589\s*\uc911.{0,30}\uac1c\ubc1c|\ud604\uc7ac.{0,80}\uac1c\ubc1c|\uc5f0\uad6c\uac1c\ubc1c\s*(?:\uc218\ud589|\uc9c4\ud589)|"
    r"(?:\uc5f0\uad6c\uacfc\uc81c|\uc790\uccb4\uacfc\uc81c|\uac1c\ubc1c\uc2e4\uc801|\uc5f0\uad6c\uac1c\ubc1c\uc2e4\uc801)|"
    r"\uacf5\uae09.{0,20}(?:\uc911|\ub418|\ud558|\uc2dc\uc791)|\ub0a9\ud488(?:\ud558|\uc911)|"
    r"(?:\uc11c\ube44\uc2a4|\uc194\ub8e8\uc158).{0,8}\uc81c\uacf5(?:\ud558\uace0|\ud574\uc624|\uc911)|"
    r"\ud310\ub9e4.{0,20}(?:\ud1b5\ud574|\ub9e4\ucd9c)|\ub9e4\ucd9c.{0,15}(?:\ucc3d\ucd9c|\ubc1c\uc0dd)|"
    r"\ud504\ub85c\uc81d\ud2b8\ub97c?\s*\uc218\ud589|\uacc4\uc57d.{0,20}(?:\uccb4\uacb0|\ub9fa)|\ucd94\uc9c4\s*(?:\ud558\uae30\ub85c|\uacc4\ud68d|\uc911|\ud558\uace0)|"
    r"(?:\uc804\ub7b5|\uacc4\ud68d|\uc608\uc815|\ubaa9\ud45c).{0,60}(?:\uad6c\ucd95|\uac15\ud654|\ud655\ub300|\uc804\ud658|\ub3c4\uc785|\uac1c\ubc1c|\ucd9c\uc2dc)|"
    r"(?:\uad6c\ucd95|\uac15\ud654|\ud655\ub300|\uc804\ud658|\ub3c4\uc785|\uac1c\ubc1c|\ucd9c\uc2dc).{0,45}(?:\uacc4\ud68d|\uc608\uc815|\ucd94\uc9c4|\ud558\uaca0\uc2b5\ub2c8\ub2e4|\ud558\uae30\ub85c)|"
    r"\uc785\uc810.{0,20}(?:\uc911|\uc644\ub8cc)|\uc9c4\ucd9c(?:\ud558|\uc911)|\ubaa8\uc0c9\s*\uc911|"
    r"(?:\uac1c\ubc1c|\uacf5\uae09|\uc6b4\uc601|\ud310\ub9e4|\uc11c\ube44\uc2a4).{0,20}\uc2dc\uc791"
)
HIGH_CONFIDENCE_ACTION_RE = re.compile(
    r"\uc5f0\uad6c\uacfc\uc81c|\uc790\uccb4\uacfc\uc81c|\uac1c\ubc1c\uc2e4\uc801|\uc5f0\uad6c\uac1c\ubc1c\uc2e4\uc801|"
    r"\uac1c\ubc1c(?:\ud558\uc600|\ud588|\ud558\uace0|\uc911|\uc644\ub8cc|\uc9c4\ud589)|\uc9c4\ud589\s*\uc911.{0,30}\uac1c\ubc1c|"
    r"\ud604\uc7ac.{0,80}\uac1c\ubc1c|\uad6c\ucd95(?:\ud558\uc600|\ud588|\ud558\uace0|\uc644\ub8cc|\uc911)|\ub3c4\uc785(?:\ud558\uc600|\ud588|\ud558\uace0|\uc644\ub8cc|\uc911)|"
    r"\uacc4\uc57d.{0,25}(?:\uccb4\uacb0|\ub9fa)|\ud22c\uc790.{0,15}(?:\uc9d1\ud589|\uc644\ub8cc|\uc2b9\uc778)|"
    r"(?<!\ucc3d)\ucd9c\uc2dc(?:\ud558|\ud588|\ud560|\uc911|\uc608\uc815)?|\ub860\uce6d|\uc624\ud508|\uac1c\uc7a5|\uc124\ub9bd|\ud569\ubcd1|\uc778\uc218|"
    r"\uc6b4\uc601(?:\ud558\uace0|\ud574\uc624|\uc911)|(?:\uc11c\ube44\uc2a4|\uc194\ub8e8\uc158).{0,8}\uc81c\uacf5(?:\ud558\uace0|\ud574\uc624|\uc911)|"
    r"(?:\uc81c\ud488|\uc11c\ube44\uc2a4|\uc194\ub8e8\uc158|\uc7a5\ube44|\uc2dc\uc2a4\ud15c).{0,35}\uacf5\uae09(?:\ud558\uace0|\uc911|\ub418)|\ub0a9\ud488(?:\ud558\uace0|\uc911)|"
    r"\uc804\ub7b5\uc801\s*\uc81c\ud734|\ucd94\uc9c4\s*\ud558\uae30\ub85c|"
    r"(?:\uc804\ub7b5|\uacc4\ud68d).{0,60}(?:\uad6c\ucd95|\uac15\ud654|\ud655\ub300|\uc804\ud658|\ub3c4\uc785|\uac1c\ubc1c|\ucd9c\uc2dc)"
)
PRODUCT_REFERENCE_RE = re.compile(
    r"\ud3ec\ud2b8\ud3f4\ub9ac\uc624\s*(?:\.jpg)?|\ud65c\uc6a9\ubd84\uc57c\s*(?:\.jpg)?|"
    r"\uc81c\ud488\uad70\s*(?:\uc18c\uac1c|\ud604\ud669)|\uc81c\ud488\ubcc4\s*\ud604\ud669"
)
NON_DIGITAL_PLATFORM_RE = re.compile(
    r"(?:\ubc14\uc774\uc624|\uc784\uc0c1|\uc2e0\uc57d|\uc81c\uc57d|\uc0c1\uc6a9\uae30\uc220|\uc5f0\uad6c).{0,35}\ud50c\ub7ab\ud3fc|"
    r"\ud50c\ub7ab\ud3fc\s*\ud615\uc2dd\uc73c\ub85c\s*\uacf5\uc815\ub77c\uc778|(?:\uacf5\uc815\ub77c\uc778|\uc784\uac00\uacf5|2\ucc28\uc804\uc9c0\uc18c\uc7ac).{0,40}\ud50c\ub7ab\ud3fc"
)
DIGITAL_PLATFORM_QUALIFIER_RE = re.compile(
    r"AI|\uc778\uacf5\uc9c0\ub2a5|\ub370\uc774\ud130|\uc628\ub77c\uc778|\uc18c\ud504\ud2b8\uc6e8\uc5b4|IT|\ud074\ub77c\uc6b0\ub4dc|\ub514\uc9c0\ud138|IoT"
)
VAGUE_STRATEGY_RE = re.compile(
    r"\ub178\ub825\ud558\uc600|\ub9e4\uc9c4|\uc2dc\uc7a5\uc744?\s*\uc120\ub3c4|\uc8fc\uc5ed|\ucc45\uc784.{0,20}\ub2e4\ud558|"
    r"\ucd5c\uace0\uc758\s*\uae30\uc220\uc744?\s*\ubcf4\uc720"
)
DIGITAL_DEMAND_EXPOSURE_RE = re.compile(
    r"(?:5G|AI|\uc778\uacf5\uc9c0\ub2a5|\uc790\uc728\s*\uc8fc\ud589).{0,120}(?:\uc218\uc694|\uc2dc\uc7a5).{0,120}"
    r"(?:\uc18c\uc7ac|\ud654\ud559|\ubd80\ud488|\uc6d0\ub8cc|\ubc18\ub3c4\uccb4\s*\uacf5\uc815)|"
    r"(?:\ubc18\ub3c4\uccb4\s*\uacf5\uc815\uc6a9\s*\ud654\ud559\s*\uc18c\uc7ac).{0,120}(?:\uc218\uc694|5G|AI)"
)
CUSTOMER_OR_COMPETITOR_CONTEXT_RE = re.compile(
    r"(?:\uace0\uac1d|\ubc29\uc1a1/\ud1b5\uc2e0\uc0ac\uc5c5\uc790).{0,240}(?:\ud22c\uc790|\ud655\ub300).{0,120}\uc804\ub9dd|"
    r"(?:\ub124\uc774\ubc84|\uce74\uce74\uc624).{0,500}(?:\ub124\uc774\ubc84|\uce74\uce74\uc624)"
)
FORMER_BUSINESS_RE = re.compile(
    r"\uc124\ub9bd\s*(?:\ub2f9\uc2dc|\ubc30\uacbd)|\uc131\uc7a5\s*\uacfc\uc815|\ud569\ubcd1\s*\uc804|\ud569\ubcd1\uc804|\uacfc\uac70.{0,40}\uc8fc\ub41c\s*\uc0ac\uc5c5|"
    r"\uc18c\ud504\ud2b8\uc6e8\uc5b4.{0,100}\ubcc0\uacbd\ub418\uc5c8"
)
CLEAN_ACCOUNTING_OR_ENTITY_LIST_RE = re.compile(
    r"\ub300\uc5ec\uae08|\ud68c\uc218\uac00\ub2a5\uc131|\ucda9\ub2f9\uae08|\ubb34\ud615\uc790\uc0b0|\ucde8\ub4dd\uc6d0\uac00|"
    r"\uc5f0\uacb0\uc81c\uc678|\uc5f0\uacb0\uc5d0\s*\ud3ec\ud568|\uc2e0\uaddc\ub85c\s*\uc5f0\uacb0|\uc9c0\ubd84\uc728\s*\ubcc0\ub3d9|"
    r"\uccad\uc0b0\uc73c\ub85c\s*\uc778\ud55c\s*\uc81c\uc678|\uc8fc\uc694\s*\uc885\uc18d\uae30\uc5c5|\ud2b9\uc218\uad00\uacc4\uc790|"
    r"\uc9c0\ubd84\ubc95|\uc720\uc0c1\uc99d\uc790|\uc8fc\uc2dd.{0,30}\ucde8\ub4dd|\uae30\ucd08.{0,30}\ucde8\ub4dd.{0,30}\ucc98\ubd84|"
    r"\uc2e0\uaddc\uc124\ub9bd\uc5d0\s*\ub530\ub978\s*\uc9c0\ubc30\ub825|\uc7ac\ubb34\uc0c1\ud0dc\ud45c|\ucd9c\uc790\ubaa9\uc801|\uc7a5\ubd80\uac00\uc561|"
    r"\uc190\uc0c1\ud3c9\uac00|\uc190\uc0c1\uc744\s*\uc778\uc2dd|\uae30\uc5c5\ud68c\uacc4\uae30\uc900\uc11c|\uc218\ud589\uc758\ubb34|"
    r"\uc9c0\ubc30\ub825\uc744\s*\ud589\uc0ac|\uc885\uc18d\uae30\uc5c5\uc73c\ub85c\s*\ubd84\ub958|\ub2f9\uae30\s*\uc911\s*\uc885\uc18d\uae30\uc5c5|\uc778\uc801\ubd84\ud560|"
    r"\ud1b5\uc81c\uc758\s*\uc774\uc804|\uc218\uc775\uc774\s*\uc778\uc2dd|\uc218\uc775\s*\uc778\uc2dd\uc2dc\uae30"
)


def clean_stale_history(sentence: str, report_year: int) -> bool:
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", sentence)]
    short_years = [
        2000 + int(value)
        for value in re.findall(r"(?<!\d)(\d{2})\s*\ub144(?:\ub3c4)?", sentence)
        if int(value) <= report_year % 100
    ]
    years.extend(short_years)
    if not years or max(years) > report_year - 2:
        return False
    current_marker = re.search(
        r"\ud604\uc7ac|\ub2f9\uae30|\uc9c0\uc18d\uc801|\uc6b4\uc601\s*(?:\uc911|\ud558\uace0)|\uac1c\ubc1c\s*\uc911|\uacf5\uae09\s*\uc911",
        sentence,
    )
    return not bool(current_marker)


def classify_sentence_v4(sentence: str, report_year: int, corp_name: str) -> dict[str, Any]:
    labels = classify_sentence_v2(sentence, report_year, corp_name)
    compact_sentence = re.sub(r"\s+", "", sentence)
    compact_name = re.sub(r"\(\uc8fc\)|\uc8fc\uc2dd\ud68c\uc0ac|\s", "", str(corp_name or ""))
    focal = bool(CLEAN_FOCAL_RE.search(sentence) or (len(compact_name) >= 2 and compact_name in compact_sentence))
    active_clean = bool(ACTIVE_CLEAN_RE.search(sentence))
    high_confidence_action = bool(HIGH_CONFIDENCE_ACTION_RE.search(sentence))
    implementation_evidence = bool(active_clean and (focal or high_confidence_action))
    # Forecasts and industry descriptions remain non-substantive even when a
    # sentence mentions the focal firm, unless it also states a concrete act.
    market_trend_without_firm = bool(
        MARKET_TREND_RE.search(sentence)
        and (not focal or not implementation_evidence)
    )
    external_actor_without_firm = bool(
        (EXTERNAL_ACTOR_RE.search(sentence) or GENERIC_EXTERNAL_ACTOR_RE.search(sentence))
        and not focal
    )
    static_scope = bool(
        (STATIC_SCOPE_RE.search(sentence) and not implementation_evidence)
        or (PRODUCT_REFERENCE_RE.search(sentence) and not focal)
    )
    segment_list = bool(SEGMENT_LIST_RE.search(sentence))
    nondigital_platform = bool(
        NON_DIGITAL_PLATFORM_RE.search(sentence)
        and not DIGITAL_PLATFORM_QUALIFIER_RE.search(sentence)
    )
    missing_implicit_implementation = bool(
        labels.get("substantive_digital_sentence_v2")
        and not implementation_evidence
    )
    vague_strategy = bool(VAGUE_STRATEGY_RE.search(sentence) and not implementation_evidence)
    demand_exposure = bool(DIGITAL_DEMAND_EXPOSURE_RE.search(sentence))
    customer_or_competitor_context = bool(CUSTOMER_OR_COMPETITOR_CONTEXT_RE.search(sentence))
    former_business = bool(FORMER_BUSINESS_RE.search(sentence))
    accounting_or_entity_list = bool(CLEAN_ACCOUNTING_OR_ENTITY_LIST_RE.search(sentence))
    stale_history = clean_stale_history(sentence, int(report_year))
    veto_substance = bool(
        market_trend_without_firm or external_actor_without_firm or static_scope or segment_list
        or nondigital_platform or missing_implicit_implementation or vague_strategy
        or demand_exposure or customer_or_competitor_context or former_business
        or accounting_or_entity_list or stale_history
    )
    if veto_substance:
        labels["substantive_digital_sentence_v2"] = 0
        labels["bmi_sentence_v2"] = 0
        labels["market_signal_sentence_v2"] = 0
    if nondigital_platform:
        labels["digital_sentence_v2"] = 0
        labels["boilerplate_digital_sentence_v2"] = 0
    labels["v4_market_trend_without_firm"] = int(market_trend_without_firm)
    labels["v4_external_actor_without_firm"] = int(external_actor_without_firm)
    labels["v4_static_scope_or_segment"] = int(static_scope or segment_list)
    labels["v4_nondigital_platform"] = int(nondigital_platform)
    labels["v4_missing_implicit_implementation"] = int(missing_implicit_implementation)
    labels["v4_vague_strategy"] = int(vague_strategy)
    labels["v4_digital_demand_exposure"] = int(demand_exposure)
    labels["v4_customer_or_competitor_context"] = int(customer_or_competitor_context)
    labels["v4_former_business"] = int(former_business)
    labels["v4_accounting_or_entity_list"] = int(accounting_or_entity_list)
    labels["v4_high_confidence_action"] = int(high_confidence_action)
    labels["v4_stale_history"] = int(stale_history)
    return labels


def api_key() -> str:
    value = os.getenv("DART_API_KEY", "").strip()
    if not value:
        raise SystemExit("DART_API_KEY is not set.")
    return value


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, CACHE_DIR, LIST_CACHE_DIR, REPORT_DIR, SECTION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def quarter_windows(start_year: int, end_date: date) -> Iterable[tuple[date, date]]:
    cursor = date(start_year, 1, 1)
    while cursor <= end_date:
        quarter_end_month = ((cursor.month - 1) // 3 + 1) * 3
        end = date(cursor.year, quarter_end_month, calendar.monthrange(cursor.year, quarter_end_month)[1])
        end = min(end, end_date)
        yield cursor, end
        cursor = end.replace(day=1)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def request_json(params: dict[str, Any], cache_path: Path, refresh: bool = False) -> dict[str, Any]:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = dict(params)
    payload["crtfc_key"] = api_key()
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(f"{API_BASE}/list.json", params=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            status = str(data.get("status", ""))
            if status not in {"000", "013"}:
                raise RuntimeError(f"OpenDART {status}: {data.get('message')}")
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.08)
            return data
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(attempt * 2, 8))
    raise RuntimeError(f"OpenDART list request failed: {last_error}")


def company_profile(corp_code: str, refresh: bool = False) -> dict[str, Any]:
    cache_path = CACHE_DIR / "company_profiles" / f"{corp_code}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(
                f"{API_BASE}/company.json",
                params={"crtfc_key": api_key(), "corp_code": corp_code}, timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("status")) != "000":
                raise RuntimeError(f"OpenDART {payload.get('status')}: {payload.get('message')}")
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.08)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    return {"status": "error", "message": str(last_error), "corp_code": corp_code}


def parse_report_period(report_name: Any) -> tuple[int | None, int | None]:
    match = re.search(r"\((20\d{2})\.(\d{2})\)", str(report_name or ""))
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def collect_delisted_companies(corp: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Download KRX/KIND delistings and map every ticker to an OpenDART corp code."""
    cache_path = CACHE_DIR / "kind_delisted_2015_2024.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, dtype={"corp_code": str, "stock_code": str})
    session = requests.Session()
    session.get(KIND_DELIST_URL, params={"method": "searchDelCompanyMain"}, timeout=30).raise_for_status()
    frames: list[pd.DataFrame] = []
    for market_code, market in (("1", "KOSPI"), ("2", "KOSDAQ")):
        form = {
            "method": "searchDelCompanySub", "currentPageSize": "3000", "pageIndex": "1",
            "orderMode": "2", "orderStat": "D", "tabType": "1", "forward": "delcompany_down",
            "marketType": market_code, "fromDate": f"{START_YEAR}-01-01", "toDate": f"{END_YEAR}-12-31",
        }
        response = session.post(KIND_DELIST_URL, data=form, timeout=60)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.content.decode("euc-kr", "replace")))
        if not tables:
            continue
        frame = tables[0].copy()
        frame.columns = ["sequence", "company_name_krx", "stock_raw", "delist_date", "delist_reason", "delist_note"]
        frame["market"] = market
        frames.append(frame)
    delisted = pd.concat(frames, ignore_index=True)
    delisted["stock_code"] = delisted["stock_raw"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    mapping = corp[["corp_code", "stock_code", "corp_name"]].copy()
    mapping["corp_code"] = mapping["corp_code"].astype(str).str.zfill(8)
    mapping["stock_code"] = mapping["stock_code"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
    delisted = delisted.merge(mapping, on="stock_code", how="left", validate="one_to_one")
    if delisted["corp_code"].isna().any():
        missing = delisted.loc[delisted["corp_code"].isna(), "stock_code"].tolist()
        raise RuntimeError(f"Could not map delisted tickers to DART corp codes: {missing[:20]}")
    delisted["corp_code"] = delisted["corp_code"].astype(str).str.zfill(8)
    delisted["is_spac"] = delisted["company_name_krx"].astype(str).str.contains("스팩", na=False)
    safe_csv(delisted, cache_path)
    return delisted


def collect_delisted_filings(delisted: pd.DataFrame, refresh: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    companies = delisted[~delisted["is_spac"]].copy()
    for sequence, row in enumerate(companies.itertuples(index=False), start=1):
        common = {
            "corp_code": str(row.corp_code).zfill(8),
            "bgn_de": f"{START_YEAR}0101", "end_de": f"{date.today():%Y%m%d}",
            "pblntf_ty": "A", "pblntf_detail_ty": "A001",
            "sort": "date", "sort_mth": "asc", "page_count": 100,
        }
        stem = f"D_{row.corp_code}"
        first = request_json({**common, "page_no": 1}, LIST_CACHE_DIR / f"{stem}_001.json", refresh)
        pages = int(first.get("total_page") or 0)
        batch = list(first.get("list") or [])
        for page in range(2, pages + 1):
            result = request_json(
                {**common, "page_no": page}, LIST_CACHE_DIR / f"{stem}_{page:03d}.json", refresh
            )
            batch.extend(result.get("list") or [])
        for filing in batch:
            filing = dict(filing)
            filing["market_override"] = row.market
            filing["delist_date"] = row.delist_date
            filing["delist_reason"] = row.delist_reason
            filing["historical_delisted_sample"] = True
            rows.append(filing)
        if sequence % 25 == 0 or sequence == len(companies):
            print(f"Delisted metadata {sequence}/{len(companies)} filings={len(rows):,}", flush=True)
    return rows


def collect_metadata(refresh: bool = False) -> None:
    ensure_dirs()
    corp = pd.read_excel(
        SOURCE_CORP_PATH,
        dtype={"corp_code": str, "stock_code": str, "industry_code": str},
    )
    corp["corp_code"] = corp["corp_code"].astype(str).str.zfill(8)
    delisted = collect_delisted_companies(corp, refresh)
    rows: list[dict[str, Any]] = []
    end_date = date.today()
    windows = list(quarter_windows(START_YEAR, end_date))
    total_jobs = len(windows) * 2
    job = 0
    for market_code in ("Y", "K"):
        for start, end in windows:
            job += 1
            stem = f"{market_code}_{start:%Y%m%d}_{end:%Y%m%d}"
            common = {
                "bgn_de": f"{start:%Y%m%d}",
                "end_de": f"{end:%Y%m%d}",
                "corp_cls": market_code,
                "pblntf_ty": "A",
                "pblntf_detail_ty": "A001",
                "sort": "date",
                "sort_mth": "asc",
                "page_count": 100,
            }
            first = request_json({**common, "page_no": 1}, LIST_CACHE_DIR / f"{stem}_001.json", refresh)
            pages = int(first.get("total_page") or 0)
            batch = list(first.get("list") or [])
            for page in range(2, pages + 1):
                result = request_json(
                    {**common, "page_no": page},
                    LIST_CACHE_DIR / f"{stem}_{page:03d}.json",
                    refresh,
                )
                batch.extend(result.get("list") or [])
            rows.extend(batch)
            print(f"Metadata {job}/{total_jobs}: {market_code} {start}..{end} rows={len(batch):,}", flush=True)

    rows.extend(collect_delisted_filings(delisted, refresh))

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No annual-report metadata returned.")
    raw["report_year"], raw["fiscal_month"] = zip(*raw["report_nm"].map(parse_report_period))
    raw["report_year"] = pd.to_numeric(raw["report_year"], errors="coerce").astype("Int64")
    is_business_report = raw["report_nm"].astype(str).str.contains(
        r"\uc0ac\uc5c5\ubcf4\uace0\uc11c\s*\(", regex=True, na=False,
    )
    raw = raw[raw["report_year"].between(START_YEAR, END_YEAR) & is_business_report].copy()
    raw["corp_code"] = raw["corp_code"].astype(str).str.zfill(8)
    raw["stock_code"] = raw["stock_code"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
    raw["rcept_no"] = raw["rcept_no"].astype(str)
    raw["market"] = raw.get("market_override", pd.Series(index=raw.index, dtype=object)).fillna(raw["corp_cls"].map(MARKET_LABELS))
    raw["historical_delisted_sample"] = raw.get("historical_delisted_sample", False).fillna(False).astype(bool)
    raw["delist_date"] = pd.to_datetime(raw.get("delist_date"), errors="coerce")
    raw = raw[raw["delist_date"].isna() | raw["report_year"].le(raw["delist_date"].dt.year)].copy()
    raw["is_correction"] = raw["report_nm"].astype(str).str.contains("정정", na=False)
    raw = raw.sort_values(["corp_code", "report_year", "rcept_dt", "rcept_no"])
    first_filings = raw.drop_duplicates(["corp_code", "report_year"], keep="first").copy()
    latest_corrections = (
        raw.drop_duplicates(["corp_code", "report_year"], keep="last")
        [["corp_code", "report_year", "rcept_no", "rcept_dt", "report_nm"]]
        .rename(columns={
            "rcept_no": "latest_corrected_rcept_no",
            "rcept_dt": "latest_corrected_rcept_dt",
            "report_nm": "latest_corrected_report_name",
        })
    )
    latest = first_filings.merge(latest_corrections, on=["corp_code", "report_year"], how="left", validate="one_to_one")
    latest["first_annual_report_rcept_no"] = latest["rcept_no"]
    latest["first_annual_report_rcept_dt"] = latest["rcept_dt"]
    latest = latest.rename(columns={"report_year": "year", "report_nm": "report_name"})

    profile_cols = [
        "corp_code", "corp_name_eng", "industry_code", "est_dt", "acc_mt",
        "adres", "hm_url", "is_financial",
    ]
    profile = corp[[column for column in profile_cols if column in corp]].drop_duplicates("corp_code")
    latest = latest.merge(profile, on="corp_code", how="left", validate="many_to_one")
    latest["industry_code"] = latest.get("industry_code", pd.Series(index=latest.index, dtype=object)).fillna("").astype(str)
    missing_profile_codes = sorted(latest.loc[latest["industry_code"].isin({"", "nan", "None"}), "corp_code"].unique())
    profile_rows: list[dict[str, Any]] = []
    for sequence, corp_code in enumerate(missing_profile_codes, start=1):
        payload = company_profile(str(corp_code).zfill(8), refresh)
        profile_rows.append({
            "corp_code": str(corp_code).zfill(8), "industry_code_api": payload.get("induty_code"),
            "est_dt_api": payload.get("est_dt"), "acc_mt_api": payload.get("acc_mt"),
            "corp_name_eng_api": payload.get("corp_name_eng"), "profile_api_status": payload.get("status"),
            "profile_api_message": payload.get("message"),
        })
        if sequence % 25 == 0 or sequence == len(missing_profile_codes):
            print(f"Historical company profiles {sequence}/{len(missing_profile_codes)}", flush=True)
    if profile_rows:
        supplemental = pd.DataFrame(profile_rows)
        latest = latest.merge(supplemental, on="corp_code", how="left", validate="many_to_one")
        latest["industry_code"] = latest["industry_code"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).fillna(latest["industry_code_api"])
        latest["est_dt"] = latest["est_dt"].fillna(latest["est_dt_api"])
        latest["acc_mt"] = latest["acc_mt"].fillna(latest["acc_mt_api"])
        latest["corp_name_eng"] = latest["corp_name_eng"].fillna(latest["corp_name_eng_api"])
    latest["industry_code"] = latest["industry_code"].fillna("").astype(str)
    derived_financial = latest["industry_code"].str.startswith(FINANCIAL_KSIC_PREFIXES)
    name_financial = latest["corp_name"].astype(str).str.contains(
        r"은행|증권|보험|캐피탈|저축은행|신용정보|자산운용", regex=True, na=False
    )
    latest["is_financial"] = latest.get("is_financial", False).fillna(False).astype(bool) | derived_financial | name_financial

    coverage = latest.groupby("corp_code").agg(
        corp_name=("corp_name", "last"),
        stock_code=("stock_code", "last"),
        market_latest=("market", "last"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        n_report_years=("year", "nunique"),
        is_financial=("is_financial", "max"),
        historical_delisted_sample=("historical_delisted_sample", "max"),
        industry_code=("industry_code", "last"),
    ).reset_index()
    year_sets = latest.groupby("corp_code")["year"].apply(lambda values: sorted(set(map(int, values))))

    def max_run(values: list[int]) -> int:
        best = current = 0
        previous: int | None = None
        for value in values:
            current = current + 1 if previous is not None and value == previous + 1 else 1
            best = max(best, current)
            previous = value
        return best

    coverage["max_consecutive_years"] = coverage["corp_code"].map(year_sets.map(max_run))
    coverage["available_years"] = coverage["corp_code"].map(year_sets.map(lambda values: ";".join(map(str, values))))
    coverage["eligible_main_min3"] = coverage["n_report_years"].ge(3) & ~coverage["is_financial"]
    coverage["eligible_robust_min5"] = coverage["n_report_years"].ge(5) & ~coverage["is_financial"]
    coverage["eligible_balanced_full"] = coverage["n_report_years"].eq(END_YEAR - START_YEAR + 1) & ~coverage["is_financial"]
    # Compatibility alias consumed by the model script; this now means all 11
    # fiscal years from 2015 through 2025, not literally ten observations.
    coverage["eligible_balanced10"] = coverage["eligible_balanced_full"]
    eligible_codes = set(coverage.loc[coverage["eligible_main_min3"], "corp_code"])
    selected = latest[latest["corp_code"].isin(eligible_codes)].copy()
    selected = selected.sort_values(["market", "corp_code", "year"]).reset_index(drop=True)

    summary = pd.DataFrame([
        {"metric": "raw_filing_records", "value": len(raw)},
        {"metric": "deduplicated_firm_years", "value": len(latest)},
        {"metric": "all_unique_firms", "value": latest["corp_code"].nunique()},
        {"metric": "historical_delisted_firms", "value": latest.loc[latest["historical_delisted_sample"], "corp_code"].nunique()},
        {"metric": "main_nonfinancial_min3_firms", "value": coverage["eligible_main_min3"].sum()},
        {"metric": "main_nonfinancial_min3_firm_years", "value": len(selected)},
        {"metric": "robust_nonfinancial_min5_firms", "value": coverage["eligible_robust_min5"].sum()},
        {"metric": "balanced_full_2015_2025_nonfinancial_firms", "value": coverage["eligible_balanced_full"].sum()},
        {"metric": "KOSPI_main_firms", "value": coverage.loc[coverage["eligible_main_min3"] & coverage["market_latest"].eq("KOSPI"), "corp_code"].nunique()},
        {"metric": "KOSDAQ_main_firms", "value": coverage.loc[coverage["eligible_main_min3"] & coverage["market_latest"].eq("KOSDAQ"), "corp_code"].nunique()},
    ])
    safe_csv(raw, OUTPUT_DIR / "01_dart_annual_report_filings_raw.csv")
    safe_csv(latest, OUTPUT_DIR / "02_kospi_kosdaq_annual_report_metadata.csv")
    safe_csv(coverage, OUTPUT_DIR / "03_firm_coverage_and_sample_flags.csv")
    safe_csv(selected, OUTPUT_DIR / "04_selected_nonfinancial_min3_firm_years.csv")
    safe_csv(summary, OUTPUT_DIR / "05_sample_summary.csv")
    print(summary.to_string(index=False), flush=True)


def legacy_report_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not LEGACY_REPORT_DIR.exists():
        return index
    for path in LEGACY_REPORT_DIR.glob("*.txt"):
        match = re.search(r"_(\d{14})\.txt$", path.name)
        if match:
            index[match.group(1)] = path
    return index


def document_to_text(content: bytes) -> str:
    if content[:2] != b"PK":
        message = dart_base.parse_open_dart_error(content)
        raise RuntimeError(message or "OpenDART document endpoint did not return ZIP")
    text_parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".xml", ".html", ".htm"))]
        for name in names:
            raw = archive.read(name)
            decoded = dart_base.decode_bytes(raw)
            root = etree.fromstring(
                decoded.encode("utf-8", errors="replace"),
                parser=etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8"),
            )
            text_parts.append("\n".join(part.strip() for part in root.itertext() if part and part.strip()))
    return "\n".join(text_parts)


def download_one(task: dict[str, Any], key: str, old_index: dict[str, Path]) -> dict[str, Any]:
    rcept_no = str(task["rcept_no"])
    target = REPORT_DIR / f"{task['stock_code']}_{int(task['year'])}_{rcept_no}.txt"
    if target.exists() and target.stat().st_size > 100:
        return {**task, "download_status": "cached_new", "text_source_rcept_no": rcept_no, "text_cache_path": str(target), "download_error": ""}
    old = old_index.get(rcept_no)
    if old and old.exists() and old.stat().st_size > 100:
        return {**task, "download_status": "reused_legacy", "text_source_rcept_no": rcept_no, "text_cache_path": str(old), "download_error": ""}
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(
                f"{API_BASE}/document.xml",
                params={"crtfc_key": key, "rcept_no": rcept_no},
                timeout=120,
            )
            response.raise_for_status()
            text = document_to_text(response.content)
            if len(text) < 100:
                raise RuntimeError("extracted report text is unexpectedly short")
            target.write_text(text, encoding="utf-8", errors="replace")
            return {**task, "download_status": "downloaded", "text_source_rcept_no": rcept_no, "text_cache_path": str(target), "download_error": ""}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if str(exc).lstrip().startswith("020") or "429" in str(exc):
                return {
                    **task, "download_status": "quota_exceeded", "text_cache_path": "",
                    "download_error": str(exc),
                }
            if str(exc).lstrip().startswith("014"):
                break
            time.sleep(min(attempt * 2, 8))
    fallback_no = str(task.get("latest_corrected_rcept_no") or "")
    if str(last_error).lstrip().startswith("014") and fallback_no and fallback_no != rcept_no:
        try:
            response = requests.get(
                f"{API_BASE}/document.xml",
                params={"crtfc_key": key, "rcept_no": fallback_no}, timeout=120,
            )
            response.raise_for_status()
            text = document_to_text(response.content)
            if len(text) < 100:
                raise RuntimeError("fallback corrected report text is unexpectedly short")
            target.write_text(text, encoding="utf-8", errors="replace")
            return {
                **task, "download_status": "downloaded_fallback_latest_correction",
                "text_source_rcept_no": fallback_no, "text_cache_path": str(target),
                "download_error": "primary first filing unavailable (014); used latest corrected annual report",
            }
        except Exception as fallback_exc:  # noqa: BLE001
            last_error = RuntimeError(f"primary={last_error}; fallback={fallback_exc}")
    return {**task, "download_status": "error", "text_cache_path": "", "download_error": str(last_error)}


def download_reports(workers: int = 4, limit: int | None = None) -> None:
    ensure_dirs()
    metadata_path = OUTPUT_DIR / "04_selected_nonfinancial_min3_firm_years.csv"
    metadata = pd.read_csv(metadata_path, dtype={"corp_code": str, "stock_code": str, "rcept_no": str})
    metadata["corp_code"] = metadata["corp_code"].str.zfill(8)
    metadata["stock_code"] = metadata["stock_code"].str.zfill(6)
    if limit:
        metadata = metadata.head(limit).copy()
    completed: dict[str, dict[str, Any]] = {}
    if CHECKPOINT_PATH.exists():
        prior = pd.read_csv(CHECKPOINT_PATH, dtype={"corp_code": str, "stock_code": str, "rcept_no": str})
        success_states = {"downloaded", "cached_new", "reused_legacy", "downloaded_fallback_latest_correction"}
        completed = {
            str(row["rcept_no"]): row.to_dict() for _, row in prior.iterrows()
            if str(row.get("download_status")) in success_states
        }
    tasks = [
        {"corp_code": str(row.corp_code).zfill(8), "corp_name": row.corp_name,
         "stock_code": str(row.stock_code).zfill(6), "market": row.market,
         "year": int(row.year), "rcept_no": str(row.rcept_no), "report_name": row.report_name,
         "latest_corrected_rcept_no": str(getattr(row, "latest_corrected_rcept_no", "") or "")}
        for row in metadata.itertuples(index=False) if str(row.rcept_no) not in completed
    ]
    results = list(completed.values())
    old_index = legacy_report_index()
    key = api_key()
    print(f"Reports total={len(metadata):,}; cached checkpoint={len(completed):,}; pending={len(tasks):,}; legacy={len(old_index):,}", flush=True)
    processed = 0
    quota_hit = False
    batch_size = 250
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(download_one, task, key, old_index) for task in batch]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                processed += 1
                quota_hit = quota_hit or result["download_status"] == "quota_exceeded"
                if processed % 50 == 0 or processed == len(tasks):
                    checkpoint = pd.DataFrame(results).drop_duplicates("rcept_no", keep="last")
                    safe_csv(checkpoint, CHECKPOINT_PATH)
                    ok = checkpoint["download_status"].isin({"downloaded", "cached_new", "reused_legacy", "downloaded_fallback_latest_correction"}).sum()
                    errors = checkpoint["download_status"].eq("error").sum()
                    print(f"Download {processed:,}/{len(tasks):,}; available={ok:,}; errors={errors:,}", flush=True)
        if quota_hit:
            print("OpenDART daily quota reached; checkpoint saved and remaining reports were not requested.", flush=True)
            break
    checkpoint = pd.DataFrame(results).drop_duplicates("rcept_no", keep="last")
    safe_csv(checkpoint, CHECKPOINT_PATH)
    merged = metadata.merge(
        checkpoint[["rcept_no", "download_status", "text_source_rcept_no", "text_cache_path", "download_error"]],
        on="rcept_no", how="left", validate="one_to_one",
    )
    safe_csv(merged, OUTPUT_DIR / "06_report_download_manifest.csv")


def parse_viewer_nodes(html: str) -> list[dict[str, str]]:
    blocks = re.findall(r"var node\d+ = \{\};(.*?treeData\.push\(node\d+\);)", html, flags=re.S)
    nodes: list[dict[str, str]] = []
    for block in blocks:
        node: dict[str, str] = {}
        for key_name in ("text", "dcmNo", "eleId", "offset", "length", "dtd", "tocNo"):
            match = re.search(rf"\['{key_name}'\]\s*=\s*\"([^\"]*)\"", block)
            node[key_name] = match.group(1) if match else ""
        if node["dcmNo"] and node["eleId"]:
            nodes.append(node)
    return nodes


def strategy_nodes(nodes: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for node in nodes:
        title = re.sub(r"[\sⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX.]+", "", node.get("text", ""))
        if any(label in title for label in ("회사의개요", "사업의내용", "경영진단및분석의견")):
            selected.append(node)
    if selected:
        return selected
    fallback = [
        node for node in nodes
        if "재무" not in node.get("text", "") and "감사" not in node.get("text", "")
        and int(node.get("length") or 0) >= 1000
    ]
    return fallback[:2]


def public_get(url: str, *, params: dict[str, Any] | None = None, timeout: int = 90) -> requests.Response:
    last_error: Exception | None = None
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://dart.fss.or.kr/"}
    for attempt in range(1, 6):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            time.sleep(0.12)
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(1.5 * attempt, 7.5))
    raise RuntimeError(f"DART public viewer failed after retries: {last_error}")


def section_text_one(task: dict[str, Any]) -> dict[str, Any]:
    rcept_no = str(task["rcept_no"])
    target = SECTION_DIR / f"{task['stock_code']}_{int(task['year'])}_{rcept_no}_sections.txt"
    if target.exists() and target.stat().st_size > 100:
        return {**task, "download_status": "cached_sections", "text_cache_path": str(target), "download_error": ""}
    try:
        main = public_get(f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}", timeout=60)
        nodes = strategy_nodes(parse_viewer_nodes(main.text))
        if not nodes:
            raise RuntimeError("no predefined strategy-bearing report sections found")
        parts: list[str] = []
        titles: list[str] = []
        for node in nodes:
            response = public_get(
                "https://dart.fss.or.kr/report/viewer.do",
                params={
                    "rcpNo": rcept_no, "dcmNo": node["dcmNo"], "eleId": node["eleId"],
                    "offset": node["offset"], "length": node["length"], "dtd": node["dtd"],
                    "tocNo": node.get("tocNo", ""),
                },
                timeout=90,
            )
            if len(response.content) < 100:
                continue
            parts.append(dart_base.markup_to_text(response.text))
            titles.append(node["text"])
        text = "\n".join(part for part in parts if part.strip())
        if len(text) < 100:
            raise RuntimeError("strategy sections produced insufficient text")
        target.write_text(text, encoding="utf-8", errors="replace")
        return {
            **task, "download_status": "downloaded_sections", "text_cache_path": str(target),
            "download_error": "", "selected_section_count": len(parts),
            "selected_section_titles": "; ".join(titles),
            "text_scope": "I company overview + II business content + IV MD&A when available",
        }
    except Exception as exc:  # noqa: BLE001
        return {**task, "download_status": "error", "text_cache_path": "", "download_error": str(exc)}


def download_strategy_sections(workers: int = 2, limit: int | None = None) -> None:
    ensure_dirs()
    metadata = pd.read_csv(
        OUTPUT_DIR / "04_selected_nonfinancial_min3_firm_years.csv",
        dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
    )
    metadata["corp_code"] = metadata["corp_code"].str.zfill(8)
    metadata["stock_code"] = metadata["stock_code"].str.zfill(6)
    if limit:
        metadata = metadata.head(limit).copy()
    completed: dict[str, dict[str, Any]] = {}
    if SECTION_CHECKPOINT_PATH.exists():
        prior = pd.read_csv(SECTION_CHECKPOINT_PATH, dtype={"corp_code": str, "stock_code": str, "rcept_no": str})
        completed = {
            str(row["rcept_no"]): row.to_dict() for _, row in prior.iterrows()
            if str(row.get("download_status")) in {"cached_sections", "downloaded_sections"}
        }
    tasks = [
        row for row in metadata.to_dict("records") if str(row["rcept_no"]) not in completed
    ]
    results = list(completed.values())
    print(f"Strategy sections total={len(metadata):,}; cached={len(completed):,}; pending={len(tasks):,}", flush=True)
    batch_size = 250
    processed = 0
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(section_text_one, task) for task in batch]
            for future in as_completed(futures):
                results.append(future.result())
                processed += 1
                if processed % 50 == 0 or processed == len(tasks):
                    checkpoint = pd.DataFrame(results).drop_duplicates("rcept_no", keep="last")
                    safe_csv(checkpoint, SECTION_CHECKPOINT_PATH)
                    ok = checkpoint["download_status"].isin({"cached_sections", "downloaded_sections"}).sum()
                    print(f"Sections {processed:,}/{len(tasks):,}; available={ok:,}; errors={checkpoint['download_status'].eq('error').sum():,}", flush=True)
    checkpoint = pd.DataFrame(results).drop_duplicates("rcept_no", keep="last")
    safe_csv(checkpoint, SECTION_CHECKPOINT_PATH)
    merged = metadata.merge(
        checkpoint[[
            "rcept_no", "download_status", "text_cache_path", "download_error",
            "selected_section_count", "selected_section_titles", "text_scope",
        ]], on="rcept_no", how="left", validate="one_to_one",
    )
    safe_csv(merged, OUTPUT_DIR / "06_strategy_section_manifest.csv")


def score_one(task: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text_path = Path(str(task.get("text_cache_path") or ""))
    counts = {
        "digital_sentence_count": 0,
        "substantive_digital_sentence_count": 0,
        "bmi_sentence_count": 0,
        "market_signal_sentence_count": 0,
        "boilerplate_digital_sentence_count": 0,
    }
    if not text_path.exists():
        return {**task, **counts, "total_sentences": 0, "score_status": "missing_text"}, []
    text = text_path.read_text(encoding="utf-8", errors="replace")
    raw_sentences = korean_base.split_korean_sentences(text)
    seen: set[str] = set()
    sentences: list[str] = []
    for sentence in raw_sentences:
        key = "".join(sentence.lower().split())
        if key and key not in seen:
            seen.add(key)
            sentences.append(sentence)
    examples: list[dict[str, Any]] = []
    report_year = int(task["year"])
    for sentence_id, sentence in enumerate(sentences, start=1):
        labels = classify_sentence_v4(sentence, report_year, str(task.get("corp_name") or ""))
        digital = int(labels["digital_sentence_v2"])
        substantive = int(labels["substantive_digital_sentence_v2"])
        bmi = int(labels["bmi_sentence_v2"])
        market_signal = int(labels["market_signal_sentence_v2"])
        boilerplate = int(labels["boilerplate_digital_sentence_v2"])
        counts["digital_sentence_count"] += digital
        counts["substantive_digital_sentence_count"] += substantive
        counts["bmi_sentence_count"] += bmi
        counts["market_signal_sentence_count"] += market_signal
        counts["boilerplate_digital_sentence_count"] += boilerplate
        if digital:
            examples.append({
                "corp_code": task["corp_code"], "corp_name": task["corp_name"],
                "stock_code": task["stock_code"], "market": task["market"],
                "year": report_year, "rcept_no": task["rcept_no"],
                "sentence_id": sentence_id,
                "priority": substantive * 100 + bmi * 30 + market_signal * 10 - boilerplate * 50,
                "sentence": sentence[:3000], "digital_sentence": digital,
                "substantive_digital_sentence": substantive, "bmi_sentence": bmi,
                "market_signal_sentence": market_signal,
                "boilerplate_digital_sentence": boilerplate,
                "v2_accounting_noise": labels.get("v2_accounting_noise", 0),
                "v2_name_collision": labels.get("v2_name_collision", 0),
                "v2_stale_history": labels.get("v2_stale_history", 0),
                "v2_generic_context": labels.get("v2_generic_context", 0),
                "v4_market_trend_without_firm": labels.get("v4_market_trend_without_firm", 0),
                "v4_external_actor_without_firm": labels.get("v4_external_actor_without_firm", 0),
                "v4_static_scope_or_segment": labels.get("v4_static_scope_or_segment", 0),
                "v4_nondigital_platform": labels.get("v4_nondigital_platform", 0),
                "v4_missing_implicit_implementation": labels.get("v4_missing_implicit_implementation", 0),
                "v4_vague_strategy": labels.get("v4_vague_strategy", 0),
                "v4_digital_demand_exposure": labels.get("v4_digital_demand_exposure", 0),
                "v4_customer_or_competitor_context": labels.get("v4_customer_or_competitor_context", 0),
                "v4_former_business": labels.get("v4_former_business", 0),
                "v4_accounting_or_entity_list": labels.get("v4_accounting_or_entity_list", 0),
                "v4_high_confidence_action": labels.get("v4_high_confidence_action", 0),
                "v4_stale_history": labels.get("v4_stale_history", 0),
            })
    examples.sort(key=lambda row: (-row["priority"], row["sentence_id"]))
    examples = examples[:20]
    for rank, row in enumerate(examples, start=1):
        row["sentence_rank"] = rank
    denominator = len(sentences)
    scores = {
        **task, **counts, "total_sentences": denominator, "score_status": "ok",
        "DigitalStrategy": counts["substantive_digital_sentence_count"] / denominator if denominator else None,
        "BMI_Reconstruction": counts["bmi_sentence_count"] / denominator if denominator else None,
        "MarketSignalDisclosure": counts["market_signal_sentence_count"] / denominator if denominator else None,
        "BoilerplateDisclosure": counts["boilerplate_digital_sentence_count"] / denominator if denominator else None,
        "classifier_version": "korean_contextual_v4_final_action_attribution_2026-08-15",
    }
    return scores, examples


def score_reports(workers: int = 4, limit: int | None = None) -> None:
    # The final indicator uses complete annual reports. The section-viewer
    # experiment is intentionally retained as a sensitivity artifact only: it
    # has incomplete coverage and must never silently replace the full corpus.
    manifest_path = OUTPUT_DIR / "06_report_download_manifest.csv"
    success_states = {"downloaded", "cached_new", "reused_legacy", "downloaded_fallback_latest_correction"}
    manifest = pd.read_csv(
        manifest_path,
        dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
    )
    manifest = manifest[
        manifest["download_status"].isin(success_states)
    ].copy()
    if limit:
        manifest = manifest.head(limit).copy()
    completed: dict[str, dict[str, Any]] = {}
    if SCORE_CHECKPOINT_PATH.exists():
        prior_scores = pd.read_csv(
            SCORE_CHECKPOINT_PATH,
            dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
            low_memory=False,
        )
        prior_scores = prior_scores[
            prior_scores.get("classifier_version", "").eq(
                "korean_contextual_v4_final_action_attribution_2026-08-15"
            )
        ].copy()
        completed = {
            str(row["rcept_no"]): row.to_dict()
            for _, row in prior_scores.iterrows()
            if str(row.get("score_status")) == "ok"
        }
    tasks = [
        task for task in manifest.to_dict("records")
        if str(task["rcept_no"]) not in completed
    ]
    scores: list[dict[str, Any]] = list(completed.values())
    examples: list[dict[str, Any]] = []
    if EXAMPLE_CHECKPOINT_PATH.exists() and completed:
        prior_examples = pd.read_csv(
            EXAMPLE_CHECKPOINT_PATH,
            dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
            low_memory=False,
        )
        examples = prior_examples[
            prior_examples["rcept_no"].astype(str).isin(completed)
        ].to_dict("records")

    print(
        f"Score total={len(manifest):,}; cached={len(completed):,}; pending={len(tasks):,}",
        flush=True,
    )
    batch_size = 1000
    processed = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]
            futures = [executor.submit(score_one, task) for task in batch]
            for future in as_completed(futures):
                score, selected = future.result()
                scores.append(score)
                examples.extend(selected)
                processed += 1
                if processed % 100 == 0 or processed == len(tasks):
                    print(
                        f"Scoring {len(completed) + processed:,}/{len(manifest):,}",
                        flush=True,
                    )
            safe_csv(
                pd.DataFrame(scores).drop_duplicates("rcept_no", keep="last"),
                SCORE_CHECKPOINT_PATH,
            )
            safe_csv(
                pd.DataFrame(examples).drop_duplicates(
                    ["rcept_no", "sentence_id"], keep="last"
                ),
                EXAMPLE_CHECKPOINT_PATH,
            )
    panel = pd.DataFrame(scores).sort_values(["corp_code", "year"]).reset_index(drop=True)
    panel["YearMean_DigitalStrategy"] = panel.groupby("year")["DigitalStrategy"].transform("mean")
    panel["RelativeDigitalSignal_Year"] = panel["DigitalStrategy"] - panel["YearMean_DigitalStrategy"]
    if "industry_code" in panel:
        panel["industry_group"] = (
            panel["industry_code"].astype(str).str.extract(r"(\d+)", expand=False)
            .fillna("").str.zfill(2).str[:2]
        )
        grouped = panel.groupby(["industry_group", "year"])["DigitalStrategy"]
        group_sum = grouped.transform("sum")
        group_n = grouped.transform("count")
        peer_mean = (group_sum - panel["DigitalStrategy"]) / (group_n - 1)
        panel["IndustryYearN"] = group_n
        panel["IndustryYearPeerMean_DigitalStrategy"] = peer_mean.where(group_n.ge(5), panel["YearMean_DigitalStrategy"])
        panel["RelativeDigitalSignal"] = panel["DigitalStrategy"] - panel["IndustryYearPeerMean_DigitalStrategy"]
    else:
        panel["RelativeDigitalSignal"] = panel["RelativeDigitalSignal_Year"]
    panel["SignalSubstanceGap"] = panel["MarketSignalDisclosure"] - panel["DigitalStrategy"]
    safe_csv(panel, OUTPUT_DIR / "07_digital_strategy_panel_kospi_kosdaq.csv")
    safe_csv(pd.DataFrame(examples), OUTPUT_DIR / "08_representative_digital_sentences.csv")
    print(f"Panel rows={len(panel):,}; firms={panel['corp_code'].nunique():,}; sentence examples={len(examples):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    metadata = sub.add_parser("metadata")
    metadata.add_argument("--refresh", action="store_true")
    download = sub.add_parser("download")
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--limit", type=int)
    sections = sub.add_parser("sections")
    sections.add_argument("--workers", type=int, default=2)
    sections.add_argument("--limit", type=int)
    score = sub.add_parser("score")
    score.add_argument("--workers", type=int, default=4)
    score.add_argument("--limit", type=int)
    sub.add_parser("all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "metadata":
        collect_metadata(args.refresh)
    elif args.command == "download":
        download_reports(args.workers, args.limit)
    elif args.command == "sections":
        download_strategy_sections(args.workers, args.limit)
    elif args.command == "score":
        score_reports(args.workers, args.limit)
    else:
        collect_metadata(False)
        download_reports(4)
        score_reports(4)


if __name__ == "__main__":
    main()
