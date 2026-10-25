"""
Korean listed-company English annual report pipeline.

Purpose
-------
Build a firm-year panel for a thesis on digital strategy in Korean listed
companies, using English OpenDART annual reports where available.

Default outputs
---------------
01_listed_corporations_from_dart.xlsx
02_english_annual_report_metadata.xlsx
03_selected_research_sample.xlsx
04_digital_strategy_panel.xlsx
05_representative_digital_sentences.xlsx

Notes
-----
Set the API key in the environment. Do not hard-code it in this file.

PowerShell:
    $env:DART_API_KEY = "YOUR_KEY"

Run:
    python korea_digital_strategy_pipeline.py run --out-dir korea_digital_strategy_output

Smoke checks:
    python korea_digital_strategy_pipeline.py doctor
    python korea_digital_strategy_pipeline.py self-test
"""

from __future__ import annotations

import argparse
import calendar
import html as html_lib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent

REQUIRED_PIP_PACKAGES = {
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "tqdm": "tqdm",
}


def ensure_runtime_packages() -> None:
    """Install missing runtime packages into the current Python environment."""
    if os.environ.get("KOREA_PIPELINE_SKIP_AUTO_INSTALL") == "1":
        return

    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_PIP_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing:
        return

    print("Missing Python packages detected:")
    print("  " + " ".join(missing))
    print("Installing them with the current Python interpreter...")

    try:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    except Exception:
        pass

    install_cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *missing]
    try:
        subprocess.check_call(install_cmd)
    except subprocess.CalledProcessError:
        print("Default pip source failed. Retrying with Tsinghua PyPI mirror...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
                *missing,
            ]
        )

    importlib.invalidate_caches()
    still_missing = [
        module_name
        for module_name in REQUIRED_PIP_PACKAGES
        if importlib.util.find_spec(module_name) is None
    ]
    if still_missing:
        raise SystemExit(
            "Package installation finished, but these modules are still missing: "
            + ", ".join(still_missing)
        )


ensure_runtime_packages()


try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - this is a startup diagnostic.
    raise SystemExit(
        "pandas is required. Install requirements or run with a Python environment "
        "that includes pandas and openpyxl."
    ) from exc

try:  # Optional. urllib fallback is implemented below.
    import requests  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment.
    requests = None  # type: ignore

try:  # Optional. lxml/regex fallback is implemented below.
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment.
    BeautifulSoup = None  # type: ignore
    XMLParsedAsHTMLWarning = None  # type: ignore

if XMLParsedAsHTMLWarning is not None:
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:
    from lxml import html as lxml_html  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment.
    lxml_html = None  # type: ignore

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment.
    _tqdm = None


ENG_API_BASE = "https://engopendart.fss.or.kr/engapi"
KOR_API_BASE = "https://opendart.fss.or.kr/api"

MARKET_LABELS = {
    "Y": "KOSPI",
    "K": "KOSDAQ",
    "N": "KONEX",
    "E": "OTHER",
}
MARKET_PRIORITY = {"Y": 0, "K": 1, "N": 2, "E": 3, "": 4, None: 4}


CORE_DIGITAL_TERMS = [
    "digital transformation",
    "digital strategy",
    "digital innovation",
    "digitalization",
    "digitization",
    "data-driven",
    "software-defined",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "generative ai",
    "big data",
    "data analytics",
    "data analysis",
    "data platform",
    "cloud computing",
    "cloud service",
    "cloud platform",
    "internet of things",
    "iot",
    "blockchain",
    "cybersecurity",
    "cyber security",
    "robotic process automation",
    "rpa",
    "smart factory",
    "smart factories",
    "smart manufacturing",
    "digital twin",
    "edge computing",
    "5g",
    "digital platform",
    "digital ecosystem",
    "digital service",
    "digital solution",
    "online channel",
    "digital channel",
    "e-commerce",
    "ecommerce",
    "over-the-air",
    "ota",
    "connected car",
    "connected service",
    "smart mobility",
    "mobility service",
    "smart home",
]

CONTEXTUAL_DIGITAL_TERMS = [
    "ai",
    "cloud",
    "data",
    "analytics",
    "algorithm",
    "platform",
    "software",
    "app",
    "mobile",
    "online",
    "automation",
    "automated",
    "robotics",
    "robot",
    "autonomous",
    "connected",
    "smart",
    "subscription",
]

ACTION_TERMS = [
    "acquire",
    "adopt",
    "advance",
    "apply",
    "automate",
    "build",
    "commercialize",
    "collaborate",
    "connect",
    "construct",
    "create",
    "deploy",
    "design",
    "develop",
    "digitalize",
    "digitize",
    "enhance",
    "establish",
    "expand",
    "implement",
    "improve",
    "integrate",
    "introduce",
    "invest",
    "launch",
    "modernize",
    "operate",
    "optimize",
    "partner",
    "provide",
    "rebuild",
    "reorganize",
    "restructure",
    "roll out",
    "strengthen",
    "transform",
    "upgrade",
    "use",
    "utilize",
]

BMI_TERMS = [
    "new product",
    "new service",
    "smart product",
    "solution",
    "data service",
    "software service",
    "personalized service",
    "customized service",
    "online channel",
    "digital channel",
    "platform",
    "ecosystem",
    "partner",
    "customer interface",
    "customer experience",
    "direct-to-consumer",
    "direct to consumer",
    "e-commerce",
    "ecommerce",
    "omni-channel",
    "omnichannel",
    "subscription",
    "recurring revenue",
    "service revenue",
    "platform revenue",
    "software revenue",
    "monetization",
    "new revenue model",
    "business model",
    "mobility service",
    "connected service",
]

OPERATIONAL_CONTEXT_TERMS = [
    "business model",
    "channel",
    "customer",
    "factory",
    "manufacturing",
    "plant",
    "production",
    "product",
    "procurement",
    "process",
    "quality",
    "revenue",
    "sales",
    "service",
    "supply chain",
    "logistics",
    "vehicle",
    "battery",
    "semiconductor",
    "network",
    "distribution",
    "commerce",
]

BOILERPLATE_PATTERNS = [
    r"\bimportant\s+trend\b",
    r"\bthe\s+digital\s+era\b",
    r"\bdigital\s+era\b",
    r"\brespond\s+to\s+(?:the\s+)?(?:changing|changes)\b",
    r"\bfuture\s+growth\s+engine\b",
    r"\bwill\s+continue\s+to\s+make\s+efforts\b",
    r"\bcontinue\s+to\s+strive\b",
    r"\bstrive\s+to\b",
    r"\bseek\s+to\b",
    r"\bprepare\s+for\s+the\s+future\b",
    r"\bparadigm\s+shift\b",
    r"\bnew\s+normal\b",
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"'(\[]))")


def compile_term_patterns(terms: Sequence[str]) -> Dict[str, re.Pattern[str]]:
    patterns: Dict[str, re.Pattern[str]] = {}
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        patterns[term] = re.compile(
            r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
    return patterns


CORE_DIGITAL_PATTERNS = compile_term_patterns(CORE_DIGITAL_TERMS)
CONTEXTUAL_DIGITAL_PATTERNS = compile_term_patterns(CONTEXTUAL_DIGITAL_TERMS)
ACTION_PATTERNS = compile_term_patterns(ACTION_TERMS)
BMI_PATTERNS = compile_term_patterns(BMI_TERMS)
OPERATIONAL_PATTERNS = compile_term_patterns(OPERATIONAL_CONTEXT_TERMS)
BOILERPLATE_REGEX = [re.compile(p, flags=re.IGNORECASE) for p in BOILERPLATE_PATTERNS]
REPRESENTATIVE_SENTENCE_COLUMNS = [
    "corp_code",
    "corp_name",
    "corp_name_eng",
    "stock_code",
    "year",
    "rcept_no",
    "report_name",
    "sentence_rank",
    "sentence_id",
    "priority",
    "sentence",
    "digital_sentence",
    "substantive_digital_sentence",
    "bmi_sentence",
    "market_signal_sentence",
    "boilerplate_digital_sentence",
    "core_digital_terms",
    "contextual_digital_terms",
    "action_terms",
    "bmi_terms",
    "operational_terms",
]
HANGUL_CHAR_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]{2,}\b")
COMPANY_DESIGNATOR_RE = re.compile(
    r"\b(?:ltd|llc|inc|corp|corporation|company|co\.?|pte|gmbh|s\.a\.|limited|private)\b",
    flags=re.IGNORECASE,
)
TABLE_OR_LIST_NOISE_RE = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"^\(?unit\s*:",
        r"^\(?in\s+(?:thousand|million|billions?)",
        r"^\(?단위\s*:",
        r"\b(?:ordinary shares|preferred shares|par value|book value|carrying amount)\b",
        r"\b(?:assets|liabilities|equity|depreciation|amortization)\b.{0,40}\b(?:assets|liabilities|equity|depreciation|amortization)\b",
        r"\b(?:krw|usd|eur|jpy)\s*[\d,]+",
        r"\b\d{4}[-./]\d{2}[-./]\d{2}\b.*\b\d{4}[-./]\d{2}[-./]\d{2}\b",
    ]
]


def progress(iterable: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc)
    return simple_progress(iterable, total=total, desc=desc)


def simple_progress(
    iterable: Iterable[Any],
    total: Optional[int] = None,
    desc: str = "",
    step: int = 100,
) -> Iterator[Any]:
    for i, item in enumerate(iterable, start=1):
        if i == 1 and desc:
            print(desc)
        if i % step == 0:
            suffix = f"/{total}" if total else ""
            print(f"  {desc}: {i}{suffix}")
        yield item


def dependency_status() -> Dict[str, bool]:
    return {
        "requests": requests is not None,
        "pandas": True,
        "beautifulsoup4": BeautifulSoup is not None,
        "lxml": lxml_html is not None,
        "tqdm": _tqdm is not None,
        "openpyxl": _has_openpyxl(),
    }


def _has_openpyxl() -> bool:
    try:
        import openpyxl  # noqa: F401

        return True
    except ImportError:
        return False


def decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "euc-kr", "cp949", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    text = html_lib.unescape(text or "")
    text = text.replace("&cr;", "\n").replace("&#13;", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean_sentence(sentence: str) -> str:
    sentence = normalize_text(sentence)
    sentence = re.sub(r"[◆◇●■□▶▷※]+", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip(" \t\r\n-;:")


def language_profile(text: str) -> Dict[str, Any]:
    text = text or ""
    latin_chars = len(LATIN_CHAR_RE.findall(text))
    hangul_chars = len(HANGUL_CHAR_RE.findall(text))
    letter_chars = latin_chars + hangul_chars
    latin_share = latin_chars / letter_chars if letter_chars else 0.0
    if letter_chars == 0:
        flag = "no_letters"
    elif latin_share >= 0.85 and (latin_chars >= 500 or hangul_chars == 0):
        flag = "mostly_english"
    elif latin_share >= 0.50:
        flag = "mixed_latin_korean"
    else:
        flag = "mostly_korean"
    return {
        "raw_text_chars": len(text),
        "raw_latin_chars": latin_chars,
        "raw_hangul_chars": hangul_chars,
        "raw_latin_share_of_letters": latin_share,
        "text_language_flag": flag,
    }


def is_table_or_list_fragment(sentence: str, english_words: Sequence[str]) -> bool:
    if any(pattern.search(sentence) for pattern in TABLE_OR_LIST_NOISE_RE):
        return True
    if sentence.count(" - ") >= 4 or sentence.count(" | ") >= 2:
        return True
    comma_count = sentence.count(",")
    if len(english_words) and comma_count / len(english_words) > 0.35:
        return True
    digit_count = len(re.findall(r"\d", sentence))
    if len(sentence) and digit_count / len(sentence) > 0.28:
        return True
    company_designators = COMPANY_DESIGNATOR_RE.findall(sentence)
    if len(company_designators) >= 3 and not match_terms(sentence, ACTION_PATTERNS):
        return True
    uppercase_words = [word for word in english_words if len(word) > 2 and word.isupper()]
    if len(english_words) >= 8 and len(uppercase_words) / len(english_words) > 0.45:
        return True
    return False


def is_english_narrative_sentence(sentence: str, min_chars: int = 40) -> bool:
    if len(sentence) < min_chars or len(sentence) > 800:
        return False
    hangul_chars = len(HANGUL_CHAR_RE.findall(sentence))
    latin_chars = len(LATIN_CHAR_RE.findall(sentence))
    if hangul_chars > 2:
        return False
    if latin_chars < 35:
        return False
    letter_chars = latin_chars + hangul_chars
    if letter_chars and latin_chars / letter_chars < 0.95:
        return False
    english_words = ENGLISH_WORD_RE.findall(sentence)
    if len(english_words) < 7:
        return False
    if is_table_or_list_fragment(sentence, english_words):
        return False
    return True


def split_sentences(text: str, min_chars: int = 40) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[•·]+", "\n", text)
    text = re.sub(r"(?<=[A-Za-z,;:])\n\s*(?=[A-Za-z])", " ", text)
    parts: List[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        parts.extend(SENTENCE_SPLIT_RE.split(block))

    sentences: List[str] = []
    for part in parts:
        sentence = clean_sentence(part)
        if not is_english_narrative_sentence(sentence, min_chars=min_chars):
            continue
        sentences.append(sentence)
    return sentences


def match_terms(text: str, patterns: Dict[str, re.Pattern[str]]) -> List[str]:
    return [term for term, pattern in patterns.items() if pattern.search(text)]


def has_boilerplate_language(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in BOILERPLATE_REGEX)


def classify_sentence(sentence: str) -> Dict[str, Any]:
    core_hits = match_terms(sentence, CORE_DIGITAL_PATTERNS)
    contextual_hits = match_terms(sentence, CONTEXTUAL_DIGITAL_PATTERNS)
    action_hits = match_terms(sentence, ACTION_PATTERNS)
    bmi_hits = match_terms(sentence, BMI_PATTERNS)
    operational_hits = match_terms(sentence, OPERATIONAL_PATTERNS)
    boilerplate_language = has_boilerplate_language(sentence)

    contextual_is_specific = bool(contextual_hits and action_hits and (bmi_hits or operational_hits))
    core_is_statement = bool(core_hits and (action_hits or boilerplate_language))
    digital_hit = bool(core_is_statement or contextual_is_specific)

    boilerplate = False
    if digital_hit:
        has_specific_evidence = bool(action_hits and (bmi_hits or operational_hits or core_hits))
        boilerplate = boilerplate_language and not has_specific_evidence
        if not has_specific_evidence and not core_hits:
            boilerplate = True

    substantive = bool(digital_hit and not boilerplate and action_hits and (bmi_hits or operational_hits or core_hits))
    bmi = bool(substantive and bmi_hits)
    market_signal = bool(digital_hit and not boilerplate and (substantive or core_hits))

    return {
        "digital_sentence": int(digital_hit),
        "substantive_digital_sentence": int(substantive),
        "bmi_sentence": int(bmi),
        "market_signal_sentence": int(market_signal),
        "boilerplate_digital_sentence": int(digital_hit and boilerplate),
        "core_digital_terms": "; ".join(core_hits),
        "contextual_digital_terms": "; ".join(contextual_hits),
        "action_terms": "; ".join(action_hits),
        "bmi_terms": "; ".join(bmi_hits),
        "operational_terms": "; ".join(operational_hits),
    }


def analyze_text(text: str, max_examples: int = 20) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile = language_profile(text)
    sentences = split_sentences(text)
    counts = {
        "digital_sentence_count": 0,
        "substantive_digital_sentence_count": 0,
        "bmi_sentence_count": 0,
        "market_signal_sentence_count": 0,
        "boilerplate_digital_sentence_count": 0,
    }
    examples: List[Dict[str, Any]] = []

    for sentence_id, sentence in enumerate(sentences, start=1):
        classified = classify_sentence(sentence)
        counts["digital_sentence_count"] += classified["digital_sentence"]
        counts["substantive_digital_sentence_count"] += classified["substantive_digital_sentence"]
        counts["bmi_sentence_count"] += classified["bmi_sentence"]
        counts["market_signal_sentence_count"] += classified["market_signal_sentence"]
        counts["boilerplate_digital_sentence_count"] += classified["boilerplate_digital_sentence"]

        if classified["digital_sentence"]:
            priority = (
                classified["substantive_digital_sentence"] * 100
                + classified["bmi_sentence"] * 30
                + classified["market_signal_sentence"] * 10
                - classified["boilerplate_digital_sentence"] * 50
            )
            examples.append(
                {
                    "sentence_id": sentence_id,
                    "priority": priority,
                    "sentence": sentence[:3000],
                    **classified,
                }
            )

    total_sentences = len(sentences)
    if total_sentences:
        digital_strategy = counts["substantive_digital_sentence_count"] / total_sentences
        bmi_reconstruction = counts["bmi_sentence_count"] / total_sentences
        market_signal = counts["market_signal_sentence_count"] / total_sentences
    else:
        digital_strategy = 0.0
        bmi_reconstruction = 0.0
        market_signal = 0.0

    scores = {
        **profile,
        "total_sentences": total_sentences,
        "analyzable_english_sentence_count": total_sentences,
        **counts,
        "DigitalStrategy": digital_strategy,
        "BMI_Reconstruction": bmi_reconstruction,
        "MarketSignalDisclosure": market_signal,
    }

    examples.sort(key=lambda row: (-row["priority"], row["sentence_id"]))
    return scores, examples[:max_examples]


def markup_to_text(raw_text: str) -> str:
    raw_text = raw_text or ""
    if BeautifulSoup is not None:
        parser = "lxml" if lxml_html is not None else "html.parser"
        soup = BeautifulSoup(raw_text, parser)
        for tag in soup(["script", "style"]):
            tag.decompose()
        return normalize_text(soup.get_text(" "))
    if lxml_html is not None:
        try:
            root = lxml_html.fromstring(raw_text)
            return normalize_text(root.text_content())
        except Exception:
            pass
    no_tags = re.sub(r"<[^>]+>", " ", raw_text)
    return normalize_text(no_tags)


def zip_document_to_text(content: bytes) -> Tuple[str, int]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".xml", ".html", ".htm", ".txt"))
        ]
        text_parts = []
        for name in sorted(names):
            raw = archive.read(name)
            text_parts.append(markup_to_text(decode_bytes(raw)))
    return "\n".join(text_parts), len(names)


def parse_open_dart_error(content: bytes) -> str:
    text = decode_bytes(content[:5000]).strip()
    try:
        root = ET.fromstring(text)
        status = root.findtext("status") or ""
        message = root.findtext("message") or ""
        if status or message:
            return f"{status} {message}".strip()
    except ET.ParseError:
        pass
    try:
        data = json.loads(text)
        status = str(data.get("status", ""))
        message = str(data.get("message", ""))
        if status or message:
            return f"{status} {message}".strip()
    except json.JSONDecodeError:
        pass
    return text[:500]


class OpenDartError(RuntimeError):
    pass


class OpenDartClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        sleep_seconds: float = 0.12,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        self.api_key = (api_key or os.getenv("DART_API_KEY") or "").strip()
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session() if requests is not None else None

    def require_key(self) -> None:
        if not self.api_key:
            try:
                entered = input("Enter your OpenDART API key (DART_API_KEY): ").strip()
            except EOFError:
                entered = ""
            if entered:
                self.api_key = entered
        if not self.api_key:
            raise OpenDartError(
                "DART_API_KEY is not set. Set it in the environment or enter it when prompted."
            )

    def get_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        content = self.get_bytes(url, params)
        text = decode_bytes(content)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenDartError(f"Expected JSON but received: {text[:300]}") from exc

    def get_zip(self, url: str, params: Dict[str, Any]) -> bytes:
        content = self.get_bytes(url, params)
        if content[:2] == b"PK":
            return content
        error_text = parse_open_dart_error(content)
        raise OpenDartError(f"Expected zip response from {url}; received: {error_text}")

    def get_bytes(self, url: str, params: Dict[str, Any]) -> bytes:
        self.require_key()
        clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
        clean_params["crtfc_key"] = self.api_key

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                if self.session is not None:
                    response = self.session.get(url, params=clean_params, timeout=self.timeout)
                    response.raise_for_status()
                    content = response.content
                else:
                    query = urllib.parse.urlencode(clean_params)
                    request = urllib.request.Request(
                        f"{url}?{query}",
                        headers={"User-Agent": "Mozilla/5.0 korea-digital-strategy-pipeline"},
                    )
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        content = response.read()
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return content
            except Exception as exc:  # noqa: BLE001 - retry wrapper needs broad catch.
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        raise OpenDartError(f"Request failed after {self.retries} attempts: {last_error}")


@dataclass
class PipelineConfig:
    start_year: int = 2015
    end_year: int = 2025
    out_dir: Path = SCRIPT_DIR / "korea_digital_strategy_output"
    markets: Tuple[str, ...] = ("Y", "K")
    min_available_years: int = 5
    max_final_firms: int = 180
    sleep_seconds: float = 0.12
    timeout: int = 60
    retries: int = 3
    refresh_metadata: bool = False
    refresh_reports: bool = False
    metadata_only: bool = False
    enrich_profiles: bool = True
    exclude_financial: bool = True
    representative_sentences_per_report: int = 20
    allow_korean_document_fallback: bool = False
    offline_cache_only: bool = False
    sample_corp_codes: Tuple[str, ...] = ()
    selected_sample_file: Optional[Path] = None

    @property
    def raw_dir(self) -> Path:
        return self.out_dir / "raw_reports"

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "_cache"


def ensure_dirs(config: PipelineConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)


def write_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = sanitize_dataframe_for_excel(df)
    max_excel_rows = 1_048_576
    if len(df) <= max_excel_rows:
        df.to_excel(path, index=False)
        return

    rows_per_sheet = max_excel_rows - 1
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for idx, start in enumerate(range(0, len(df), rows_per_sheet), start=1):
            end = min(start + rows_per_sheet, len(df))
            df.iloc[start:end].to_excel(writer, index=False, sheet_name=f"part_{idx}")


def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Excel export robust if an API field arrives as a nested value."""
    if df.empty:
        return df

    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype != "object":
            continue

        def clean_value(value: Any) -> Any:
            if isinstance(value, (list, tuple, set, dict)):
                return json.dumps(value, ensure_ascii=False)
            if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                converted = value.tolist()
                if isinstance(converted, (list, tuple, dict)):
                    return json.dumps(converted, ensure_ascii=False)
            return value

        cleaned[col] = cleaned[col].map(clean_value)
    return cleaned


def read_excel_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_excel(path, dtype={"corp_code": str, "stock_code": str, "rcept_no": str})
    return None


def parse_corp_code_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        xml_name = archive.namelist()[0]
        xml_bytes = archive.read(xml_name)

    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.findall("list"):
        rows.append(
            {
                "corp_code": item.findtext("corp_code"),
                "corp_name": item.findtext("corp_name"),
                "stock_code": item.findtext("stock_code"),
                "modify_date": item.findtext("modify_date"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["corp_code"] = df["corp_code"].fillna("").astype(str).str.zfill(8)
    df["stock_code"] = df["stock_code"].fillna("").astype(str).str.strip()
    df = df[df["stock_code"].str.match(r"^\d{6}$", na=False)].copy()
    df["stock_code"] = df["stock_code"].str.zfill(6)
    return df.drop_duplicates("corp_code").reset_index(drop=True)


def get_listed_corporations(client: OpenDartClient) -> pd.DataFrame:
    urls = [
        f"{ENG_API_BASE}/corpCode.xml",
        f"{KOR_API_BASE}/corpCode.xml",
    ]
    last_error = None
    for url in urls:
        try:
            content = client.get_zip(url, {})
            return parse_corp_code_zip(content)
        except Exception as exc:  # noqa: BLE001 - fallback to Korean endpoint.
            last_error = exc
    raise OpenDartError(f"Could not retrieve corporation codes: {last_error}")


def quarter_windows(start: date, end: date) -> Iterator[Tuple[date, date]]:
    current = start
    while current <= end:
        quarter_end_month = ((current.month - 1) // 3 + 1) * 3
        last_day = calendar.monthrange(current.year, quarter_end_month)[1]
        window_end = date(current.year, quarter_end_month, last_day)
        if window_end > end:
            window_end = end
        yield current, window_end
        if quarter_end_month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, quarter_end_month + 1, 1)


def infer_fiscal_year(report_name: Any, filing_date: Any) -> Optional[int]:
    report_text = str(report_name or "")
    filing_text = re.sub(r"\D", "", str(filing_date or ""))
    filing_year = None
    filing_month = None
    if len(filing_text) >= 8:
        filing_year = int(filing_text[:4])
        filing_month = int(filing_text[4:6])

    candidates: List[int] = []
    for match in re.finditer(r"(?<!\d)(20\d{2})(?:[.\-/]\s*(0?[1-9]|1[0-2]))?(?!\d)", report_text):
        year = int(match.group(1))
        if filing_year is None or year <= filing_year:
            candidates.append(year)
    if candidates:
        return max(candidates)

    if filing_year is None:
        return None
    if filing_month is not None and filing_month <= 6:
        return filing_year - 1
    return filing_year


def normalize_report_rows(items: Any, market: str, window_start: date, window_end: date) -> List[Dict[str, Any]]:
    if not items:
        return []
    if isinstance(items, dict):
        items = [items]
    rows = []
    for item in items:
        report_name = item.get("report_nm") or item.get("report_name")
        rcept_dt = item.get("rcept_dt")
        fiscal_year = infer_fiscal_year(report_name, rcept_dt)
        rows.append(
            {
                "corp_code": str(item.get("corp_code") or "").zfill(8),
                "corp_name": item.get("corp_name"),
                "stock_code": str(item.get("stock_code") or "").strip(),
                "corp_cls": item.get("corp_cls") or market,
                "market": MARKET_LABELS.get(item.get("corp_cls") or market, item.get("corp_cls") or market),
                "year": fiscal_year,
                "rcept_no": str(item.get("rcept_no") or ""),
                "report_name": report_name,
                "flr_nm": item.get("flr_nm"),
                "rcept_dt": rcept_dt,
                "rm": item.get("rm"),
                "search_market": market,
                "search_window_start": window_start.strftime("%Y%m%d"),
                "search_window_end": window_end.strftime("%Y%m%d"),
            }
        )
    return rows


def normalize_reports_dataframe(reports: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if reports.empty:
        return reports
    reports = reports.copy()
    reports["rcept_no"] = reports["rcept_no"].fillna("").astype(str).str.strip()
    reports = reports[reports["rcept_no"].str.len() > 0].copy()
    reports = reports.drop_duplicates("rcept_no").reset_index(drop=True)
    reports["year"] = pd.to_numeric(reports["year"], errors="coerce")
    reports = reports[reports["year"].between(config.start_year, config.end_year, inclusive="both")].copy()
    reports["year"] = reports["year"].astype(int)
    stock_code = reports["stock_code"].fillna("").astype(str).str.strip()
    reports["stock_code"] = stock_code.where(stock_code.str.match(r"^\d{6}$"), "")
    reports["stock_code"] = reports["stock_code"].where(reports["stock_code"] == "", reports["stock_code"].str.zfill(6))
    reports["corp_code"] = reports["corp_code"].fillna("").astype(str).str.zfill(8)
    return reports.reset_index(drop=True)


def load_completed_search_windows(progress_json: Optional[Path]) -> set[str]:
    if not progress_json or not progress_json.exists():
        return set()
    try:
        data = json.loads(progress_json.read_text(encoding="utf-8"))
        return set(data.get("completed_windows", []))
    except Exception:
        return set()


def save_completed_search_windows(progress_json: Optional[Path], completed: set[str]) -> None:
    if not progress_json:
        return
    progress_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_windows": sorted(completed),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_search_checkpoint(checkpoint_csv: Optional[Path], rows: List[Dict[str, Any]]) -> None:
    if not checkpoint_csv:
        return
    checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(checkpoint_csv, index=False, encoding="utf-8-sig")


def search_task_key(market: str, start: date, end: date) -> str:
    return f"{market}:{start.strftime('%Y%m%d')}:{end.strftime('%Y%m%d')}"


def search_english_annual_reports_bulk(
    client: OpenDartClient,
    config: PipelineConfig,
    checkpoint_csv: Optional[Path] = None,
    progress_json: Optional[Path] = None,
) -> pd.DataFrame:
    filing_start = date(config.start_year, 1, 1)
    filing_end = min(date(config.end_year + 1, 12, 31), date.today())
    if filing_end < filing_start:
        filing_end = date(config.end_year, 12, 31)

    rows: List[Dict[str, Any]] = []
    if checkpoint_csv and checkpoint_csv.exists():
        existing = pd.read_csv(
            checkpoint_csv,
            dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
        )
        rows = existing.to_dict("records")
        print(f"Loaded partial English annual report search cache: {len(rows):,} rows")

    completed_windows = load_completed_search_windows(progress_json)
    windows = list(quarter_windows(filing_start, filing_end))
    tasks = [(market, start, end) for market in config.markets for start, end in windows]

    for market, start, end in progress(tasks, total=len(tasks), desc="Searching English annual reports"):
        task_key = search_task_key(market, start, end)
        if task_key in completed_windows:
            continue

        page_no = 1
        task_rows: List[Dict[str, Any]] = []
        while True:
            params = {
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "last_reprt_at": "Y",
                "pblntf_ty": "A",
                "pblntf_detail_ty": "A001",
                "corp_cls": market,
                "sort": "date",
                "sort_mth": "desc",
                "page_no": page_no,
                "page_count": 100,
            }
            data = client.get_json(f"{ENG_API_BASE}/list.json", params)
            status = str(data.get("status", ""))
            if status == "013":
                break
            if status != "000":
                raise OpenDartError(
                    f"Search failed for market={market}, {start}-{end}, page={page_no}: "
                    f"{status} {data.get('message')}"
                )
            task_rows.extend(normalize_report_rows(data.get("list"), market, start, end))
            total_page = int(data.get("total_page") or 1)
            if page_no >= total_page:
                break
            page_no += 1

        rows.extend(task_rows)
        completed_windows.add(task_key)
        save_search_checkpoint(checkpoint_csv, rows)
        save_completed_search_windows(progress_json, completed_windows)

    reports = pd.DataFrame(rows)
    return normalize_reports_dataframe(reports, config)


def choose_one_report_per_firm_year(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return reports
    work = reports.copy()
    work["rcept_dt_sort"] = pd.to_numeric(work["rcept_dt"], errors="coerce")
    work["rcept_no_sort"] = pd.to_numeric(work["rcept_no"], errors="coerce")
    work = work.sort_values(
        ["corp_code", "year", "rcept_dt_sort", "rcept_no_sort"],
        ascending=[True, True, False, False],
    )
    work = work.drop_duplicates(["corp_code", "year"], keep="first")
    return work.drop(columns=["rcept_dt_sort", "rcept_no_sort"]).reset_index(drop=True)


def fetch_company_profiles(
    client: OpenDartClient,
    corp_codes: Sequence[str],
    cache_path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        return pd.read_excel(cache_path, dtype={"corp_code": str, "stock_code": str})

    rows: List[Dict[str, Any]] = []
    unique_codes = sorted({str(code).zfill(8) for code in corp_codes if str(code).strip()})
    for corp_code in progress(unique_codes, total=len(unique_codes), desc="Fetching company profiles"):
        try:
            data = client.get_json(f"{ENG_API_BASE}/company.json", {"corp_code": corp_code})
            status = str(data.get("status", ""))
            if status == "000":
                rows.append(
                    {
                        "corp_code": corp_code,
                        "corp_name_eng": data.get("corp_name_eng"),
                        "corp_cls_profile": data.get("corp_cls"),
                        "ceo_nm": data.get("ceo_nm"),
                        "jurir_no": data.get("jurir_no"),
                        "bizr_no": data.get("bizr_no"),
                        "adres": data.get("adres"),
                        "hm_url": data.get("hm_url"),
                        "ir_url": data.get("ir_url"),
                        "phn_no": data.get("phn_no"),
                        "fax_no": data.get("fax_no"),
                        "industry_code": data.get("induty_code"),
                        "est_dt": data.get("est_dt"),
                        "acc_mt": data.get("acc_mt"),
                        "profile_error": None,
                    }
                )
            else:
                rows.append({"corp_code": corp_code, "profile_error": f"{status} {data.get('message')}"})
        except Exception as exc:  # noqa: BLE001 - keep pipeline going.
            rows.append({"corp_code": corp_code, "profile_error": str(exc)})

    profiles = pd.DataFrame(rows)
    if not profiles.empty:
        write_excel(profiles, cache_path)
    return profiles


def available_year_blocks(years: Sequence[int]) -> Tuple[str, int]:
    clean_years = sorted({int(y) for y in years if pd.notna(y)})
    if not clean_years:
        return "", 0
    blocks: List[str] = []
    start = prev = clean_years[0]
    max_len = 1
    current_len = 1
    for year in clean_years[1:]:
        if year == prev + 1:
            current_len += 1
        else:
            blocks.append(f"{start}" if start == prev else f"{start}-{prev}")
            max_len = max(max_len, current_len)
            start = year
            current_len = 1
        prev = year
    blocks.append(f"{start}" if start == prev else f"{start}-{prev}")
    max_len = max(max_len, current_len)
    return ", ".join(blocks), max_len


def first_nonempty(values: Sequence[Any]) -> Any:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return value
    return None


def mode_market(values: Sequence[Any]) -> Any:
    clean = [value for value in values if pd.notna(value) and str(value).strip()]
    if not clean:
        return None
    counts: Dict[str, int] = {}
    for value in clean:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], MARKET_PRIORITY.get(item[0], 9)))[0][0]


def is_financial_industry(industry_code: Any, corp_name: Any = "", corp_name_eng: Any = "") -> bool:
    code = re.sub(r"\D", "", str(industry_code or ""))
    if code.startswith(("64", "65", "66")):
        return True
    name = f"{corp_name or ''} {corp_name_eng or ''}".lower()
    name_patterns = [
        "bank",
        "financial",
        "finance",
        "insurance",
        "securities",
        "asset management",
        "capital",
        "card co",
        "credit",
        "investment trust",
        "mutual savings",
    ]
    return any(pattern in name for pattern in name_patterns)


def build_metadata_grid(
    corp_df: pd.DataFrame,
    reports: pd.DataFrame,
    profiles: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    one_report = choose_one_report_per_firm_year(reports)
    years_df = pd.DataFrame({"year": list(range(config.start_year, config.end_year + 1))})
    base = corp_df[["corp_code", "corp_name", "stock_code", "modify_date"]].drop_duplicates("corp_code").copy()
    base["key"] = 1
    years_df["key"] = 1
    grid = base.merge(years_df, on="key").drop(columns=["key"])

    report_cols = [
        "corp_code",
        "year",
        "corp_cls",
        "market",
        "rcept_no",
        "report_name",
        "flr_nm",
        "rcept_dt",
        "rm",
    ]
    if one_report.empty:
        metadata = grid.copy()
    else:
        metadata = grid.merge(one_report[report_cols], on=["corp_code", "year"], how="left")
    metadata["has_english_annual_report"] = metadata["rcept_no"].notna().astype(int)
    metadata["corp_cls"] = metadata["corp_cls"].fillna("")
    metadata["market"] = metadata["market"].fillna(metadata["corp_cls"].map(MARKET_LABELS)).fillna("")

    if not profiles.empty:
        profiles_small = profiles.drop_duplicates("corp_code")
        metadata = metadata.merge(profiles_small, on="corp_code", how="left")
        metadata["corp_cls"] = metadata["corp_cls"].where(
            metadata["corp_cls"].astype(str).str.len() > 0,
            metadata.get("corp_cls_profile"),
        )
        metadata["market"] = metadata["corp_cls"].map(MARKET_LABELS).fillna(metadata["market"])
    else:
        metadata["industry_code"] = None
        metadata["corp_name_eng"] = None

    metadata["is_financial"] = [
        is_financial_industry(row.get("industry_code"), row.get("corp_name"), row.get("corp_name_eng"))
        for row in metadata.to_dict("records")
    ]

    availability = build_corporation_availability(corp_df, metadata, profiles)
    return metadata, availability


def build_corporation_availability(
    corp_df: pd.DataFrame,
    metadata: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    available = metadata[metadata["has_english_annual_report"] == 1].copy()
    rows = []
    grouped = available.groupby("corp_code") if not available.empty else []
    availability_by_code: Dict[str, Dict[str, Any]] = {}
    for corp_code, group in grouped:
        years = sorted(group["year"].dropna().astype(int).unique().tolist())
        blocks, max_consecutive = available_year_blocks(years)
        corp_cls = mode_market(group["corp_cls"].tolist())
        availability_by_code[str(corp_code).zfill(8)] = {
            "available_english_annual_report_years": ", ".join(str(y) for y in years),
            "available_year_blocks": blocks,
            "n_available_english_annual_report_years": len(years),
            "max_consecutive_english_annual_report_years": max_consecutive,
            "first_english_annual_report_year": min(years) if years else None,
            "last_english_annual_report_year": max(years) if years else None,
            "corp_cls": corp_cls,
            "market": MARKET_LABELS.get(corp_cls, corp_cls),
        }

    for _, corp in corp_df.iterrows():
        corp_code = str(corp["corp_code"]).zfill(8)
        row = corp.to_dict()
        row.update(
            availability_by_code.get(
                corp_code,
                {
                    "available_english_annual_report_years": "",
                    "available_year_blocks": "",
                    "n_available_english_annual_report_years": 0,
                    "max_consecutive_english_annual_report_years": 0,
                    "first_english_annual_report_year": None,
                    "last_english_annual_report_year": None,
                    "corp_cls": None,
                    "market": None,
                },
            )
        )
        rows.append(row)

    availability = pd.DataFrame(rows)
    if not profiles.empty:
        availability = availability.merge(profiles.drop_duplicates("corp_code"), on="corp_code", how="left")
        availability["corp_cls"] = availability["corp_cls"].where(
            availability["corp_cls"].notna(),
            availability.get("corp_cls_profile"),
        )
        availability["market"] = availability["corp_cls"].map(MARKET_LABELS).fillna(availability["market"])
    if "industry_code" not in availability.columns:
        availability["industry_code"] = None
    if "corp_name_eng" not in availability.columns:
        availability["corp_name_eng"] = None
    availability["is_financial"] = [
        is_financial_industry(row.get("industry_code"), row.get("corp_name"), row.get("corp_name_eng"))
        for row in availability.to_dict("records")
    ]
    return availability


def build_selected_sample(metadata: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    available = metadata[metadata["has_english_annual_report"] == 1].copy()
    if config.sample_corp_codes:
        sample_codes = {code.zfill(8) for code in config.sample_corp_codes}
        available = available[available["corp_code"].isin(sample_codes)].copy()
    if available.empty:
        return pd.DataFrame()

    grouped_rows = []
    for (corp_code, corp_name, stock_code), group in available.groupby(["corp_code", "corp_name", "stock_code"]):
        years = sorted(group["year"].dropna().astype(int).unique().tolist())
        blocks, max_consecutive = available_year_blocks(years)
        corp_cls = mode_market(group["corp_cls"].tolist())
        corp_name_eng = first_nonempty(group.get("corp_name_eng", pd.Series(dtype=object)).tolist())
        industry_code = first_nonempty(group.get("industry_code", pd.Series(dtype=object)).tolist())
        is_financial = is_financial_industry(industry_code, corp_name, corp_name_eng)
        grouped_rows.append(
            {
                "corp_code": corp_code,
                "corp_name": corp_name,
                "corp_name_eng": corp_name_eng,
                "stock_code": stock_code,
                "corp_cls": corp_cls,
                "market": MARKET_LABELS.get(corp_cls, corp_cls),
                "industry_code": industry_code,
                "is_financial": is_financial,
                "available_english_annual_report_years": ", ".join(str(y) for y in years),
                "available_year_blocks": blocks,
                "n_available_years": len(years),
                "max_consecutive_years": max_consecutive,
                "first_year": min(years) if years else None,
                "last_year": max(years) if years else None,
            }
        )

    selected = pd.DataFrame(grouped_rows)
    selected = selected[selected["n_available_years"] >= config.min_available_years].copy()
    selected = selected[selected["corp_cls"].isin(config.markets)].copy()
    if config.exclude_financial:
        selected = selected[selected["is_financial"] != True].copy()  # noqa: E712 - pandas bool comparison.
    selected["market_priority"] = selected["corp_cls"].map(MARKET_PRIORITY).fillna(9)
    selected = selected.sort_values(
        ["market_priority", "n_available_years", "max_consecutive_years", "last_year", "stock_code"],
        ascending=[True, False, False, False, True],
    )
    if config.max_final_firms:
        selected = selected.head(config.max_final_firms)
    return selected.drop(columns=["market_priority"]).reset_index(drop=True)


def load_selected_sample_override(path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Selected sample file not found: {path}")
    selected = pd.read_excel(path, dtype={"corp_code": str, "stock_code": str})
    if "corp_code" not in selected.columns:
        raise ValueError(f"Selected sample file must contain corp_code: {path}")
    selected["corp_code"] = selected["corp_code"].astype(str).str.zfill(8)
    selected = selected[selected["corp_code"].str.match(r"^\d{8}$", na=False)].copy()
    selected = selected.drop_duplicates("corp_code").reset_index(drop=True)

    if "stock_code" not in selected.columns or "corp_name" not in selected.columns:
        firm_info_cols = [
            col
            for col in [
                "corp_code",
                "corp_name",
                "corp_name_eng",
                "stock_code",
                "corp_cls",
                "market",
                "industry_code",
                "is_financial",
            ]
            if col in metadata.columns
        ]
        firm_info = metadata[firm_info_cols].drop_duplicates("corp_code")
        selected = selected.merge(firm_info, on="corp_code", how="left", suffixes=("", "_metadata"))
        for col in ["corp_name", "corp_name_eng", "stock_code", "corp_cls", "market", "industry_code", "is_financial"]:
            meta_col = f"{col}_metadata"
            if meta_col in selected.columns:
                selected[col] = selected.get(col).where(selected.get(col).notna(), selected[meta_col]) if col in selected.columns else selected[meta_col]
                selected = selected.drop(columns=[meta_col])

    if "stock_code" not in selected.columns:
        selected["stock_code"] = ""
    selected["stock_code"] = selected["stock_code"].fillna("").astype(str).str.zfill(6)
    return selected


def safe_filename_piece(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80] or "unknown"


def download_report_text(
    client: OpenDartClient,
    rcept_no: str,
    config: PipelineConfig,
) -> Tuple[Optional[str], Optional[str], Optional[str], int]:
    endpoints = [(f"{ENG_API_BASE}/document.xml", "engopendart")]
    if config.allow_korean_document_fallback:
        endpoints.append((f"{KOR_API_BASE}/document.xml", "opendart_korean_fallback"))

    last_error = None
    for url, endpoint_label in endpoints:
        try:
            content = client.get_zip(url, {"rcept_no": rcept_no})
            text, file_count = zip_document_to_text(content)
            return text, None, endpoint_label, file_count
        except Exception as exc:  # noqa: BLE001 - endpoint fallback.
            last_error = str(exc)
    return None, last_error, None, 0


def build_panel(
    metadata: pd.DataFrame,
    selected: pd.DataFrame,
    client: OpenDartClient,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected_codes = set(selected["corp_code"].astype(str).str.zfill(8))
    work = metadata[
        (metadata["corp_code"].astype(str).str.zfill(8).isin(selected_codes))
        & (metadata["has_english_annual_report"] == 1)
        & (metadata["rcept_no"].notna())
    ].copy()
    work = choose_one_report_per_firm_year(work)

    rows: List[Dict[str, Any]] = []
    examples_rows: List[Dict[str, Any]] = []

    for _, report in progress(work.iterrows(), total=len(work), desc="Downloading and scoring"):
        report_dict = report.to_dict()
        rcept_no = str(report_dict.get("rcept_no") or "")
        stock_code = safe_filename_piece(report_dict.get("stock_code"))
        year = safe_filename_piece(report_dict.get("year"))
        txt_path = config.raw_dir / f"{stock_code}_{year}_{rcept_no}.txt"

        error = None
        endpoint_label = None
        file_count = 0
        if txt_path.exists() and not config.refresh_reports:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            download_status = "cached"
        elif config.offline_cache_only:
            text = ""
            error = "Missing local text cache; offline-cache-only mode skipped download."
            download_status = "missing_cache"
        else:
            text, error, endpoint_label, file_count = download_report_text(client, rcept_no, config)
            if text:
                txt_path.write_text(text, encoding="utf-8")
                download_status = "ok"
            else:
                text = ""
                download_status = "failed"

        scores, examples = analyze_text(
            text,
            max_examples=config.representative_sentences_per_report,
        )

        panel_row = {
            **report_dict,
            **scores,
            "download_status": download_status,
            "download_error": error,
            "document_endpoint": endpoint_label,
            "document_file_count": file_count,
            "text_cache_path": str(txt_path),
        }
        rows.append(panel_row)

        for rank, example in enumerate(examples, start=1):
            examples_rows.append(
                {
                    "corp_code": report_dict.get("corp_code"),
                    "corp_name": report_dict.get("corp_name"),
                    "corp_name_eng": report_dict.get("corp_name_eng"),
                    "stock_code": report_dict.get("stock_code"),
                    "year": report_dict.get("year"),
                    "rcept_no": rcept_no,
                    "report_name": report_dict.get("report_name"),
                    "sentence_rank": rank,
                    **example,
                }
            )

    panel = pd.DataFrame(rows)
    examples_df = pd.DataFrame(examples_rows, columns=REPRESENTATIVE_SENTENCE_COLUMNS)
    if not panel.empty:
        panel = add_relative_signals(panel)
        panel = order_panel_columns(panel)
    return panel, examples_df


def add_relative_signals(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()

    metric_cols = [
        "raw_text_chars",
        "raw_latin_chars",
        "raw_hangul_chars",
        "raw_latin_share_of_letters",
        "total_sentences",
        "analyzable_english_sentence_count",
        "digital_sentence_count",
        "substantive_digital_sentence_count",
        "bmi_sentence_count",
        "market_signal_sentence_count",
        "boilerplate_digital_sentence_count",
        "DigitalStrategy",
        "BMI_Reconstruction",
        "MarketSignalDisclosure",
    ]
    for col in metric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    result["YearMean_DigitalStrategy"] = result.groupby("year")["DigitalStrategy"].transform("mean")
    result["RelativeDigitalSignal_YearOnly"] = (
        result["DigitalStrategy"] - result["YearMean_DigitalStrategy"]
    )

    if "industry_code" in result.columns:
        result["industry_group"] = (
            result["industry_code"]
            .fillna("")
            .astype(str)
            .str.extract(r"(\d{2})", expand=False)
            .fillna("")
        )
    else:
        result["industry_group"] = ""

    has_industry = result["industry_group"].astype(str).str.len() > 0
    result["IndustryYearMean_DigitalStrategy"] = float("nan")
    if has_industry.any():
        industry_means = (
            result.loc[has_industry]
            .groupby(["industry_group", "year"])["DigitalStrategy"]
            .mean()
            .to_dict()
        )
        industry_values = [
            industry_means.get((industry_group, year), float("nan"))
            for industry_group, year in zip(
                result.loc[has_industry, "industry_group"],
                result.loc[has_industry, "year"],
            )
        ]
        result.loc[has_industry, "IndustryYearMean_DigitalStrategy"] = industry_values

    result["RelativeDigitalSignal_IndustryYear"] = (
        result["DigitalStrategy"] - result["IndustryYearMean_DigitalStrategy"]
    )
    result["RelativeDigitalSignal"] = result["RelativeDigitalSignal_YearOnly"]
    has_industry_mean = has_industry & result["IndustryYearMean_DigitalStrategy"].notna()
    result.loc[has_industry_mean, "RelativeDigitalSignal"] = (
        result.loc[has_industry_mean, "RelativeDigitalSignal_IndustryYear"].to_numpy()
    )
    result["relative_signal_reference"] = "year"
    result.loc[has_industry_mean, "relative_signal_reference"] = "industry-year"
    return result


def order_panel_columns(panel: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "corp_code",
        "corp_name",
        "corp_name_eng",
        "stock_code",
        "corp_cls",
        "market",
        "industry_code",
        "industry_group",
        "is_financial",
        "year",
        "rcept_no",
        "report_name",
        "flr_nm",
        "rcept_dt",
        "raw_text_chars",
        "raw_latin_chars",
        "raw_hangul_chars",
        "raw_latin_share_of_letters",
        "text_language_flag",
        "total_sentences",
        "analyzable_english_sentence_count",
        "digital_sentence_count",
        "substantive_digital_sentence_count",
        "bmi_sentence_count",
        "market_signal_sentence_count",
        "boilerplate_digital_sentence_count",
        "DigitalStrategy",
        "BMI_Reconstruction",
        "MarketSignalDisclosure",
        "YearMean_DigitalStrategy",
        "IndustryYearMean_DigitalStrategy",
        "RelativeDigitalSignal_YearOnly",
        "RelativeDigitalSignal_IndustryYear",
        "RelativeDigitalSignal",
        "relative_signal_reference",
        "download_status",
        "download_error",
        "document_endpoint",
        "document_file_count",
        "text_cache_path",
    ]
    ordered = [col for col in preferred if col in panel.columns]
    remaining = [col for col in panel.columns if col not in ordered]
    return panel[ordered + remaining]


def run_pipeline(config: PipelineConfig) -> None:
    ensure_dirs(config)
    client = OpenDartClient(
        sleep_seconds=config.sleep_seconds,
        timeout=config.timeout,
        retries=config.retries,
    )

    corp_path = config.out_dir / "01_listed_corporations_from_dart.xlsx"
    metadata_path = config.out_dir / "02_english_annual_report_metadata.xlsx"
    selected_path = config.out_dir / "03_selected_research_sample.xlsx"
    panel_path = config.out_dir / "04_digital_strategy_panel.xlsx"
    examples_path = config.out_dir / "05_representative_digital_sentences.xlsx"
    raw_reports_path = config.cache_dir / "english_annual_reports_found_raw.xlsx"
    raw_reports_checkpoint_csv = config.cache_dir / "english_annual_reports_found_raw.partial.csv"
    raw_reports_progress_path = config.cache_dir / "english_annual_reports_search_progress.json"
    profiles_path = config.cache_dir / "company_profiles.xlsx"

    if config.offline_cache_only:
        if not metadata_path.exists() or not corp_path.exists():
            raise RuntimeError(
                "offline-cache-only mode requires existing 01 and 02 metadata files."
            )
        print("Offline cache-only mode enabled; DART downloads will be skipped.")
    else:
        client.require_key()

    legacy_cache_dir = Path.cwd() / "korea_digital_strategy_output" / "_cache"
    for legacy_path, target_path in [
        (legacy_cache_dir / "english_annual_reports_found_raw.xlsx", raw_reports_path),
        (legacy_cache_dir / "company_profiles.xlsx", profiles_path),
    ]:
        try:
            same_file = legacy_path.resolve() == target_path.resolve()
        except OSError:
            same_file = False
        if legacy_path.exists() and not target_path.exists() and not same_file:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, target_path)
            print(f"Migrated legacy cache: {legacy_path} -> {target_path}")

    if metadata_path.exists() and corp_path.exists() and not config.refresh_metadata:
        print("Loading cached corporation list and metadata.")
        corp_df = pd.read_excel(corp_path, dtype={"corp_code": str, "stock_code": str})
        metadata = pd.read_excel(metadata_path, dtype={"corp_code": str, "stock_code": str, "rcept_no": str})
    else:
        print("Retrieving listed corporations.")
        corp_df = get_listed_corporations(client)
        print(f"Listed corporations with stock codes: {len(corp_df):,}")

        if raw_reports_path.exists() and not config.refresh_metadata:
            print("Loading cached raw English annual report search results.")
            reports = pd.read_excel(
                raw_reports_path,
                dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
            )
            reports = normalize_reports_dataframe(reports, config)
        else:
            print("Searching English OpenDART annual report metadata in quarterly windows.")
            reports = search_english_annual_reports_bulk(
                client,
                config,
                checkpoint_csv=raw_reports_checkpoint_csv,
                progress_json=raw_reports_progress_path,
            )
            write_excel(reports, raw_reports_path)
        print(f"English annual report filings found: {len(reports):,}")

        profile_codes: Sequence[str]
        if config.enrich_profiles and not reports.empty:
            profile_codes = sorted(reports["corp_code"].dropna().astype(str).str.zfill(8).unique())
            profiles = fetch_company_profiles(
                client,
                profile_codes,
                profiles_path,
                refresh=config.refresh_metadata,
            )
        else:
            profiles = pd.DataFrame()

        metadata, corp_availability = build_metadata_grid(corp_df, reports, profiles, config)
        write_excel(corp_availability, corp_path)
        write_excel(metadata, metadata_path)

    if config.selected_sample_file:
        selected = load_selected_sample_override(config.selected_sample_file, metadata)
        print(f"Loaded selected sample override: {config.selected_sample_file}")
    else:
        selected = build_selected_sample(metadata, config)
    write_excel(selected, selected_path)
    print(f"Selected firms: {len(selected):,}")

    if config.metadata_only:
        print("Metadata-only mode enabled; skipping downloads and sentence scoring.")
        return
    if selected.empty:
        print("No firms passed the availability filter; panel scoring skipped.")
        return

    panel, examples = build_panel(metadata, selected, client, config)
    write_excel(panel, panel_path)
    write_excel(examples, examples_path)

    print("Done.")
    print(f"Output directory: {config.out_dir.resolve()}")
    print(f"Firm-year panel rows: {len(panel):,}")
    print(f"Representative sentence rows: {len(examples):,}")


def doctor() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"DART_API_KEY set: {bool(os.getenv('DART_API_KEY'))}")
    print("Dependency status:")
    for name, ok in dependency_status().items():
        marker = "OK" if ok else "MISSING"
        fallback = ""
        if name == "requests" and not ok:
            fallback = " (urllib fallback will be used)"
        if name == "beautifulsoup4" and not ok and lxml_html is not None:
            fallback = " (lxml fallback will be used)"
        if name == "tqdm" and not ok:
            fallback = " (simple progress fallback will be used)"
        print(f"  {name}: {marker}{fallback}")
    if not _has_openpyxl():
        print("openpyxl is required for Excel output.")
    return 0


def self_test() -> int:
    text = """
    Digital transformation is important in the new digital era.
    In 2023, the company invested in an AI-based data platform to optimize
    production planning and supply chain operations.
    We launched a subscription software service and online channel to improve
    customer experience and create recurring revenue.
    The company will continue to strive for digital innovation as a future growth engine.
    """
    scores, examples = analyze_text(text, max_examples=10)
    print(json.dumps(scores, indent=2))
    print("Representative sentences:")
    for row in examples:
        print(
            f"- substantive={row['substantive_digital_sentence']} "
            f"bmi={row['bmi_sentence']} boilerplate={row['boilerplate_digital_sentence']} "
            f"| {row['sentence']}"
        )
    return 0


def parse_markets(text: str) -> Tuple[str, ...]:
    values = tuple(part.strip().upper() for part in text.split(",") if part.strip())
    invalid = [value for value in values if value not in MARKET_LABELS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid market code(s): {', '.join(invalid)}")
    return values


def parse_corp_codes(text: str) -> Tuple[str, ...]:
    if not text:
        return ()
    return tuple(re.sub(r"\D", "", part).zfill(8) for part in text.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run metadata collection, sample selection, and scoring.")
    run.add_argument("--start-year", type=int, default=2015)
    run.add_argument("--end-year", type=int, default=2025)
    run.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "korea_digital_strategy_output")
    run.add_argument("--markets", type=parse_markets, default=("Y", "K"), help="Comma-separated: Y,K,N,E")
    run.add_argument("--min-available-years", type=int, default=5)
    run.add_argument("--max-final-firms", type=int, default=180)
    run.add_argument("--sleep-seconds", type=float, default=0.12)
    run.add_argument("--timeout", type=int, default=60)
    run.add_argument("--retries", type=int, default=3)
    run.add_argument("--refresh-metadata", action="store_true")
    run.add_argument("--refresh-reports", action="store_true")
    run.add_argument("--metadata-only", action="store_true")
    run.add_argument("--skip-profile-enrichment", action="store_true")
    run.add_argument("--include-financial", action="store_true")
    run.add_argument("--allow-korean-document-fallback", action="store_true")
    run.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="Recompute scoring from local raw_reports cache only; do not call DART for missing reports.",
    )
    run.add_argument("--representative-sentences-per-report", type=int, default=20)
    run.add_argument(
        "--sample-corp-codes",
        type=parse_corp_codes,
        default=(),
        help="Optional comma-separated 8-digit corp_code list for a small scoring run.",
    )
    run.add_argument(
        "--selected-sample-file",
        type=Path,
        default=None,
        help="Optional Excel file with a corp_code column to override automatic sample selection.",
    )

    subparsers.add_parser("doctor", help="Check API key and Python dependency status.")
    subparsers.add_parser("self-test", help="Run the sentence classifier on sample text.")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        out_dir=args.out_dir,
        markets=args.markets,
        min_available_years=args.min_available_years,
        max_final_firms=args.max_final_firms,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retries=args.retries,
        refresh_metadata=args.refresh_metadata,
        refresh_reports=args.refresh_reports,
        metadata_only=args.metadata_only,
        enrich_profiles=not args.skip_profile_enrichment,
        exclude_financial=not args.include_financial,
        representative_sentences_per_report=args.representative_sentences_per_report,
        allow_korean_document_fallback=args.allow_korean_document_fallback,
        offline_cache_only=args.offline_cache_only,
        sample_corp_codes=args.sample_corp_codes,
        selected_sample_file=args.selected_sample_file,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    commands = {"run", "doctor", "self-test"}
    if not argv_list or argv_list[0] not in commands:
        argv_list = ["run", *argv_list]
    args = parser.parse_args(argv_list)

    if args.command == "run":
        try:
            run_pipeline(config_from_args(args))
            return 0
        except Exception as exc:  # noqa: BLE001 - command-line user message.
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if args.command == "doctor":
        return doctor()
    if args.command == "self-test":
        return self_test()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
