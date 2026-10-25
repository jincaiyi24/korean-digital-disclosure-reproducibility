"""
Korean annual-report digital strategy scoring pipeline.

This script reuses the already-downloaded DART report text cache and builds
Korean sentence-level indicators for digital strategy, business-model
reconstruction, and market-visible digital signals.

Default inputs
--------------
korea_digital_strategy_output/04_digital_strategy_panel.xlsx

Default outputs
---------------
korea_digital_strategy_output/04_digital_strategy_panel_korean.xlsx
korea_digital_strategy_output/05_representative_digital_sentences_korean.xlsx
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - command-line diagnostic.
    raise SystemExit("pandas and openpyxl are required. Run: python -m pip install pandas openpyxl tqdm") from exc

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - simple fallback is below.
    _tqdm = None


DIGITAL_CORE_TERMS = [
    "디지털 전환",
    "디지털전환",
    "디지털 전략",
    "디지털 혁신",
    "디지털화",
    "디지타이제이션",
    "디지털 트랜스포메이션",
    "인공지능",
    "머신러닝",
    "딥러닝",
    "생성형 ai",
    "생성 ai",
    "ai",
    "빅데이터",
    "데이터 분석",
    "데이터 플랫폼",
    "데이터 기반",
    "클라우드",
    "사물인터넷",
    "iot",
    "블록체인",
    "사이버보안",
    "사이버 보안",
    "정보보안",
    "rpa",
    "로봇 프로세스 자동화",
    "스마트팩토리",
    "스마트 팩토리",
    "스마트공장",
    "스마트 공장",
    "지능형 공장",
    "스마트 제조",
    "디지털 트윈",
    "5g",
    "디지털 플랫폼",
    "디지털 서비스",
    "디지털 솔루션",
    "디지털 시대",
    "4차 산업혁명",
    "온라인 채널",
    "디지털 채널",
    "전자상거래",
    "이커머스",
    "e-commerce",
    "ecommerce",
    "모바일 앱",
    "모바일 플랫폼",
    "앱 서비스",
    "커넥티드카",
    "커넥티드 카",
    "커넥티드 서비스",
    "스마트 모빌리티",
    "모빌리티 서비스",
    "스마트홈",
    "스마트 홈",
    "erp",
    "mes",
    "scm",
    "crm",
]

CONTEXTUAL_DIGITAL_TERMS = [
    "데이터",
    "플랫폼",
    "소프트웨어",
    "모바일",
    "온라인",
    "자동화",
    "스마트",
    "클라우드",
    "앱",
    "어플리케이션",
    "애플리케이션",
    "알고리즘",
    "로봇",
    "자율주행",
    "커넥티드",
    "구독",
]

ACTION_TERMS = [
    "구축",
    "도입",
    "개발",
    "투자",
    "적용",
    "활용",
    "운영",
    "추진",
    "확대",
    "고도화",
    "개선",
    "최적화",
    "자동화",
    "디지털화",
    "전환",
    "전산화",
    "통합",
    "출시",
    "제공",
    "강화",
    "협력",
    "제휴",
    "상용화",
    "운용",
    "도입하",
    "구축하",
    "개발하",
    "투자하",
    "적용하",
    "활용하",
    "운영하",
    "추진하",
    "확대하",
    "고도화하",
    "개선하",
    "최적화하",
    "제공하",
    "출시하",
]

BMI_TERMS = [
    "신규 서비스",
    "새로운 서비스",
    "디지털 플랫폼",
    "플랫폼",
    "소프트웨어 서비스",
    "데이터 서비스",
    "구독",
    "정기구독",
    "반복매출",
    "반복 매출",
    "온라인 채널",
    "디지털 채널",
    "판매채널",
    "판매 채널",
    "고객경험",
    "고객 경험",
    "생태계",
    "에코시스템",
    "스마트 제품",
    "모빌리티 서비스",
    "커넥티드 서비스",
    "전자상거래",
    "이커머스",
    "모바일 서비스",
    "앱 서비스",
    "솔루션",
    "비즈니스 모델",
    "사업모델",
    "사업 모델",
    "수익모델",
    "수익 모델",
    "서비스형",
]

OPERATIONAL_TERMS = [
    "생산",
    "제조",
    "공장",
    "설비",
    "품질",
    "물류",
    "공급망",
    "공급 체인",
    "구매",
    "조달",
    "재고",
    "판매",
    "영업",
    "고객",
    "제품",
    "서비스",
    "차량",
    "배터리",
    "반도체",
    "유통",
    "프로세스",
    "업무",
    "원가",
    "채널",
    "매출",
    "마케팅",
    "연구개발",
    "r&d",
]

BOILERPLATE_PATTERNS = [
    r"디지털\s*시대",
    r"4차\s*산업혁명",
    r"미래\s*성장\s*동력",
    r"미래\s*성장동력",
    r"패러다임\s*변화",
    r"급변하는\s*환경",
    r"변화하는\s*환경",
    r"지속적으로\s*노력",
    r"최선을\s*다하",
    r"중요성이\s*증대",
    r"선제적으로\s*대응",
    r"적극적으로\s*대응",
]

TABLE_NOISE_PATTERNS = [
    r"^\(?단위\s*:",
    r"^\(?주\s*:",
    r"^구\s*분\s",
    r"^구분\s",
    r"^목\s*차$",
    r"^해당사항\s*없음",
    r"^감사의견",
    r"^연결재무제표",
    r"^재무제표",
    r"^주석",
    r"^자본금\s*변동사항",
    r"^주식의\s*총수",
    r"^임원\s*및\s*직원",
    r"^이사회",
    r"^감사위원회",
    r"^타법인\s*출자",
    r"^특수관계자",
    r"^파생상품",
    r"^공정가치",
    r"^범주별\s*금융상품",
    r"^당기\s+전기",
    r"국내외\s*주요기업",
    r"서울특별시|경기도|부산광역시|대구광역시|인천광역시|대한민국",
    r"지급보증|우발부채|보증보험|공탁보증|이행\(지급\)|이행\(계약\)|이행\(하자\)",
    r"재원\s*조달방법",
    r"소요자금\s*미정",
    r"사업자등록번호|법인등록번호",
    r"대표자명|소재지|설립일",
]

NON_STRATEGIC_CONTEXT_PATTERNS = [
    r"온라인\s*채널\s*확대\s*영향",
    r"확대\s*영향으로",
    r"수요\s*(?:증가|감소|확대)",
    r"시장(?:은|이|의)?.{0,30}(?:성장|확대|증가|감소|전망)",
    r"산업(?:은|이|의)?.{0,30}(?:성장|확대|증가|감소|전망|추세)",
    r"증권사들은",
    r"정부(?:는|에서| 또한)",
]

HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")
KOREAN_TOKEN_RE = re.compile(r"[\uac00-\ud7a3A-Za-z0-9&+-]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")
KOREAN_END_RE = re.compile(r"(다|니다|한다|된다|하였다|했다|있다|없다|이다|임|함|됨|음)[.!?]?$")


def compile_terms(terms: Sequence[str]) -> Dict[str, re.Pattern[str]]:
    patterns: Dict[str, re.Pattern[str]] = {}
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s*")
        if re.fullmatch(r"[A-Za-z0-9&.+-]+", term):
            pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        else:
            pattern = escaped
        patterns[term] = re.compile(pattern, flags=re.IGNORECASE)
    return patterns


DIGITAL_CORE_PATTERNS = compile_terms(DIGITAL_CORE_TERMS)
CONTEXTUAL_DIGITAL_PATTERNS = compile_terms(CONTEXTUAL_DIGITAL_TERMS)
ACTION_PATTERNS = compile_terms(ACTION_TERMS)
BMI_PATTERNS = compile_terms(BMI_TERMS)
OPERATIONAL_PATTERNS = compile_terms(OPERATIONAL_TERMS)
BOILERPLATE_REGEX = [re.compile(pattern, flags=re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS]
TABLE_NOISE_REGEX = [re.compile(pattern, flags=re.IGNORECASE) for pattern in TABLE_NOISE_PATTERNS]
NON_STRATEGIC_CONTEXT_REGEX = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in NON_STRATEGIC_CONTEXT_PATTERNS
]


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
    "digital_core_terms_ko",
    "contextual_digital_terms_ko",
    "action_terms_ko",
    "bmi_terms_ko",
    "operational_terms_ko",
]


@dataclass
class KoreanPipelineConfig:
    source_panel: Path
    output_dir: Path
    max_examples_per_report: int = 20


def progress(iterable: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc)
    return simple_progress(iterable, total=total, desc=desc)


def simple_progress(iterable: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterator[Any]:
    for idx, item in enumerate(iterable, start=1):
        if idx == 1 and desc:
            print(desc)
        if idx % 100 == 0:
            suffix = f"/{total}" if total else ""
            print(f"  {desc}: {idx}{suffix}")
        yield item


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("&cr;", "\n").replace("&#13;", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean_segment(segment: str) -> str:
    segment = normalize_text(segment)
    segment = re.sub(r"[◆◇●■□▶▷※]+", " ", segment)
    segment = re.sub(r"\s+", " ", segment)
    return segment.strip(" \t\r\n-;:·ㆍ")


def text_profile(text: str) -> Dict[str, Any]:
    hangul_chars = len(HANGUL_RE.findall(text or ""))
    latin_chars = len(LATIN_RE.findall(text or ""))
    letter_chars = hangul_chars + latin_chars
    hangul_share = hangul_chars / letter_chars if letter_chars else 0.0
    if letter_chars == 0:
        flag = "no_letters"
    elif hangul_share >= 0.75:
        flag = "mostly_korean"
    elif hangul_share >= 0.30:
        flag = "mixed_korean_latin"
    else:
        flag = "mostly_latin"
    return {
        "raw_text_chars": len(text or ""),
        "raw_hangul_chars": hangul_chars,
        "raw_latin_chars": latin_chars,
        "raw_hangul_share_of_letters": hangul_share,
        "text_language_flag": flag,
    }


def match_terms(text: str, patterns: Dict[str, re.Pattern[str]]) -> List[str]:
    return [term for term, pattern in patterns.items() if pattern.search(text)]


def is_table_noise(segment: str) -> bool:
    if any(pattern.search(segment) for pattern in TABLE_NOISE_REGEX):
        return True
    if "-----------" in segment or "========" in segment:
        return True
    if segment.count("|") >= 2 or segment.count("\t") >= 2:
        return True
    digit_count = len(re.findall(r"\d", segment))
    if len(segment) and digit_count / len(segment) > 0.35:
        return True
    comma_count = segment.count(",")
    tokens = KOREAN_TOKEN_RE.findall(segment)
    if tokens and comma_count / len(tokens) > 0.35:
        return True
    if len(tokens) >= 8:
        numeric_tokens = [token for token in tokens if re.fullmatch(r"[\d,./()%+-]+", token)]
        if len(numeric_tokens) / len(tokens) > 0.35:
            return True
    return False


def is_korean_narrative_segment(segment: str) -> bool:
    if len(segment) < 25 or len(segment) > 700:
        return False
    hangul_chars = len(HANGUL_RE.findall(segment))
    if hangul_chars < 15:
        return False
    tokens = KOREAN_TOKEN_RE.findall(segment)
    if len(tokens) < 5:
        return False
    if is_table_noise(segment):
        return False
    return True


def split_korean_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[•]+", "\n", text)
    raw_lines = [clean_segment(line) for line in text.splitlines()]
    raw_lines = [line for line in raw_lines if line]

    candidates: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        combined = clean_segment(" ".join(buffer))
        buffer.clear()
        if not combined:
            return
        parts = SENTENCE_SPLIT_RE.split(combined)
        for part in parts:
            cleaned = clean_segment(part)
            if is_korean_narrative_segment(cleaned):
                candidates.append(cleaned)

    for line in raw_lines:
        if is_table_noise(line) and not match_terms(line, DIGITAL_CORE_PATTERNS):
            flush_buffer()
            continue
        buffer.append(line)
        current = clean_segment(" ".join(buffer))
        if KOREAN_END_RE.search(line) or len(current) >= 350:
            flush_buffer()
    flush_buffer()

    return candidates


def has_boilerplate_language(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in BOILERPLATE_REGEX)


def has_non_strategic_context(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in NON_STRATEGIC_CONTEXT_REGEX)


def classify_sentence(sentence: str) -> Dict[str, Any]:
    core_hits = match_terms(sentence, DIGITAL_CORE_PATTERNS)
    contextual_hits = match_terms(sentence, CONTEXTUAL_DIGITAL_PATTERNS)
    action_hits = match_terms(sentence, ACTION_PATTERNS)
    bmi_hits = match_terms(sentence, BMI_PATTERNS)
    operational_hits = match_terms(sentence, OPERATIONAL_PATTERNS)
    boilerplate_language = has_boilerplate_language(sentence)
    non_strategic_context = has_non_strategic_context(sentence)

    contextual_is_specific = bool(contextual_hits and action_hits and (bmi_hits or operational_hits))
    core_is_statement = bool(core_hits and (action_hits or operational_hits or bmi_hits or boilerplate_language))
    digital_hit = bool(core_is_statement or contextual_is_specific)
    if non_strategic_context and not boilerplate_language:
        digital_hit = False

    has_specific_evidence = bool(action_hits and (bmi_hits or operational_hits or core_hits))
    boilerplate = bool(digital_hit and boilerplate_language and not has_specific_evidence)
    if digital_hit and not has_specific_evidence and not core_hits:
        boilerplate = True

    substantive = bool(digital_hit and not boilerplate and has_specific_evidence)
    bmi = bool(substantive and bmi_hits)
    market_signal = bool(digital_hit and not boilerplate)

    return {
        "digital_sentence": int(digital_hit),
        "substantive_digital_sentence": int(substantive),
        "bmi_sentence": int(bmi),
        "market_signal_sentence": int(market_signal),
        "boilerplate_digital_sentence": int(digital_hit and boilerplate),
        "digital_core_terms_ko": "; ".join(core_hits),
        "contextual_digital_terms_ko": "; ".join(contextual_hits),
        "action_terms_ko": "; ".join(action_hits),
        "bmi_terms_ko": "; ".join(bmi_hits),
        "operational_terms_ko": "; ".join(operational_hits),
    }


def analyze_text(text: str, max_examples: int = 20) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    profile = text_profile(text)
    sentences = split_korean_sentences(text)
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
        "analyzable_korean_sentence_count": total_sentences,
        **counts,
        "DigitalStrategy": digital_strategy,
        "BMI_Reconstruction": bmi_reconstruction,
        "MarketSignalDisclosure": market_signal,
    }
    examples.sort(key=lambda row: (-row["priority"], row["sentence_id"]))
    return scores, examples[:max_examples]


def add_relative_signals(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    metric_cols = [
        "raw_text_chars",
        "raw_hangul_chars",
        "raw_latin_chars",
        "raw_hangul_share_of_letters",
        "total_sentences",
        "analyzable_korean_sentence_count",
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
    result["RelativeDigitalSignal_YearOnly"] = result["DigitalStrategy"] - result["YearMean_DigitalStrategy"]

    if "industry_code" in result.columns:
        result["industry_group"] = (
            result["industry_code"].fillna("").astype(str).str.extract(r"(\d{2})", expand=False).fillna("")
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
        result.loc[has_industry, "IndustryYearMean_DigitalStrategy"] = [
            industry_means.get((industry_group, year), float("nan"))
            for industry_group, year in zip(
                result.loc[has_industry, "industry_group"],
                result.loc[has_industry, "year"],
            )
        ]

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
        "raw_hangul_chars",
        "raw_latin_chars",
        "raw_hangul_share_of_letters",
        "text_language_flag",
        "total_sentences",
        "analyzable_korean_sentence_count",
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


def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype != "object":
            continue

        def clean_value(value: Any) -> Any:
            if isinstance(value, (list, tuple, set, dict)):
                return json.dumps(value, ensure_ascii=False)
            return value

        cleaned[col] = cleaned[col].map(clean_value)
    return cleaned


def write_excel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_dataframe_for_excel(df).to_excel(path, index=False)


def build_korean_panel(config: KoreanPipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_excel(
        config.source_panel,
        dtype={"corp_code": str, "stock_code": str, "rcept_no": str},
    )

    metric_cols_to_drop = [
        "raw_text_chars",
        "raw_hangul_chars",
        "raw_latin_chars",
        "raw_hangul_share_of_letters",
        "raw_latin_share_of_letters",
        "text_language_flag",
        "total_sentences",
        "analyzable_english_sentence_count",
        "analyzable_korean_sentence_count",
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
    ]

    rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for _, report in progress(source.iterrows(), total=len(source), desc="Scoring Korean reports"):
        report_dict = report.to_dict()
        base_row = {key: value for key, value in report_dict.items() if key not in metric_cols_to_drop}
        text_cache_path = Path(str(report_dict.get("text_cache_path") or ""))

        if text_cache_path.exists():
            text = text_cache_path.read_text(encoding="utf-8", errors="ignore")
            status = "cached"
            error = ""
        else:
            text = ""
            status = "missing_cache"
            error = "Missing local text cache."

        scores, examples = analyze_text(text, max_examples=config.max_examples_per_report)

        row = {
            **base_row,
            **scores,
            "download_status": status,
            "download_error": error,
            "text_cache_path": str(text_cache_path) if str(text_cache_path) != "." else "",
        }
        rows.append(row)

        for rank, example in enumerate(examples, start=1):
            example_rows.append(
                {
                    "corp_code": report_dict.get("corp_code"),
                    "corp_name": report_dict.get("corp_name"),
                    "corp_name_eng": report_dict.get("corp_name_eng"),
                    "stock_code": report_dict.get("stock_code"),
                    "year": report_dict.get("year"),
                    "rcept_no": report_dict.get("rcept_no"),
                    "report_name": report_dict.get("report_name"),
                    "sentence_rank": rank,
                    **example,
                }
            )

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = add_relative_signals(panel)
        panel = order_panel_columns(panel)
    examples_df = pd.DataFrame(example_rows, columns=REPRESENTATIVE_SENTENCE_COLUMNS)
    return panel, examples_df


def run_pipeline(config: KoreanPipelineConfig) -> None:
    if not config.source_panel.exists():
        raise FileNotFoundError(f"Source panel not found: {config.source_panel}")

    panel, examples = build_korean_panel(config)
    panel_path = config.output_dir / "04_digital_strategy_panel_korean.xlsx"
    examples_path = config.output_dir / "05_representative_digital_sentences_korean.xlsx"
    write_excel(panel, panel_path)
    write_excel(examples, examples_path)

    print("Done.")
    print(f"Output directory: {config.output_dir.resolve()}")
    print(f"Korean firm-year panel rows: {len(panel):,}")
    print(f"Korean representative sentence rows: {len(examples):,}")
    if not panel.empty:
        print(f"Positive DigitalStrategy rows: {(panel['DigitalStrategy'].fillna(0) > 0).sum():,}")
        print(f"Positive BMI_Reconstruction rows: {(panel['BMI_Reconstruction'].fillna(0) > 0).sum():,}")


def self_test() -> int:
    text = """
    회사는 생산 공정의 효율성을 높이기 위해 AI 기반 데이터 분석 플랫폼을 구축하고 스마트팩토리 시스템을 도입하였다.
    디지털 시대의 변화에 선제적으로 대응하고 미래 성장동력을 확보하기 위해 노력하고 있습니다.
    당사는 모바일 앱 서비스를 출시하여 온라인 채널을 확대하고 고객 경험을 개선하였다.
    """
    scores, examples = analyze_text(text, max_examples=10)
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    for row in examples:
        print(
            f"- substantive={row['substantive_digital_sentence']} "
            f"bmi={row['bmi_sentence']} boilerplate={row['boilerplate_digital_sentence']} "
            f"| {row['sentence']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    default_output = SCRIPT_DIR / "korea_digital_strategy_output"
    run = subparsers.add_parser("run", help="Build Korean sentence-level digital strategy panel.")
    run.add_argument(
        "--source-panel",
        type=Path,
        default=default_output / "04_digital_strategy_panel.xlsx",
        help="Existing panel containing text_cache_path.",
    )
    run.add_argument("--output-dir", type=Path, default=default_output)
    run.add_argument("--max-examples-per-report", type=int, default=20)

    subparsers.add_parser("self-test", help="Run Korean classifier on sample sentences.")
    return parser


def config_from_args(args: argparse.Namespace) -> KoreanPipelineConfig:
    return KoreanPipelineConfig(
        source_panel=args.source_panel,
        output_dir=args.output_dir,
        max_examples_per_report=args.max_examples_per_report,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if not argv_list or argv_list[0] not in {"run", "self-test"}:
        argv_list = ["run", *argv_list]
    args = parser.parse_args(argv_list)
    if args.command == "run":
        try:
            run_pipeline(config_from_args(args))
            return 0
        except Exception as exc:  # noqa: BLE001 - user-facing command-line message.
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    if args.command == "self-test":
        return self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
