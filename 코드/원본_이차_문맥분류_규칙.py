"""Second-generation Korean sentence classifier and audit evaluator.

This module uses contextual gates rather than standalone keyword counts:
technology meaning -> focal-firm attribution -> verifiable action -> BMI scope.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "korea_digital_strategy_output"
AUDIT_PATH = OUT_DIR / "06_korean_sentence_validation_audit.xlsx"
EVAL_PATH = OUT_DIR / "08_korean_classifier_v2_audit.xlsx"


def rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


ACCOUNTING_RE = rx(
    r"무형자산|상각(?:누계액|비|이|을)|장부금액|역사적\s*원가|내용연수|"
    r"재무제표|회계정책|손상차손|공정가치|이연법인세|주석\s*\d|"
    r"차입금명칭|전자단기사채|출자약정금액|리스채권|특수관계자.{0,80}거래|"
    r"수익인식|자본화된\s*개발비|컴퓨터소프트웨어|"
    r"\([12]\)\s*(?:제품|상품|재화)의\s*판매\s*(?:연결)?회사는|"
    r"영업이익\s*감소.{0,100}대손충당금|구\s*분.{0,120}영업이익|"
    r"산출기준.{0,100}(?:단가|가격변동)|주요\s*가격변동요인|가격변동추이|"
    r"유형자산\s*취득약정|관계기업.{0,120}ERP\s*구축\s*컨설팅"
)
ENTITY_OR_NAME_RE = rx(
    r"투자조합|벤처투자조합|사모투자|스마트신세계|스마트\s*CKD|"
    r"나이스비즈니스플랫폼|요즈마글로벌AI펀드|메디클라우드.{0,40}전환사채|"
    r"스마트사이드주식회사|스마트로더\s*전자단기사채|스마트역삼"
)
NON_DIGITAL_COLLISION_RE = rx(
    r"당사\s*맥주인\s*['\"]?클라우드|클라우드\s*제품\s*리뉴얼|"
    r"브랜드\s*플랫폼|CPU\s*플랫폼|터빈.{0,20}플랫폼|날개\s*구조물\s*플랫폼|"
    r"마우스\s*플랫폼|항체.{0,30}플랫폼|기술\s*플랫폼으로\s*폴리|"
    r"신약개발\s*플랫폼(?!.*AI)|화학적으로\s*현실화시켜.{0,30}플랫폼|"
    r"모바일\s*화면.{0,30}레진|모바일\s*스마트기기\s*시장.{0,30}소재|"
    r"AI\s*실린더\s*헤드|PBV.{0,60}플랫폼|플랫폼.{0,60}PBV|"
    r"사외이사.{0,180}4차\s*산업혁명|4차\s*산업혁명.{0,180}사외이사"
)
FINANCIAL_ENTITY_RE = rx(
    r"(?:AI|디지털|플랫폼).{0,25}(?:펀드|투자조합).{0,100}"
    r"(?:청산|지분증권|금융자산|처분손실|특수관계자\s*거래)"
)
NEGATION_RE = rx(r"미영위|추진을\s*중단|영위하지\s*않|사업을\s*영위하지|미추진")
BUSINESS_PURPOSE_RE = rx(
    r"사업\s*목적|목적\s*사업|정관에\s*기재된|목\s*적\s*사\s*업|"
    r"사업\s*분야\(업종|변경\s*후\s*사업목적|각호에\s*관련된|"
    r"진출\s*목적|사업분야\s*및\s*진출목적"
)
BUSINESS_LIST_RE = rx(
    r"제조\s*및\s*판매업|개발\s*및\s*판매업|소프트웨어\s*개발업|"
    r"전자상거래업|통신판매업|온라인\s*판매업|정보서비스업|"
    r"판매조직.{0,160}스마트팜사업부문|사업부문\s*주요\s*매출품목|"
    r"(?:개발|판매)사업/?제품\s*판매사업|판매업\s*\d+\.?$|"
    r"물류업\s*엔터테인먼트.{0,120}미디어플랫폼\s*사업|"
    r"전자상거래를\s*통한\s*상품\s*용역의\s*판매|"
    r"사업부문\s*요약\s*재무현황|사업부.{0,80}(?:소프트웨어|시스템)|"
    r"시스템소프트(?:웨어)?개발.{0,100}(?:사업부|수출|제품)"
)
SCOPE_ENUM_RE = rx(
    r"영위하고\s*있는\s*사업|사업부문\s*주요\s*매출품목|"
    r"시스템\s*소프트웨어\s*개발\s*및\s*공급업|"
    r"소프트웨어\s*(?:연구)?개발\s*및\s*(?:판매|공급)업?|"
    r"응용소프트웨어\s*개발\s*및\s*공급업|"
    r"소프트웨어\s*개발\s*및\s*판매\s*\(주\)|"
    r"소프트웨어.{0,50}연결조정.{0,30}영업이익"
)
CHRONOLOGY_RE = rx(
    r"(?:회사\s*)?연혁|(?:19|20)\d{2}\.\d{1,2}.{0,180}(?:19|20)\d{2}\.\d{1,2}|"
    r"(?:19|20)\d{2}년\s*\d{1,2}월.{0,180}(?:19|20)\d{2}년\s*\d{1,2}월|"
    r"경영상의\s*주요계약.{0,240}(?:도입계약|전산용역)"
)
BOARD_LIST_RE = rx(r"보고사항|의안내용|가결\s*찬성|이사회.{0,40}(?:결의|승인)")
GENERIC_CONTEXT_RE = rx(
    r"산업의\s*(?:현황|정의|개요|특성|성장성)|시장의\s*향후\s*전망|성장할\s*것으로\s*예상|"
    r"수요가?\s*(?:증가|확대)|시장(?:의\s*특성|\s*특성)|"
    r"시장(?:은|이|의)?.{0,50}(?:성장|확대|전망|전환|증가)|"
    r"산업(?:은|이|의)?.{0,50}(?:성장|확대|전망|변화|전환)|"
    r"정부(?:는|가|의|에서|\s*또한)|국가적\s*차원|우리\s*기업들도|"
    r"모든\s*기업|대기업들도|업체들이|글로벌\s*(?:기업|브랜드)들은|"
    r"업계는|최근\s*원양어업은|네덜란드의\s*시스템|"
    r"제작사들은|플랫폼\s*사는|유통\s*환경의\s*변화|활성화되고\s*있|"
    r"증가하고\s*있는\s*상황|경쟁이\s*심화되고|디지털\s*TV는|"
    r"참고자료\s*:|지속적인\s*관심이\s*필요|주목을\s*받고\s*있는|"
    r"발생하고\s*있는\s*추세|공급과\s*수요.{0,40}성장|"
    r"판매경로\s*및\s*판매방법|판매조직.{0,60}온라인담당|"
    r"급속히\s*확대되는\s*추세|산업.{0,70}방향이\s*새롭게\s*떠오르|"
    r"향후.{0,60}추이.{0,30}전망|경쟁\s*환경.{0,80}중요성|"
    r"중요한\s*성공요인|진화에\s*따라.{0,60}변화|"
    r"성장에\s*따라.{0,100}(?:급증|적용.{0,20}확대)|"
    r"브랜드들.{0,120}큰\s*관심"
)
FOCAL_FIRM_RE = rx(r"당사|우리\s*회사|회사(?:는|가|의)|연결회사|연결실체|그룹(?:은|내)")
EXPLICIT_FOCAL_RE = rx(
    r"당사|우리\s*회사|연결회사|연결실체|그룹(?:은|내)|"
    r"회사(?:는|가)\s*(?:새로운|디지털|스마트|온라인|데이터)"
)
EXTERNAL_COMPANY_RE = rx(
    r"마이크로소프트|구글|아마존|알리바바|테슬라|인텔|"
    r"(?<![A-Za-z])Intel(?![A-Za-z])|(?<![A-Za-z])AMD(?![A-Za-z])|"
    r"메리바라|\bGM\b|General\s*Motors|해외\s*경쟁사|경쟁업체들은"
)

FOURTH_IR_RE = rx(r"4차\s*산업혁명|제4차\s*산업혁명|디지털\s*시대")
STRONG_DIGITAL_RE = rx(
    r"디지털(?:화|\s*(?:전환|트랜스포메이션|혁신|전략|플랫폼|서비스|솔루션|Amp))|"
    r"인공지능|생성형?\s*AI|(?<![A-Za-z])AI(?![A-Za-z])|머신러닝|딥러닝|"
    r"빅데이터|데이터\s*(?:분석|플랫폼|기반)|클라우드\s*(?:서비스|컴퓨팅|플랫폼)|"
    r"사물인터넷|(?<![A-Za-z])IoT(?![A-Za-z])|블록체인|디지털\s*트윈|"
    r"스마트\s*(?:팩토리|공장|제조|생태공장|상수도)|지능형.{0,25}시스템|"
    r"(?<![A-Za-z])RPA(?![A-Za-z])|로봇(?:\s*프로세스\s*자동화|\s*기술|\s*시스템|\s*개발)|"
    r"사이버\s*보안|정보보안|(?<![A-Za-z])(?:ERP|MES|WMS|TMS|CRM|BIM)(?![A-Za-z])|"
    r"자율주행|알고리즘|데이터베이스|5G|Bluetooth|UWB|3D\s*스캔|드론|"
    r"(?<![A-Za-z])TES(?![A-Za-z])|"
    r"구독경제.{0,30}결제|휴대폰\s*결제|커넥티드카"
)
ONLINE_DIGITAL_RE = rx(
    r"온라인\s*(?:채널|판매|쇼핑몰|몰|사업|플랫폼|마케팅|부문)|"
    r"전자상거래|이커머스|e-?커머스|모바일\s*(?:앱|플랫폼|서비스|사이트|경로)|"
    r"디지털\s*채널|옴니\s*채널|비대면\s*(?:서비스|채널)|(?<![A-Za-z])OTT(?![A-Za-z])|"
    r"온라인.{0,45}(?:판매|매출|수익|채널|사업|몰|마케팅|진출|입점|확대)|"
    r"모바일.{0,35}(?:서비스|플랫폼|판매|결제|개발|운영)|"
    r"온라인으로\s*연계|온라인을\s*근간|인터넷\s*온라인\s*광고"
)
AUTOMATION_RE = rx(
    r"설비\s*자동화|설비자동화|자동화\s*(?:설비|시설|라인|시스템|공정|기술|투자)|"
    r"자동화를\s*(?:지속|추진|적용)|자동화(?:의|에)?\s*(?:지속적인\s*)?투자|"
    r"자동화\s*(?:신기술|솔루션)|자동화(?:하|했|하여|되|된)|"
    r"설비를?\s*자동화|로봇을?\s*(?:활용|적용)|공정\s*자동화|"
    r"설계\s*자동화|무인\s*자동화|로봇\s*시스템|서비스\s*로봇|"
    r"스마트\s*(?:팜|홈|시티|가구)|원격\s*(?:측정|검침|모니터링)"
)
PLATFORM_RE = rx(r"플랫폼")
SOFTWARE_RE = rx(r"소프트웨어|IT\s*(?:서비스|시스템|솔루션)|정보시스템")
DATA_OPERATION_RE = rx(
    r"데이터.{0,35}(?:수집|적재|집적|축적|분석|활용|관리|통합|연결|검증)|"
    r"(?:수집|적재|집적|축적|분석|활용|관리|통합).{0,35}데이터"
)

ACTION_RE = rx(
    r"구축|도입|개발|투자|적용|활용|운영|추진|확대|고도화|개선|최적화|"
    r"자동화|통합|출시|제공|강화|협력|제휴|상용화|인수|취득|설립|체결|"
    r"전환|개편|확보|발굴|공급|판매|론칭|오픈|신설|충원|참여|"
    r"수집|적재|집적|축적|측정|관리|조회|모니터링|자리매김|매출이\s*발생|"
    r"성장하였습니다|증가하고\s*있|집중|도모|시행|진행|기획|제작|보급|"
    r"소개|중개|발생|자리매김|서비스를\s*제공"
)
IMPLEMENTED_RE = rx(
    r"구축(?:하|했|하여|되어|중)|도입(?:하|했|하여|되어|중)|"
    r"개발(?:하|했|하여|완료|중)|운영(?:하|했|하여|중)|출시|론칭|오픈|"
    r"인수|취득|설립|투자(?:하|했|하여|중)|체결|적용(?:하|했|하여|중)|"
    r"활용(?:하|했|하여|중)|제공(?:하|했|하여|중)|판매(?:하|했|하여|중)|"
    r"확대(?:하|했|하여|중)|강화(?:하|했|하여|중)|고도화(?:하|했|하여|중)|"
    r"진출|전환(?:하|했|하여|중)|시행|수취하였습니다|운용"
)
PLAN_RE = rx(r"계획|예정|목표|추구|하고자|하겠습니다|추진\s*중|준비\s*중|검토")
GENERIC_HYPE_RE = rx(
    r"4차\s*산업혁명|제4차\s*산업혁명|디지털\s*시대|선제적으로\s*대응|"
    r"적극적으로\s*대응|발맞춰|총력을\s*다하|노력하겠습니다|메가트렌드"
)
VAGUE_FIRM_CLAIM_RE = rx(
    r"디지털\s*플랫폼\s*서비스\s*차별화로.{0,80}시너지|"
    r"수준\s*높은\s*솔루션을\s*제공하고.{0,100}수익을\s*확보|"
    r"미래먹거리를\s*선점.{0,80}총력을\s*다하|"
    r"유망분야.{0,80}신규사업을\s*적극\s*발굴|"
    r"유통구조\s*변화에\s*대처하여.{0,80}성장세|"
    r"다양한\s*플랫폼과\s*협업.{0,120}기업으로\s*거듭|"
    r"온라인\s*수익성\s*강화.{0,100}외형\s*성장"
)

BMI_RE = rx(
    r"비즈니스\s*모델|사업\s*모델|수익\s*모델|구독|정기구독|반복\s*매출|"
    r"고객\s*(?:경험|편의|가치)|온라인\s*(?:채널|판매|쇼핑몰|몰|사업)|"
    r"전자상거래|이커머스|e-?커머스|모바일\s*(?:앱|플랫폼|서비스|사이트)|"
    r"디지털\s*(?:플랫폼|서비스|솔루션|채널)|플랫폼\s*(?:사업|서비스)|"
    r"솔루션\s*(?:사업|서비스|제공|공급|개발)|신규\s*서비스|"
    r"모빌리티\s*서비스|커넥티드|핀테크|전자결제|간편결제|D2C|O2O|"
    r"스마트\s*(?:제품|홈|팜)|온라인과\s*오프라인|온/?오프라인|옴니|"
    r"CRM|맞춤형\s*마케팅|데이터.{0,30}고객|고객.{0,30}데이터|"
    r"온라인\s*부문|온라인\s*비즈니스|온라인몰|웹사이트|모바일\s*사이트|"
    r"IT\s*서비스|ERP\s*솔루션|소프트웨어\s*(?:개발|서비스)|"
    r"스마트.{0,30}솔루션|비즈니스\s*플랫폼|덴탈\s*비타민"
)
DIRECT_BMI_RE = rx(
    r"비즈니스\s*모델|사업\s*모델|수익\s*모델|구독|정기구독|"
    r"온라인.{0,45}(?:채널|판매|쇼핑몰|몰|사업|비즈니스|마케팅|입점|진출|매출|수익)|"
    r"전자상거래|이커머스|e-?커머스|모바일.{0,35}(?:서비스|플랫폼|사이트|결제)|"
    r"디지털\s*(?:플랫폼|서비스|솔루션|채널)|CRM|맞춤형\s*마케팅|"
    r"(?:고객.{0,35}데이터|데이터.{0,35}고객)|핀테크|전자결제|간편결제|휴대폰\s*결제|"
    r"IT\s*서비스|ERP\s*솔루션|소프트웨어\s*(?:서비스|판매)|"
    r"플랫폼.{0,45}(?:사업|서비스|고객|매출|수익|판매|거래|취득|인수)|"
    r"(?:사업|서비스|고객|매출|수익|판매|거래|취득|인수).{0,45}플랫폼|"
    r"스마트.{0,35}솔루션|자동화\s*솔루션|온라인과\s*오프라인|온/?오프라인|"
    r"D2C|O2O|덴탈\s*비타민"
)
SMART_PRODUCT_RE = rx(
    r"IoT.{0,45}(?:제품|서비스|기기)|(?:제품|서비스|기기).{0,45}IoT|"
    r"자율주행.{0,35}(?:개발|솔루션|제품)|로봇.{0,35}(?:개발|솔루션|서비스)|"
    r"AI.{0,35}(?:제품|서비스|비즈니스\s*모델)|"
    r"(?:제품|서비스|비즈니스\s*모델).{0,35}AI"
)
CUSTOMER_OR_REVENUE_RE = rx(
    r"고객|판매|매출|수익|채널|서비스|사업|거래|결제|마케팅|유통|시장\s*진출"
)
CURRENT_LANGUAGE_RE = rx(
    r"현재|지속|운영\s*중|진행\s*중|추진\s*중|확대\s*중|\d{4}\s*~|"
    r"이후|그동안|강화한\s*결과|진출한\s*결과"
)
PLATFORM_QUALIFIER_RE = rx(
    r"디지털|온라인|모바일|데이터|AI|인공지능|블록체인|소프트웨어|전자상거래|"
    r"이커머스|결제|핀테크|물류|헬스케어|미래농업|OTT|NDC|고객|비즈니스|"
    r"수익|거래|대출비교|콘텐츠|덴탈|유통|판매|채널|사업|기록관리|동영상"
)
ACCOUNTING_EXCEPTION_RE = rx(
    r"당기\s*중.{0,180}(?:자동화\s*설비|디지털\s*플랫폼|WMS).{0,120}(?:백만원|대체)|"
    r"ERP\s*구축\s*컨설팅.{0,100}(?:개발약정|계약)"
)
PERSONNEL_BIO_RE = rx(
    r"미등기\s*임원|미등기\s*상근|주요\s*경력.{0,120}(?:이커머스|AI)|글로벌이커머스\s*상무"
)
SOFTWARE_SPECIFIC_RE = rx(r"서비스|솔루션|시스템|개발|운영|공급|판매|구축|사업")
ACTIVE_OPERATION_RE = rx(
    r"영위하고\s*있|운영하고\s*있|판매를\s*시작|매출이\s*발생|매출을\s*달성|"
    r"서비스를\s*제공|사업부문은|온라인을\s*근간|판매하고\s*있|자리매김|"
    r"운영중|핵심\s*사업부문으로\s*성장|온라인몰.{0,40}매출"
)
NAMED_COMPANY_RE = rx(r"\(주\)|㈜|주식회사")
ATTRIBUTION_RE = rx(r"연구개발\s*실적|개발기술|정부보조금을\s*수취|특허명|연구과제")
OWNED_STRATEGY_RE = rx(r"전략을\s*추진|중점\s*추진")
FOCAL_GENERIC_ACTION_RE = rx(r"강화에\s*집중|개선을\s*도모|개발하여|구축하여")
BOARD_ACTION_RE = rx(r"계약을\s*체결|투자|인수|취득")
CONCRETE_INTENT_RE = rx(r"진출\s*목적|하고자|위하여|판매\s*채널\s*확대")
BUSINESS_IMPLEMENTATION_RE = rx(
    r"영위\s*중|운영\s*중|매출이?\s*(?:발생|달성)|매출을\s*달성|"
    r"판매를\s*시작|론칭|보급하고\s*있|합작법인을\s*설립|투자\s*하는\s*약정|"
    r"당사.{0,100}플랫폼을?\s*통(?:해|하여).{0,140}(?:중개|제공|판매)"
)
INTENT_ADD_RE = rx(r"(?:개발|운영|판매)\s*추가\s*$")
PLATFORM_VALUE_RE = rx(r"서비스|사업|고객|판매|매출|수익|거래|제공|공급")
STATIC_EQUIPMENT_RE = rx(r"자동화\s*설비를?\s*갖추(?:고|어|었)")
EQUIPMENT_LIST_RE = rx(
    r"(?:공장|물류).{0,20}자동화\s*설비\s*등|"
    r"자동화\s*설비\s*등\s*(?:\(주\)|㈜|$)"
)
AUTOMATION_CHANGE_RE = rx(r"자동화.{0,60}(?:투자|확대|구축|도입|증설|신설|전환|개선)")


def stale_historical(sentence: str, report_year: int | None) -> bool:
    if not report_year:
        return False
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", sentence)]
    if not years:
        return False
    current_language = CURRENT_LANGUAGE_RE.search(sentence)
    return max(years) <= report_year - 2 and not current_language


def stale_chronology(sentence: str, report_year: int | None) -> bool:
    if not CHRONOLOGY_RE.search(sentence):
        return False
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", sentence)]
    if not years:
        return True
    if not report_year:
        return False
    return max(years) < report_year


def platform_is_digital(sentence: str) -> bool:
    if not PLATFORM_RE.search(sentence) or NON_DIGITAL_COLLISION_RE.search(sentence):
        return False
    return bool(PLATFORM_QUALIFIER_RE.search(sentence))


def normalized_company_tokens(corp_name: str | None) -> list[str]:
    if not corp_name:
        return []
    cleaned = re.sub(r"[()（）㈜주식회사\s]", "", str(corp_name))
    tokens = [cleaned]
    if cleaned.endswith("홀딩스"):
        tokens.append(cleaned.removesuffix("홀딩스"))
    return [token for token in tokens if len(token) >= 2]


def classify_sentence_v2(
    sentence: str,
    report_year: int | None = None,
    corp_name: str | None = None,
) -> Dict[str, Any]:
    sentence = " ".join(str(sentence).split())
    accounting = bool(ACCOUNTING_RE.search(sentence))
    entity_name = bool(ENTITY_OR_NAME_RE.search(sentence))
    collision = bool(NON_DIGITAL_COLLISION_RE.search(sentence))
    financial_entity = bool(FINANCIAL_ENTITY_RE.search(sentence))
    negated = bool(NEGATION_RE.search(sentence))
    business_purpose = bool(BUSINESS_PURPOSE_RE.search(sentence))
    board_list = bool(BOARD_LIST_RE.search(sentence))
    generic_context = bool(GENERIC_CONTEXT_RE.search(sentence))
    company_tokens = normalized_company_tokens(corp_name)
    focal_name_present = any(token in re.sub(r"\s+", "", sentence) for token in company_tokens)
    focal_firm = bool(FOCAL_FIRM_RE.search(sentence) or focal_name_present)
    explicit_focal_language = bool(EXPLICIT_FOCAL_RE.search(sentence))
    explicit_focal = bool(explicit_focal_language or focal_name_present)
    external_company_context = bool(EXTERNAL_COMPANY_RE.search(sentence) and not explicit_focal)

    strong_digital = bool(STRONG_DIGITAL_RE.search(sentence))
    online_digital = bool(ONLINE_DIGITAL_RE.search(sentence))
    automation = bool(AUTOMATION_RE.search(sentence))
    data_operation = bool(DATA_OPERATION_RE.search(sentence))
    platform_digital = platform_is_digital(sentence)
    software = bool(SOFTWARE_RE.search(sentence))
    fourth_ir = bool(FOURTH_IR_RE.search(sentence))

    accounting_exception = bool(ACCOUNTING_EXCEPTION_RE.search(sentence))
    name_only = entity_name and not (IMPLEMENTED_RE.search(sentence) and not accounting)
    personnel_bio = bool(PERSONNEL_BIO_RE.search(sentence))
    hard_non_digital = (
        (accounting and not accounting_exception)
        or collision
        or financial_entity
        or negated
        or name_only
        or personnel_bio
    )
    technology_specific = strong_digital or online_digital or automation or data_operation or platform_digital
    software_specific = software and bool(SOFTWARE_SPECIFIC_RE.search(sentence))
    digital_relevance = technology_specific or software_specific or fourth_ir
    strategy_relevance = technology_specific or software_specific
    if hard_non_digital:
        digital_relevance = False
        strategy_relevance = False

    action = bool(ACTION_RE.search(sentence))
    implemented = bool(IMPLEMENTED_RE.search(sentence))
    stale = stale_historical(sentence, report_year) or stale_chronology(sentence, report_year)
    static_equipment_description = bool(
        STATIC_EQUIPMENT_RE.search(sentence) and not AUTOMATION_CHANGE_RE.search(sentence)
    )
    active_operation = bool(ACTIVE_OPERATION_RE.search(sentence))
    action = bool(action or implemented or active_operation)
    implemented = bool(implemented or active_operation)
    list_like_scope = bool(
        (BUSINESS_LIST_RE.search(sentence) and not active_operation)
        or SCOPE_ENUM_RE.search(sentence)
    )

    named_company = bool(NAMED_COMPANY_RE.search(sentence))
    attributable = (
        focal_firm
        or named_company
        or implemented
        or bool(ATTRIBUTION_RE.search(sentence))
        or bool(OWNED_STRATEGY_RE.search(sentence))
        or (action and not generic_context)
    )
    if generic_context and not (
        explicit_focal_language
        or OWNED_STRATEGY_RE.search(sentence)
        or (
            focal_name_present
            and (
                implemented
                or FOCAL_GENERIC_ACTION_RE.search(sentence)
            )
        )
    ):
        attributable = False

    # Fourth-industrial-revolution rhetoric is disclosure, not evidence of a
    # digital strategy unless a concrete technology or software is also named.
    substantive = bool(strategy_relevance and action and attributable)
    if stale or list_like_scope:
        substantive = False
    if static_equipment_description or EQUIPMENT_LIST_RE.search(sentence):
        substantive = False
    if external_company_context:
        substantive = False
    if business_purpose and not BUSINESS_IMPLEMENTATION_RE.search(sentence):
        substantive = False
    if INTENT_ADD_RE.search(sentence):
        substantive = False
    if board_list and not BOARD_ACTION_RE.search(sentence):
        substantive = False
    if VAGUE_FIRM_CLAIM_RE.search(sentence):
        substantive = False

    # A specific implementation plan counts as strategy; a charter-only intention does not.
    specific_plan = bool(
        strategy_relevance
        and PLAN_RE.search(sentence)
        and focal_firm
        and IMPLEMENTED_RE.search(sentence)
        and not business_purpose
        and not generic_context
        and not hard_non_digital
        and not VAGUE_FIRM_CLAIM_RE.search(sentence)
    )
    if specific_plan and not stale:
        substantive = True

    bmi = bool(
        substantive
        and (
            DIRECT_BMI_RE.search(sentence)
            or
            (BMI_RE.search(sentence) and CUSTOMER_OR_REVENUE_RE.search(sentence))
            or SMART_PRODUCT_RE.search(sentence)
            or (
                platform_digital
                and PLATFORM_VALUE_RE.search(sentence)
            )
        )
    )

    firm_signal = bool(
        digital_relevance
        and focal_firm
        and not business_purpose
        and (PLAN_RE.search(sentence) or GENERIC_HYPE_RE.search(sentence))
        and not accounting
        and not negated
    )
    implemented_business_signal = bool(
        digital_relevance
        and business_purpose
        and BUSINESS_IMPLEMENTATION_RE.search(sentence)
        and not negated
    )
    concrete_intent_signal = bool(
        digital_relevance
        and business_purpose
        and focal_firm
        and CONCRETE_INTENT_RE.search(sentence)
        and not negated
    )
    market_signal = bool(
        substantive or firm_signal or implemented_business_signal or concrete_intent_signal
    )

    generic_hype = bool(digital_relevance and GENERIC_HYPE_RE.search(sentence) and not substantive)
    firm_hype = bool(firm_signal and not substantive and not business_purpose)
    boilerplate = bool(generic_hype or firm_hype)

    digital_sentence = bool(digital_relevance)
    if business_purpose and (strong_digital or online_digital or platform_digital):
        digital_sentence = True
    if fourth_ir:
        digital_sentence = True

    return {
        "digital_sentence_v2": int(digital_sentence),
        "substantive_digital_sentence_v2": int(substantive),
        "bmi_sentence_v2": int(bmi),
        "market_signal_sentence_v2": int(market_signal),
        "boilerplate_digital_sentence_v2": int(boilerplate),
        "v2_accounting_noise": int(accounting),
        "v2_name_collision": int(name_only or collision),
        "v2_stale_history": int(stale),
        "v2_business_purpose": int(business_purpose),
        "v2_generic_context": int(generic_context),
        "v2_specific_plan": int(specific_plan),
        "v2_external_company_context": int(external_company_context),
    }


def metric(df: pd.DataFrame, audit: str, prediction: str) -> Dict[str, Any]:
    actual = pd.to_numeric(df[audit], errors="coerce").fillna(0).astype(int)
    predicted = pd.to_numeric(df[prediction], errors="coerce").fillna(0).astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "construct": audit,
        "n": len(df),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(df),
    }


def evaluate() -> pd.DataFrame:
    df = pd.read_excel(AUDIT_PATH, sheet_name="audit_sample")
    predictions = [
        classify_sentence_v2(row["sentence"], int(row["year"]), row.get("corp_name"))
        for _, row in df.iterrows()
    ]
    predicted = pd.DataFrame(predictions)
    combined = pd.concat([df.reset_index(drop=True), predicted], axis=1)
    pairs = [
        ("audit_digital_sentence", "digital_sentence_v2"),
        ("audit_substantive_digital_sentence", "substantive_digital_sentence_v2"),
        ("audit_bmi_sentence", "bmi_sentence_v2"),
        ("audit_market_signal_sentence", "market_signal_sentence_v2"),
        ("audit_boilerplate_or_hype", "boilerplate_digital_sentence_v2"),
    ]
    metrics = pd.DataFrame([metric(combined, audit, prediction) for audit, prediction in pairs])
    errors = combined[
        combined["audit_substantive_digital_sentence"]
        != combined["substantive_digital_sentence_v2"]
    ].copy()
    with pd.ExcelWriter(EVAL_PATH, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="v2_metrics", index=False)
        errors.to_excel(writer, sheet_name="substantive_errors", index=False)
        combined.to_excel(writer, sheet_name="all_predictions", index=False)
    print(metrics.to_string(index=False))
    print(f"Wrote: {EVAL_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate()
