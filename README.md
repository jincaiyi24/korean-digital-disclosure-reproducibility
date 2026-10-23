# 한국 사업보고서 디지털 공시 분류와 실증분석 재현

이 저장소는 한국 상장기업의 사업보고서를 문장 단위로 분류하고, 기업연도별 BMODD·NonBMODD 지표와 논문의 실증분석 결과를 산출하는 실행 코드다. 정해진 원문 입력과 지원 환경을 준비하면 `code/public_run.py`가 원문 추출, 분류, 집계, 분석표본 생성과 회귀분석을 순서대로 실행한다.

전체 실행 절차와 분류 기준은 한국어 설명서에서 확인할 수 있다: [Word 내려받기](docs/BMODD_Reproducibility_Manual_KO.docx), [PDF 내려받기](docs/BMODD_Reproducibility_Manual_KO.pdf).

현재 방법 버전은 `locked_source_aware_plus_metadata_viewer15_20261023_v1`이다. 분류 규칙의 버전은 `korean_contextual_v4_final_action_attribution_2026-08-15`로 유지했다. 원본 코드 내부의 `BMI`·`NonBMI`는 공개 명칭 `BMODD`·`NonBMODD`에 각각 대응한다. `TotalDigital`과 `DigitalDisclosure`는 이름만 보고 합치지 않는다.

## 규칙 기반 문장분류를 사용하는 이유

이 연구는 기업의 디지털 공시가 어떤 방향을 담고 있는지를 측정한다. 디지털 기술을 언급한 문장이라도 기술 소개, 외부 시장 동향, 해당 기업의 구체적 활동은 연구상 의미가 다르다. 특히 고객·서비스·채널·거래·수익과 연결된 공시를 구분해야 비즈니스 모델 지향 디지털 공시(BMODD)와 후속 경영효율의 관계를 분석할 수 있다.

Python 분류는 이러한 개념 구분을 현재 잠금자료의 25,041,971개 문장 단위에 같은 절차로 적용하기 위한 구현이다. 문장 내용과 함께 기업명·보고연도에 관한 조건을 적용하고, 판정 결과를 기업연도별 공시비율로 집계한다. 연구자는 각 공시변수가 어떤 원문과 규칙에서 산출되었는지 확인하면서 기업과 연도에 걸친 분석을 수행할 수 있다.

판정에서는 디지털 관련성(C), 기업 활동의 구체성과 실질성(S), 사업모델 연결(B)을 구분해 기록한다. NonBMODD는 실질적 디지털 문장 중 BMODD 규칙에 해당하지 않는 잔여범주다. 이 구분은 디지털 공시의 총량과 그 안에서 사업모델을 지향하는 공시를 나누어 살펴보기 위한 것으로, NonBMODD 전체를 내부 시스템이나 기술 인프라 활동으로 해석하지 않는다.

규칙 기반 방법에서는 독자가 어휘, 문맥 조건, 제외 규칙과 적용 순서를 직접 검토할 수 있다. 공개된 코드와 기준을 대조하여 개별 문장의 판정과 기업연도 지표의 산출 과정을 확인할 수 있다는 점이 이 설계의 중요한 이유다. Python은 이 절차를 일관되게 실행하는 도구이며, 코드북의 개념적 판단은 어휘와 정규표현식으로 근사한다. 따라서 실행의 재현성과 인간 코딩에 대한 분류 성능은 구분해 평가한다.

## 1. 검증된 실행 범위

공식 Windows AMD64용 CPython 3.13.12와 해시로 고정한 의존성 환경에서 **전체 절차를 빈 출력 폴더부터 세 번 독립 실행**했다. 매번 같은 원본 ZIP과 저장된 HTML을 다시 읽어 추출·분류하고, 기업연도 지표와 현재 논문의 모든 분석 모형을 다시 추정했다. 세 번 모두 다음 결과가 일치했다.

| 항목 | 각 실행의 결과 |
| --- | ---: |
| 기업연도 보고서 | 22,654 |
| 유효문장 D 및 실제 분류 함수 호출 | 25,041,971 |
| 디지털 후보문장 | 338,468 |
| 실질적 디지털 문장 S | 49,704 |
| BMODD | 25,070 |
| NonBMODD | 24,634 |
| 보충분석 BH 검정 집합 | 19 |
| 재실행 중 HTTP 요청 | 0 |

보고서별 네 가지 원시 문장 수, 추출문 해시와 전체 문장 예측의 canonical 해시를 먼저 봉인한 뒤 참조값과 비교했다. 각 실행의 표본, 수치 CSV와 그림 파일도 동일했다. 자세한 해시와 검증 범위는 [공개 검증 요약](evidence/PUBLIC_VALIDATION_SUMMARY.json)에 있다.

이 검증은 **현재 고정 입력에 대한 오프라인 재현**이다. 재다운로드, 임의 플랫폼에서의 비트 단위 동일성, 과거 25,080,866문장 측정치의 복원은 검증 범위에 포함하지 않았다. 기존 300문장 인간 코딩의 κ와 해당 표본을 이용한 분류기의 Accuracy·F1은 이번 실행에서 다시 계산하지 않았다.

## 2. 실행에 필요한 입력

저장소에는 코드, 분류 규칙 목록, 출처 계획, 15개 경로 독립형 원문 기술서와 두 개의 공개용 금융 스냅샷이 들어 있다. 원문 ZIP과 249개 HTML 본문, API 키, 개인 수집 기록, 가상환경 및 BERT 자료는 포함하지 않는다.

외부 원문은 아래처럼 준비한다. `SOURCE_PLAN.json`에 기록된 파일명과 SHA-256이 기준이다.

```text
locked_data/
  raw_dart/                  # 고정 사업보고서 ZIP
  supplemental_raw/          # 명시된 실제 정정 원문의 ZIP
  viewer_capture/
    report_*/http_responses/  # 고정된 원래 HTML 응답 본문
```

22,654개 보고서 중 22,639개는 ZIP 경로, 15개는 보존된 뷰어 HTML 경로로 처리한다. 각 경로는 출처 메타데이터로 정했으며, 목표 문장 수에 맞추어 파서를 선택하지 않는다. 보고서의 표본 식별자와 실제 요청한 정정 원문의 식별자는 별도로 보존한다. 15개 HTML 기술서는 기존 기록의 공개용 투영이며 새 HTTP 수집 기록이 아니다.

두 금융 입력은 `bundle/finance/expanded_panel.csv`와 `bundle/finance/frozen_augmented_panel.csv`다. 공개용 사본에서는 비분석 필드인 오류 메모와 로컬 캐시 경로만 비웠다. 다른 셀의 원래 문자열, 행 순서와 분석 값은 유지했다. 데이터의 출처·이용 범위는 [DATA_NOTICE.md](DATA_NOTICE.md)를 확인한다. MIT는 소프트웨어에 적용되며 제3자 원문이나 데이터의 권리를 새로 부여하지 않는다.

### 원문 확인과 선택적 취득 도구

이미 보유한 입력은 네트워크 없이 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe -B code\acquire_locked_reports.py --bundle bundle --data-root ..\locked_data --check-only
```

향후 원문 취득을 시도하려면 같은 명령의 `--check-only` 대신 `--fetch`를 명시한다. ZIP 취득에는 본인의 `DART_API_KEY` 환경변수가 필요하다. 응답이 고정 해시와 다르면 도구는 중단하며 다른 보고서로 대체하지 않는다. 이 보조 도구는 mock 전송 테스트로 확인했고, 이번 세 번의 재현에서는 실제 신규 다운로드를 수행하지 않았다. 취득 성공과 분류·회귀 재현 성공은 별도 상태로 기록된다.

## 3. 환경 준비와 한 번에 실행

공식 **CPython 3.13.12 Windows AMD64**로 격리된 가상환경을 만든 뒤, 저장소 루트에서 실행한다. `py -3.13`이 다른 패치 버전을 가리키면 정확한 3.13.12 실행파일로 가상환경을 생성한다.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements_windows_cp313_hashed.txt
```

실행기는 Python DLL, 34개 패키지 버전, 설치된 9,658개 패키지 파일의 해시와 lxml의 실제·컴파일 libxml/libxslt 버전을 확인한다. 다른 환경은 자동으로 같은 환경이라고 간주하지 않는다. 각 작업자는 BLAS 스레드 1개를 사용하며, 전체 분류는 작업자 8개로 실행된다.

입력과 환경만 먼저 확인하려면 다음 명령을 사용한다.

```powershell
.\.venv\Scripts\python.exe -B code\public_run.py --data-root ..\locked_data --output ..\preflight_01
```

한 번의 전체 재현:

```powershell
.\.venv\Scripts\python.exe -B code\public_run.py --data-root ..\locked_data --output ..\replay_01 --run --rounds 1
```

세 번의 독립 전체 재현:

```powershell
.\.venv\Scripts\python.exe -B code\public_run.py --data-root ..\locked_data --output ..\replay_3rounds_01 --run --rounds 3
```

각 출력 경로는 **아직 존재하지 않는 폴더**여야 한다. 사전 확인에 쓴 출력 폴더를 전체 실행에 재사용하지 않는다. 원문이 여러 위치에 있으면 `--data-root` 대신 `--raw-dir`, `--supplemental-raw-dir`, `--viewer-capture-root`를 모두 지정한다. 금융 스냅샷은 기본적으로 `bundle/finance`를 사용한다. 승인 JSON이나 개인 컴퓨터의 과거 프로젝트 경로를 수동으로 수정할 필요는 없다.

## 4. 분류와 집계 기준을 확인하는 방법

전체 실행은 고정 출처 확인 → 원문 재추출 → 기존 분문·유효문장 처리 → 기존 규칙 분류 → 보고서별 결과 봉인 → 기업연도 집계 → 분석표본·모형 추정 순서다. 목표 계수나 참조 문장 수는 예측 입력을 생성하지 않으며, 실제 결과를 봉인한 다음에만 비교한다.

- `D`: 기존 전처리·유효문장 조건을 통과한 문장 수.
- `candidate`: 기존 디지털 후보문장 규칙을 통과한 문장 수.
- `S`: 기존 실질적 디지털 문장 조건을 통과한 수.
- `BMODD`: S 중 기존 비즈니스 모델 지향 규칙을 충족한 수.
- `NonBMODD = S − BMODD`: S 중 BMODD 규칙에 해당하지 않는 잔여 범주.
- 원시 비율은 각각 `BMODD/D`, `NonBMODD/D`, `S/D`다. BMODD·NonBMODD의 기본 z값은 전체 22,654개 기업연도의 원시 비율과 표본표준편차(`ddof=1`)로 산출한다.

수작업 개념 기준과 Python의 키워드·정규식 근사 구현은 구분한다. NonBMODD를 별도의 내부 시스템 분류기가 만든 순수 내부 디지털화 범주로 해석하지 않는다. 실제 조건의 순서와 문맥·배제 규칙은 다음 자료에서 코드 위치까지 확인할 수 있다.

- [실제 로드된 규칙](rules/ACTUALLY_LOADED_RULE_DECLARATIONS.json)
- [유효 함수와 조건 순서](rules/EFFECTIVE_FUNCTIONS_AND_ORDERED_CONDITIONS.json)
- [키워드·정규식 목록](rules/REGEX_AND_KEYWORD_INDEX.csv)
- [우선순위와 측정 범위](rules/DECISION_PRIORITY_AND_MEASUREMENT_BOUNDARIES.json)

원본 소스는 `bundle/method`에 해시를 유지한 채 보존했다. 일부 과거 소스에는 당시 로컬 경로 상수가 남아 있지만, 공개 실행기는 선택한 원본 함수와 상대경로 설정을 사용하므로 그 과거 폴더를 만들 필요가 없다.

## 5. 주요 출력과 실패 처리

세 번 실행하면 출력 루트의 `runs/round_01`, `round_02`, `round_03`에 각각 새 결과가 생성된다.

| 경로(각 round 아래) | 내용 |
| --- | --- |
| `classification/fresh_report_predictions.jsonl` | 보고서별 실제 문장 수·출처·추출/문장예측 해시 |
| `classification/PREDICTION_SEAL.json` | 실제 호출 수, 전체 처리 여부와 예측 봉인 |
| `empirical/BMODD_FIRM_YEAR_CLASSIFICATION.csv` | 22,654행·14열 BMODD/NonBMODD 분류 지표 |
| `empirical/primary_h1_h2.csv`, `empirical/h2_wald.csv` | H1·H2와 차이검정 |
| `empirical/current_core`, `empirical/h3_joint` | 시점별·공동표본·강건성 및 공식 H3 차이검정 |
| `empirical/trend.csv`, `placebo.csv`, `ipw` | 기업별 추세·위약·IPW 결과 |
| `empirical/supplement_with_original_BH19.csv` | COVID·K-IFRS 1116·분해와 원래 19개 BH 검정 집합 |
| `empirical/MODEL_OUTPUTS_SEAL.json` | 참조 계수를 열기 전에 봉인한 실제 모형 출력 |
| `FULL_CURRENT_METHOD_RUN_RECEIPT.json` | 해당 전체 실행의 판정 |

세 번의 전체 실행이 모두 통과하고 수치·표본·그림 해시가 같을 때만 `runs/THREE_ROUND_RECEIPT.json`이 생성된다. 예측 체크포인트를 공유하지 않으며, 첫 실패가 나면 다음 실행을 시작하지 않는다. 오류 기록과 부분 결과를 보존하고 원문·규칙·인간 라벨을 고쳐서 목표 수치에 맞추지 않는다. `CURRENT_MEASUREMENT_PANEL_INTERNAL.csv`는 실행 중 생성되는 내부 분석 입력이며, 공개 결과 CSV의 대체물이 아니다.

## 中文快速入口

先准备与 `bundle/SOURCE_PLAN.json` 哈希一致的原ZIP及249份HTML，并使用官方Windows AMD64 CPython 3.13.12和锁定依赖。仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -B code\public_run.py --data-root ..\locked_data --output ..\replay_3rounds_01 --run --rounds 3
```

这是在原文和环境已备条件下的一键全流程：每轮重新解析、分类、汇总并估计当前论文全部模型。三轮已实际完成并经独立核验，每轮22,654份报告、25,041,971次原分类函数调用，原始比率、样本、数值CSV和图形一致，0次HTTP。它不等于重新从网站下载或重新核算人工κ及分类器Accuracy/F1。原文、API密钥和BERT不在仓库中；金融输入为仅清空两项非分析字段的安全副本。原始精度以CSV为准，Excel查看版不替代CSV。软件MIT与数据权利范围见 `LICENSE` 和 `DATA_NOTICE.md`。
