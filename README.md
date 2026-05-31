##
본 파일은 2026 글로벌리더학부X글로벌융합학부 융합학술제 제출 시 쓰였던 데모 파일입니다.

# Korean AI Safety Guardrail Demo

한국어 프롬프트의 위험도를 실시간으로 분석하고, 위험 수준에 따라 안전 응답 정책을 적용하는 Streamlit 기반 AI Safety Guardrail 데모 프로젝트입니다.

본 프로젝트는 단순한 키워드 필터링을 넘어, 문장 임베딩 기반 의미 유사도, BERT prototype 기반 위험 분류, 세션 흐름 분석을 결합하여 사용자의 입력 문장을 `YELLOW`, `ORANGE`, `RED` 단계로 분류합니다.

## 1. 프로젝트 개요

LLM 서비스에서는 사용자의 입력이 안전한 요청인지, 위험 가능성이 있는 요청인지 판단하는 guardrail 시스템이 중요합니다.
이 프로젝트는 한국어 입력 문장을 대상으로 위험도를 점수화하고, 위험 단계에 따라 다른 응답 정책을 적용하는 데모 애플리케이션입니다.

주요 목표는 다음과 같습니다.

* 한국어 프롬프트의 위험도 실시간 분석
* 위험 수준에 따른 단계별 응답 정책 적용
* 키워드 기반 탐지와 의미 기반 탐지의 결합
* 세션 내 반복적 위험 패턴 추적
* 고위험 입력 발생 시 PDF 리포트 생성

## 2. 주요 기능

### 실시간 채팅 시뮬레이션

사용자가 한국어 문장을 입력하면 시스템이 즉시 위험도를 계산하고, 단계별 정책에 따라 응답합니다.

### 위험도 단계 분류

입력 문장은 다음 세 단계 중 하나로 분류됩니다.

| 단계     | 설명                                |
| ------ | --------------------------------- |
| YELLOW | 저위험 또는 안전한 입력으로 판단하여 일반 응답 제공     |
| ORANGE | 위험 가능성이 있어 구체적인 답변을 제한하고 안전 응답 제공 |
| RED    | 고위험 요청으로 판단하여 응답을 차단하고 보고서 생성     |

### 다중 위험 신호 분석

본 프로젝트는 다음 요소들을 함께 고려하여 최종 위험도를 계산합니다.

* Keyword Risk
* Embedding Risk
* BERT Prototype Risk
* Temporal GRU-style Risk
* Session Carryover Risk

### 위험도 대시보드

Streamlit UI에서 현재 세션의 위험 상태를 시각적으로 확인할 수 있습니다.

* Current Stage
* Final Risk Score
* 세션 누적 리스크
* 누적 턴 수
* 위험 신호별 점수
* 상위 분류 레이블
* 판단 근거
* 시스템 로그

### PDF 리포트 생성

입력이 `RED` 단계로 판단되면 세션 위험도와 입력 로그를 바탕으로 PDF 리포트를 생성합니다.

## 3. 프로젝트 구조

```bash
.
├── app.py                  # Streamlit 기반 웹 애플리케이션
├── guardrail_engine.py      # 위험도 분석 및 guardrail 로직
├── requirements.txt         # Python 패키지 의존성
├── devcontainer.json        # 개발 컨테이너 설정
└── reports/                 # RED 단계 발생 시 PDF 리포트 저장 폴더
```

## 4. 사용 기술

* Python
* Streamlit
* Pandas
* NumPy
* PyTorch
* Transformers
* Sentence-Transformers
* Google GenAI
* ReportLab

## 5. 위험도 판단 방식

### 5.1 Keyword Risk

입력 문장에 포함된 위험 키워드, 절차 요청 표현, 긴급성 표현, 은닉 의도 표현 등을 기반으로 위험도를 계산합니다.

예시 신호:

* 긴급성 표현
* 구체적인 절차나 도구 요청
* 특정 대상 지시
* 회피 또는 은닉 의도
* 위험 행위 관련 키워드

### 5.2 Embedding Risk

SentenceTransformer 모델을 사용할 수 있는 경우, 입력 문장과 위험 예시 문장 간의 의미적 유사도를 계산합니다.

모델을 사용할 수 없는 환경에서는 문자 n-gram 기반 fallback 방식을 사용합니다.

### 5.3 BERT Prototype Classifier

BERT 또는 KoBERT 계열 모델을 사용할 수 있는 경우, 위험 카테고리별 prototype vector와 입력 문장 간의 유사도를 계산합니다.

모델을 사용할 수 없는 경우에는 prototype 기반 fallback 방식으로 동작합니다.

분류 카테고리는 다음과 같습니다.

* self_harm
* violent_illegal
* cyber_abuse
* bypass_abuse

### 5.4 Temporal Sequence Analysis

단일 입력만 보는 것이 아니라, 세션 내 이전 입력들과의 흐름을 함께 고려합니다.
이를 위해 GRU 스타일의 수동 가중치 기반 시퀀스 분석 로직을 사용하여 반복적이거나 누적되는 위험 패턴을 반영합니다.

## 6. 설치 및 실행 방법

### 6.1 패키지 설치

```bash
pip install -r requirements.txt
```

### 6.2 Streamlit 실행

```bash
streamlit run app.py
```

실행 후 브라우저에서 Streamlit 앱이 열리며, 한국어 문장을 입력해 위험도 분석 결과를 확인할 수 있습니다.

## 7. 환경 변수 설정

Gemini API를 사용하려면 다음 환경 변수를 설정할 수 있습니다.

```bash
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
```

모델 다운로드를 허용하려면 다음 환경 변수를 설정합니다.

```bash
ALLOW_MODEL_DOWNLOADS=true
```

이메일 전송 기능을 사용하려면 다음 환경 변수를 설정할 수 있습니다.

```bash
DEMO_EMAIL_SENDER=your_email@gmail.com
DEMO_EMAIL_PASSWORD=your_app_password
DEMO_EMAIL_RECEIVER=receiver_email@gmail.com
```

환경 변수가 설정되지 않은 경우에도 프로젝트는 fallback 방식으로 동작하며, API 없이 데모 응답을 반환합니다.

## 8. 실행 화면 구성

앱은 크게 다음 영역으로 구성됩니다.

* 실시간 채팅 시뮬레이션
* 단계별 정책 설명
* 위험도 대시보드
* 모델 상태 확인
* 시스템 로그
* 세션 이력
* PDF 보고서 다운로드

## 9. 프로젝트 특징

이 프로젝트의 핵심 특징은 단일 방식에 의존하지 않고 여러 위험 신호를 결합한다는 점입니다.

단순한 키워드 필터는 우회 표현이나 문맥적 위험을 놓칠 수 있습니다.
따라서 본 프로젝트에서는 키워드 탐지, 의미 유사도, prototype 기반 분류, 세션 흐름 분석을 함께 사용하여 보다 안정적인 guardrail 구조를 구현했습니다.

또한 실제 서비스 환경을 가정하여 위험 단계별 응답 정책을 분리했습니다.
저위험 입력은 일반 응답을 허용하고, 중간 위험 입력은 안전 응답으로 전환하며, 고위험 입력은 차단 및 보고서 생성으로 이어지도록 설계했습니다.

## 10. 느낀 점

이번 프로젝트를 통해 AI Safety Guardrail이 단순히 금지어를 찾는 문제가 아니라, 사용자의 의도와 문맥, 반복적인 입력 흐름까지 함께 고려해야 하는 문제라는 점을 배웠습니다.

특히 한국어 입력에서는 표현 방식이 다양하고 우회적인 문장이 등장할 수 있기 때문에, 키워드 기반 탐지만으로는 한계가 있습니다.
이를 보완하기 위해 의미 기반 분석과 세션 흐름 분석을 함께 사용하는 구조를 구현하면서, LLM 서비스에서 안전성을 확보하기 위한 시스템 설계의 중요성을 경험할 수 있었습니다.

## 11. 향후 개선 방향

* 실제 학습된 위험 분류 모델 적용
* 한국어 safety dataset 기반 fine-tuning
* 더 정교한 세션 기반 LSTM/GRU 모델 적용
* 위험 카테고리 세분화
* 관리자용 dashboard 기능 강화
* 실제 서비스 API와 연동 가능한 구조로 확장

## 12. 주의사항

본 프로젝트는 학습 및 데모 목적의 AI Safety Guardrail 시스템입니다.
실제 서비스 환경에 적용하기 위해서는 더 많은 데이터 기반 검증, 전문가 검토, 정책 기준 정교화, 오탐 및 미탐 분석이 필요합니다.
