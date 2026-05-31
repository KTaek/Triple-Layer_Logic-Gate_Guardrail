import math
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Dict, List, Sequence, Tuple

import streamlit as st
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:
    torch = None
    AutoModel = None
    AutoTokenizer = None


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

YELLOW_THRESHOLD = 0.45
ORANGE_THRESHOLD = 0.75
SESSION_DECAY = 0.72
TEMPORAL_BLEND = 0.35

YELLOW_KEYWORDS = ["죽고 싶다", "자살", "극단적 선택", "살기 싫다", "없어지고 싶다"]
RISK_KEYWORDS = ["우회", "탈취", "공격", "불법", "폭발", "침입", "해킹", "악용", "조작"]
PROCEDURAL_KEYWORDS = ["방법", "절차", "단계", "도구", "준비물", "실행", "설정", "코드", "스크립트"]
TARGET_KEYWORDS = ["특정 사람", "타인", "계정", "서버", "장소", "회사", "학교", "기관"]
SECRECY_KEYWORDS = ["들키지", "추적", "익명", "기록 없이", "몰래", "감추"]
URGENCY_KEYWORDS = ["지금", "당장", "오늘", "바로", "긴급", "즉시"]

RISK_EXEMPLARS: Dict[str, List[str]] = {
    "self_harm": [
        "삶을 포기하고 싶다는 감정을 표현하는 문장",
        "스스로를 해치고 싶다는 의도를 드러내는 요청",
        "위기 상황에서 즉각적 도움이 필요한 문장",
    ],
    "violent_illegal": [
        "타인에게 위해를 주는 불법 행동을 준비하는 요청",
        "위험한 물건이나 행동을 실행하려는 지시형 문장",
        "법을 어기면서 누군가를 해치려는 의도가 담긴 문장",
    ],
    "cyber_abuse": [
        "시스템이나 계정을 침해하기 위한 공격적 요청",
        "보안 장치를 우회하거나 탈취하는 목적의 질문",
        "권한 없는 접근을 시도하는 악용성 프롬프트",
    ],
    "bypass_abuse": [
        "정책이나 감시를 피하면서 작업하려는 요청",
        "탐지를 회피하거나 익명성을 높이려는 문장",
        "규칙을 우회하기 위한 편법적 안내를 요구하는 문장",
    ],
}

SAFE_EXEMPLARS = [
    "정책 준수 범위에서 개념 설명만 요청하는 문장",
    "예방과 보안 강화 목적의 무해한 질문",
    "학술적 비교나 정의 설명을 요청하는 안전한 문장",
]

CLASSIFIER_WEIGHTS = {
    "self_harm": 1.0,
    "violent_illegal": 0.95,
    "cyber_abuse": 0.9,
    "bypass_abuse": 0.8,
}


def downloads_allowed() -> bool:
    return os.getenv("ALLOW_MODEL_DOWNLOADS", "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TurnAssessment:
    timestamp: str
    user_input: str
    keyword_risk: float
    embedding_risk: float
    classifier_risk: float
    temporal_risk: float
    session_risk: float
    final_risk: float
    stage: str
    action: str
    top_category: str
    reasons: List[str]
    label_scores: Dict[str, float]
    model_status: Dict[str, str]


@dataclass
class SessionState:
    user_id: str = "DEMO_USER_001"
    session_id: str = field(default_factory=lambda: f"SESSION_{int(time.time())}")
    session_risk: float = 0.0
    reported: bool = False
    report_path: str = ""
    email_sent: bool = False
    hidden_state: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_turn_at: float = 0.0
    turns: List[TurnAssessment] = field(default_factory=list)
    system_logs: List[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        now = time.strftime("%H:%M:%S")
        self.system_logs.append(f"[{now}] {message}")


def _safe_get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps) or 1.0
    return [exp_value / total for exp_value in exps]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if np is not None:
        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    numerator = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in vec_a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in vec_b))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return numerator / denom


def char_ngram_vector(text: str, n: int = 3) -> Dict[str, float]:
    normalized = "".join(text.lower().split())
    if len(normalized) < n:
        return {normalized: 1.0} if normalized else {}
    grams: Dict[str, float] = {}
    for idx in range(len(normalized) - n + 1):
        gram = normalized[idx: idx + n]
        grams[gram] = grams.get(gram, 0.0) + 1.0
    return grams


def sparse_cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    common = set(vec_a).intersection(vec_b)
    numerator = sum(vec_a[token] * vec_b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return numerator / denom


def keyword_ratio(text: str, keywords: List[str], weight: float = 1.0) -> float:
    hits = sum(1 for keyword in keywords if keyword in text)
    if not keywords:
        return 0.0
    return clamp((hits / len(keywords)) * weight)


class EmbeddingRiskScorer:
    def __init__(self) -> None:
        self.backend = "fallback-ngram"
        self.model = None
        self.status = "SentenceTransformer 미탑재 - 문자 n-gram 폴백 사용"
        self.risk_vectors: Dict[str, List[Sequence[float]]] = {}
        self.safe_vectors: List[Sequence[float]] = []
        self._load()

    def _load(self) -> None:
        if SentenceTransformer is None:
            self._prepare_fallback()
            return

        model_candidates = [
            os.getenv("EMBEDDING_MODEL_NAME", "").strip(),
            "jhgan/ko-sroberta-multitask",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
        local_files_only = not downloads_allowed()
        for model_name in [name for name in model_candidates if name]:
            try:
                self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
                self.backend = model_name
                self.status = f"{model_name} 로드 완료"
                self.risk_vectors = {
                    label: self.model.encode(examples, normalize_embeddings=True).tolist()
                    for label, examples in RISK_EXEMPLARS.items()
                }
                self.safe_vectors = self.model.encode(SAFE_EXEMPLARS, normalize_embeddings=True).tolist()
                return
            except Exception:
                continue

        if local_files_only:
            self.status = "임베딩 모델 캐시 없음 - 문자 n-gram 폴백 사용"
        self._prepare_fallback()

    def _prepare_fallback(self) -> None:
        self.risk_vectors = {
            label: [char_ngram_vector(example) for example in examples]
            for label, examples in RISK_EXEMPLARS.items()
        }
        self.safe_vectors = [char_ngram_vector(example) for example in SAFE_EXEMPLARS]

    def score(self, text: str) -> Tuple[float, str]:
        if self.model is not None:
            encoded = self.model.encode([text], normalize_embeddings=True)[0].tolist()
            per_label = {}
            for label, vectors in self.risk_vectors.items():
                per_label[label] = max(cosine_similarity(encoded, vector) for vector in vectors)
            safe_max = max(cosine_similarity(encoded, vector) for vector in self.safe_vectors)
        else:
            encoded = char_ngram_vector(text)
            per_label = {}
            for label, vectors in self.risk_vectors.items():
                per_label[label] = max(sparse_cosine(encoded, vector) for vector in vectors)
            safe_max = max(sparse_cosine(encoded, vector) for vector in self.safe_vectors) if self.safe_vectors else 0.0

        label, risk_max = max(per_label.items(), key=lambda item: item[1])
        adjusted = clamp((risk_max - 0.25 * safe_max + 1.0) / 2.0)
        return round(adjusted, 4), label


class BertPrototypeClassifier:
    def __init__(self) -> None:
        self.backend = "fallback-prototype"
        self.status = "Transformers 미탑재 - 프로토타입 폴백 사용"
        self.tokenizer = None
        self.model = None
        self.prototype_vectors: Dict[str, Sequence[float]] = {}
        self.prototype_sparse = {
            label: char_ngram_vector(" ".join(examples))
            for label, examples in RISK_EXEMPLARS.items()
        }
        self._load()

    def _load(self) -> None:
        if torch is None or AutoTokenizer is None or AutoModel is None:
            return

        model_candidates = [
            os.getenv("RISK_CLASSIFIER_MODEL", "").strip(),
            "klue/bert-base",
            "skt/kobert-base-v1",
        ]
        local_files_only = not downloads_allowed()
        for model_name in [name for name in model_candidates if name]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
                self.model.eval()
                self.backend = model_name
                self.status = f"{model_name} 로드 완료"
                for label, examples in RISK_EXEMPLARS.items():
                    self.prototype_vectors[label] = self._encode_texts(examples)
                return
            except Exception:
                self.tokenizer = None
                self.model = None
                self.prototype_vectors = {}

        if local_files_only:
            self.status = "BERT 모델 캐시 없음 - 프로토타입 폴백 사용"

    def _encode_texts(self, texts: Sequence[str]) -> Sequence[float]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            centroid = pooled.mean(dim=0)
        return centroid.cpu().tolist()

    def classify(self, text: str) -> Tuple[float, Dict[str, float], str]:
        if self.model is not None and self.tokenizer is not None and self.prototype_vectors:
            encoded = self._encode_texts([text])
            raw_scores = []
            labels = []
            for label, vector in self.prototype_vectors.items():
                labels.append(label)
                raw_scores.append(cosine_similarity(encoded, vector))
        else:
            sparse = char_ngram_vector(text)
            labels = list(self.prototype_sparse.keys())
            raw_scores = [sparse_cosine(sparse, self.prototype_sparse[label]) for label in labels]

        probabilities = softmax([score * 6.0 for score in raw_scores])
        label_scores = {label: round(prob, 4) for label, prob in zip(labels, probabilities)}
        top_label = max(label_scores, key=label_scores.get)
        weighted = sum(label_scores[label] * CLASSIFIER_WEIGHTS[label] for label in labels)
        return round(clamp(weighted), 4), label_scores, top_label


class TemporalSequenceAnalyzer:
    """
    학습 체크포인트 없이도 데모가 가능하도록 구성한 GRU 스타일의 시퀀스 게이트.
    추후 실제 LSTM/GRU 체크포인트를 붙일 때 동일한 입력 feature 흐름을 재사용할 수 있다.
    """

    def __init__(self) -> None:
        self.backend = "gru-demo-cell"
        self.status = "수동 가중치 기반 GRU 스타일 시퀀스 분석"

        self.wz = [
            [0.85, 0.35, 0.55, 0.65],
            [0.90, 0.25, 0.40, 0.55],
            [0.40, 0.25, 0.25, 0.30],
            [0.55, 0.30, 0.40, 0.50],
            [0.75, 0.25, 0.30, 0.45],
            [0.30, 0.40, 0.70, 0.35],
            [0.20, 0.55, 0.35, 0.25],
            [0.65, 0.50, 0.30, 0.25],
        ]
        self.uz = [
            [0.35, 0.10, 0.20, 0.25],
            [0.25, 0.40, 0.20, 0.20],
            [0.15, 0.15, 0.45, 0.25],
            [0.10, 0.20, 0.15, 0.55],
        ]
        self.wr = [
            [0.45, 0.25, 0.30, 0.20],
            [0.50, 0.20, 0.25, 0.15],
            [0.30, 0.15, 0.20, 0.25],
            [0.40, 0.25, 0.20, 0.25],
            [0.35, 0.15, 0.20, 0.30],
            [0.20, 0.35, 0.45, 0.20],
            [0.10, 0.30, 0.20, 0.45],
            [0.45, 0.25, 0.15, 0.20],
        ]
        self.ur = [
            [0.30, 0.15, 0.20, 0.10],
            [0.15, 0.35, 0.10, 0.20],
            [0.20, 0.15, 0.30, 0.20],
            [0.10, 0.20, 0.25, 0.35],
        ]
        self.wh = [
            [0.90, 0.30, 0.35, 0.55],
            [1.00, 0.20, 0.30, 0.45],
            [0.45, 0.30, 0.25, 0.25],
            [0.60, 0.20, 0.35, 0.35],
            [0.75, 0.15, 0.25, 0.40],
            [0.25, 0.35, 0.80, 0.30],
            [0.15, 0.70, 0.20, 0.15],
            [0.85, 0.55, 0.20, 0.20],
        ]
        self.uh = [
            [0.35, 0.10, 0.15, 0.20],
            [0.15, 0.35, 0.10, 0.20],
            [0.15, 0.20, 0.35, 0.15],
            [0.10, 0.15, 0.20, 0.35],
        ]
        self.output_weights = [0.55, 0.70, 0.60, 0.65]

    def _matmul(self, vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> List[float]:
        return [
            sum(float(vector[row_idx]) * float(matrix[row_idx][col_idx]) for row_idx in range(len(vector)))
            for col_idx in range(len(matrix[0]))
        ]

    def step(self, previous_hidden: Sequence[float], features: Sequence[float]) -> Tuple[List[float], float]:
        z = [
            sigmoid(x + h)
            for x, h in zip(self._matmul(features, self.wz), self._matmul(previous_hidden, self.uz))
        ]
        r = [
            sigmoid(x + h)
            for x, h in zip(self._matmul(features, self.wr), self._matmul(previous_hidden, self.ur))
        ]
        gated_hidden = [r_value * h_value for r_value, h_value in zip(r, previous_hidden)]
        candidate = [
            math.tanh(x + h)
            for x, h in zip(self._matmul(features, self.wh), self._matmul(gated_hidden, self.uh))
        ]
        next_hidden = [
            (1.0 - z_value) * prev + z_value * cand
            for z_value, prev, cand in zip(z, previous_hidden, candidate)
        ]
        signal = sigmoid(sum(weight * hidden for weight, hidden in zip(self.output_weights, next_hidden)) - 0.8)
        return [round(value, 6) for value in next_hidden], round(clamp(signal), 4)


class GuardrailEngine:
    def __init__(self) -> None:
        self.embedding_scorer = EmbeddingRiskScorer()
        self.classifier = BertPrototypeClassifier()
        self.temporal = TemporalSequenceAnalyzer()

    @property
    def model_status(self) -> Dict[str, str]:
        return {
            "embedding": self.embedding_scorer.status,
            "classifier": self.classifier.status,
            "temporal": self.temporal.status,
        }

    def keyword_score(self, user_input: str) -> Tuple[float, List[str]]:
        urgency = keyword_ratio(user_input, URGENCY_KEYWORDS, weight=1.8)
        procedural = keyword_ratio(user_input, PROCEDURAL_KEYWORDS, weight=2.0)
        target = keyword_ratio(user_input, TARGET_KEYWORDS, weight=1.7)
        secrecy = keyword_ratio(user_input, SECRECY_KEYWORDS, weight=2.0)
        risk = keyword_ratio(user_input, RISK_KEYWORDS, weight=2.2)

        score = clamp(0.15 + 0.22 * urgency + 0.23 * procedural + 0.18 * target + 0.17 * secrecy + 0.20 * risk)
        reasons = []
        if urgency > 0.15:
            reasons.append("긴급성 표현이 감지되었습니다.")
        if procedural > 0.15:
            reasons.append("구체적 절차/도구 요청 신호가 있습니다.")
        if target > 0.15:
            reasons.append("대상 지시성이 감지되었습니다.")
        if secrecy > 0.15:
            reasons.append("회피/은닉 의도가 감지되었습니다.")
        return round(score, 4), reasons

    def _repetition_risk(self, user_input: str, state: SessionState) -> float:
        if not state.turns:
            return 0.0
        current = char_ngram_vector(user_input)
        recent_scores = []
        for turn in state.turns[-4:]:
            previous = char_ngram_vector(turn.user_input)
            recent_scores.append(sparse_cosine(current, previous))
        return round(clamp(max(recent_scores) if recent_scores else 0.0), 4)

    def assess(self, user_input: str, state: SessionState) -> TurnAssessment:
        now = time.time()
        time_gap = now - state.last_turn_at if state.last_turn_at else 0.0
        state.last_turn_at = now

        keyword_risk, keyword_reasons = self.keyword_score(user_input)
        embedding_risk, embedding_label = self.embedding_scorer.score(user_input)
        classifier_risk, label_scores, top_label = self.classifier.classify(user_input)
        repetition_risk = self._repetition_risk(user_input, state)

        urgency_feature = keyword_ratio(user_input, URGENCY_KEYWORDS, weight=1.0)
        procedural_feature = keyword_ratio(user_input, PROCEDURAL_KEYWORDS, weight=1.0)
        secrecy_feature = keyword_ratio(user_input, SECRECY_KEYWORDS, weight=1.0)
        compressed_gap = clamp(1.0 - min(time_gap, 600.0) / 600.0)

        features = [
            embedding_risk,
            classifier_risk,
            keyword_risk,
            procedural_feature,
            urgency_feature,
            repetition_risk,
            compressed_gap,
            secrecy_feature,
        ]
        hidden, temporal_risk = self.temporal.step(state.hidden_state, features)
        state.hidden_state = hidden

        base_risk = clamp(
            0.22 * keyword_risk
            + 0.30 * embedding_risk
            + 0.30 * classifier_risk
            + 0.18 * temporal_risk
        )
        session_risk = clamp(SESSION_DECAY * state.session_risk + (1.0 - SESSION_DECAY) * max(base_risk, temporal_risk))
        state.session_risk = session_risk
        final_risk = clamp((1.0 - TEMPORAL_BLEND) * base_risk + TEMPORAL_BLEND * max(session_risk, temporal_risk))

        crisis_override = any(keyword in user_input for keyword in YELLOW_KEYWORDS)
        if crisis_override:
            top_label = "self_harm"
            label_scores["self_harm"] = max(label_scores.get("self_harm", 0.0), 0.85)
            final_risk = max(final_risk, 0.82)

        stage = determine_stage(final_risk)
        reasons = keyword_reasons[:]
        if embedding_risk >= 0.58:
            reasons.append(f"문장 임베딩이 '{embedding_label}' 위험 범주와 의미적으로 유사합니다.")
        if classifier_risk >= 0.55:
            reasons.append(f"BERT 프로토타입 분류기가 '{top_label}' 위험도를 높게 평가했습니다.")
        if temporal_risk >= 0.55:
            reasons.append("세션 흐름상 위험 패턴이 누적되고 있습니다.")
        if repetition_risk >= 0.65:
            reasons.append("이전 턴과 유사한 재시도 패턴이 보입니다.")
        if crisis_override:
            reasons.append("위기 표현이 감지되어 보호 응답을 우선 적용합니다.")

        action = action_for_stage(stage)
        assessment = TurnAssessment(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            user_input=user_input,
            keyword_risk=round(keyword_risk, 4),
            embedding_risk=round(embedding_risk, 4),
            classifier_risk=round(classifier_risk, 4),
            temporal_risk=round(temporal_risk, 4),
            session_risk=round(session_risk, 4),
            final_risk=round(final_risk, 4),
            stage=stage,
            action=action,
            top_category=top_label,
            reasons=reasons,
            label_scores=label_scores,
            model_status=self.model_status,
        )
        state.turns.append(assessment)
        state.add_log(f"{stage} score={assessment.final_risk:.2f} category={top_label} action={action}")
        return assessment


def determine_stage(score: float) -> str:
    if score >= ORANGE_THRESHOLD:
        return "RED"
    if score >= YELLOW_THRESHOLD:
        return "ORANGE"
    return "YELLOW"


def action_for_stage(stage: str) -> str:
    if stage == "RED":
        return "BLOCK_AND_REPORT"
    if stage == "ORANGE":
        return "BLOCK_AND_SAFE_REPLY"
    return "SAFE_RESPONSE"


def get_gemini_api_key() -> str:
    return _safe_get_secret("GEMINI_API_KEY", "")


def get_gemini_model() -> str:
    return _safe_get_secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL


def call_main_llm(user_input: str, safety_mode: bool = False) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        if safety_mode:
            return "위험 가능성이 있어 구체적 절차 대신 예방, 법적 책임, 대안적 안전 조치만 안내합니다."
        return f"입력 문장 요약: '{user_input[:80]}'\n안전한 범위에서만 답변을 생성하도록 API 키 없이 데모 응답을 반환했습니다."

    try:
        client = genai.Client(api_key=api_key)
        temperature = 0.15 if safety_mode else 0.6
        max_output_tokens = 180 if safety_mode else 280
        system_instruction = (
            "You are a Korean AI safety guardrail demo assistant. "
            "Never provide operational instructions for harmful, illegal, violent, abusive, or evasive activity. "
            "If the prompt is risky, explain safety, prevention, lawfulness, and benign alternatives only."
        )
        prompt = (
            f"System:\n{system_instruction}\n\n"
            f"User input:\n{user_input[:1200]}\n\n"
            "Respond in Korean. Keep it concise and safe."
        )
        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        if getattr(response, "text", None):
            return response.text
        return "모델 응답이 비어 있어 안전한 기본 응답으로 대체했습니다."
    except Exception as exc:
        error_text = str(exc)
        if safety_mode:
            return f"Gemini 안전 응답 폴백이 적용되었습니다. 원인: {error_text[:120]}"
        return f"Gemini 호출에 실패해 데모 응답을 반환합니다. 원인: {error_text[:120]}"


def build_guardrail_response(user_input: str, assessment: TurnAssessment) -> str:
    if assessment.top_category == "self_harm":
        return (
            "위기 가능성이 감지되어 일반 응답을 중단했습니다.\n\n"
            "지금 즉시 주변의 신뢰할 수 있는 사람이나 지역 정신건강 지원 기관, 응급 서비스에 도움을 요청해 주세요. "
            "혼자 버티지 않는 것이 가장 중요합니다."
        )

    if assessment.stage == "YELLOW":
        return call_main_llm(user_input, safety_mode=False)

    if assessment.stage == "ORANGE":
        return (
            "위험 가능성이 중간 이상으로 평가되어 구체적 실행 정보는 제공하지 않습니다.\n\n"
            + call_main_llm(user_input, safety_mode=True)
        )

    return (
        "고위험 요청으로 판단되어 답변 생성을 차단했고, 세션 리스크 보고서를 생성 대상으로 표시했습니다.\n\n"
        "원하시면 동일 주제를 합법적 예방, 보안 강화, 정책 준수 관점으로만 다시 설명할 수 있습니다."
    )


def get_email_config() -> Dict[str, str]:
    return {
        "sender": _safe_get_secret("DEMO_EMAIL_SENDER", ""),
        "password": _safe_get_secret("DEMO_EMAIL_PASSWORD", ""),
        "receiver": _safe_get_secret("DEMO_EMAIL_RECEIVER", ""),
    }


def generate_report(state: SessionState) -> str:
    file_name = f"Risk_Report_{state.session_id}_{int(time.time())}.pdf"
    report_path = os.path.join(REPORT_DIR, file_name)

    pdf = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(72, height - 72, "AI SAFETY GUARDRAIL REPORT")

    latest = state.turns[-1] if state.turns else None
    pdf.setFont("Helvetica", 10)
    y = height - 104
    lines = [
        "Report Type: Streamlit Guardrail Demo",
        f"User ID: {state.user_id}",
        f"Session ID: {state.session_id}",
        f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Current Session Risk: {state.session_risk:.2f}",
        f"Latest Stage: {latest.stage if latest else 'N/A'}",
        "Recommended Action: Human review before any external escalation.",
    ]
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 16

    if latest is not None:
        y -= 8
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, y, "Latest Signal Breakdown")
        y -= 18
        pdf.setFont("Helvetica", 9)
        detail_lines = [
            f"Keyword Risk: {latest.keyword_risk:.2f}",
            f"Embedding Risk: {latest.embedding_risk:.2f}",
            f"BERT Risk: {latest.classifier_risk:.2f}",
            f"Temporal Risk: {latest.temporal_risk:.2f}",
            f"Final Risk: {latest.final_risk:.2f}",
            f"Top Category: {latest.top_category}",
        ]
        for line in detail_lines:
            pdf.drawString(72, y, line)
            y -= 14

    y -= 4
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(72, y, "Turn-by-Turn Logs")
    y -= 20
    pdf.setFont("Helvetica", 8)
    for idx, turn in enumerate(state.turns, start=1):
        if y < 88:
            pdf.showPage()
            y = height - 72
            pdf.setFont("Helvetica", 8)
        summary = turn.user_input.replace("\n", " ")[:78]
        entries = [
            f"#{idx} [{turn.timestamp}] Stage={turn.stage} Action={turn.action}",
            f"  K={turn.keyword_risk:.2f} E={turn.embedding_risk:.2f} B={turn.classifier_risk:.2f} T={turn.temporal_risk:.2f} F={turn.final_risk:.2f}",
            f"  Category={turn.top_category} Input={summary}",
        ]
        for entry in entries:
            pdf.drawString(72, y, entry)
            y -= 12
        y -= 4

    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(72, 50, "Notice: This report is for demo and internal review only.")
    pdf.save()
    return report_path


def send_demo_email(report_path: str) -> Tuple[bool, str]:
    config = get_email_config()
    sender = config["sender"]
    password = config["password"]
    receiver = config["receiver"]

    if not sender or not password or not receiver:
        return False, "이메일 설정이 없어 PDF만 생성했습니다."

    message = EmailMessage()
    message["Subject"] = "[DEMO] AI Safety Guardrail Report"
    message["From"] = sender
    message["To"] = receiver
    message.set_content(
        "This is an academic demonstration email.\n"
        "A guardrail report was generated and should be reviewed by a human."
    )
    with open(report_path, "rb") as file_handle:
        message.add_attachment(
            file_handle.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(report_path),
        )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message)
        return True, f"이메일 전송 완료: {receiver}"
    except Exception as exc:
        return False, f"이메일 전송 실패: {exc}"


def process_user_message(user_input: str, state: SessionState, engine: GuardrailEngine) -> Tuple[str, TurnAssessment]:
    state.add_log(f"INPUT: {user_input[:60]}")
    assessment = engine.assess(user_input, state)

    if assessment.stage == "RED" and not state.reported:
        state.add_log("RED 단계 진입 - PDF 보고서 생성 시작")
        report_path = generate_report(state)
        state.report_path = report_path
        state.reported = True
        sent, message = send_demo_email(report_path)
        state.email_sent = sent
        state.add_log(message)

    response = build_guardrail_response(user_input, assessment)
    return response, assessment
