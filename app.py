import pandas as pd
import streamlit as st

from guardrail_engine import (
    GuardrailEngine,
    SessionState,
    get_gemini_model,
    process_user_message,
)


st.set_page_config(page_title="Korean AI Safety Guardrail", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg-main: #f4efe6;
        --bg-panel: #fffaf2;
        --ink: #1d1f1e;
        --muted: #5b5e58;
        --accent: #0e7490;
        --yellow: #d97706;
        --orange: #c2410c;
        --red: #b91c1c;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(14, 116, 144, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 30%),
            linear-gradient(180deg, #f9f4ea 0%, #f2ede4 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-card, .panel-card {
        background: rgba(255, 250, 242, 0.82);
        border: 1px solid rgba(120, 113, 108, 0.18);
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(33, 24, 12, 0.08);
        backdrop-filter: blur(10px);
    }
    .hero-card {
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
        margin-bottom: 1rem;
    }
    .panel-card {
        padding: 1.1rem 1.1rem 0.6rem 1.1rem;
        margin-top: 0.6rem;
    }
    .eyebrow {
        font-size: 0.82rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
    }
    .hero-title {
        font-size: 2.1rem;
        line-height: 1.1;
        font-weight: 800;
        margin: 0.25rem 0 0.6rem 0;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.5;
    }
    .risk-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.72rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.45rem;
        margin-bottom: 0.4rem;
    }
    .pill-yellow { background: rgba(217, 119, 6, 0.12); color: var(--yellow); }
    .pill-orange { background: rgba(194, 65, 12, 0.12); color: var(--orange); }
    .pill-red { background: rgba(185, 28, 28, 0.12); color: var(--red); }
    .metric-label {
        color: var(--muted);
        font-size: 0.84rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }
    .tiny-note {
        color: var(--muted);
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "engine" not in st.session_state:
    st.session_state.engine = GuardrailEngine()
if "guardrail_state" not in st.session_state:
    st.session_state.guardrail_state = SessionState()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

engine: GuardrailEngine = st.session_state.engine
state: SessionState = st.session_state.guardrail_state


def pill_for_stage(stage: str) -> str:
    if stage == "RED":
        return '<span class="risk-pill pill-red">RED · Block + Report</span>'
    if stage == "ORANGE":
        return '<span class="risk-pill pill-orange">ORANGE · Block + Safe Reply</span>'
    return '<span class="risk-pill pill-yellow">YELLOW · Safe Response</span>'


def stage_help(stage: str) -> str:
    if stage == "RED":
        return "고위험으로 판단되어 응답을 차단하고 보고서 생성을 수행합니다."
    if stage == "ORANGE":
        return "위험 가능성이 있어 구체 답변을 차단하고 안전 응답으로 전환합니다."
    return "저위험 또는 탐색 단계로 보고 안전 범위 내 답변을 제공합니다."


latest_turn = state.turns[-1] if state.turns else None
current_stage = latest_turn.stage if latest_turn else "YELLOW"

st.markdown(
    f"""
    <div class="hero-card">
        <div class="eyebrow">Streamlit AI Guardrail Demo</div>
        <div class="hero-title">한국어 프롬프트 위험도를 실시간 점수화하는<br/>YELLOW · ORANGE · RED 가드레일</div>
        <div class="hero-copy">
            기존 키워드 필터를 넘어 문장 임베딩 기반 의미 유사도, BERT 프로토타입 분류,
            GRU 스타일 세션 흐름 분석을 합쳐 한국어 위험 프롬프트를 탐지합니다.
        </div>
        <div style="margin-top: 0.9rem;">
            {pill_for_stage(current_stage)}
            <span class="risk-pill pill-yellow">Gemini Model · {get_gemini_model()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("실시간 채팅 시뮬레이션")
    st.caption("입력 즉시 위험도를 재계산하고 단계별 응답 정책을 적용합니다.")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    user_input = st.chat_input("한국어 문장을 입력하세요")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        reply, assessment = process_user_message(user_input, state, engine)
        st.session_state.chat_history.append(("assistant", reply))
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("단계별 정책")
    st.markdown(
        """
        `YELLOW`는 안전 응답을 허용합니다.  
        `ORANGE`는 구체 답변을 차단하고 안전한 대체 안내로 전환합니다.  
        `RED`는 응답 차단과 함께 PDF 보고서 생성을 트리거합니다.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("위험도 대시보드")
    st.markdown(f'<div class="metric-label">Current Stage</div>{pill_for_stage(current_stage)}', unsafe_allow_html=True)
    st.markdown(
        f'<div class="tiny-note" style="margin-top:0.4rem;">{stage_help(current_stage)}</div>',
        unsafe_allow_html=True,
    )

    current_score = latest_turn.final_risk if latest_turn else 0.0
    st.markdown('<div class="metric-label">Final Risk Score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{current_score:.2f} / 1.00</div>', unsafe_allow_html=True)
    st.progress(min(max(current_score, 0.0), 1.0))

    col1, col2 = st.columns(2)
    col1.metric("세션 누적 리스크", f"{state.session_risk:.2f}")
    col2.metric("누적 턴 수", f"{len(state.turns)}")

    if latest_turn:
        st.markdown("**신호 분해**")
        signal_df = pd.DataFrame(
            [
                ["Keyword", latest_turn.keyword_risk],
                ["Embedding", latest_turn.embedding_risk],
                ["BERT Prototype", latest_turn.classifier_risk],
                ["Temporal GRU", latest_turn.temporal_risk],
                ["Session Carryover", latest_turn.session_risk],
            ],
            columns=["Signal", "Score"],
        )
        st.dataframe(signal_df, use_container_width=True, hide_index=True)

        st.markdown("**상위 분류 레이블**")
        st.json(latest_turn.label_scores)

        if latest_turn.reasons:
            st.markdown("**판단 근거**")
            for reason in latest_turn.reasons:
                st.write(f"- {reason}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("모델 상태")
    st.json(engine.model_status)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("시스템 로그")
    st.text_area("logs", value="\n".join(state.system_logs[-18:]), height=220, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if state.report_path:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("보고서")
        with open(state.report_path, "rb") as file_handle:
            st.download_button(
                "PDF 보고서 다운로드",
                data=file_handle,
                file_name=state.report_path.split("/")[-1].split("\\")[-1],
                mime="application/pdf",
            )
        if state.email_sent:
            st.success("데모 이메일 전송이 완료되었습니다.")
        else:
            st.info("이메일은 전송되지 않았거나 환경변수가 설정되지 않았습니다.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.subheader("세션 이력")
if state.turns:
    history_df = pd.DataFrame(
        [
            {
                "time": turn.timestamp,
                "stage": turn.stage,
                "category": turn.top_category,
                "risk": round(turn.final_risk, 3),
                "action": turn.action,
                "summary": turn.user_input[:46],
            }
            for turn in reversed(state.turns[-10:])
        ]
    )
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("아직 입력이 없습니다.")

if st.button("세션 초기화"):
    st.session_state.guardrail_state = SessionState()
    st.session_state.chat_history = []
    st.session_state.engine = GuardrailEngine()
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
