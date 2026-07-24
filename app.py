# app.py — AI 공감 다이어리 (Streamlit)
# 오늘 있었던 일을 한 줄로 쓰면, OpenAI(gpt-4o-mini)가 감정을 분석하고 공감·위로해 준다.
# 원본 Node 프로토타입(node-prototype/src/openai.js)의 프롬프트/8범주/폴백 로직을 그대로 옮겼다.
# API 키는 절대 코드/저장소에 두지 않고 st.secrets["OPENAI_API_KEY"] 또는 환경변수에서만 읽는다.

import html
import json
import os
from datetime import datetime

import streamlit as st

# 로컬 개발 편의: .env 가 있으면 로드(있을 때만). 배포(Streamlit Cloud)에서는 st.secrets 사용.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from openai import OpenAI

# --------------------------------------------------------------------------
# 상수 · 감정 정의 (PRD 6장 고정 8범주)
# --------------------------------------------------------------------------
MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 15  # 초

EMOTIONS = ["기쁨", "설렘", "평온", "슬픔", "불안", "분노", "지침", "기타"]

# 감정 → (이모지, 배경색, 글자색) — 밝은 배경 + 어두운 글자로 대비(WCAG AA) 확보
EMOTION_STYLE = {
    "기쁨": ("😊", "#fdf3c7", "#8a6d0b"),
    "설렘": ("🥰", "#fbe1ea", "#a83b63"),
    "평온": ("🙂", "#d9f2e6", "#2f7a5a"),
    "슬픔": ("😢", "#dbe8fb", "#37588f"),
    "불안": ("😟", "#e7e0f5", "#5a4a8a"),
    "분노": ("😠", "#fbdcd7", "#a8402f"),
    "지침": ("😮‍💨", "#e6e3df", "#5f574d"),
    "기타": ("🌫️", "#eee9e1", "#6a635b"),
}

FALLBACK_EMPATHY = (
    "지금 이 순간의 마음을 이렇게 적어주셔서 고마워요. "
    "어떤 하루였든 당신의 감정은 소중합니다. 오늘도 충분히 잘 지내셨어요."
)

SYSTEM_PROMPT = """당신은 따뜻하고 공감 능력이 뛰어난 감정 코치입니다.
사용자가 오늘 있었던 일을 한 줄로 적으면, 그 감정을 읽고 위로해 줍니다.

반드시 지켜야 할 규칙:
1. 대표 감정을 다음 8가지 중 정확히 하나로만 분류합니다: 기쁨, 설렘, 평온, 슬픔, 불안, 분노, 지침, 기타.
   - 위 목록에 없는 단어나 변형은 절대 사용하지 마세요.
2. 공감·위로 메시지(empathy)는 2~4문장의 한국어로 작성합니다.
   - 판단하거나 훈계하지 말고, 사용자의 감정을 있는 그대로 인정하고 따뜻하게 다독여 주세요.
   - 진부한 표현을 피하고 사용자의 구체적인 상황에 반응하세요.
3. 반드시 아래 JSON 형식으로만 답하세요. 그 외의 텍스트, 설명, 마크다운은 절대 출력하지 마세요.

출력 형식:
{"emotion": "<8가지 중 하나>", "empathy": "<2~4문장의 공감 메시지>"}"""


# --------------------------------------------------------------------------
# API 키 · OpenAI 클라이언트
# --------------------------------------------------------------------------
def get_api_key():
    """st.secrets → 환경변수 순으로 키를 찾는다. 키 값은 화면/로그에 노출하지 않는다."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)


def parse_response(raw: str):
    """모델 응답(JSON 문자열)을 파싱하고 감정 범주를 검증한다."""
    data = json.loads(raw)
    emotion = (data.get("emotion") or "").strip()
    empathy = (data.get("empathy") or "").strip()
    if not empathy:
        raise ValueError("empathy 필드가 비어 있음")
    if emotion not in EMOTIONS:
        emotion = "기타"
    return emotion, empathy


def analyze_entry(content: str):
    """한 줄 일기를 분석해 dict(emotion, empathy, fallback, notice)를 반환.
    실패해도 예외를 던지지 않고 안전한 폴백을 돌려준다."""
    api_key = get_api_key()
    if not api_key:
        return {
            "emotion": "기타",
            "empathy": FALLBACK_EMPATHY,
            "fallback": True,
            "notice": "OPENAI_API_KEY가 설정되지 않아 기본 위로 메시지를 보여드려요.",
        }

    client = get_client(api_key)
    last_error = None
    for attempt in range(2):  # 최초 + 재시도 1회
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0.8,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
            )
            raw = completion.choices[0].message.content or ""
            emotion, empathy = parse_response(raw)
            return {"emotion": emotion, "empathy": empathy, "fallback": False}
        except Exception as err:  # noqa: BLE001
            last_error = err
            status = getattr(err, "status_code", None) or getattr(err, "status", None)
            if status in (401, 403):
                break  # 키 무효/권한 없음: 재시도해도 소용없음

    return {
        "emotion": "기타",
        "empathy": FALLBACK_EMPATHY,
        "fallback": True,
        "notice": "AI 응답을 불러오지 못해 기본 위로 메시지를 보여드려요. 잠시 후 다시 시도해 주세요.",
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI 공감 다이어리", page_icon="🌿", layout="centered")

CUSTOM_CSS = """
<style>
:root { --accent:#e0876b; --accent-strong:#cf6f52; --text:#3a352f; --muted:#9a9088; }
.stApp {
  background:
    radial-gradient(1200px 600px at 15% -10%, #f7ede0 0%, transparent 55%),
    radial-gradient(1000px 500px at 110% 10%, #f3eef6 0%, transparent 50%),
    #fbf6ee;
}
.block-container { max-width: 680px; padding-top: 2.2rem; }
.diary-title { font-size: 2rem; font-weight: 800; color: var(--text); margin: 0; letter-spacing: -.5px; }
.diary-sub { color: var(--muted); margin: .3rem 0 1.4rem; font-size: .98rem; }
.stTextArea textarea {
  background: #fffdf9; border: 1px solid #ece3d6 !important; border-radius: 16px !important;
  font-size: 1.02rem; color: var(--text); box-shadow: 0 6px 18px rgba(160,120,90,.06);
}
div.stButton > button {
  background: var(--accent); color: #fff; border: none; border-radius: 14px;
  padding: .55rem 1.4rem; font-weight: 700; box-shadow: 0 6px 16px rgba(224,135,107,.35);
  transition: transform .12s ease, background .12s ease;
}
div.stButton > button:hover { background: var(--accent-strong); transform: translateY(-1px); }
.card {
  background: #fffdf9; border-radius: 18px; padding: 1.15rem 1.25rem; margin: .7rem 0;
  border: 1px solid #f2ebe0; box-shadow: 0 8px 22px rgba(160,120,90,.08);
  border-left: 6px solid var(--stripe, #ece3d6);
}
.badge {
  display: inline-block; padding: .22rem .7rem; border-radius: 999px;
  font-weight: 700; font-size: .9rem; margin-bottom: .55rem;
}
.card .orig { color: var(--text); font-size: 1.05rem; margin: .1rem 0 .7rem; line-height: 1.55; }
.card .orig::before { content: "“"; color: var(--muted); }
.card .orig::after { content: "”"; color: var(--muted); }
.card .empathy-label { color: var(--accent-strong); font-weight: 700; font-size: .86rem; margin-bottom: .2rem; }
.card .empathy { color: #57514a; line-height: 1.65; }
.card .date { color: var(--muted); font-size: .8rem; margin-top: .1rem; }
.card .notice { color: #a06a4a; font-size: .82rem; margin-top: .5rem; }
.empty { text-align:center; color: var(--muted); padding: 2.4rem 1rem;
  border: 1.5px dashed #e3d8c8; border-radius: 18px; background: rgba(255,253,249,.5); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---- 세션 상태 ----
# Streamlit Cloud는 파일시스템이 임시(재시작 시 초기화)라 히스토리는 세션 메모리에 보관한다.
if "history" not in st.session_state:
    st.session_state.history = []
if "next_id" not in st.session_state:
    st.session_state.next_id = 1
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def emotion_badge_html(emotion: str) -> str:
    emoji, bg, fg = EMOTION_STYLE.get(emotion, EMOTION_STYLE["기타"])
    return f'<span class="badge" style="background:{bg};color:{fg}">{emoji} {html.escape(emotion)}</span>'


def card_html(entry: dict, show_date: bool = True) -> str:
    _, _, fg = EMOTION_STYLE.get(entry["emotion"], EMOTION_STYLE["기타"])
    date_html = (
        f'<div class="date">{html.escape(entry["created_at"])}</div>' if show_date else ""
    )
    notice_html = (
        f'<div class="notice">ℹ️ {html.escape(entry["notice"])}</div>'
        if entry.get("notice")
        else ""
    )
    # 사용자 원문·AI 응답은 반드시 escape 하여 HTML 주입(XSS)을 막는다.
    return (
        f'<div class="card" style="--stripe:{fg}">'
        f"{emotion_badge_html(entry['emotion'])}"
        f'<div class="orig">{html.escape(entry["content"])}</div>'
        f'<div class="empathy-label">💌 마음을 담은 한마디</div>'
        f'<div class="empathy">{html.escape(entry["empathy"])}</div>'
        f"{notice_html}{date_html}"
        f"</div>"
    )


# ---- 헤더 ----
st.markdown('<div class="diary-title">🌿 AI 공감 다이어리</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="diary-sub">오늘 있었던 일을 한 줄로 남겨보세요. 마음을 읽어드릴게요.</div>',
    unsafe_allow_html=True,
)

# 키 미설정 안내(값은 노출하지 않음)
if not get_api_key():
    st.warning(
        "OpenAI API 키가 설정되지 않았어요. 로컬은 `.streamlit/secrets.toml` 또는 `.env`, "
        "배포는 Streamlit Cloud의 **Secrets**에 `OPENAI_API_KEY`를 넣어주세요. "
        "지금은 기본 위로 메시지로 동작합니다."
    )

# ---- 작성 폼 ----
with st.form("entry_form", clear_on_submit=True):
    content = st.text_area(
        "오늘 하루, 한 줄로 남긴다면?",
        max_chars=300,
        height=90,
        placeholder="예) 오랜만에 친구를 만나서 마음이 따뜻해졌다.",
    )
    submitted = st.form_submit_button("기록하기")

if submitted:
    text = (content or "").strip()
    if not text:
        st.error("한 줄이라도 적어 주세요.")
    else:
        with st.spinner("마음을 읽는 중..."):
            result = analyze_entry(text)
        entry = {
            "id": st.session_state.next_id,
            "content": text,
            "emotion": result["emotion"],
            "empathy": result["empathy"],
            "notice": result.get("notice"),
            "created_at": datetime.now().strftime("%Y.%m.%d %H:%M"),
        }
        st.session_state.next_id += 1
        st.session_state.history.insert(0, entry)  # 최신순
        st.session_state.last_result = entry

# ---- 최근 결과 ----
if st.session_state.last_result is not None:
    st.markdown("#### 오늘의 공감")
    st.markdown(card_html(st.session_state.last_result, show_date=False), unsafe_allow_html=True)

# ---- 히스토리 ----
st.markdown("---")
st.markdown("### 📖 지난 기록")

if not st.session_state.history:
    st.markdown(
        '<div class="empty">아직 남긴 일기가 없어요.<br>첫 한 줄을 기록해보는 건 어때요?</div>',
        unsafe_allow_html=True,
    )
else:
    for entry in st.session_state.history:
        st.markdown(card_html(entry), unsafe_allow_html=True)
        if st.button("🗑️ 삭제", key=f"del-{entry['id']}"):
            st.session_state.history = [
                e for e in st.session_state.history if e["id"] != entry["id"]
            ]
            if (
                st.session_state.last_result
                and st.session_state.last_result["id"] == entry["id"]
            ):
                st.session_state.last_result = None
            st.rerun()

st.caption("💡 배포 환경(Streamlit Cloud)에서는 앱이 재시작되면 기록이 초기화됩니다.")
