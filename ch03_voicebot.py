import hashlib
import os

import openai
import streamlit as st

from voice_service import answer, synthesize, transcribe


def main():
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")
    st.header("조예원's 음성 비서 프로그램")
    st.caption("녹음 → 음성 인식 → 인식 내용 확인 → 질문 전송")

    for key, value in {"messages": [], "recorder_id": 0, "recording_hash": None,
                       "transcript": "", "reply_audio": None}.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        api_key = st.text_input("OpenAI API 키", type="password").strip()
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except st.errors.StreamlitSecretNotFoundError:
                pass
        model = st.text_input("답변 모델", value="gpt-4o-mini").strip()
        language = st.selectbox("녹음 언어", ["한국어", "영어", "자동 감지"])
        language_code = {"한국어": "ko", "영어": "en", "자동 감지": None}[language]
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.session_state.recorder_id += 1
            st.session_state.recording_hash = None
            st.session_state.transcript = ""
            st.session_state.reply_audio = None
            st.rerun()

    left, right = st.columns(2)
    with left:
        st.subheader("질문하기")
        recording = st.audio_input("녹음 후 정지 버튼을 누르세요", sample_rate=16000,
                                   key=f"recording_{st.session_state.recorder_id}")
        if recording is not None:
            audio_bytes = recording.getvalue()
            digest = hashlib.sha256(audio_bytes).hexdigest()
            if digest != st.session_state.recording_hash:
                st.session_state.recording_hash = digest
                st.session_state.transcript = ""
            if st.button("음성 인식", disabled=not api_key):
                try:
                    with st.spinner("음성을 인식하고 있습니다..."):
                        client = openai.OpenAI(api_key=api_key, timeout=45, max_retries=1)
                        st.session_state.transcript = transcribe(client, audio_bytes, language_code)
                except ValueError as error:
                    st.warning(str(error))
                except openai.AuthenticationError:
                    st.error("API 키를 확인해 주세요.")
                except openai.RateLimitError:
                    st.error("API 사용 한도 또는 잔액을 확인하고 잠시 후 다시 시도해 주세요.")
                except openai.OpenAIError:
                    st.error("음성 인식 요청에 실패했습니다. 연결 상태와 API 설정을 확인해 주세요.")

        st.text_area("인식된 질문 (수정하거나 직접 입력할 수 있습니다)", key="transcript")
        if not api_key:
            st.info("사이드바에 API 키를 입력해 주세요.")
        if st.button("질문 전송", disabled=not api_key or not model):
            text = st.session_state.transcript.strip()
            if not text:
                st.warning("음성을 인식하거나 질문을 입력해 주세요.")
            else:
                try:
                    with st.spinner("답변을 작성하고 있습니다..."):
                        client = openai.OpenAI(api_key=api_key, timeout=45, max_retries=1)
                        reply = answer(client, st.session_state.messages, text, model)
                    st.session_state.messages.extend([
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": reply},
                    ])
                    st.session_state.reply_audio = None
                    try:
                        st.session_state.reply_audio = synthesize(reply)
                    except Exception:
                        st.warning("음성 재생 파일을 만들지 못했습니다. 아래 답변을 확인해 주세요.")
                except openai.AuthenticationError:
                    st.error("API 키를 확인해 주세요.")
                except openai.RateLimitError:
                    st.error("API 사용 한도 또는 잔액을 확인해 주세요.")
                except (openai.OpenAIError, ValueError):
                    st.error("답변 요청에 실패했습니다. 모델 접근 권한과 연결 상태를 확인해 주세요.")

    with right:
        st.subheader("질문 / 답변")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if st.session_state.reply_audio:
            st.audio(st.session_state.reply_audio, format="audio/mp3")


if __name__ == "__main__":
    main()
