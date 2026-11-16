"""음성 처리: 사용자 간 공유 파일 없이 메모리에서 처리합니다."""
import io
import wave


def transcribe(client, audio_bytes, language="ko"):
    if len(audio_bytes) > 24 * 1024 * 1024:
        raise ValueError("녹음이 너무 깁니다. 짧게 나누어 녹음해 주세요.")
    with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
        if audio.getnframes() / audio.getframerate() < 0.5:
            raise ValueError("녹음이 너무 짧습니다. 1초 이상 말씀해 주세요.")
        frames = audio.readframes(audio.getnframes())
        if not any(frames):
            raise ValueError("소리가 기록되지 않았습니다. 마이크를 확인해 주세요.")
    options = {"model": "whisper-1", "file": ("recording.wav", audio_bytes, "audio/wav")}
    if language:
        options["language"] = language
    result = client.audio.transcriptions.create(**options)
    text = result.text.strip()
    if not text:
        raise ValueError("말소리를 인식하지 못했습니다. 다시 녹음해 주세요.")
    return text


def answer(client, history, question, model):
    response = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": "질문에 친절하고 간결한 한국어로 답하세요."},
        *history[-20:],
        {"role": "user", "content": question},
    ])
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("빈 답변을 받았습니다.")
    return text.strip()


def synthesize(text):
    from gtts import gTTS
    buffer = io.BytesIO()
    gTTS(text=text, lang="ko", timeout=20).write_to_fp(buffer)
    return buffer.getvalue()
