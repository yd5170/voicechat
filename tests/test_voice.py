import io
import sys
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_service import answer, transcribe
from streamlit.testing.v1 import AppTest


def recording(duration=1, silent=False):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((b'\x00\x00' if silent else b'\x01\x00') * int(16000 * duration))
    return out.getvalue()


class VoiceRegressionTests(unittest.TestCase):
    def test_transcription_uses_original_wav_and_korean_hint(self):
        client = MagicMock()
        client.audio.transcriptions.create.return_value.text = ' 안녕하세요 '
        raw = recording()
        self.assertEqual(transcribe(client, raw), '안녕하세요')
        options = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(options['file'][1], raw)
        self.assertEqual(options['language'], 'ko')

    def test_empty_and_short_audio_do_not_call_paid_api(self):
        client = MagicMock()
        for raw in [recording(0.1), recording(silent=True)]:
            with self.assertRaises(ValueError):
                transcribe(client, raw)
        client.audio.transcriptions.create.assert_not_called()

    def test_answer_preserves_assistant_role(self):
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [SimpleNamespace(message=SimpleNamespace(content=' 답변 '))]
        history = [{'role': 'user', 'content': '이전 질문'}, {'role': 'assistant', 'content': '이전 답변'}]
        self.assertEqual(answer(client, history, '새 질문', 'test-model'), '답변')
        self.assertEqual(client.chat.completions.create.call_args.kwargs['messages'][2]['role'], 'assistant')

    def test_app_rerun_does_not_repeat_request_and_reset_allows_next_turn(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'ch03_voicebot.py')).run()
        self.assertEqual(len(app.exception), 0)
        app.sidebar.text_input[0].set_value('test-key').run()
        with patch('openai.OpenAI', return_value=MagicMock()), patch('voice_service.answer', return_value='테스트 답변') as generate, patch('voice_service.synthesize', return_value=b'test-audio'):
            app.text_area[0].set_value('첫 질문').run()
            next(b for b in app.button if b.label == '질문 전송').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state['messages']), 2)
            app.run()
            self.assertEqual(generate.call_count, 1)
            next(b for b in app.button if b.label == '대화 초기화').click().run()
            self.assertEqual(app.session_state['messages'], [])
            app.text_area[0].set_value('두 번째 질문').run()
            next(b for b in app.button if b.label == '질문 전송').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(app.session_state['messages'][1]['role'], 'assistant')


if __name__ == '__main__':
    unittest.main()
