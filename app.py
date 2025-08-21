import os
import json
import tempfile
import gradio as gr
from llama_cpp import Llama
from faster_whisper import WhisperModel
from TTS.api import TTS

# -----------------------------
# Auto-accept Coqui license
# -----------------------------
os.environ["COQUI_TOS_AGREED"] = "1"

# -----------------------------
# Model paths
# -----------------------------
LLM_PATH = "models/gguf_model.gguf"
TTS_MODEL_PATH = "models/TTS_model"
HISTORY_FILE = "history.json"

# -----------------------------
# Load models
# -----------------------------
llm = Llama(model_path=LLM_PATH)
asr = WhisperModel("small")  # small/medium/base/tiny for faster CPU
tts = TTS(TTS_MODEL_PATH, gpu=False)  # set gpu=True if CUDA

# -----------------------------
# Load chat history
# -----------------------------
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        chat_history = json.load(f)
else:
    chat_history = []

# -----------------------------
# Helper functions
# -----------------------------
def transcribe_audio(audio_file):
    if not audio_file:
        return ""
    segments, info = asr.transcribe(audio_file)
    return " ".join([seg.text for seg in segments])

def chat_with_ai(user_text, voice_file=None):
    global chat_history

    # If voice provided, transcribe
    if voice_file:
        user_text = transcribe_audio(voice_file)

    if not user_text.strip():
        return None, chat_history

    chat_history.append({"role": "user", "content": user_text})

    # Prepare prompt
    prompt = "\n".join([f"{x['role']}: {x['content']}" for x in chat_history]) + "\nassistant: "
    output = llm(prompt, max_tokens=256)
    response = output['choices'][0]['text'].strip()

    chat_history.append({"role": "assistant", "content": response})

    # Save history
    with open(HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=2)

    # Generate TTS audio
    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path, speaker_wav=None, language="en")

    return audio_path, chat_history

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Local AI Companion") as demo:
    gr.Markdown("# 🤖 Local AI Companion (Text + Voice)")
    gr.Markdown("Chat with a local LLM, use voice input, and get TTS output. History is saved automatically.")

    with gr.Row():
        text_input = gr.Textbox(label="Enter text", placeholder="Type something...", lines=3)
        voice_input = gr.Audio(label="Record / Upload voice", type="filepath", source="microphone")
        send_btn = gr.Button("Send")

    chatbox = gr.Chatbot()

    send_btn.click(
        fn=chat_with_ai,
        inputs=[text_input, voice_input],
        outputs=[gr.Audio(label="Assistant Speech"), chatbox]
    )

if __name__ == "__main__":
    demo.launch()
