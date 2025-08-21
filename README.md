# 🤖 Local AI Companion

A fully offline AI assistant in Python, inspired by Ollama.

**Features:**
- Chat with a **local LLM** (LLaMA/GGUF model)
- **Voice input** using Whisper ASR
- **Voice output** using Coqui TTS
- Conversation history saved locally
- Gradio-based web interface for easy interaction
- Supports **GPU acceleration** if available
- Fully offline operation, preserving privacy

---

## Requirements

- Python 3.10+
- PyTorch (CPU or GPU)
- Models:
  - GGUF/ggml LLaMA model in `models/gguf_model.gguf`
  - Coqui TTS model in `models/TTS_model/`  

### Install Dependencies
```bash
pip install -r requirements.txt
