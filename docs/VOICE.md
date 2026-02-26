# Voice Input/Output (Local)

This project supports **local** voice input (STT) and voice output (TTS) across platforms.

## Voice Input (STT)

We record audio locally and transcribe with a local Whisper engine. You must install:

```
pip install sounddevice numpy
```

And **one** of the following:

```
pip install faster-whisper
```

or

```
pip install openai-whisper
```

### CLI

```
uv run ollamacode --voice-in --voice-seconds 5 --voice-model base "ignored"
```

### TUI

```
/listen 5
```

Push‑to‑talk hotkey (default `Ctrl+Space`):

```
OLLAMACODE_TUI_PTT_KEY=c-space uv run ollamacode
```

## Voice Output (TTS)

### macOS
Uses built‑in `say`.

### Linux
Install either:

```
sudo apt-get install espeak
```

or:

```
sudo apt-get install speech-dispatcher
```

### Windows
Uses built‑in PowerShell System.Speech.

### CLI

```
uv run ollamacode --voice-out "Say this"
```

### TUI

```
/say hello
```

Enable auto‑speak:

```
OLLAMACODE_TUI_VOICE_OUT=1 uv run ollamacode
```
