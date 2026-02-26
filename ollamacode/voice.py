"""
Local voice input/output utilities.

Voice input:
- records WAV locally (prefers sounddevice; otherwise raises a helpful error)
- transcribes with faster-whisper or openai-whisper if installed

Voice output:
- macOS: say
- Linux: espeak or spd-say
- Windows: PowerShell System.Speech
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import wave
import audioop
from pathlib import Path
from typing import Any


class VoiceError(RuntimeError):
    pass


def _write_wav(path: str, data: bytes, sample_rate: int, channels: int = 1) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def record_wav(
    path: str,
    *,
    seconds: float = 5.0,
    sample_rate: int = 16000,
    device: int | None = None,
    meter_cb: callable | None = None,
) -> str:
    """Record audio to WAV. Requires sounddevice (pip install sounddevice)."""
    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
    except Exception as e:
        raise VoiceError(
            "Voice input requires the optional dependency 'sounddevice' (and numpy). "
            "Install: pip install sounddevice numpy"
        ) from e

    frames = int(seconds * sample_rate)
    chunks: list[bytes] = []
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=device,
        ) as stream:
            remaining = frames
            while remaining > 0:
                block = min(1024, remaining)
                data, _ = stream.read(block)
                raw = data.tobytes()
                chunks.append(raw)
                if meter_cb is not None:
                    level = audioop.rms(raw, 2) / 32768.0
                    meter_cb(level)
                remaining -= block
    except Exception as e:
        raise VoiceError(f"Audio recording failed: {e}") from e
    data = b"".join(chunks)
    _write_wav(path, data, sample_rate, channels=1)
    return path


def _transcribe_with_faster_whisper(path: str, model: str) -> str | None:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None
    try:
        device = "cpu"
        compute_type = "int8"
        m = WhisperModel(model, device=device, compute_type=compute_type)
        segments, _info = m.transcribe(path)
        return "".join(seg.text for seg in segments).strip()
    except Exception as e:
        raise VoiceError(f"faster-whisper failed: {e}") from e


def _transcribe_with_openai_whisper(path: str, model: str) -> str | None:
    try:
        import whisper  # type: ignore
    except Exception:
        return None
    try:
        m = whisper.load_model(model)
        result: dict[str, Any] = m.transcribe(path)
        text = result.get("text") if isinstance(result, dict) else None
        return (text or "").strip()
    except Exception as e:
        raise VoiceError(f"openai-whisper failed: {e}") from e


def transcribe_wav(path: str, model: str = "base") -> str:
    """Transcribe WAV with faster-whisper or openai-whisper (local)."""
    text = _transcribe_with_faster_whisper(path, model)
    if text is not None:
        return text
    text = _transcribe_with_openai_whisper(path, model)
    if text is not None:
        return text
    raise VoiceError(
        "No local whisper engine available. Install one of:\n"
        "  pip install faster-whisper\n"
        "  pip install openai-whisper"
    )


def record_and_transcribe(
    *,
    seconds: float = 5.0,
    sample_rate: int = 16000,
    model: str = "base",
    meter_cb: callable | None = None,
) -> str:
    """Record from mic then transcribe; returns text."""
    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "input.wav")
        record_wav(wav_path, seconds=seconds, sample_rate=sample_rate, meter_cb=meter_cb)
        return transcribe_wav(wav_path, model=model)


def speak_text(text: str, *, voice: str | None = None, rate: int | None = None) -> None:
    """Speak text using local OS tools."""
    if not text:
        return
    system = platform.system().lower()
    if "darwin" in system:
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(rate)]
        cmd.append(text)
        subprocess.run(cmd, check=False)
        return
    if "windows" in system:
        # PowerShell System.Speech
        rate_expr = f"$s.Rate={rate};" if rate is not None else ""
        voice_expr = f'$s.SelectVoice("{voice}");' if voice else ""
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"{voice_expr}{rate_expr}"
            f'$s.Speak("{text.replace("\"", "\\\"")}");'
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
        )
        return
    # Linux / other
    if shutil.which("espeak"):
        cmd = ["espeak"]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-s", str(rate)]
        cmd.append(text)
        subprocess.run(cmd, check=False)
        return
    if shutil.which("spd-say"):
        cmd = ["spd-say"]
        if voice:
            cmd += ["-v", voice]
        if rate:
            cmd += ["-r", str(rate)]
        cmd.append(text)
        subprocess.run(cmd, check=False)
        return
    raise VoiceError(
        "No local TTS engine found. Install espeak/spd-say (Linux) or use macOS/Windows built-ins."
    )
