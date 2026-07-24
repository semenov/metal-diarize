#!/usr/bin/env python3
"""fast-whisper-diarize: speaker-diarized transcription, fast on Apple Silicon.

Hybrid pipeline:
  1. whisper.cpp (Metal GPU)      -> transcript segments with timestamps
  2. pyannote community-1 (MPS)   -> speaker turns
  3. merge by timestamp overlap   -> speaker-labeled transcript

Usage:
  python fast_diarize.py interview.mp3 --speakers 2
  python fast_diarize.py talk.wav --model ~/models/ggml-large-v3-turbo.bin --language ru
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def read_hf_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    token_file = Path.home() / ".hf_token"
    if token_file.exists():
        return token_file.read_text().strip()
    die(
        "no Hugging Face token found. Diarization needs one (free):\n"
        "  1. accept the license at huggingface.co/pyannote/speaker-diarization-community-1\n"
        "  2. create a Read token at huggingface.co/settings/tokens\n"
        "  3. export HF_TOKEN=hf_... (or save it to ~/.hf_token, or pass --hf-token)"
    )


def to_wav16k(src: Path, tmpdir: Path) -> Path:
    """Convert any audio/video input to 16 kHz mono PCM WAV via ffmpeg."""
    out = tmpdir / "audio16k.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        die(f"ffmpeg failed:\n{r.stderr[-2000:]}")
    return out


def transcribe(wav: Path, model: Path, language: str, tmpdir: Path) -> list[tuple[float, float, str]]:
    """Run whisper.cpp (Metal) and return [(start_s, end_s, text), ...]."""
    base = tmpdir / "transcript"
    r = subprocess.run(
        ["whisper-cli", "-m", str(model), "-l", language, "-f", str(wav),
         "-oj", "-of", str(base), "-np"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        die(f"whisper-cli failed:\n{r.stderr[-2000:]}")
    with open(f"{base}.json") as f:
        data = json.load(f)
    segments = []
    for s in data["transcription"]:
        text = s["text"].strip()
        if text:
            segments.append(
                (s["offsets"]["from"] / 1000.0, s["offsets"]["to"] / 1000.0, text)
            )
    return segments


def diarize(wav: Path, token: str, min_speakers: int, max_speakers: int) -> list[tuple[float, float, str]]:
    """Run pyannote speaker diarization (MPS if available) and return [(start, end, speaker), ...]."""
    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)

    device = "cpu"
    if torch.backends.mps.is_available():
        try:
            pipe.to(torch.device("mps"))
            device = "mps"
        except Exception:
            pass
    print(f"diarization running on: {device}")

    # Load waveform manually — avoids the torchcodec dependency
    with wave.open(str(wav)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    waveform = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)

    diar = pipe(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=min_speakers, max_speakers=max_speakers,
    )
    ann = getattr(diar, "speaker_diarization", diar)  # pyannote 4.x wraps the Annotation
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]


def merge(segments, turns) -> str:
    """Assign each transcript segment the speaker with maximal time overlap."""
    def best_speaker(s, e):
        overlap = {}
        for ts, te, spk in turns:
            ov = min(e, te) - max(s, ts)
            if ov > 0:
                overlap[spk] = overlap.get(spk, 0) + ov
        return max(overlap, key=overlap.get) if overlap else "UNKNOWN"

    def fmt(sec):
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    lines, prev = [], None
    for start, end, text in segments:
        spk = best_speaker(start, end)
        if spk != prev:
            lines.append(f"\n[{fmt(start)}] {spk}:")
            prev = spk
        lines.append(text)
    return " ".join(lines).replace("\n ", "\n").strip() + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Speaker-diarized transcription, fast on Apple Silicon")
    p.add_argument("audio", type=Path, help="input audio/video file (any ffmpeg-readable format)")
    p.add_argument("--model", type=Path, default=Path("ggml-large-v3-turbo.bin"),
                   help="path to whisper.cpp ggml model (default: ./ggml-large-v3-turbo.bin)")
    p.add_argument("--language", default="en", help="audio language code (default: en)")
    p.add_argument("--speakers", type=int, default=None,
                   help="exact number of speakers, if known")
    p.add_argument("--min-speakers", type=int, default=1)
    p.add_argument("--max-speakers", type=int, default=8)
    p.add_argument("--hf-token", default=None, help="Hugging Face token (default: $HF_TOKEN or ~/.hf_token)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output file (default: <input>.diarized.txt)")
    args = p.parse_args()

    if not args.audio.exists():
        die(f"input not found: {args.audio}")
    if not args.model.exists():
        die(f"whisper model not found: {args.model}\n"
            "download one, e.g.:\n"
            "  curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin")
    for tool in ("ffmpeg", "whisper-cli"):
        if not shutil.which(tool):
            die(f"{tool} not found — install with: brew install ffmpeg whisper-cpp")

    if args.speakers:
        args.min_speakers = args.max_speakers = args.speakers
    token = read_hf_token(args.hf_token)
    out = args.output or args.audio.with_suffix(".diarized.txt")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        wav = to_wav16k(args.audio, tmpdir)
        print(f"[{time.time()-t0:5.1f}s] converted to 16 kHz WAV")
        segments = transcribe(wav, args.model, args.language, tmpdir)
        print(f"[{time.time()-t0:5.1f}s] transcribed: {len(segments)} segments (whisper.cpp / Metal)")
        turns = diarize(wav, token, args.min_speakers, args.max_speakers)
        speakers = sorted({t[2] for t in turns})
        print(f"[{time.time()-t0:5.1f}s] diarized: {len(turns)} turns, {len(speakers)} speakers")
    out.write_text(merge(segments, turns))
    print(f"[{time.time()-t0:5.1f}s] wrote {out}")


if __name__ == "__main__":
    main()
