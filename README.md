# crosstalk

**Speaker-diarized transcription that's actually fast on Apple Silicon.**

WhisperX is the standard tool for diarized transcripts, but on a Mac its transcription
backend (CTranslate2) can't use the GPU — a 30-minute file takes 20+ minutes on CPU.
Meanwhile whisper.cpp transcribes the same file in ~2 minutes on Metal, but has no
diarization.

This tool glues the fast halves of both worlds together:

```
whisper.cpp (Metal GPU)          pyannote community-1 (MPS GPU)
   transcript + timestamps    +     speaker turns
                └──── merge by time overlap ────┘
                   speaker-labeled transcript
```

**Benchmark** (M-series MacBook, 31-minute panel discussion, 4 speakers):

| Pipeline | Time |
|---|---|
| WhisperX (large-v3, CPU) | 21+ min (killed before finishing) |
| **crosstalk** (large-v3-turbo Metal + pyannote MPS) | **~3 min** |

Output looks like:

```
[00:00] SPEAKER_03: Welcome everybody, and welcome to those of you joining us on livestream...

[01:13] SPEAKER_00: It's always hard to know exactly when something will happen, but...
```

## Install

### Homebrew (recommended)

```bash
brew tap semenov/crosstalk
brew install crosstalk
```

This pulls in `ffmpeg` and `whisper-cpp` automatically and installs the
`crosstalk` command. The whisper model is downloaded on first use and
cached in `~/Library/Caches/crosstalk/` — no manual model setup.

### pipx (alternative)

```bash
brew install ffmpeg whisper-cpp        # system deps
pipx install git+https://github.com/semenov/crosstalk
```

### Hugging Face token (one-time, free)

The diarization model is gated. Once:

1. Accept the license at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. `export HF_TOKEN=hf_...` (or save it to `~/.hf_token`)

## Use

```bash
crosstalk interview.mp3                 # any ffmpeg-readable format
crosstalk debate.wav --speakers 2      # exact speaker count, if known
crosstalk talk.mp4 --language ru        # non-English audio
crosstalk ep1.m4a --model medium -o ep1.txt   # smaller/faster model
```

`--model` accepts a name (`large-v3-turbo` default, `large-v3`, `medium`, `small`,
`base` — auto-downloaded and cached) or a path to your own ggml `.bin`.

Output: `<input>.crosstalk.txt` with `[MM:SS] SPEAKER_XX:` turn labels.
Speaker labels are anonymous (`SPEAKER_00`, `SPEAKER_01`, ...) — mapping them to real
names is up to you (usually obvious from the first few turns).

## How it works

1. **Transcribe** — `whisper-cli` (whisper.cpp) runs on the Metal GPU, emits JSON
   segments with millisecond offsets.
2. **Diarize** — `pyannote/speaker-diarization-community-1` clusters "who spoke when."
   Runs on Apple's GPU via PyTorch MPS (with CPU fallback). The WAV is loaded manually
   and passed as a raw waveform tensor, avoiding the `torchcodec` dependency.
3. **Merge** — each transcript segment gets the speaker whose diarization turns
   overlap it most in time. Same assignment logic WhisperX uses, without its slow
   transcription front end.

## Limitations

- Segment-level (not word-level) speaker assignment. Fine for interviews and panels;
  very rapid crosstalk can blur at turn boundaries.
- Diarization quality is pyannote's — great on 2–4 clear voices, harder on large
  panels or heavy overlap.
- macOS/Apple Silicon focus. It runs elsewhere (CPU fallback), but the speedup is
  the point.

## License

MIT
