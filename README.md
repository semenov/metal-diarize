# crosstalk

**Turn any recording into a transcript that knows who said what — fast, on your Mac.**

`crosstalk` transcribes audio *and* labels each speaker, so an interview, podcast, or
panel becomes a clean, readable script. It runs entirely on your machine and is built
to be quick on Apple Silicon.

![crosstalk demo](demo.gif)

---

## Quick start

```bash
brew tap semenov/crosstalk
brew trust semenov/crosstalk      # one-time — see "Trusting the tap" below
brew install crosstalk

export HF_TOKEN=hf_your_token     # free token — see "Hugging Face token" below

crosstalk interview.mp3 --speakers 2
```

That's it. You get `interview.crosstalk.txt`:

```
[00:00] SPEAKER_01: So explain to me, in practice — how does your system work?

[00:11] SPEAKER_00: You studied organic chemistry, right? A molecule can be
        written as a linear string — a SMILES string.
```

`brew install` is instant. The first time you actually transcribe something, `crosstalk`
does a one-time setup (downloads the speech + speaker models, ~2 GB) and caches them —
every run after that is fast.

---

## Why it exists

The popular tool for diarized transcripts, **WhisperX**, can't use the GPU on a Mac —
its speech engine runs on the CPU, so a 30-minute file takes 20+ minutes. Meanwhile
**whisper.cpp** transcribes that same file in ~2 minutes on Apple's Metal GPU, but has
no speaker labels.

`crosstalk` stitches the fast half of each together:

```
whisper.cpp (Metal GPU)            pyannote (Apple GPU)
   transcript + timestamps    +      speaker turns
                └──── merged by who's talking when ────┘
                     speaker-labeled transcript
```

**Benchmark** — 31-minute, 4-speaker panel discussion on an M-series MacBook:

| Tool | Time |
|---|---|
| WhisperX (CPU) | 21+ min (never finished) |
| **crosstalk** | **~3 min** |

---

## Trusting the tap

The first time you tap `crosstalk`, Homebrew asks you to *trust* it:

```bash
brew trust semenov/crosstalk
```

**Why?** Homebrew treats third-party taps as untrusted by default — a formula is just
code, and running `brew install` from a stranger's tap runs their install script. The
`brew trust` step is you saying "yes, I've chosen to install from this tap." You only
do it once per tap, and you can inspect exactly what you're trusting first:

```bash
# see the formula before trusting it
brew cat semenov/crosstalk/crosstalk
```

**Can I make it trusted for everyone, with no prompt?** Not from my side — trust is a
decision each user makes on their own machine, by design. The only way to get the fully
prompt-free `brew install crosstalk` (no tap, no trust) is to land the formula in
Homebrew's central **homebrew-core**, which requires the project to first build up some
notability (stars, real usage). Until then, the one-time `brew tap` + `brew trust` is
the normal, expected flow for any independent tap.

---

## Hugging Face token

Speaker separation uses a model that's free but gated behind a quick sign-up. One time:

1. **Accept the license** — click *Agree* at
   [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
2. **Create a token** — a *Read* token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Make it available** — either:
   ```bash
   export HF_TOKEN=hf_your_token     # add to ~/.zshrc to make it permanent
   ```
   or save it to a file crosstalk checks automatically:
   ```bash
   echo -n hf_your_token > ~/.hf_token && chmod 600 ~/.hf_token
   ```

---

## Usage

```bash
crosstalk interview.mp3                 # any audio or video file
crosstalk debate.wav --speakers 2       # tell it the exact speaker count (more accurate)
crosstalk talk.mp4 --language ru        # non-English audio
crosstalk episode.m4a -o episode.txt    # choose the output filename
crosstalk long.mp3 --model medium       # smaller model = faster, slightly less accurate
```

| Option | What it does |
|---|---|
| `--speakers N` | Exact number of speakers, if you know it |
| `--min-speakers` / `--max-speakers` | Bounds when you don't know the exact count |
| `--language XX` | Audio language code (default `en`) |
| `--model NAME` | `large-v3-turbo` (default), `large-v3`, `medium`, `small`, `base` — or a path to your own ggml `.bin`. Named models auto-download and cache. |
| `-o FILE` | Output path (default `<input>.crosstalk.txt`) |

Speakers come out labeled `SPEAKER_00`, `SPEAKER_01`, … — anonymous, but who's who is
usually obvious from the first few lines. (Diarization tells you *how many* distinct
voices and *when* each talks; putting real names on them is a quick find-and-replace.)

---

## Install without Homebrew

Prefer `pipx`? You just need the two system tools first:

```bash
brew install ffmpeg whisper-cpp
pipx install git+https://github.com/semenov/crosstalk
```

---

## Record your own demo

The GIF above was made with [asciinema](https://asciinema.org) +
[agg](https://github.com/asciinema/agg). To capture a real run on your machine:

```bash
brew install asciinema agg
asciinema rec demo.cast -c "crosstalk interview.mp3 --speakers 2"
agg --theme monokai demo.cast demo.gif
```

---

## Limitations

- Speaker labels are per-segment, not per-word — great for interviews and panels,
  less precise on rapid crosstalk where people talk over each other.
- Separation quality is only as good as the underlying model (pyannote): excellent on
  2–4 clear voices, harder on big panels or heavy overlap.
- Built for macOS / Apple Silicon. It runs elsewhere (falling back to CPU), but the
  speed is the whole point.

---

## Built on

Standing on the shoulders of [whisper.cpp](https://github.com/ggerganov/whisper.cpp),
[pyannote.audio](https://github.com/pyannote/pyannote-audio), and the approach pioneered
by [WhisperX](https://github.com/m-bain/whisperX). `crosstalk` is the glue that makes
them fast together on a Mac.

## License

MIT — see [LICENSE](LICENSE).
