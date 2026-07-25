"""Generates demo.cast, the source for the README's demo.gif.

Regenerate the GIF after editing:
    python3 gen_cast.py
    agg --theme monokai --font-size 26 --cols 92 --rows 20 demo.cast demo.gif

(agg: `brew install agg`. To capture a real run instead of this scripted one:
`brew install asciinema` then `asciinema rec demo.cast -c "crosstalk file.mp3"`.)
"""
import json

E = "\x1b"  # ESC; json.dumps emits it as the valid  escape


def c(*codes):
    return E + "[" + ";".join(map(str, codes)) + "m"


green, cyan, mag, dim, bold, rst = c(1, 32), c(1, 36), c(1, 35), c(2), c(1), c(0)
prompt = green + "$" + rst + " "

events = [
    [0.4,  prompt],
    [1.0,  "crosstalk interview.mp3"],
    [1.7,  "\r\n"],
    [2.0,  "[  0.1s] converted to 16 kHz WAV\r\n"],
    [3.6,  "[  8.4s] transcribed: 41 segments " + dim + "(whisper.cpp / Metal)" + rst + "\r\n"],
    [3.9,  "diarization running on: " + cyan + "mps" + rst + "\r\n"],
    [5.6,  "[ 19.6s] diarized: 14 turns, 2 speakers\r\n"],
    [5.8,  "[ 19.6s] wrote " + bold + "interview.crosstalk.txt" + rst + "\r\n"],
    [6.3,  prompt],
    [7.0,  "head -5 interview.crosstalk.txt"],
    [7.6,  "\r\n"],
    [7.9,  "\r\n[00:00] " + cyan + "SPEAKER_01" + rst + ": So explain to me, in practice - how does your\r\n         system actually work?\r\n"],
    [8.9,  "\r\n[00:11] " + mag + "SPEAKER_00" + rst + ": You studied organic chemistry, right? A molecule\r\n         can be written as a linear string - a SMILES string.\r\n"],
    [10.1, "\r\n[00:27] " + cyan + "SPEAKER_01" + rst + ": And your model reads that string and predicts\r\n         whether it binds the target protein?\r\n"],
    [11.3, "\r\n[00:34] " + mag + "SPEAKER_00" + rst + ": Exactly. Target-agnostic - works on a protein\r\n         it has never seen before.\r\n"],
    [12.6, "\r\n" + prompt],
    [13.6, ""],
]

with open("demo.cast", "w") as f:
    f.write(json.dumps({"version": 2, "width": 92, "height": 20,
                        "title": "crosstalk", "env": {"TERM": "xterm-256color"}}) + "\n")
    for t, d in events:
        f.write(json.dumps([t, "o", d]) + "\n")
print("wrote demo.cast")
