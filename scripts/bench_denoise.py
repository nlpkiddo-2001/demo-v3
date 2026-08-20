#!/usr/bin/env python3
"""Score denoisers on the only thing this pipeline asks of them: VAD separation.

Why this exists
---------------
``denoise.py`` says denoised audio goes to the VAD and raw audio goes to the
recogniser, and that the question of which denoiser is better should be left to
measurement. This is that measurement. It exists because the obvious metric is
the wrong one.

A denoiser is normally judged on enhancement quality — PESQ, STOI, how clean it
sounds. None of that is what we need. Denoised audio in this pipeline is only
ever seen by the VAD, and the VAD is asking one question: speech or not. So the
figure of merit is how far apart the denoiser pushes those two classes, and a
model can improve every enhancement metric while making that gap *worse*.

That is not hypothetical. GTCRN, the original default, is a speech enhancement
model: fed room noise it does not return silence, it returns its best
reconstruction of the speech it assumes is buried in there, which is
speech-shaped by construction. The VAD then scores it as speech-like — correctly,
given what it was handed. On our own recordings that lifts room noise from 0.40
to 0.76 against a 0.60 bar, and in ``echo_mode="off"`` a noise burst over the bar
cancels whatever the agent was saying.

What it measures
----------------
Every recording with a sidecar ``.jsonl`` is replayed through each candidate in
80 ms chunks, exactly as the live session would, and each chunk is scored by
:class:`TenVAD` and run through :class:`SpeechGate`. Three populations come out:

``room``    every chunk where the agent was audible and no word landed. This is
            the population that must stay UNDER the bar, because anything in it
            that crosses cancels the agent mid-sentence.
``voice``   chunks where the recogniser emitted a new word.
``silence`` everything else.

Both labels come from the recording, not from any VAD: agent audibility from the
playback rows, words from the recogniser — which is fed RAW audio off the ring.
Neither depends on the denoiser under test. That independence is the whole point
and it is easy to lose: score ``room`` over "chunks the gate opened on" instead
and you are only sampling frames the candidate already called speech, so every
model scores ~0.9 and the table says nothing.

    python scripts/bench_denoise.py
    python scripts/bench_denoise.py --glob 'recordings/agent-*.jsonl'
    python scripts/bench_denoise.py --models none,gtcrn,dpdfnet4 --threshold 0.6

Read ``margin`` first
---------------------
``margin`` is median(voice) − **p95**(room): the gap the threshold has to live
in. p95 and not the median, because barge-in is fired by the loudest moment of
the room, not its typical one — a model with a quiet median and a spiky tail
interrupts you just as often. A denoiser that lifts both classes equally has
bought nothing, however good it sounds. ``barge`` counts simulated false
interruptions, the symptom the margin explains.

Honesty about the labels
------------------------
``voice`` is only the chunks where a word *landed*, so it misses the quiet head
and tail of each word — it is a sample of speech, not all of it. And the
recogniser is only fed while the gate is open, so a denoiser that gated open more
at recording time got more chances to produce words; every recording on disk was
captured with GTCRN fitted. ``room`` has no such dependency.

So this ranks candidates on *our* audio; it does not establish ground truth.
Clip-level labels, the way ``bench_vad.py`` takes them, would settle it
properly.
"""

from __future__ import annotations

import argparse
import asyncio
import glob as globlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.speech.denoise import CHOICES, make_denoiser          # noqa: E402
from server.speech.speech_gate import GateConfig, SpeechGate      # noqa: E402
from server.speech.vad_ten import TenVAD                          # noqa: E402

TARGET_SR = 16000
CHUNK = 1280            # 80 ms, the browser's chunk
AGENT_HOLD_MS = 200.0   # a ref chunk means the agent is audible for this long after


def load_sidecar(jsonl: Path):
    """Per-chunk telemetry from a measurement recording.

    Returns (t_ms, agent_audible, word_landed) or None if there is nothing to
    score — an empty capture, or one with no agent playback in it.
    """
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    chunks = [r for r in rows if r["type"] == "chunk"]
    refs = np.array(sorted(r["t_ms"] for r in rows if r["type"] == "ref"))
    if not chunks or not len(refs):
        return None

    t = np.array([c["t_ms"] for c in chunks])
    i = np.searchsorted(refs, t)
    agent = (i > 0) & ((t - refs[np.maximum(i - 1, 0)]) <= AGENT_HOLD_MS)

    # A new word appearing in the running hypothesis. asr_text is cumulative
    # within a turn, so "changed and non-empty" is one word's worth of evidence
    # that this chunk carried speech.
    word, prev = [], ""
    for c in chunks:
        txt = (c.get("asr_text") or "").strip()
        word.append(bool(txt) and txt != prev)
        if txt:
            prev = txt
    return t, agent, np.array(word)


async def score(model: str, files: list[Path], threshold: float, min_frames: int):
    """Replay every file through `model` and collect the three populations."""
    room, voice, silence, fires = [], [], [], []
    audio_s = wall_s = 0.0

    for jsonl in files:
        side = load_sidecar(jsonl)
        if side is None:
            continue
        wav = jsonl.with_name(jsonl.stem + ".mic.wav")
        if not wav.is_file():
            continue
        y, sr = sf.read(str(wav), dtype="float32")
        if sr != TARGET_SR or len(y) < TARGET_SR:
            continue
        t, agent, word = side

        den = make_denoiser(model)
        await den.start()
        vad = TenVAD()
        await vad.start()
        gate = SpeechGate(GateConfig(threshold=threshold, mode="count",
                                     min_frames=min_frames, release_threshold=0.35))

        probs, speech = [], []
        t0 = time.perf_counter()
        for i in range(0, len(y), CHUNK):
            clean = den.process(y[i:i + CHUNK])
            frames = vad.frames(clean)
            probs.append(max(frames.probs, default=0.0))
            speech.append(gate.update(frames).in_speech)
        wall_s += time.perf_counter() - t0
        audio_s += len(y) / TARGET_SR

        n = min(len(probs), len(t))
        p = np.array(probs[:n])
        sp = np.array(speech[:n])
        t, agent, word = t[:n], agent[:n], word[:n]

        # Both populations are defined by the RECORDING, never by `sp` — see the
        # module docstring. Conditioning on the candidate's own gate is the one
        # mistake that makes this table meaningless.
        room    += list(p[agent & ~word])
        voice   += list(p[word])
        silence += list(p[~agent & ~word])

        # A barge-in fires on the gate opening while the agent is audible.
        k = 0
        while k < n:
            if sp[k] and agent[k]:
                m = k
                while m + 1 < n and sp[m + 1]:
                    m += 1
                fires.append(t[m] - t[k] + 80.0)
                k = m + 1
            else:
                k += 1

        await vad.stop()
        await den.stop()

    return {
        "room": np.array(room), "voice": np.array(voice), "silence": np.array(silence),
        "fires": fires, "rtfx": audio_s / wall_s if wall_s else float("nan"),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="recordings/*.jsonl",
                    help="measurement recordings to score (default: all of them)")
    ap.add_argument("--models", default=",".join(CHOICES),
                    help=f"comma-separated subset of {', '.join(CHOICES)}")
    ap.add_argument("--threshold", type=float, default=0.6, help="SpeechGate bar")
    ap.add_argument("--min-frames", type=int, default=2, help="frames needed per chunk")
    args = ap.parse_args()

    files = sorted(Path(p) for p in globlib.glob(args.glob))
    if not files:
        sys.exit(f"no recordings matched {args.glob!r}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"{len(files)} recordings · gate threshold {args.threshold} "
          f"· min_frames {args.min_frames}\n")

    print(f"{'denoiser':18} {'room med':>9} {'room p95':>9} {'voice':>6} {'silence':>8} "
          f"{'margin':>7} {'barge':>6} {'speed':>7}")
    print("-" * 82)
    rows = []
    for m in models:
        r = await score(m, files, args.threshold, args.min_frames)
        if not r["voice"].size or not r["room"].size:
            print(f"{m:18} no scoreable audio")
            continue
        p95 = float(np.percentile(r["room"], 95))
        margin = float(np.median(r["voice"]) - p95)
        rows.append((margin, m))
        print(f"{m:18} {np.median(r['room']):9.2f} {p95:9.2f} {np.median(r['voice']):6.2f} "
              f"{np.median(r['silence']):8.2f} {margin:7.2f} "
              f"{len(r['fires']):6} {r['rtfx']:6.0f}x")

    if rows:
        rows.sort(reverse=True)
        print(f"\nwidest margin: {rows[0][1]} ({rows[0][0]:.2f})")
        print("margin is median(voice) - p95(room); bigger is more room for the bar.")


if __name__ == "__main__":
    asyncio.run(main())
