#!/usr/bin/env python3
"""Phase 0 — measure S1 on real audio, then set the threshold from the numbers.

This is the tool handoff.md Part 3 asks for when it says "measure before you
change anything", and it is what finally answers Fix 3: the inherited 0.65 was
chosen for Silero and means nothing on TEN's scale, so the cut-off has to come
out of a measurement rather than a guess.

What it does
------------
Feeds clips through :class:`TenVAD` in 80 ms chunks exactly as the live pipeline
would, keeps every per-frame score, and then sweeps thresholds and Fix 1 rules
over those stored scores. The VAD runs **once** per clip; every combination
afterwards is arithmetic on the same numbers. That is not just a speed trick —
it means each rule is judged on identical evidence, so a difference in the table
is a difference in the rule and never a difference in what the model happened to
hear that run.

Labels are the ground truth
---------------------------
Pass clips grouped by what they contain. ``--speech`` clips are ones the VAD
*should* fire on, ``--noise`` clips are ones it should stay quiet through
(coughs, claps, keyboard, door slams, room tone). The tables then read directly
as "how much real speech did we keep" against "how much noise leaked through",
which is the only trade that matters here.

    # the Phase 1 clips the handoff asks for
    python scripts/bench_vad.py --speech clips/speech/*.wav --noise clips/cough/*.wav

    # sanity baseline with no recording of your own, using the cached dataset
    python scripts/bench_vad.py --librispeech 6

    # also write a measurement recording (wav + jsonl) for later re-analysis
    python scripts/bench_vad.py --speech a.wav --noise b.wav --record my-room

Read the output as a starting point, not an answer. A threshold picked on six
LibriSpeech clips in a silent studio is still a guess about your room; the point
of ``--record`` is that once you have real audio, every table here can be re-run
over it without touching a microphone again.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import sys
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.speech.measure import SessionRecorder, rms_dbfs  # noqa: E402
from server.speech.speech_gate import MODES, GateConfig, SpeechGate  # noqa: E402
from server.speech.vad_ten import TenVAD, VADFrames  # noqa: E402

TARGET_SR = 16000
CHUNK = 1280  # 80 ms @ 16 kHz — how the browser streams


# ── input ──
def load_wav16k(path: str) -> np.ndarray:
    """Load any WAV as 16 kHz mono float32.

    Polyphase resampling, matching what the live path does, so the model sees
    the same spectral content here as it will in production.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_librispeech(n: int) -> list[tuple[str, np.ndarray]]:
    """Clean read speech from the locally cached dataset — a control, not a room."""
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    ).cast_column("audio", Audio(decode=False))
    out = []
    for i in range(min(n, len(ds))):
        blob = ds[i]["audio"]
        raw = blob["bytes"] if blob.get("bytes") else open(blob["path"], "rb").read()
        audio, _ = sf.read(io.BytesIO(raw), dtype="float32")
        out.append((f"librispeech[{i}]", audio.astype(np.float32)))
    return out


def expand(patterns: list[str]) -> list[str]:
    """Expand globs, keeping order stable and dropping duplicates."""
    seen, files = set(), []
    for pat in patterns or []:
        hits = sorted(glob.glob(pat)) or ([pat] if Path(pat).is_file() else [])
        if not hits:
            print(f"  ! no file matched {pat!r}", file=sys.stderr)
        for h in hits:
            if h not in seen:
                seen.add(h)
                files.append(h)
    return files


# ── scoring ──
async def score_clip(
    vad: TenVAD, audio: np.ndarray, recorder: SessionRecorder | None, label: str
) -> list[list[float]]:
    """Run one clip through the VAD in 80 ms chunks; return per-chunk frame scores.

    The model handle is reset first so history from the previous clip cannot
    bleed into this one — separate clips are separate streams, unlike the live
    microphone where continuity is deliberate.
    """
    vad.reset_model_state()
    if recorder is not None:
        recorder.mark(f"clip:{label}", clip=label, samples=int(audio.size))

    chunks: list[list[float]] = []
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        chunk = audio[i : i + CHUNK]
        frames = vad.frames(chunk)
        chunks.append(frames.probs)
        if recorder is not None:
            recorder.chunk(chunk, frames=frames.probs, clip=label)
    return chunks


def replay(chunks: list[list[float]], cfg: GateConfig, hop_ms: float) -> dict:
    """Replay stored frame scores through a gate config. No model involved."""
    gate = SpeechGate(cfg)
    n_speech = onsets = 0
    for probs in chunks:
        upd = gate.update(VADFrames(probs=probs, flags=[], hop_ms=hop_ms))
        n_speech += bool(upd.is_speech)
        onsets += upd.event in ("start", "resume")
    return {
        "chunks": len(chunks),
        "speech_chunks": n_speech,
        "pct": 100.0 * n_speech / max(1, len(chunks)),
        "onsets": onsets,
    }


# ── reporting ──
def print_distribution(groups: dict[str, list[list[float]]]) -> None:
    """Raw frame-score percentiles per class — the shape behind every threshold."""
    print("\n=== frame-score distribution (the evidence any threshold cuts) ===")
    print(f"{'class':>10} | {'frames':>7} | {'mean':>6} | {'p10':>6} | {'p50':>6} | {'p90':>6} | {'p99':>6}")
    print("-" * 70)
    for name, chunks in groups.items():
        flat = np.array([p for c in chunks for p in c], dtype=np.float64)
        if flat.size == 0:
            print(f"{name:>10} | {'—':>7}")
            continue
        q = np.percentile(flat, [10, 50, 90, 99])
        print(
            f"{name:>10} | {flat.size:>7} | {flat.mean():>6.3f} | "
            f"{q[0]:>6.3f} | {q[1]:>6.3f} | {q[2]:>6.3f} | {q[3]:>6.3f}"
        )


def print_sweep(
    groups: dict[str, list[list[float]]],
    thresholds: list[float],
    modes: list[str],
    min_frames: int,
    confirm_chunks: int,
    hop_ms: float,
) -> dict:
    """Threshold × rule sweep. Returns the best threshold per mode by margin."""
    have_noise = bool(groups.get("noise"))
    print(
        f"\n=== threshold sweep (min_frames={min_frames}, confirm_chunks={confirm_chunks}) ==="
    )
    print("  speech% = real speech kept (HIGH is good)"
          + ("   noise% = noise leaked through (LOW is good)" if have_noise else ""))
    header = f"{'thr':>5} | {'mode':>6} | {'speech%':>8}"
    if have_noise:
        header += f" | {'noise%':>7} | {'margin':>7}"
    print(header)
    print("-" * len(header))

    best: dict[str, tuple[float, float]] = {}
    for thr in thresholds:
        for mode in modes:
            cfg = GateConfig(
                threshold=thr, mode=mode, min_frames=min_frames,
                confirm_chunks=confirm_chunks,
            )
            sp = replay(groups.get("speech", []), cfg, hop_ms)["pct"] if groups.get("speech") else float("nan")
            row = f"{thr:>5.2f} | {mode:>6} | {sp:>7.1f}%"
            if have_noise:
                nz = replay(groups["noise"], cfg, hop_ms)["pct"]
                margin = sp - nz
                row += f" | {nz:>6.1f}% | {margin:>7.1f}"
                if mode not in best or margin > best[mode][1]:
                    best[mode] = (thr, margin)
            print(row)
        print()
    return best


def print_recommendation(best: dict[str, tuple[float, float]], have_noise: bool) -> None:
    if not have_noise:
        print("=== no --noise clips supplied ===")
        print("  Without audio the VAD should stay quiet through, a threshold cannot be")
        print("  set: every table above only shows what we keep, never what leaks. Record")
        print("  the Phase 1 clips the handoff asks for (cough, clap, click, keyboard,")
        print("  room tone) and re-run before trusting any number here.")
        return
    print("=== starting point, by widest margin ===")
    for mode, (thr, margin) in sorted(best.items()):
        print(f"  mode={mode:<6} threshold={thr:.2f}   (margin {margin:.1f} points)")
    print("\n  Treat these as a hypothesis to test live, not a setting to ship. Margin")
    print("  weighs a lost word the same as a leaked cough, and they are not equally")
    print("  costly: a missed word breaks the sentence, a leaked cough costs a wait.")
    print("  Where two thresholds are close, prefer the lower one and let Fix 1 do the")
    print("  rejecting — that is the whole point of not raising the threshold.")


# ── main ──
async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 0: measure TEN VAD + the Fix 1 rules on real clips."
    )
    ap.add_argument("--speech", nargs="*", default=[], help="clips that ARE speech (globs ok)")
    ap.add_argument("--noise", nargs="*", default=[], help="clips that are NOT speech (globs ok)")
    ap.add_argument("--librispeech", type=int, default=0, help="add N cached clean-speech clips")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.65,0.7,0.8",
                    help="comma-separated thresholds to sweep")
    ap.add_argument("--modes", default=",".join(MODES), help=f"comma-separated subset of {MODES}")
    ap.add_argument("--min-frames", type=int, default=2, help="Fix 1: frames needed per chunk")
    ap.add_argument("--confirm-chunks", type=int, default=2, help="Fix 2: chunks needed to declare onset")
    ap.add_argument("--record", default=None, metavar="NAME",
                    help="also write a measurement recording (wav + jsonl) under --out")
    ap.add_argument("--out", default="recordings", help="directory for --record")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            ap.error(f"unknown mode {m!r}; choose from {MODES}")

    clips: dict[str, list[tuple[str, np.ndarray]]] = {"speech": [], "noise": []}
    for name in expand(args.speech):
        clips["speech"].append((Path(name).name, load_wav16k(name)))
    for name in expand(args.noise):
        clips["noise"].append((Path(name).name, load_wav16k(name)))
    if args.librispeech:
        clips["speech"].extend(load_librispeech(args.librispeech))

    if not clips["speech"] and not clips["noise"]:
        ap.error("no clips: pass --speech / --noise files, or --librispeech N")

    print("=== clips ===")
    for cls, items in clips.items():
        for name, audio in items:
            print(f"  {cls:>6}  {name:<34} {len(audio)/TARGET_SR:>6.1f}s  "
                  f"rms={rms_dbfs(audio):>7.1f} dBFS  peak={np.max(np.abs(audio)):.3f}")
    if not clips["noise"]:
        print("  (no --noise clips — false-alarm columns will be omitted)")

    recorder = None
    if args.record:
        recorder = SessionRecorder(
            args.out, args.record,
            # No agent in an offline run, so the ref track would be nothing but
            # zeros the same size as the mic recording.
            record_ref=False,
            meta={"tool": "bench_vad", "thresholds": thresholds, "modes": modes},
        )

    vad = TenVAD()
    await vad.start()
    try:
        groups: dict[str, list[list[float]]] = {}
        for cls, items in clips.items():
            scored: list[list[float]] = []
            for name, audio in items:
                scored.extend(await score_clip(vad, audio, recorder, f"{cls}/{name}"))
            if scored:
                groups[cls] = scored
        hop_ms = vad.hop_ms
    finally:
        await vad.stop()
        if recorder is not None:
            summary = recorder.close()
            print(f"\n  recording written: {summary['jsonl']}")

    print_distribution(groups)
    best = print_sweep(groups, thresholds, modes, args.min_frames, args.confirm_chunks, hop_ms)
    print()
    print_recommendation(best, bool(groups.get("noise")))


if __name__ == "__main__":
    asyncio.run(main())
