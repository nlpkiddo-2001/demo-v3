#!/usr/bin/env python3
"""Compare recognisers on the accents we actually serve — not on LibriSpeech.

Why this exists
---------------
Every number we had for the recogniser came from LibriSpeech: American English,
read aloud from books, in a studio. Our users are Indian, American and European,
speaking spontaneously, over a laptop microphone. A model that wins on the first
distribution can easily lose on the second, and the public leaderboards are
mostly the first.

So this measures on EdAcc — the Edinburgh International Accents of English
Corpus — which is *conversational* English labelled by the speaker's accent. The
clips are real dialogue: hesitations, overlaps, self-corrections. That is the
distribution we care about.

    python scripts/bench_asr.py --n 60
    python scripts/bench_asr.py --models nemotron,parakeet --n 120
    python scripts/bench_asr.py --groups india,us --n 200

What it reports
---------------
Word error rate per accent group, plus real-time factor. Read the *gap between
groups* as much as the absolute number: a model that is excellent on US English
and poor on Indian English is the wrong model for us however well it scores on
average.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_SR = 16000

# The three groups we sell into, mapped onto EdAcc's accent labels.
GROUPS: dict[str, tuple[str, ...]] = {
    "india": ("Indian English",),
    "us": ("Mainstream US English",),
    "europe": (
        "Southern British English", "Irish English", "Scottish English",
        "Eastern European", "Italian", "Spanish", "Catalan", "Dutch",
        "Bulgarian", "European", "German", "French",
    ),
}

_PUNCT = re.compile(r"[^a-z0-9' ]+")
# EdAcc marks segments that must be EXCLUDED from scoring with this token, the
# usual Kaldi/ESPnet convention. Score them and you compare a correct transcript
# against a piece of metadata: the model is charged 100% error for getting it
# right. Missing this made US English look like the worst accent in the set.
_SKIP_MARKER = "ignore_time_segment_in_scoring"
# Annotation tags for non-speech events. Not words, and no recogniser emits them.
_TAG = re.compile(r"<[^>]*>")
# Conversational transcripts are full of these and no two models agree on them;
# scoring them measures annotation style, not recognition.
_FILLER = {"uh", "um", "mm", "hmm", "er", "erm", "ah", "eh", "mhm", "uh-huh"}


def normalise(text: str, drop_fillers: bool = True) -> str:
    """Lowercase, strip tags and punctuation, optionally drop hesitation tokens."""
    words = _PUNCT.sub(" ", _TAG.sub(" ", (text or "").lower())).split()
    if drop_fillers:
        words = [w for w in words if w not in _FILLER]
    return " ".join(words)


def load_edacc(n_per_group: int, groups: list[str], min_sec: float, max_sec: float):
    """Return {group: [(audio16k, reference_text)]} from EdAcc."""
    import soundfile as sf
    from datasets import Audio, load_dataset
    from huggingface_hub import HfFolder

    print("loading EdAcc (conversational, accent-labelled)...", flush=True)
    ds = load_dataset("edinburghcstr/edacc", split="validation", token=HfFolder.get_token())
    # decode=False keeps the raw bytes so we do not need torchcodec.
    ds = ds.cast_column("audio", Audio(decode=False))

    want = {g: set(GROUPS[g]) for g in groups}
    out: dict[str, list] = {g: [] for g in groups}
    for row in ds:
        acc = row["accent"]
        grp = next((g for g in groups if acc in want[g]), None)
        if grp is None or len(out[grp]) >= n_per_group:
            continue
        if all(len(v) >= n_per_group for v in out.values()):
            break
        blob = row["audio"]
        audio_bytes = blob["bytes"] if blob.get("bytes") else open(blob["path"], "rb").read()
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            from math import gcd

            import scipy.signal as sps

            g_ = gcd(sr, TARGET_SR)
            audio = sps.resample_poly(audio, TARGET_SR // g_, sr // g_)
        dur = len(audio) / TARGET_SR
        if not (min_sec <= dur <= max_sec):
            continue
        raw = (row["text"] or "").lower()
        if _SKIP_MARKER in raw:       # excluded by the corpus itself
            continue
        ref = normalise(row["text"])
        if len(ref.split()) < 3:      # too short to score meaningfully
            continue
        out[grp].append((np.ascontiguousarray(audio, dtype=np.float32), ref))

    for g, items in out.items():
        secs = sum(len(a) for a, _ in items) / TARGET_SR
        print(f"  {g:<7} {len(items):>4} clips, {secs/60:.1f} min", flush=True)
    return out


# ── the candidates ──
def run_nemotron(clips, att=(70, 1), device="cuda"):
    """Our current live model, driven exactly as the pipeline drives it."""
    from server.speech.stt_nemotron import NemotronStreamingSTT

    stt = NemotronStreamingSTT(att_context=att, device=device)
    import asyncio

    asyncio.run(stt.start())
    CH = 1280
    hyps, elapsed = [], 0.0
    for audio, _ in clips:
        stt.reset()
        t0 = time.time()
        for i in range(0, len(audio) - CH + 1, CH):
            stt.accept_audio(audio[i : i + CH])
        stt.flush()
        elapsed += time.time() - t0
        hyps.append(stt.partial())
    return hyps, elapsed


def run_nemo_offline(clips, model_id, device="cuda"):
    """Any NeMo checkpoint, whole-clip (the accurate-pass style)."""
    import nemo.collections.asr as nemo_asr
    import torch

    m = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    m.eval().to(device)
    hyps, elapsed = [], 0.0
    for audio, _ in clips:
        t0 = time.time()
        with torch.no_grad():
            out = m.transcribe([audio], batch_size=1, verbose=False)
        elapsed += time.time() - t0
        first = out[0] if out else ""
        hyps.append(first.text if hasattr(first, "text") else str(first))
    del m
    torch.cuda.empty_cache()
    return hyps, elapsed


def run_whisper(clips, model_id="openai/whisper-large-v3-turbo", device="cuda"):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    m = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()
    hyps, elapsed = [], 0.0
    for audio, _ in clips:
        t0 = time.time()
        feats = proc(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            ids = m.generate(
                feats.input_features.to(device, torch.float16),
                language="en", task="transcribe", max_new_tokens=200,
            )
        elapsed += time.time() - t0
        hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    del m
    torch.cuda.empty_cache()
    return hyps, elapsed


RUNNERS = {
    "nemotron":     lambda c, d: run_nemotron(c, (70, 1), d),
    "nemotron-mid": lambda c, d: run_nemotron(c, (70, 6), d),
    "nemotron-slow": lambda c, d: run_nemotron(c, (70, 13), d),
    "parakeet":     lambda c, d: run_nemo_offline(c, "nvidia/parakeet-tdt-0.6b-v3", d),
    "canary":       lambda c, d: run_nemo_offline(c, "nvidia/canary-qwen-2.5b", d),
    "whisper":      lambda c, d: run_whisper(c, "openai/whisper-large-v3-turbo", d),
    "whisper-lg":   lambda c, d: run_whisper(c, "openai/whisper-large-v3", d),
}


def main() -> None:
    from jiwer import wer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="clips per accent group")
    ap.add_argument("--groups", default="india,us,europe")
    ap.add_argument("--models", default="nemotron,parakeet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--show", type=int, default=0, help="print N ref/hyp pairs per group")
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip() in GROUPS]
    models = [m.strip() for m in args.models.split(",") if m.strip() in RUNNERS]
    if not models:
        ap.error(f"pick from: {', '.join(RUNNERS)}")

    data = load_edacc(args.n, groups, args.min_sec, args.max_sec)

    results: dict[str, dict[str, float]] = defaultdict(dict)
    rtf: dict[str, float] = {}
    for name in models:
        print(f"\n=== {name} ===", flush=True)
        total_audio = total_time = 0.0
        for g in groups:
            clips = data[g]
            if not clips:
                continue
            try:
                hyps, el = RUNNERS[name](clips, args.device)
            except Exception as e:
                print(f"  {g}: FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            refs = [r for _, r in clips]
            hyps = [normalise(h) or "<empty>" for h in hyps]
            w = wer(refs, hyps) * 100
            if args.show:
                for r_, h_ in list(zip(refs, hyps))[: args.show]:
                    print(f"      REF {r_[:88]}")
                    print(f"      HYP {h_[:88]}")
            results[name][g] = w
            secs = sum(len(a) for a, _ in clips) / TARGET_SR
            total_audio += secs
            total_time += el
            print(f"  {g:<7} WER {w:6.2f}%   ({len(clips)} clips)", flush=True)
        rtf[name] = total_time / max(total_audio, 1e-6)

    print("\n" + "=" * 62)
    print("WORD ERROR RATE BY ACCENT  (lower is better)")
    hdr = f"{'model':<14} " + " ".join(f"{g:>9}" for g in groups) + f" {'RTF':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name in models:
        row = f"{name:<14} " + " ".join(
            f"{results[name].get(g, float('nan')):>8.2f}%" for g in groups
        )
        print(row + f" {rtf.get(name, 0):>7.3f}")
    print("\nThe gap between columns matters as much as the numbers: a model that")
    print("is strong on US English and weak on Indian English is the wrong model")
    print("for this product, whatever its leaderboard average says.")


if __name__ == "__main__":
    main()
