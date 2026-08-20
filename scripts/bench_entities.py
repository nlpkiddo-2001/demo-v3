#!/usr/bin/env python3
"""Score the recognisers on the words that actually matter — names, places, mail ids.

Why WER is the wrong number here
--------------------------------
``bench_asr.py`` measures word error rate per accent group, and on our own audio
the accurate pass sits around 9-10% on Indian English. That number is fine. It is
also completely blind to the thing people complain about, because proper nouns
are a rounding error in the token count: a turn can score 95% WER-correct while
getting the customer's name, their city and their email address all wrong. One
name in a forty-word sentence is 2.5% of the words and 100% of the point.

So this measures **entity recall** instead: of the names, places and mail ids
that were genuinely said, how many came back intact? Nothing else is scored.
Published work on exactly this — the Contextual Earnings-22 benchmark — found
that context biasing moves keyword F-score sharply while leaving WER flat, which
is the same observation from the other direction: the two metrics do not track
each other, and only one of them is about the user's name.

The ground truth is real, not synthetic
---------------------------------------
The seeded cases below are lifted from our own ``recordings/``, and they were
found by looking for the moment a speaker repeats himself. That turns out to be
a reliable tell: a person who has just been misheard says the word again, louder.

    "It's Koyambatur bro. It's Koyambatur."      <- Coimbatore, twice
    "so it's not Leil Adak it's Leh Ladakh"      <- corrected mid-sentence
    "No pro I'm not flying from Quambatur"       <- Coimbatore again, third spelling

Three attempts at two Indian place names, zero hits. That is the baseline this
script exists to move, and it is worth remembering that none of it shows up in a
WER table.

Adding your own
---------------
The seeded manifest has no personal names and no mail ids, because none were
ever said into a microphone here — the system prompt tells the agent to ask
people to *type* those rather than say them, which is a sensible mitigation and
also why we have no audio of them. To cover that half:

    python scripts/bench_entities.py --phrases          # the script to read
    ./run_agent.sh                                      # read it into the UI
    python scripts/bench_entities.py --ingest agent-20260816-120000

There is no microphone on this machine, so the recorder is the pipeline itself.
That is the right one anyway: it captures through the exact path production
uses, browser resampling and all.

Both halves matter. A recogniser that handles places and fails on surnames is
still broken for a CRM — and the expectation is that surnames score worse, since
"Coimbatore" is at least somewhere in the training data and "Preetam Rangudu"
is not.

Reading the output
------------------
Per-entity hit/miss with the transcript underneath, then recall per model. Look
at the misses, not the percentage: they tell you *how* the model failed, and a
consistent mishearing ("Koyambatur" every time) is a different problem from a
random one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "entities.json"
TARGET_SR = 16000


# Every case here was verified by listening: the entity really is spoken in the
# clip, and the transcript we shipped really did get it wrong. ``heard`` records
# what the live model produced, so a regression is visible as a return to it.
SEED = [
    {
        "file": "recordings/capture-20260813-105609.mic.wav",
        "start": 67.9, "end": 93.4,
        "entities": ["Coimbatore", "Chennai"],
        "heard": {"Coimbatore": "Koyambatur"},
        "note": "speaker repeats the city twice after being misheard",
    },
    {
        "file": "recordings/agent-20260814-181647.mic.wav",
        "start": 76.5, "end": 93.6,
        "entities": ["Coimbatore", "Chennai"],
        "heard": {"Coimbatore": "Quambatur"},
        "note": "same city, a third spelling; Chennai lands, Coimbatore does not",
    },
    {
        "file": "recordings/agent-20260814-164301.mic.wav",
        "start": 6.3, "end": 46.2,
        "entities": ["Leh Ladakh"],
        "heard": {"Leh Ladakh": "Leiladak / Leil Adak"},
        "note": "speaker corrects the model mid-sentence and is misheard again",
    },
]

# Entity matching is deliberately forgiving about everything EXCEPT the letters.
# Case and punctuation are noise; "coimbatore" and "Coimbatore," are both hits.
# "Koyambatur" is not, and no amount of normalisation should make it one.
_PUNCT = re.compile(r"[^a-z0-9@.\s]+")


def _norm(text: str) -> str:
    return _PUNCT.sub(" ", (text or "").lower())


def hit(transcript: str, entity: str) -> bool:
    """Did the entity survive transcription?

    Substring match on normalised text, so a name inside a sentence counts and
    word-boundary bookkeeping does not become its own source of bugs. Multi-word
    entities ("Leh Ladakh") have their internal spacing collapsed so a model
    that writes "Leh  Ladakh" is not punished for whitespace.
    """
    t = " ".join(_norm(transcript).split())
    e = " ".join(_norm(entity).split())
    return bool(e) and e in t


def load_clip(path: Path, start: float, end: float) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    clip = audio[int(start * sr):int(end * sr)]
    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), TARGET_SR)
        clip = resample_poly(clip, TARGET_SR // g, int(sr) // g)
    return np.ascontiguousarray(clip, dtype=np.float32)


class _NeMoASR:
    """Any plain NeMo ``ASRModel`` checkpoint, for models we only want to measure.

    The production providers live in ``server/speech/`` and carry the timeouts,
    failure handling and no-opinion contract the pipeline needs. Nothing here
    does: a bench wants the raw answer and wants a crash to be loud. So the
    candidates that are not shipping get thin adapters, and the two that ARE
    shipping are measured through their real production classes — otherwise the
    bench would be scoring code that never runs.
    """

    def __init__(self, model_id: str, device: str, label: str = ""):
        self.model_id, self.device = model_id, device
        self.name = label or model_id.split("/")[-1]
        self._m = None

    async def start(self) -> None:
        import nemo.collections.asr as nemo_asr

        self._m = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
        self._m.eval()
        self._m.to(self.device)

    async def stop(self) -> None:
        self._m = None

    def transcribe(self, audio: np.ndarray) -> str | None:
        import torch

        with torch.no_grad():
            out = self._m.transcribe([audio], batch_size=1, verbose=False)
        if not out:
            return None
        first = out[0]
        return (getattr(first, "text", None) or str(first)).strip() or None


class _HFSeq2Seq:
    """Whisper-family checkpoints through transformers.

    ``condition_on_prev_tokens`` is left off and the temperature fallback
    disabled: both make Whisper more fluent and more inventive, and on a bench
    about whether a name survived, fluency is the enemy.
    """

    def __init__(self, model_id: str, device: str, label: str = ""):
        self.model_id, self.device = model_id, device
        self.name = label or model_id.split("/")[-1]
        self._pipe = None

    async def start(self) -> None:
        import torch
        from transformers import pipeline

        self._pipe = pipeline(
            "automatic-speech-recognition", model=self.model_id,
            torch_dtype=torch.float16, device=self.device,
            chunk_length_s=30,
        )

    async def stop(self) -> None:
        self._pipe = None

    def transcribe(self, audio: np.ndarray) -> str | None:
        out = self._pipe({"raw": audio, "sampling_rate": TARGET_SR},
                         generate_kwargs={"language": "en", "task": "transcribe"})
        return ((out or {}).get("text") or "").strip() or None


class _CohereASR:
    """Cohere Transcribe, via the modelling code shipped in its own repo.

    The model card asks for ``transformers>=5.4.0``, which we do not have and
    will not install — NeMo pins this environment and upgrading it to bench a
    candidate would risk the recogniser that is actually in production. The repo
    carries ``auto_map`` plus ``modeling_cohere_asr.py``, so ``trust_remote_code``
    loads it on 4.57.6 without touching the rest of the stack.

    Cohere's own release notes say this model "is eager to transcribe, even
    non-speech sounds" and recommend a VAD in front of it. Every clip here is
    already VAD-gated speech from the pipeline, so the bench is a fair reading of
    it — but that eagerness is why it was rejected for the live seat.
    """

    def __init__(self, device: str, label: str = "cohere-transcribe"):
        self.device, self.name = device, label
        self._m = self._p = None

    async def start(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        mid = "CohereLabs/cohere-transcribe-03-2026"
        self._p = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        # AutoModel maps to CohereAsrModel — the bare encoder-decoder, with no
        # ``generate``. The repo's auto_map points AutoModelForSpeechSeq2Seq at
        # CohereAsrForConditionalGeneration, which is the one that can decode.
        # bfloat16, not float16: the repo's attention masks with a hardcoded
        # -1e9, which overflows fp16's ~65504 range and raises. bf16 keeps
        # fp32's exponent range, so the same constant is representable.
        self._m = AutoModelForSpeechSeq2Seq.from_pretrained(
            mid, trust_remote_code=True, dtype=torch.bfloat16)
        self._m.eval().to(self.device)

    async def stop(self) -> None:
        self._m = self._p = None

    def transcribe(self, audio: np.ndarray) -> str | None:
        """Use the repo's own ``transcribe`` helper, not raw ``generate``.

        ``generate`` is a trap on transformers 4.x here: their wrapper sets
        ``generation_kwargs["decoder_attention_mask"]`` unconditionally but only
        *builds* it when ``decoder_input_ids`` is supplied, so calling it plainly
        leaves a None under a key 4.57 expects to be a tensor, and generation
        dies in ``_update_model_kwargs_for_generation``. ``transcribe`` builds
        both properly.

        It also splits audio longer than ``max_audio_clip_s`` (35 s) into
        overlapping chunks and reassembles — which several clips in this manifest
        need, and which hand-rolled batching would get wrong.
        """
        import torch

        with torch.no_grad():
            out = self._m.transcribe(
                processor=self._p, language="en",
                audio_arrays=[np.asarray(audio, dtype=np.float32)],
                sample_rates=[TARGET_SR],
            )
        if not out:
            return None
        return (out[0] or "").strip() or None


# Short name -> how to build it. Kept as a table so ``--models`` stays a flat
# list and adding a candidate is one line rather than another elif.
MODELS = {
    # the two that actually ship, measured through their production classes
    "parakeet":   lambda d, l: _prod("parakeet", d, l),
    "canary":     lambda d, l: _prod("canary", d, l),
    "sarvam":     lambda d, l: _prod("sarvam", d, l),
    # candidates, thin adapters
    "parakeet-v2": lambda d, l: _NeMoASR("nvidia/parakeet-tdt-0.6b-v2", d),
    "canary-1b":   lambda d, l: _NeMoASR("nvidia/canary-1b-v2", d),
    "whisper-v3":  lambda d, l: _HFSeq2Seq("openai/whisper-large-v3", d),
    "cohere":      lambda d, l: _CohereASR(d),
}


def _prod(name: str, device: str, lang: str):
    if name == "parakeet":
        from server.speech.stt_parakeet import ParakeetAccurate

        return ParakeetAccurate(device=device)
    if name == "canary":
        from server.speech.stt_canary import CanaryAccurate

        return CanaryAccurate(device=device)
    from server.config import settings
    from server.speech.stt_sarvam import SarvamAccurate

    return SarvamAccurate(api_key=settings.sarvam_api_key, language=lang,
                          model_id=settings.sarvam_model,
                          mode=settings.sarvam_mode, timeout_sec=30.0)


def build(name: str, device: str, lang: str):
    """One recogniser, by short name."""
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r} — have: {', '.join(MODELS)}")
    return MODELS[name](device, lang)


# The phrases to read into the agent, one per turn. Sentences rather than bare
# words on purpose: a recogniser uses context, and "Jagan" alone is a harder and
# less realistic test than "Jagan" inside the kind of sentence a caller says.
#
# Grouped so a miss is diagnosable. Personal names are expected to score worst —
# "Coimbatore" is at least in the training data somewhere, "Preetam Rangudu" is
# not — and mail ids worst of all, because they are names plus spelling plus
# punctuation the model has to render as symbols.
PHRASES = [
    # ── personal names, the half we have no audio of ──
    ("My name is Jagan",                                        ["Jagan"]),
    ("This is Vipin speaking",                                  ["Vipin"]),
    ("Can you connect me to Nandhini",                          ["Nandhini"]),
    ("Please assign the ticket to Sasi",                        ["Sasi"]),
    ("The account owner is Preetam Rangudu",                    ["Preetam Rangudu"]),
    ("Jagan and Nandhini are both on the call",                 ["Jagan", "Nandhini"]),
    # ── places ──
    ("I am flying from Coimbatore tomorrow",                    ["Coimbatore"]),
    ("The customer is based in Thiruvananthapuram",             ["Thiruvananthapuram"]),
    ("We have an office in Madurai and one in Bengaluru",       ["Madurai", "Bengaluru"]),
    ("The site visit is in Kanchipuram",                        ["Kanchipuram"]),
    ("We are planning a trip to Leh Ladakh",                    ["Leh Ladakh"]),
    ("Our branch is in Tiruchirappalli",                        ["Tiruchirappalli"]),
    # ── mail ids: say them the way a caller would, "dot" and "at" spoken ──
    ("My mail id is jagan dot v at zohocorp dot com",           ["jagan.v@zohocorp.com"]),
    ("Send it to nandhini dot s at zoho dot com",               ["nandhini.s@zoho.com"]),
    ("His email is preetam dot rangudu at gmail dot com",       ["preetam.rangudu@gmail.com"]),
]


def phrases() -> None:
    """Print the script to read into the agent UI, one sentence per turn."""
    print(__doc__.split("Adding your own")[0].strip()[:0] or "", end="")
    print(f"\nRead these into the agent, ONE PER TURN — pause after each so the\n"
          f"endpointer closes it, then start the next. Any order, but keep them\n"
          f"separate. {len(PHRASES)} turns, about two minutes.\n")
    for i, (text, ents) in enumerate(PHRASES, 1):
        print(f"  {i:2d}. {text}")
    print(f"\nThen note the session name the UI printed (agent-YYYYMMDD-HHMMSS) and run:\n"
          f"    python scripts/bench_entities.py --ingest <session-name>\n")


def ingest(session: str) -> None:
    """Turn a recorded session into manifest entries, one per spoken turn.

    Why this rather than a microphone: there is no audio input on this machine.
    The speech arrives from a browser over a WebSocket, which means the only
    recorder we have is the one already in the pipeline — and that is the right
    one anyway, because it captures the audio through exactly the path
    production uses, browser resampling and all. Test material recorded any
    other way would be measuring a signal we never actually serve.

    The pairing is positional: the Nth turn in the session is the Nth phrase in
    :data:`PHRASES`. That is why the instructions say one phrase per turn and
    pause between. It is checked, not assumed — a count mismatch stops here
    rather than silently scoring the wrong ground truth against the wrong audio.
    """
    stem = session.replace(".jsonl", "")
    meta = ROOT / "recordings" / f"{stem}.jsonl"
    wav = ROOT / "recordings" / f"{stem}.mic.wav"
    if not meta.is_file() or not wav.is_file():
        raise SystemExit(f"need both {meta.name} and {wav.name} in recordings/")

    turns, prev = [], 0.0
    for line in meta.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "mark" and d.get("label") == "turn_end":
            t = d.get("t_ms", 0) / 1000.0
            turns.append((prev, t, d.get("text", "")))
            prev = t

    if len(turns) != len(PHRASES):
        print(f"found {len(turns)} turns but {len(PHRASES)} phrases — pairing by "
              f"position needs them equal.\nturns heard:")
        for i, (a, b, txt) in enumerate(turns, 1):
            print(f"  {i:2d}. {a:6.1f}-{b:6.1f}s  {txt[:70]}")
        raise SystemExit("re-record, or edit scripts/entities.json by hand")

    cases = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else []
    for (start, end, heard), (text, ents) in zip(turns, PHRASES):
        cases.append({
            "file": f"recordings/{wav.name}",
            "start": round(start, 1), "end": round(end, 1),
            "entities": ents,
            # What the LIVE model made of it, kept so the accurate pass has
            # something to beat and a regression is visible rather than implied.
            "heard": {e: heard for e in ents if not hit(heard, e)},
            "note": text,
        })
    MANIFEST.write_text(json.dumps(cases, indent=2) + "\n")
    n = sum(len(p[1]) for p in PHRASES)
    print(f"added {len(PHRASES)} clips / {n} entities to {MANIFEST.name}\n"
          f"now run: python scripts/bench_entities.py --models parakeet,canary,...")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="parakeet,canary",
                    help="comma-separated; see MODELS table (parakeet, canary, sarvam, parakeet-v2, canary-1b, whisper-v3)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lang", default="ta-IN", help="language code for sarvam")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--phrases", action="store_true",
                    help="print the script to read into the agent UI, then exit")
    ap.add_argument("--ingest", metavar="SESSION",
                    help="turn a recorded session (agent-YYYYMMDD-HHMMSS) into "
                         "manifest entries, one per spoken turn")
    args = ap.parse_args()

    if args.phrases:
        phrases()
        return
    if args.ingest:
        ingest(args.ingest)
        return

    path = Path(args.manifest)
    if not path.is_file():
        path.write_text(json.dumps(SEED, indent=2) + "\n")
        print(f"wrote starter manifest to {path} ({len(SEED)} clips from real sessions)")
    cases = json.loads(path.read_text())

    # Load the audio once and share it, so a slow model cannot be blamed for
    # disk time and every model sees byte-identical input.
    clips = []
    for c in cases:
        f = ROOT / c["file"]
        if not f.is_file():
            print(f"  skip {c['file']} — not found")
            continue
        clips.append((c, load_clip(f, c["start"], c["end"])))
    if not clips:
        raise SystemExit("no usable clips in the manifest")

    total = sum(len(c["entities"]) for c, _ in clips)
    print(f"\n{len(clips)} clips, {total} entities to recover\n")

    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        # One candidate failing to load must not cost us the whole sweep — these
        # runs take model-download minutes and the point is the comparison.
        try:
            model = build(name, args.device, args.lang)
            asyncio.run(model.start())
        except Exception as e:
            print(f"\n{'=' * 72}\n{name}: UNAVAILABLE — {type(e).__name__}: "
                  f"{str(e)[:160]}\n{'=' * 72}")
            continue
        found, elapsed = 0, 0.0
        print(f"{'=' * 72}\n{model.name}\n{'=' * 72}")
        for c, audio in clips:
            t0 = time.perf_counter()
            text = model.transcribe(audio) or ""
            elapsed += (time.perf_counter() - t0) * 1000.0
            marks = []
            for e in c["entities"]:
                ok = hit(text, e)
                found += ok
                was = c.get("heard", {}).get(e)
                marks.append(f"  {'HIT ' if ok else 'MISS'}  {e}"
                             + ("" if ok or not was else f"   (live model heard: {was})"))
            print(f"\n{Path(c['file']).name}  {c['start']:.0f}-{c['end']:.0f}s"
                  f"  — {c.get('note','')}")
            print("\n".join(marks))
            print(f"  transcript: {text[:300] or '(nothing)'}")
        results[model.name] = (found, total, elapsed)
        asyncio.run(model.stop())

    print(f"\n{'=' * 72}\nentity recall — the only number this script is about\n{'=' * 72}")
    for name, (found, tot, ms) in results.items():
        print(f"  {name:22s} {found}/{tot}  {100.0 * found / max(1, tot):5.1f}%"
              f"   {ms / len(clips):7.0f} ms/clip")
    print("\nWER is not reported on purpose: it barely moves between these models "
          "and it is not what is broken.\n")


if __name__ == "__main__":
    main()
