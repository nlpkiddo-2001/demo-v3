"""Phase 0 measurement — record exactly what the pipeline hears, and decide nothing.

Why this file exists
--------------------
handoff.md Part 3 ends with "Measure before you change anything", and Part 4
Phase 0 is explicitly "add the logging, no behaviour change yet". This module is
that phase. It is deliberately *passive*: it never gates audio, never moves a
threshold, never ends a turn. It only writes down what happened, so that the
numbers we tune with later are measured rather than guessed.

What it records, per session
----------------------------
1. ``<name>.mic.wav`` — the raw microphone audio, 16 kHz float32, exactly as the
   VAD/STT layers receive it. Recording the *audio* and not just the scores is
   the whole point: it lets us re-run a different VAD, a different threshold or
   a denoiser over the same take offline and compare fairly, without asking
   anyone to re-record. Measure once, analyse many times.

2. ``<name>.ref.wav`` — the agent's own TTS output on the same timebase,
   zero-padded through the gaps where the agent was silent, so it stays
   sample-aligned with the mic track. That alignment is what makes offline
   echo-leak analysis possible: line the two up and you can see how much of the
   agent's voice survived the browser's AEC3 into our microphone signal.

3. ``<name>.jsonl`` — one line per audio chunk: level, the VAD score, the raw
   per-frame scores behind that score, whether ASR saw a real word, and whether
   the agent was speaking at the time.

Which handoff question each field answers
-----------------------------------------
- ``frames``            → Fix 1. Does one 16 ms spike drag a whole 80 ms chunk
                          over the line? Compare max(frames) against how many
                          frames actually cleared the threshold.
- ``vad`` + ``dbfs``    → Fix 3. What do TEN's scores really look like on our
                          audio and our room, so 0.65 stops being a guess
                          inherited from Silero.
- ``asr_blank``         → Fix 5. How often does VAD fire while ASR produces no
                          word? That is the false-alarm rate, and it is the
                          number that decides whether gating the silence-timer
                          reset on real words is worth it.
- ``agent`` + ref track → the echo question. Mic energy well above the noise
                          floor while the agent is speaking and the user is not
                          is AEC3 leakage.

This module computes no verdicts and draws no conclusions. Analysis lives in
``scripts/`` and runs over the recording afterwards.

Usage
-----
    rec = SessionRecorder("recordings", "kitchen-noise-take1")
    rec.mark("about to cough")
    rec.chunk(audio16k, vad=0.81, frames=[0.1, 0.2, 0.9, 0.3, 0.1], agent=False)
    rec.ref(tts_audio16k)          # whenever agent audio is sent to the browser
    summary = rec.close()
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16000

# Bumped whenever the JSONL field set changes, so an analysis script can tell
# which layout it is reading instead of guessing from the keys present.
SCHEMA_VERSION = 1

# Level reported for digital silence (and for an empty chunk). Deliberately a
# finite sentinel rather than -inf: -inf serialises as `-Infinity`, which Python
# will re-read but which is not valid JSON, and it would break jq and every other
# standard tool we might point at these logs. Analysis should treat anything at
# or below this as "no signal at all".
SILENCE_DBFS = -240.0


def rms_dbfs(audio: np.ndarray) -> float:
    """Return the RMS level of ``audio`` in dBFS (0 dB = full scale).

    Rough reference points for 16 kHz mic audio: a quiet room floor sits around
    -60 dBFS, normal speech around -25 dBFS, clipping at 0 dBFS. This is the
    cheap "how loud was it" number we log for every chunk — it is what lets the
    analysis pass estimate the room's noise floor and signal-to-noise ratio.

    Digital silence returns :data:`SILENCE_DBFS`.
    """
    if audio.size == 0:
        return SILENCE_DBFS
    mean_sq = float(np.mean(audio.astype(np.float64) ** 2))
    return max(SILENCE_DBFS, 20.0 * math.log10(math.sqrt(mean_sq) + 1e-12))


def peak_dbfs(audio: np.ndarray) -> float:
    """Return the peak (loudest single sample) level of ``audio`` in dBFS.

    Logged alongside RMS because they disagree in a useful way: a click or a pop
    has a high peak but a low RMS, whereas steady speech has both. That gap is
    one of the cheapest signals for spotting the transient noises in Fix 1.

    Digital silence returns :data:`SILENCE_DBFS`.
    """
    if audio.size == 0:
        return SILENCE_DBFS
    loudest = float(np.max(np.abs(audio.astype(np.float64))))
    return max(SILENCE_DBFS, 20.0 * math.log10(loudest + 1e-12))


@dataclass
class ChunkRecord:
    """One row of the measurement log — a single audio chunk as we saw it.

    Every field is either something we observed or something a caller handed us.
    Nothing here is derived from a decision the pipeline made.
    """

    i: int                          # chunk index, from 0
    t_ms: float                     # position in the session, from the audio clock
    n: int                          # samples in this chunk
    dbfs: float                     # RMS level
    peak: float                     # peak level
    vad: float | None = None        # the chunk-level VAD score the caller computed
    frames: list[float] = field(default_factory=list)   # raw per-frame scores behind it
    asr_text: str | None = None     # what ASR emitted for this chunk ("" = blank)
    agent: bool = False             # was the agent playing audio at this moment
    extra: dict = field(default_factory=dict)   # room to add signals without a schema bump

    def to_json(self) -> dict:
        """Compact dict for the JSONL file — rounded, and empty fields dropped.

        ``asr_text`` is deliberately kept when it is an empty string but dropped
        when it is ``None``: "ASR ran and produced a blank" and "ASR was not
        wired up yet" are different facts, and Fix 5 depends on telling them
        apart.
        """
        row: dict = {
            "type": "chunk",
            "i": self.i,
            "t_ms": round(self.t_ms, 1),
            "n": self.n,
            "dbfs": round(self.dbfs, 2),
            "peak": round(self.peak, 2),
        }
        if self.vad is not None:
            row["vad"] = round(self.vad, 4)
        if self.frames:
            row["frames"] = [round(float(f), 4) for f in self.frames]
        if self.asr_text is not None:
            row["asr_text"] = self.asr_text
            row["asr_blank"] = not self.asr_text.strip()
        if self.agent:
            row["agent"] = True
        if self.extra:
            row["extra"] = self.extra
        return row


class SessionRecorder:
    """Writes one measurement session to disk: mic wav + ref wav + jsonl.

    Cost and safety
    ---------------
    The JSONL rows are buffered and written every ``flush_every`` chunks (default
    50, about 4 s of audio) as a single joined write, so the common path costs a
    list append. WAV frames stream straight to an open handle. That is fast
    enough to sit in the audio path, but it is still blocking file I/O — callers
    on the asyncio event loop should hand ``chunk()`` to an executor, exactly as
    the session already does for VAD and STT inference.

    Failure policy: recording is diagnostics, never the product. Every write is
    guarded, and a disk error disables the recorder and logs once rather than
    taking the voice pipeline down with it.
    """

    def __init__(
        self,
        out_dir: str | Path,
        name: str,
        *,
        sample_rate: int = TARGET_SR,
        record_audio: bool = True,
        record_ref: bool = True,
        flush_every: int = 50,
        meta: dict | None = None,
    ):
        """``record_ref=False`` skips the agent-audio track.

        Worth turning off for offline work: a bench run has no agent, so the ref
        track would be an exact-length file of pure zeros — same size as the mic
        recording, and no information in it at all.
        """
        self.dir = Path(out_dir)
        self.name = name
        self.sample_rate = sample_rate
        self.record_audio = record_audio
        self.record_ref = record_ref
        self.flush_every = max(1, flush_every)
        self.enabled = True

        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.dir / f"{name}.jsonl"
        self.mic_path = self.dir / f"{name}.mic.wav"
        self.ref_path = self.dir / f"{name}.ref.wav"

        self._buf: list[str] = []
        self._summary: dict | None = None
        self._n_chunks = 0
        self._n_marks = 0
        self._mic_samples = 0   # session length in samples — the shared timebase
        self._ref_samples = 0   # how far the ref track has been written
        self._t0 = time.monotonic()

        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")
        self._mic = self._open_wav(self.mic_path) if record_audio else None
        self._ref = self._open_wav(self.ref_path) if (record_audio and record_ref) else None

        self._write(
            {
                "type": "header",
                "schema": SCHEMA_VERSION,
                "name": name,
                "sample_rate": sample_rate,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "meta": meta or {},
            }
        )
        log.info("measurement recording -> %s", self.jsonl_path)

    # ── wav handles ──
    def _open_wav(self, path: Path):
        """Open a streaming float32 WAV writer.

        float32 rather than int16 on purpose: the recording is the input to
        offline denoiser and VAD comparisons, and we do not want a quantisation
        step of our own sitting between the microphone and those experiments.
        """
        import soundfile as sf

        return sf.SoundFile(
            str(path), mode="w", samplerate=self.sample_rate,
            channels=1, subtype="FLOAT",
        )

    # ── recording ──
    def chunk(
        self,
        audio16k: np.ndarray,
        *,
        vad: float | None = None,
        frames: list[float] | None = None,
        asr_text: str | None = None,
        agent: bool = False,
        **extra,
    ) -> ChunkRecord | None:
        """Record one chunk of mic audio plus whatever the caller observed about it.

        ``audio16k`` is 16 kHz mono float32 — the same array the VAD and STT
        layers are handed, so what we store is exactly what they judged.

        Returns the :class:`ChunkRecord` (handy in tests) or ``None`` if
        recording has been disabled by an earlier failure.
        """
        if not self.enabled:
            return None

        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        rec = ChunkRecord(
            i=self._n_chunks,
            # Time comes from the sample count, not the wall clock: it is immune
            # to scheduling jitter, so a chunk's position in the JSONL always
            # matches its position in the WAV. Analysis relies on that.
            t_ms=1000.0 * self._mic_samples / self.sample_rate,
            n=int(audio.size),
            dbfs=rms_dbfs(audio),
            peak=peak_dbfs(audio),
            vad=vad,
            frames=list(frames or []),
            asr_text=asr_text,
            agent=agent,
            extra=extra,
        )

        self._n_chunks += 1
        self._mic_samples += audio.size
        if self._mic is not None:
            self._safe(lambda: self._mic.write(audio))
        self._write(rec.to_json())
        return rec

    def ref(self, audio16k: np.ndarray) -> None:
        """Record a piece of agent (TTS) audio on the mic timebase.

        Call this whenever agent audio is handed to the browser. The stretch
        since the last call — where the agent was silent — is filled with zeros
        first, so the ref track stays aligned with the mic track instead of
        drifting apart the moment the agent pauses.

        Two honest caveats about that alignment:

        - This timestamps audio at the moment we *send* it, which is earlier than
          the moment it actually leaves the speaker — the browser buffers it. So
          the ref track leads the true echo by an unknown playback latency.
        - TTS generates faster than real time, so chunks arrive in bursts and the
          ref track can run ahead of the mic clock. When that happens we simply
          append (no negative padding), which keeps the ref *waveform*
          contiguous and correct even though its absolute position slips.

        Both mean the analysis pass must estimate the true lag by
        cross-correlating the two tracks rather than trusting sample 0 to line
        up. What this method guarantees is the part that matters for that: an
        unbroken ref waveform, plus a ``ref`` event in the JSONL recording the
        exact mic position of every handover, so the correlation has a starting
        point to search around.

        Audio must already be 16 kHz mono float32; 24 kHz TTS output should be
        resampled by the caller (``base.to_float16k`` does this) so a single rate
        is fixed for the whole recording.
        """
        if not self.enabled or self._ref is None:
            return
        audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
        gap = max(0, self._mic_samples - self._ref_samples)
        if gap > 0:
            self._safe(lambda: self._ref.write(np.zeros(gap, dtype=np.float32)))
            self._ref_samples += gap
        self._safe(lambda: self._ref.write(audio))
        self._write(
            {
                "type": "ref",
                "t_ms": round(1000.0 * self._mic_samples / self.sample_rate, 1),
                "ref_ms": round(1000.0 * self._ref_samples / self.sample_rate, 1),
                "n": int(audio.size),
            }
        )
        self._ref_samples += audio.size

    def mark(self, label: str, **fields) -> None:
        """Drop a labelled marker at the current position.

        This is how a recording gets ground truth. Call it from the bench script
        (or a UI button) right before coughing, clapping, whispering or letting a
        pause run long, and the analysis can then score VAD against what actually
        happened instead of against a human listening back to the take.
        """
        if not self.enabled:
            return
        self._n_marks += 1
        self._write(
            {
                "type": "mark",
                "i": self._n_chunks,
                "t_ms": round(1000.0 * self._mic_samples / self.sample_rate, 1),
                "label": label,
                **fields,
            }
        )

    # ── writing ──
    def _write(self, row: dict) -> None:
        self._buf.append(json.dumps(row, ensure_ascii=False))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows out. Safe to call at any time."""
        if not self._buf:
            return
        rows, self._buf = self._buf, []
        self._safe(lambda: self._jsonl.write("\n".join(rows) + "\n"))

    def _safe(self, fn) -> None:
        """Run a write, and disable recording (once, loudly) if the disk objects.

        A measurement harness must never be the reason a call drops.
        """
        if not self.enabled:
            return
        try:
            fn()
        except Exception:
            self.enabled = False
            log.exception("measurement recording disabled after write error (%s)", self.name)

    # ── teardown ──
    def close(self) -> dict:
        """Finish the recording and return a small summary of what was captured.

        The ref track is padded out to the mic length first, so both WAVs are the
        same duration and can be loaded and lined up without any offset maths.

        Idempotent: closing twice (say, an explicit ``close()`` inside a ``with``
        block) returns the original summary rather than a second, emptier one
        describing handles that are already shut.
        """
        if self._summary is not None:
            return self._summary

        summary = {
            "name": self.name,
            "chunks": self._n_chunks,
            "marks": self._n_marks,
            "duration_sec": round(self._mic_samples / self.sample_rate, 2),
            "wall_sec": round(time.monotonic() - self._t0, 2),
            "jsonl": str(self.jsonl_path),
            "mic_wav": str(self.mic_path) if self._mic is not None else None,
            "ref_wav": str(self.ref_path) if self._ref is not None else None,
        }

        if self._ref is not None:
            tail = self._mic_samples - self._ref_samples
            if tail > 0:
                self._safe(lambda: self._ref.write(np.zeros(tail, dtype=np.float32)))
                self._ref_samples += tail

        self._write({"type": "summary", **summary})
        self.flush()

        for handle in (self._jsonl, self._mic, self._ref):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    log.exception("error closing measurement handle (%s)", self.name)
        self._mic = self._ref = None
        self.enabled = False
        self._summary = summary

        log.info(
            "measurement done: %s — %d chunks, %.1fs audio, %d marks",
            self.name, summary["chunks"], summary["duration_sec"], summary["marks"],
        )
        return summary

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
