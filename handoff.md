# Voice Pipeline Handoff — VAD Layer + Two-ASR Plan

This doc explains the plan for two parts of the pipeline: the VAD layer (deciding when someone is talking and when they're done) and the ASR layer (turning speech into text). Written in plain words so anyone can pick it up.

Noise suppression, TTS, and barge-in are out of scope for now. We fix VAD and ASR first.

---

## Part 1 — The big picture in one paragraph

We have three separate jobs, and we keep them separate on purpose. First, decide if there is speech right now (for fast reactions). Second, decide if the person sounds finished (from the tone of their voice). Third, decide if the person is actually finished (from the meaning of their words). Then we combine the last two into one "are they done?" answer. For text, we run two speech-to-text models: a fast one that runs live and drives the conversation, and a slow but more accurate one that produces the final clean text. Fast model = Parakeet. Accurate model = Cohere.

---

## Part 2 — The VAD layer (three signals)

We do NOT have just one VAD. We have three signals. Give each one a clear job and test them one at a time.

**S1 — Acoustic VAD (is there speech?)**
- Input: audio, in 80ms chunks.
- Output: yes/no, is this speech.
- Model: TEN or Silero.
- Job: catch interruptions fast, and count how long speech and silence have lasted. These counts drive everything downstream.
- It knows nothing about words or whether the person is "done." It only knows "sound that looks like speech is happening."

**S2 — Acoustic turn (do they sound done?)**
- Input: audio (the recent part of what they said).
- Output: a score, how likely they're finished.
- Model: smart-turn-v2.
- Job: listen to the tone and rhythm. A falling pitch usually means "done." A mid-sentence pause usually means "not done." No text involved.

**S3 — Semantic turn (are they actually done?)**
- Input: the text from ASR.
- Output: a score, how likely they're finished.
- Model: the turn model that reads text.
- Job: judge the meaning. "Book a flight to..." sounds done by tone but is clearly not done by meaning. This is the only VAD-side piece that needs ASR text.

**Fusion + endpoint**
- Combine S2 and S3 into one "are they done?" score.
- Only act on it based on S1's silence count.
- Keep a hard cap (about 8 seconds) so a turn always ends eventually, even if everything else is unsure.

---

## Part 3 — VAD fixes we already agreed on

These are the specific problems in the current code and how to fix them.

**Fix 1 — Stop letting one tiny spike count as speech.**
Right now each 80ms chunk is made of five 16ms frames, and we take the highest score of the five. So one short noise (a click, a cough start, a pop) in one frame makes the whole 80ms count as speech. Change this: require at least 2 frames (or 2 in a row) over the line before calling it speech. Real speech easily clears this. A short noise does not.

**Fix 2 — Confirm speech start with 2 chunks.**
Don't declare "speech started" on the very first chunk. Wait for about 2 chunks. We won't lose the beginning of the word because we already keep a 250ms buffer of earlier audio that fills it back in. Don't wait longer than needed — soft word beginnings and whispers are short.

**Fix 3 — Re-check the 0.65 threshold for TEN.**
The 0.65 number was likely set for Silero (which used 0.45). We now run TEN, and TEN's scores are on a different scale. So 0.65 is currently a guess. Measure it on TEN and set it properly. Don't just raise it to block coughs — a loud cough can score higher than quiet speech, so raising it would block the quiet speech and still let the cough through.

**Fix 4 — Fix the re-onset / double-audio bug (highest priority — this is a real bug, not a tune-up).**
Problem: when someone pauses for more than 350ms and starts again, the system treats it as a brand new start. It tears down the STT stream, may waste a speculative call, and sends the 250ms buffer again — some of which was already sent. This causes duplicate text and wasted work.
Fix: tell the difference between two cases.
- Cold start = a truly new utterance (no live STT stream). Send the buffer, reset the stream. Do this once per utterance.
- Warm resume = same utterance, short pause, stream still alive. Just keep feeding audio. No buffer resend, no reset.

**Fix 5 — Only reset the silence timer when there are real words.**
Problem today: a tiny noise during a pause makes VAD say "speech," which resets the silence timer to zero, so the system waits another 800ms for no reason.
Fix (only possible with the new ASR): reset the silence timer only when VAD says speech AND the ASR actually produced a real word in that window. Parakeet stays quiet (produces "blank") on non-speech, so a cough that fools VAD but produces no words will not reset the timer. This is a big latency win.

**Measure before you change anything.**
Log, for each chunk: the VAD score, the five raw frame scores, and whether ASR produced a real word or blank for that same moment. From this, find out: how often does VAD spike while ASR sees no words (that's your false-alarm rate), and how often does warm-resume actually happen (that tells you how bad the re-onset bug really is). Tune with these numbers, not guesses. Extend the existing bench scripts instead of writing new ones.

---

## Part 4 — VAD build order (test each piece alone)

1. **Phase 0 — Measurement.** Add the logging above. No behavior change yet.
2. **Phase 1 — S1 alone.** Do Fixes 1, 2, 3. Test with cough/clap/click clips (should NOT hold as speech) and whisper/soft clips (SHOULD still trigger). Stub out S2 and S3.
3. **Phase 2 — S2 alone.** Feed it clean sentences with known endings. Check falling-pitch = high score, mid-pause = low score. Stub S3.
4. **Phase 3 — S3 alone.** Feed it live text from ASR. Test "book a flight to..." and letter-by-letter spelling. Check it correctly overrules tone when the sentence isn't finished.
5. **Phase 4 — Fuse + wire.** Reconnect S2+S3 and the endpoint logic. Now also fix the re-onset bug (Fix 4), since it lives here. Any weirdness now is fusion weirdness, because the pieces were each tested alone.

Two things to accept, not fight:
- S1 will never fully reject laughter and heavy breathing. Those sound like speech. That's fine — S2 and S3 are the backstop for them. Don't try to fix laughter in Phase 1.
- S3 is only as good as the ASR text it reads. Some S3 mistakes are really ASR mistakes wearing a costume. Note them, don't chase them here.

---

## Part 5 — The two-ASR plan (Parakeet + Cohere)

**The idea in one line:** Parakeet runs live and drives the conversation. Cohere runs once per turn and gives the final clean text. Each does the one thing it's good at.

**Parakeet 0.6B v3 — the fast live model.**
- Streams word by word, in real time.
- Stays quiet on non-speech (produces "blank"), so it does not make up words. This is the key property.
- Its live text feeds: S3 (semantic turn), speculation, and interruption detection.
- Weak spot: number/date formatting is inconsistent. We add a formatting step for that (see Part 7).
- It never has to win on accuracy. It has to be fast and quiet-on-silence.

**Cohere Transcribe 2B — the accurate final model.**
- Lowest word error rate available, open source (Apache-2.0), runs on a normal GPU (~4GB), via a vLLM HTTP endpoint.
- BUT — two important limits we design around:
  1. It does NOT stream. It transcribes a whole clip at once. (It's "3x faster than real time" as throughput, but that is not the same as live streaming.)
  2. It is "eager" — like the old Qwen model, it will try to transcribe non-speech and can make up words. Same hallucination risk we were trying to escape.
  3. It's weak on mixed-language (code-switching) speech and has no auto language detection.
- It never runs live and never sees a pause. It only ever gets a clean, finished clip. That is how we avoid both of its problems.

**Why the VAD layer makes Cohere safe:** Cohere only invents words when it's fed silence or noise. Our VAD decides exactly where speech starts and ends, so we only ever send Cohere a clean speech clip. No silence in, no made-up words out. This is why we built VAD first.

---

## Part 6 — How the two models work together (the important part)

**New thing we need to add: an utterance audio buffer.**
Keep a copy of the raw audio for the current utterance — from the start (including the 250ms lead-in buffer) to the end (including the 350ms trailing bit). One buffer per utterance. It fills across warm-resumes. It clears when the turn is committed. This buffer is what we hand to Cohere.

**When does Cohere run?**
On a confirmed turn end (not on a guess/speculative end — we don't want it firing and re-firing).

**The latency problem (must understand this):**
A 3-second clip takes Cohere about 1 second to process. If we wait for Cohere before replying, every turn gets a full second of dead air. That kills the conversation feel. So we must NOT do "turn ends -> call Cohere -> wait -> reply."

**The fix — use speculation as a bridge:**
1. The moment the turn is confirmed, start the reply using Parakeet's text right away (we already have it, so almost no delay).
2. Run Cohere at the same time, in parallel.
3. When Cohere comes back (~1s later), compare it to Parakeet's text:
   - **They basically match** (the normal case): the reply we already started is fine. Let it go. Cohere's delay is completely hidden.
   - **They clearly differ:** stop the reply, restart it using Cohere's better text. We only pay the ~1s delay in this case — which is exactly the case where accuracy matters enough to be worth it.

**Cohere is allowed to fail. Never block on it.**
Set a hard timeout (~800ms to 1s). If Cohere is slow or errors, just use Parakeet's final text and move on. Cohere is a bonus layer for quality, not something the pipeline depends on. Worst case, we fall back to a fully working Parakeet-only system.

---

## Part 7 — ASR build order

1. **Phase 1 — Swap in Parakeet.** Replace the current streaming STT with Parakeet, change nothing else. Make sure turns, live text, speculation, and interruptions all still work. Stabilize here.
2. **Phase 2 — Add the utterance buffer.** No behavior change. Just save the clips and listen to a few, to confirm the start/end points are correct.
3. **Phase 3 — Cohere in shadow mode (do NOT skip).** On each turn end, run Cohere on the saved clip and log its text next to Parakeet's — but do NOT send it to the LLM yet. Measure three things on our own audio and our own GPU:
   - real per-clip Cohere delay,
   - how often Cohere and Parakeet agree,
   - where they disagree (watch accented and mixed-language turns).
   This data tells us if the whole plan works before we go live.
4. **Phase 4 — Turn Cohere on for real.** Wire the compare-and-pick rule, the timeout fallback, and send Cohere's text to the LLM. We tune using Phase 3 numbers, not guesses.
5. **Phase 5 — Formatting + custom words.** Add a formatting step (NeMo ITN) for numbers, dates, money on the final text. Check if Cohere lets us bias toward our own words (product names, "Zoho", etc.). If it doesn't, Parakeet might actually do better on those specific words — test it.

---

## Part 8 — How it runs

- Two ASR services now.
- Parakeet: a streaming service over WebSocket, same pattern as the current asr_server.
- Cohere: a vLLM HTTP endpoint, called once per clip (batch style).
- Both launched conditionally in run.sh, same as the current setup — just add the second.
- Separate GPUs, or split GPU memory if tight.

---

## Part 9 — Risks to watch (be honest about these)

1. **The whole latency trick depends on Parakeet and Cohere agreeing most of the time.** If Phase 3 shows they disagree a lot, we'd be restarting the reply constantly and the speed benefit disappears. Measure agreement first — this number makes or breaks the plan.
2. **Cohere is weak on mixed-language speech, and nothing here fixes that.** For bilingual turns, Parakeet's text may be better. Consider picking the transcript per-turn based on a language signal, instead of always using Cohere.
3. **Cohere's number/date formatting isn't strong.** Plan on the NeMo formatting step no matter what.
4. **S1 will never fully block laughter/breathing.** That's expected. S2/S3 handle it.
5. **S3 quality is capped by ASR quality.** Some turn mistakes are really text mistakes. Don't fix them in the turn layer.

---

## Quick summary

- Three VAD signals, each tested alone: is-there-speech (S1), sounds-done (S2), means-done (S3).
- Fix the max()-spike, the start confirmation, the TEN threshold, the re-onset double-audio bug, and the silence-reset-on-noise problem.
- Two ASR models: Parakeet live and fast (drives everything), Cohere slow and accurate (final text only).
- Hide Cohere's 1-second delay by starting the reply on Parakeet and swapping to Cohere only if the text really differs.
- Cohere can fail safely; Parakeet alone is a working system.
- Measure in shadow mode before going live.