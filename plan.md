# ClipAgent: Local AI Stream Clipper (Plan)

## 1. Objective

A free tool that runs entirely on my own PC. It does what Vizard AI does, without the cloud.

**Input:** a YouTube, Twitch or Kick stream or VOD link, or an uploaded video file.
**Output:** the best viral-worthy moments cut out as separate clips, each with:
- the `.mp4` clip, cut from the original video and unchanged
- 3 title options written for YouTube Shorts
- a plain description of what actually happens in the clip
- a score with a breakdown of why it was picked

**Quality target:** Vizard-level clip selection, meaning complete moments (setup, then peak, then payoff) with a real human reaction, not random gameplay.

**Out of scope for now:** vertical 9:16 layout (facecam on top, gameplay below) and burned-in captions. Both were deliberately postponed until clip selection is good.

**Content source:** mostly other streamers' VODs (for example Kai Cenat).

---

## 2. Hardware and environment

| Item | Value |
|---|---|
| OS | Windows (PowerShell) |
| GPU | NVIDIA RTX 3050, **4 GB VRAM** |
| RAM | 16 GB |
| CPU | Ryzen 5 |
| Project folder | `C:\clipagent` |
| Python | 3.11.9 (always install with `python -m pip ...`) |
| FFmpeg | 9.0.2, built with the `whisper` filter (whisper.cpp) |
| yt-dlp | 2026.10.26 |
| Ollama | 0.40.16 |
| Text model | `qwen2.5:3b` (1.9 GB) |
| Vision model | `qwen2.5vl:3b` (3.2 GB) |
| Whisper model | `ggml-small.bin` (for the FFmpeg whisper filter) |
| Speaker model | `wespeaker_en_voxceleb_resnet34.onnx` via `sherpa-onnx` 2.0.1 (CPU) |
| Other Python libraries | numpy, faster-whisper + CUDA libraries (installed, not used yet), opencv-python (used by the retired findcam.py), chat-downloader |

**Key constraints**
- 4 GB of VRAM means **only one model is loaded at a time**. Each stage unloads its model before the next stage loads one.
- Smart App Control was turned **off**. It had been blocking `pip.exe` and the ctranslate2 DLLs. Native ML libraries now load, and the `ctranslate2` check found 1 CUDA device.
- Decision: **stay on native Windows, not WSL2.** WSL2 isn't faster (the GPU does the work either way) and adds complexity.

---

## 3. How it works: signal-first, multimodal

Lesson learned: a small LLM reading a transcript **cannot** find viral moments. It picks game dialogue and makes up reactions. So the design is:

> **Measure real reactions first, then let AI judge only the strongest windows.**
> The AI never decides *whether* the streamer reacted. Measured signals decide that. The AI judges story, hook and context, and writes text.

This mirrors what Vizard publicly describes: multimodal analysis (sentiment, pacing, storytelling), complete clips (setup, then insight or peak, then payoff), and a scored breakdown per clip.

### Pipeline stages

```
fetch.py  ->  analyze.py  ->  judge.py  ->  cut_clips.py
(download)    (signals)       (AI judge)    (export)
```

**1. `fetch.py`: download**
- Downloads the first N hours (default 2) at up to 720p. If the file is over the size cap (default 5 GB), it retries at 480p, then 360p.
- Deletes leftover files from the previous stream so stale caches can't be reused.
- Saves `camcheck.jpg` (the facecam crop) for a visual check.
- Saves YouTube "Most replayed" heatmap data (`stream.heatmap.json`) for a future upgrade.
- Downloads the chat with `getchat.py` (chat-downloader, limited to the first N hours). Falls back to the yt-dlp full chat replay.

**2. `analyze.py`: measure the whole video**
1. Transcribe everything with the FFmpeg whisper filter, in 10-minute chunks with resume support. Drops common Whisper hallucinations.
2. **Separate the streamer's voice from game audio.** A speaker-embedding fingerprint (WeSpeaker via sherpa-onnx) finds the voice that talks the most and labels each line `STREAMER` or `GAME/OTHER`. Override with `--ref start,end` or `--voice-thr`.
3. Measure signals per second:
   - **voice**: streamer loudness in the 300–3400 Hz band, only on streamer lines
   - **rate**: streamer speech speed
   - **laugh**: laughter tags from Whisper
   - **hype**: exclamation words ("no way", "let's go", "what the"...)
   - **game**: game-sound spikes while the streamer is silent
   - **chat**: chat message-rate spikes plus hype/laugh messages, and super chats weighted ×5
4. Combine them (weights: voice 1.0, chat 1.0, laugh 0.8, hype 0.6, rate 0.4, game 0.4), then pick the top 40 candidate peaks. Each is a ±14 s window, at least 45 s apart, and scores below 2.5 are dropped.

Outputs: `stream_lines.json`, `stream_chat.json`, `stream_cands.json`, `stream_meta.json`.

**3. `judge.py`: multimodal judgment**
1. **Face:** facecam motion compared with the streamer's own baseline, plus the vision model reading 5 facecam frames (peak −8, −4, 0, +4, +8 s). It labels emotion (laughing, excited, shocked, angry, frustrated, sad, crying, scared, neutral, absent) and intensity from 0 to 10. Results are cached in `stream_face.json`.
2. **Reaction score (computed from measurements, not by the LLM):** combines voice, face and chat on a 0–10 scale. Moments under the gate (2.5) are **rejected before the LLM sees them**.
3. **LLM judge** (text model), with the virality playbook in its prompt. It sees the transcript with S/G line labels, the measured evidence and sample chat messages. It returns:
   - hook, story, share and self-contained scores (0–10)
   - a type: Funny fail, Jump scare, Rage, Clutch, Plot twist, Hot take, Chat moment, Hype, Funny line, or Not a moment (rejected)
   - start, peak and end **by transcript line number** (it can't invent timestamps)
   - a verified quote (checked against the streamer's lines; −5 if it can't be verified)
   - a one-sentence description of what happens
4. **Final score (0–100)** = 35% reaction + 20% hook + 15% story + 10% share + 10% self-contained, plus an agreement bonus (2 signals +4, 3 signals +7, 4 or more +10).
5. Keeps clips scoring at least 50, with less than 40% overlap between them, up to 8 clips.
6. Writes 3 titles per clip (35–60 characters, hook words first, game or streamer name included, no clickbait the clip doesn't deliver). It tries JSON first, then plain text, then falls back to the description.

**4. `cut_clips.py`: export**
- Re-encodes each clip for frame-accurate cuts (x264 CRF 20, AAC 160k). `--copy` does a fast cut that only lands on keyframes.
- Writes `clips/NN_type_time.mp4` plus `clips/titles.txt` with the score, type, description and titles.

**`run.py`** runs all four steps from one command.

---

## 4. Virality rubric (what the AI is taught)

Based on what YouTube says publicly about Shorts (viewed vs. swiped away, watch percentage, replays, engagement) and on observed clip-tool behaviour:

1. **Hook in 1–3 s:** tension, a bold line or a visible event right away. No "so, um" start.
2. **Complete arc:** short setup (3–8 s), then the peak, then the payoff. End 1–3 s after the payoff.
3. **A real human reaction is the product:** shouting, laughing, disbelief, rage, panic, hype, chat exploding. A calm streamer narrating gameplay is not a moment.
4. **Self-contained:** a stranger understands it within 5 s.
5. **Length:** 15–35 s (punchy) or 45–58 s (story). Never over 58 s.
6. **Strong types:** fail, jump scare, rage, clutch, plot twist, hot take, chat moment, hype, funny line.
7. **Weak:** menus, loading, cutscenes, planning, mechanics explanations, silence, game characters talking with no reaction from the streamer.
8. **Titles:** describe the stakes or the event and create curiosity, include the game and streamer name (searchable), never over-promise.

Honest limit: no one outside YouTube knows the algorithm, Vizard included. The closest you can get is **real viewer data**: chat, Most replayed, and what clip channels already made go viral (see the roadmap).

---

## 5. Commands

```powershell
cd C:\clipagent
python fetch.py "<url>" --hours 2
start camcheck.jpg
python analyze.py stream.mp4
python judge.py stream.mp4 --game "Kai Cenat Playing Every GTA"
python cut_clips.py stream.mp4
# or, once everything works:
python run.py "<url>" --game "..." --hours 2 --keep 8
```

Useful flags:
- `analyze.py`: `--ref 300,330` (a stretch where only the streamer talks), `--voice-thr 0.4`, `--topn 40`, `--min-comp 2.5`, `--lang en`
- `judge.py`: `--cam x,y,w,h`, `--top 24`, `--keep 8`, `--min-score 50`, `--gate 2.5`, `--text-model qwen2.5:7b`, `--no-vision`, `--fresh`, `--debug`
- `cut_clips.py`: `--n 8`, `--copy`

---

## 6. Known limits

- **Mixed audio:** the mic and game share one track. The voice fingerprint labels lines, but game sound under the streamer's voice can't be removed, so loudness is approximate.
- **Guests or co-streamers** are labelled "other" (fine for solo streams).
- **3B models:** judgment and titles are decent but generic. A 7B judge for the finalists is planned.
- **Facecam position is hardcoded** (`DEFAULT_CAM = 0, 0.245, 0.208, 0.207`). Auto-detection per segment is needed for "any stream".
- **The judge can't see gameplay**, only the face, transcript and signals, so descriptions can be vague.
- **Transcript timestamps are per sentence** (FFmpeg whisper), not per word.
- **Reused content:** YouTube demotes reposted streams with nothing added. Credit the streamer, add commentary or edits, or get permission.

---

## 7. Roadmap (in priority order)

1. **Evaluation set (quality metric).** Find Shorts that clip channels already posted from a test stream, note their VOD timestamps, and measure the pipeline's **hit rate**. This is the "are we at Vizard level" number, used to judge every change.
2. **Most Replayed heatmap** as a signal: real viewers' rewatch data. YouTube only shows it on videos with enough views. It has about 100 points across a whole video, so it's a coarse bonus.
3. **Gameplay understanding:** a vision caption of the full frame at each peak, so the description and titles say what actually happened.
4. **Auto facecam detection** per segment (bring back a better `findcam.py`). Handles layout changes and different streamers.
5. **faster-whisper on GPU** with word timestamps and voice-activity detection, for cleaner cut points.
6. **7B judge** (`qwen2.5:7b`) for the finalists only.
7. **Targeted mode:** "find every rage quit" or "every clutch", searching the transcript and signals for a requested moment type.
8. **Local app (Gradio):** paste a link or upload a file, see progress, browse clips with titles and descriptions, approve or reject.
9. **Twitch and Kick:** video via yt-dlp. Twitch chat via chat-downloader. Kick runs without chat.
10. **Learning loop:** log approved and posted clips and their Shorts performance (vidIQ or YouTube Analytics), then retune the weights from data.
11. **Later:** vertical 9:16 facecam/gameplay layout, animated captions, SEO-optimised titles and tags (vidIQ keyword data).
