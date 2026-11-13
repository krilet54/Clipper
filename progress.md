# ClipAgent: Progress Log

_Last updated: 13 Nov 2026_

## Status at a glance

| Area | Status |
|---|---|
| Environment and installs | ✅ Done |
| Download (video + chat) | ✅ Working (first 2h of the Kai Cenat VOD downloaded) |
| Pipeline v2 scripts | ✅ Ran end to end on the Kai Cenat VOD (first 2h) |
| Streamer vs. game voice separation | ✅ Runs (accuracy still to be measured) |
| Multimodal judge (voice + face + chat + transcript) | ✅ Runs |
| Clip quality at Vizard level | ❌ Not yet: first v2 run gave **mostly junk, a few fair clips** |
| App / UI | ✅ Built: local Gradio app (`app.py`, `ClipAgent.bat`), ⏳ first run on your PC |
| GitHub | ⏳ Files ready (`.gitignore`, `README.md`, `requirements.txt`), first push pending |

**Current step:** push to GitHub and start the app. Then go back to improving clip quality.

---

## ✅ Done

### Setup
- [x] Installed Python 3.11.9, FFmpeg 9.0.2, yt-dlp, Ollama via winget
- [x] Fixed the **Application Control block**: `pip.exe` was blocked, so we used `python -m pip`. The ctranslate2 DLLs were blocked too, so **Smart App Control was turned off**. Now `ctranslate2` loads with 1 CUDA device.
- [x] Installed faster-whisper + cuBLAS/cuDNN (works now, not used yet)
- [x] Downloaded `ggml-small.bin`. The FFmpeg whisper filter runs at about 8.35× realtime (5 min of audio in about 36 s).
- [x] Pulled `qwen2.5:3b` and `qwen2.5vl:3b`
- [x] Installed `sherpa-onnx` 2.0.1 and upgraded yt-dlp to 2026.10.26
- [x] Decided to stay on Windows, not WSL2 (no speed gain)

### Version 1: transcript-only picker (retired)
- [x] `transcribe.py`, `pick_clips.py`, `cut_clips.py` (v1)
- [x] Fixed several small-model failures along the way: 1–3 s single-line picks (now padded), the model copying prompt examples, empty titles (split into separate pick and title passes, plus a plain-text retry), hangs (streaming, token cap, warm-up)
- [x] **Result: clips had no value.** It picked game cutscene dialogue, not streamer reactions. Lesson: transcript-only can't work, so the design became signal-first.

### Version 1.5: signals + facecam (retired)
- [x] `signals.py`: audio loudness + facecam motion
- [x] `findcam.py`: automatic facecam detection per 5-minute segment (OpenCV)
- [x] `judge.py` v1: vision emotion labels + LLM scoring
- [x] Facecam fixed to `0, 0.245, 0.208, 0.207` (bottom-left) for this streamer
- [x] **Result:** emotion labels matched the faces, but (a) candidates came from total loudness, which mostly meant game explosions and music, and (b) the judge **made up reactions** ("Rage", "reacts violently") while every face was neutral. Game and streamer voices were also mixed in the transcript.

### Version 2: current pipeline
- [x] Researched how Vizard, OpusClip and AutoClip score clips (multimodal; complete setup, peak, payoff; score breakdown)
- [x] `common.py`: shared helpers
- [x] `fetch.py`: first N hours, size cap, lower-resolution retry, cleans old stream files, camcheck, heatmap, chat
- [x] `getchat.py`: chat for the first N hours only (chat-downloader)
- [x] `analyze.py`: full transcription, **streamer vs. game voice fingerprint**, voice-band loudness, speech rate, laughter, hype words, game spikes, chat spikes
- [x] `judge.py`: face motion + vision emotion, **reaction score computed from measurements**, evidence gate, LLM judge with virality playbook, boundaries by line number, quote check, 0–100 score, titles
- [x] `cut_clips.py`: named clips + `titles.txt`
- [x] `run.py`: one-command pipeline
- [x] All 7 scripts bundled in `clipagent_scripts.zip` and installed in `C:\clipagent`

### Test data
- [x] **Test 1:** `test.mp4` (youtu.be/MX8wnu3xXeM, 1h03m, GTA V, facecam bottom-left, mic quiet under game audio)
- [x] **Test 2:** `stream.mp4`, Kai Cenat "Playing Every GTA" (youtube.com/live/6LB7vibQ5q0, VOD length 11h10m)
  - first 2h downloaded at 720p60 (AV1), **1.58 GB**, about 76 minutes to download
  - chat replay `stream.live_chat.json`, **104 MB** (whole stream; the analysis only uses the first 2h)
  - `camcheck.jpg` saved
  - **v2 result:** mostly junk clips, a few fair ones. My view: the first 2h of this stream may simply hold few strong moments, and there's no evaluation set yet to measure against.

### Local app
- [x] `app.py`: Gradio web app at http://127.0.0.1:7860. Takes a link, an upload or a local file path. Options: streamer/game name, hours, max clips, minimum score, facecam box. Shows live progress, step status and a log, has a Stop button, shows each clip with a player, score, what happens, evidence and titles. Every run is archived in `outputs\<date>_<name>\` and appears in a **Past runs** tab.
- [x] `ClipAgent.bat` (double-click launcher) + desktop shortcut command
- [x] GitHub files: `.gitignore` (keeps videos, models and per-stream data out of git), `README.md`, `requirements.txt`
- [x] Tested in a sandbox: UI builds, server starts, a full run flow (with stand-in scripts) archives and displays clips

---

## ⏳ Doing now (next commands)

- [ ] Move the retired scripts into `legacy\`
- [ ] `git init`, first commit, push to a private GitHub repo `clipagent`
- [ ] `python -m pip install -r requirements.txt`, then `python app.py` (or double-click `ClipAgent.bat`)
- [ ] Create the desktop shortcut
- [ ] Do a first real run inside the app

**Then, for quality:** send the `analyze.py` voice check and top-15 list, the `judge.py` output, and a verdict per clip, so we can tune the pipeline and start the evaluation set.

---

## 📋 To do (roadmap)

### Next: quality
- [ ] Fix whatever the first v2 run shows (bugs, voice threshold, weights, gate)
- [ ] **Evaluation set:** collect existing viral Shorts from the test stream, map them to VOD timestamps, measure hit rate
- [ ] Use the **Most Replayed** heatmap as a signal (data is already saved by the new `fetch.py`)
- [ ] **Gameplay vision caption** at each peak, for concrete descriptions and titles
- [ ] **Auto facecam detection** per segment (replace the hardcoded `DEFAULT_CAM`)
- [ ] Switch transcription to **faster-whisper (GPU) with word timestamps**
- [ ] Try the **`qwen2.5:7b` judge** on the finalists

### Then: features
- [ ] Targeted mode ("find every rage quit")
- [ ] App: approve or reject buttons per clip (feeds the learning loop), own desktop window instead of the browser
- [ ] Twitch and Kick support (video + chat)
- [ ] Learning loop from posted-clip performance (vidIQ / YouTube Analytics)
- [ ] SEO titles and tags (vidIQ keyword data)

### Later
- [ ] Vertical 9:16 layout (facecam on top, gameplay below)
- [ ] Animated burned-in captions

---

## ⚠️ Known issues and watch-outs
- yt-dlp shows "JS challenge / deno" warnings. Downloads still work. If formats go missing, add `--remote-components ejs:github` (not yet in `fetch.py`).
- The video is **AV1**, which is slower to decode on the CPU (affects frame grabs and cutting speed, not correctness).
- The app has been tested in a sandbox, not yet on your PC. Expect a small fix or two on the first run.
- `DEFAULT_CAM` is set for this streamer's layout only.
- Mixed audio track: voice separation is approximate.
- Reposting other streamers' content: credit them and add value, or YouTube may demote or demonetize it.

## 📁 Files in `C:\clipagent`
| File | Role |
|---|---|
| `common.py` | shared helpers, Ollama calls, `DEFAULT_CAM` |
| `fetch.py` | download video + chat + heatmap |
| `getchat.py` | chat for the first N hours |
| `analyze.py` | transcription, voice separation, signals, candidates |
| `judge.py` | face + vision + LLM judge + titles |
| `cut_clips.py` | export clips + `titles.txt` |
| `run.py` | runs all steps (command line) |
| `app.py` | local app (Gradio UI) |
| `ClipAgent.bat` | double-click launcher for the app |
| `README.md`, `requirements.txt`, `.gitignore` | GitHub repo files |
| `plan.md`, `progress.md` | project plan and status |
| retired: `transcribe.py`, `pick_clips.py`, `signals.py`, `findcam.py`, `addcam.py`, `patch.py` | old versions, safe to delete |
