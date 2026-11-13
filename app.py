import os, sys, re, json, glob, shutil, subprocess, time, urllib.request
from datetime import datetime
import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PY = sys.executable
NAME = "stream"
MAX_SLOTS = 8
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
STATE = {"proc": None, "stop": False}


# ---------- helpers ----------

def mmss(s):
    s = int(float(s or 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:30] or "run"


def ollama_ok():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3):
            return True
    except Exception:
        return False


def clean_old():
    pats = [NAME + ".mp4", NAME + ".mkv", NAME + ".webm", NAME + ".part", NAME + ".*.part",
            NAME + ".live_chat.json", NAME + ".heatmap.json", NAME + "_*.json",
            os.path.join("work", NAME + "_*"), os.path.join("work", "cam_*.jpg"), "camcheck.jpg"]
    for pat in pats:
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except Exception:
                pass


def kill_tree(p):
    if p is None or p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        p.kill()


def list_runs():
    runs = [d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d))]
    return sorted(runs, reverse=True)


def run_step(cmd, log):
    """Run one pipeline script, streaming its output into the log."""
    log.append("")
    log.append(">>> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env, bufsize=1)
    STATE["proc"] = p
    last = 0.0
    for line in p.stdout:
        line = line.rstrip()
        if line:
            log.append(line)
        if time.time() - last > 0.5:
            last = time.time()
            yield
    p.wait()
    STATE["proc"] = None
    return p.returncode


# ---------- UI output builders ----------

def no_change():
    return [gr.update() for _ in range(2 * MAX_SLOTS)] + [gr.update(), gr.update()]


def hidden_slots():
    return [gr.update(value=None, visible=False) for _ in range(2 * MAX_SLOTS)] + \
           [gr.update(value=None, visible=False), gr.update()]


def clip_card(i, c):
    ev = c.get("evidence") or {}
    lines = [f"### #{i} · {c.get('final', '?')}/100 · {c.get('tag', 'clip')}",
             f"**{mmss(c.get('start'))} → {mmss(c.get('end'))}** "
             f"({float(c.get('end', 0)) - float(c.get('start', 0)):.0f}s)",
             "",
             f"**What happens:** {c.get('reason', '')}"]
    if ev:
        lines.append(f"**Evidence:** reaction {ev.get('reaction')}/10 · face: {ev.get('face_best', 'n/a')} · "
                     f"signals agreeing: {ev.get('agree')}")
    lines.append("")
    lines.append("**Title options:**")
    lines += [f"- {t}" for t in c.get("titles", [])] or ["- (none)"]
    return "\n".join(lines)


def load_run(run):
    """Returns updates for the clip slots, titles file and history dropdown."""
    if not run:
        return hidden_slots()
    d = os.path.join(OUT_DIR, run)
    try:
        clips = json.load(open(os.path.join(d, "clips.json"), encoding="utf-8"))
    except Exception:
        clips = []
    vids = sorted(glob.glob(os.path.join(d, "*.mp4")))
    vid_ups, md_ups = [], []
    for i in range(MAX_SLOTS):
        if i < len(vids):
            c = clips[i] if i < len(clips) else {}
            vid_ups.append(gr.update(value=vids[i], visible=True, label=os.path.basename(vids[i])))
            md_ups.append(gr.update(value=clip_card(i + 1, c), visible=True))
        else:
            vid_ups.append(gr.update(value=None, visible=False))
            md_ups.append(gr.update(value=None, visible=False))
    slots = []
    for v, m in zip(vid_ups, md_ups):
        slots += [v, m]
    tfile = os.path.join(d, "titles.txt")
    t_up = gr.update(value=tfile, visible=True) if os.path.exists(tfile) else gr.update(value=None, visible=False)
    return slots + [t_up, gr.update(choices=list_runs(), value=run)]


def archive(label, game):
    """Copy this run's clips into outputs/<date>_<name>/ so they aren't overwritten next time."""
    run = datetime.now().strftime("%Y-%m-%d_%H%M") + "_" + slug(label)
    d = os.path.join(OUT_DIR, run)
    os.makedirs(d, exist_ok=True)
    for f in sorted(glob.glob(os.path.join("clips", "*.mp4"))):
        shutil.copy2(f, d)
    if os.path.exists(os.path.join("clips", "titles.txt")):
        shutil.copy2(os.path.join("clips", "titles.txt"), d)
    if os.path.exists(NAME + "_clips.json"):
        shutil.copy2(NAME + "_clips.json", os.path.join(d, "clips.json"))
    json.dump({"source": label, "game": game, "date": datetime.now().isoformat(timespec="minutes")},
              open(os.path.join(d, "info.json"), "w", encoding="utf-8"), indent=1)
    return run


# ---------- the pipeline ----------

def process(url, upload, local_path, game, hours, keep, min_score, cam):
    log = []
    STATE["stop"] = False

    def ui(status, extra=None):
        return [status, "\n".join(log[-400:])] + (extra if extra is not None else no_change())

    url = (url or "").strip()
    src = upload or (local_path or "").strip().strip('"')
    game = (game or "").strip()
    cam = (cam or "").strip()

    if not url and not src:
        yield ui("❌ Paste a link, upload a video, or enter a file path.")
        return
    if not ollama_ok():
        yield ui("❌ Ollama isn't running. Start Ollama (llama icon in the system tray) and try again.")
        return
    if cam and len(cam.split(",")) != 4:
        yield ui("❌ Facecam box needs 4 numbers: x,y,w,h, e.g. 0,0.245,0.208,0.207")
        return

    yield ui("⏳ Preparing ...", hidden_slots())
    steps = []
    if url:
        steps.append(("Downloading video + chat", [PY, "fetch.py", url, "--hours", str(hours)]))
        label = url.rstrip("/").split("/")[-1].split("?")[0]
    else:
        if not os.path.exists(src):
            yield ui(f"❌ File not found: {src}")
            return
        label = os.path.splitext(os.path.basename(src))[0]
        clean_old()
        log.append(f"Copying {src} -> {NAME}.mp4 ...")
        yield ui("⏳ Copying video ...")
        shutil.copyfile(src, NAME + ".mp4")

    judge = [PY, "judge.py", NAME + ".mp4", "--keep", str(int(keep)), "--min-score", str(min_score)]
    if game:
        judge += ["--game", game]
    if cam:
        judge += ["--cam", cam]
    steps += [("Analysing transcript, voice and chat", [PY, "analyze.py", NAME + ".mp4"]),
              ("Judging moments (face + AI)", judge),
              ("Cutting clips", [PY, "cut_clips.py", NAME + ".mp4"])]

    t_start = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        status = f"⏳ Step {i}/{len(steps)}: {name} ... (log below)"
        yield ui(status)
        gen = run_step(cmd, log)
        while True:
            try:
                next(gen)
                yield ui(status)
            except StopIteration as e:
                rc = e.value
                break
        if STATE["stop"]:
            yield ui("⏹ Stopped.")
            return
        if rc != 0:
            yield ui(f"❌ Step failed: {name}. Open the log to see the error.")
            return
        if name.startswith("Downloading") and not os.path.exists(NAME + ".mp4"):
            yield ui("❌ Download failed. Open the log to see the yt-dlp error.")
            return
        if name.startswith("Judging") and not os.path.exists(NAME + "_clips.json"):
            yield ui("❌ The judge produced no clips file. Open the log.")
            return

    run = archive(label, game)
    mins = (time.time() - t_start) / 60
    n = len(glob.glob(os.path.join(OUT_DIR, run, "*.mp4")))
    msg = (f"✅ Done in {mins:.0f} min: {n} clip(s). Saved in outputs\\{run}"
           if n else "⚠️ Finished, but no moment passed the filters. Try a lower minimum score.")
    yield ui(msg, load_run(run))


def stop():
    STATE["stop"] = True
    kill_tree(STATE["proc"])
    return "⏹ Stopping ..."


def open_outputs():
    if os.name == "nt":
        os.startfile(OUT_DIR)
    return gr.update()


# ---------- layout ----------

with gr.Blocks(title="ClipAgent") as demo:
    gr.Markdown("# 🎬 ClipAgent\nPaste a stream link or add a video. It finds the best moments, "
                "cuts them out, and writes titles. Everything runs on this PC.")
    with gr.Tab("New job"):
        with gr.Row():
            with gr.Column(scale=2):
                url_in = gr.Textbox(label="Stream / VOD link (YouTube, Twitch, Kick)",
                                    placeholder="https://www.youtube.com/live/...")
                with gr.Accordion("...or use a video file", open=False):
                    upload_in = gr.File(label="Upload a video", file_types=["video"], type="filepath")
                    local_in = gr.Textbox(label="...or the path of a video on this PC (faster for big files)",
                                          placeholder=r"C:\videos\my_stream.mp4")
            with gr.Column(scale=1):
                game_in = gr.Textbox(label="Streamer / game (used in titles)",
                                     placeholder="Kai Cenat Playing Every GTA")
                hours_in = gr.Slider(0.5, 6, value=2, step=0.5, label="Hours to download (links only)")
                keep_in = gr.Slider(1, MAX_SLOTS, value=8, step=1, label="Max clips")
                score_in = gr.Slider(30, 80, value=50, step=5, label="Minimum score (lower = more clips)")
                cam_in = gr.Textbox(label="Facecam box x,y,w,h (optional)", placeholder="0,0.245,0.208,0.207")
        with gr.Row():
            run_btn = gr.Button("🔎 Find clips", variant="primary")
            stop_btn = gr.Button("⏹ Stop", variant="stop")
        status_md = gr.Markdown("Ready.")
        with gr.Accordion("Log", open=False):
            log_box = gr.Textbox(lines=18, max_lines=18, show_label=False, autoscroll=True)
    with gr.Tab("Past runs"):
        with gr.Row():
            hist_dd = gr.Dropdown(choices=list_runs(), label="Previous runs", scale=3)
            load_btn = gr.Button("Show clips", scale=1)
            open_btn = gr.Button("📂 Open outputs folder", scale=1)

    gr.Markdown("## Clips")
    titles_file = gr.File(label="titles.txt", visible=False)
    slots = []
    for r in range(0, MAX_SLOTS, 2):
        with gr.Row():
            for k in (r, r + 1):
                with gr.Column():
                    v = gr.Video(visible=False, height=360)
                    m = gr.Markdown(visible=False)
                    slots += [v, m]

    outputs = [status_md, log_box] + slots + [titles_file, hist_dd]
    run_btn.click(process, [url_in, upload_in, local_in, game_in, hours_in, keep_in, score_in, cam_in], outputs)
    stop_btn.click(stop, None, status_md)
    load_btn.click(load_run, hist_dd, slots + [titles_file, hist_dd])
    open_btn.click(open_outputs, None, None)

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, server_name="127.0.0.1", allowed_paths=[OUT_DIR])
