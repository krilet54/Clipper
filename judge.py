import sys, os, re, json, time, base64, subprocess
import numpy as np
from common import *

TEXT_MODEL = "qwen2.5:3b"
VISION_MODEL = "qwen2.5vl:3b"
EMOTIONS = ["laughing", "excited", "shocked", "angry", "frustrated", "sad", "crying", "scared", "neutral", "absent"]
TYPES = ["Funny fail", "Jump scare", "Rage", "Clutch", "Plot twist", "Hot take", "Chat moment", "Hype",
         "Funny line", "Not a moment"]

JUDGE_SYSTEM = """You are the head editor of a viral gaming Shorts channel. You pick clips from long livestream VODs.
What makes a clip go viral:
1. The first 1-3 seconds already contain tension, a bold statement or a visible event. A slow "so um" start loses viewers.
2. Clear arc: short setup (3-8 s), a peak (reaction, fail, win, twist, jump scare, hot take), then the payoff or reaction right after it. End 1-3 s after the payoff.
3. A strong human reaction is the product: shouting, laughing, disbelief, rage, panic, hype, plus chat exploding. A calm streamer narrating gameplay is NOT a viral moment.
4. A stranger must understand what is happening within 5 seconds, with zero context.
5. Best lengths: 15-35 s (punchy) or 45-58 s (story). Never above 58 s.
6. Strong types: Funny fail, Jump scare, Rage, Clutch, Plot twist, Hot take, Chat moment, Hype, Funny line.
7. Weak moments (type "Not a moment"): menus, loading, cutscenes, planning, explaining mechanics, silence, game characters talking with no reaction from the streamer.
Evidence rules:
- Lines marked S are the streamer's own voice. Lines marked G are game audio or other voices, NOT the streamer reacting.
- The measured evidence block is real. Never describe a scream, laugh, expression or reaction that the evidence does not show.
- Be strict. Most candidates are 3-6 out of 10. Use 8+ only for clearly exceptional moments.
Pick boundaries using the line numbers in brackets. Score each 0-10:
hook (first 3 seconds stop a scroller), story (setup, peak, payoff), share (people would send or comment), selfcontained (understandable without context).
quote = up to 10 exact words copied from an S line at the peak, or empty if there is none.
Return ONLY JSON: {"peak_id":<int>,"start_id":<int>,"end_id":<int>,"hook":<0-10>,"story":<0-10>,"share":<0-10>,"selfcontained":<0-10>,"type":"<one of: Funny fail, Jump scare, Rage, Clutch, Plot twist, Hot take, Chat moment, Hype, Funny line, Not a moment>","quote":"<words>","reason":"<one sentence saying what concretely happens>"}"""

VISION_PROMPT = ("This is a webcam frame of a video game streamer. Look at the face and body. "
                 "Calm seated gamers are neutral unless the expression is clearly visible. "
                 "Reply ONLY with JSON: {\"emotion\":\"<exactly one of: laughing, excited, shocked, angry, frustrated, sad, crying, scared, neutral, absent>\","
                 "\"intensity\":<number 0-10, 0 = calm, 10 = extreme>,\"look\":\"<max 12 words about face and body>\"} "
                 "Use absent if no person is visible.")

TITLE_SYSTEM = """You write YouTube Shorts titles for gaming clips.
Write exactly 3 different titles, each 35-60 characters, as normal readable English sentences with spaces.
Describe what happens or the stakes, create curiosity, put the hook words first, include the game name if given.
Do NOT use angle brackets, hyphenated slugs, hashtags or quotes. Do NOT quote dialogue.
Do NOT promise anything the clip does not show. No ALL CAPS.
Return ONLY JSON: {"titles":["<title 1>","<title 2>","<title 3>"]}"""

TITLE_SYSTEM_TEXT = """You write YouTube Shorts titles for gaming clips.
Write exactly 3 different titles, one per line, each 35-60 characters, as normal readable English with spaces.
Describe what happens or the stakes, hook words first, include the game name if given.
No angle brackets, no hyphenated slugs, no quotes, no numbering. Do NOT quote dialogue.
Output only the 3 titles, one per line, no extra text."""


def grab_cam(video, t, box, out):
    x, y, w, h = box
    vf = f"crop=trunc(iw*{w}/2)*2:trunc(ih*{h}/2)*2:trunc(iw*{x}/2)*2:trunc(ih*{y}/2)*2,scale=448:-2"
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, t):.2f}",
                        "-i", video, "-frames:v", "1", "-update", "1", "-vf", vf, out],
                       capture_output=True, text=True)
    if not os.path.exists(out):
        print("\n  frame grab failed:", (r.stderr or "").strip()[:300])
        return False
    return True


def look(img):
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    d = post("/api/chat", {"model": VISION_MODEL, "stream": False, "keep_alive": "10m", "format": "json",
                           "options": {"temperature": 0.1, "num_ctx": 2048, "num_predict": 120},
                           "messages": [{"role": "user", "content": VISION_PROMPT, "images": [b64]}]}, 300)
    j = extract_json(d.get("message", {}).get("content", ""))
    em = str(j.get("emotion", "")).lower()
    emotion = next((e for e in EMOTIONS if e in em), "neutral")
    return {"emotion": emotion, "intensity": int(max(0, min(10, num(j.get("intensity"), 0)))),
            "look": str(j.get("look", ""))[:100]}


def cam_motion(video, t0, dur, box):
    x, y, w, h = box
    vf = (f"crop=trunc(iw*{w}/2)*2:trunc(ih*{h}/2)*2:trunc(iw*{x}/2)*2:trunc(ih*{y}/2)*2,"
          f"fps=4,scale=96:54,format=gray")
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t0):.2f}", "-t", f"{dur:.2f}", "-i", video,
                        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    buf = np.frombuffer(p.stdout, np.uint8)
    n = len(buf) // (96 * 54)
    if n < 4:
        return None
    fr = buf[:n * 96 * 54].reshape(n, 54, 96).astype(np.float32)
    return float(np.percentile(np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2)), 90))


def motion_baseline(video, total, box):
    vals = []
    for t in np.linspace(120, max(180, total - 120), 16):
        m = cam_motion(video, float(t), 12, box)
        if m is not None:
            vals.append(m)
    if len(vals) < 4:
        return None
    a = np.array(vals)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)) * 1.4826)
    return {"med": med, "scale": max(mad, 0.3 * med, 0.05)}


def compute_evidence(c, f, bm, has_chat):
    s = c["sig"]
    frames = f.get("frames", [])
    vlm = max([fr["intensity"] for fr in frames if fr["emotion"] not in ("neutral", "absent")] or [0])
    mz = None
    if f.get("motion") is not None and bm:
        mz = (f["motion"] - bm["med"]) / bm["scale"]
    parts = []
    voice10 = min(10, max(s["voice"] * 2, s["laugh"] * 1.6))
    parts.append(voice10)
    face10 = None
    if frames or mz is not None:
        face10 = max(vlm, min(10, max(0, mz) * 2.5) if mz is not None else 0)
        parts.append(face10)
    chat10 = None
    if has_chat:
        chat10 = min(10, s["chat"] * 2)
        parts.append(chat10)
    reaction = 0.5 * max(parts) + 0.5 * (sum(parts) / len(parts))
    fired = sum([s["voice"] >= 2, bool(has_chat and s["chat"] >= 2), bool(face10 is not None and face10 >= 4),
                 s["laugh"] >= 2.5, s["game"] >= 2.5, s["hype"] >= 1.5])
    bonus = {0: 0, 1: 0, 2: 4, 3: 7}.get(int(fired), 10)
    best = max([fr for fr in frames if fr["emotion"] not in ("neutral", "absent")],
               key=lambda fr: fr["intensity"], default=None)
    return {"reaction": round(reaction, 2), "voice10": round(voice10, 1),
            "face10": None if face10 is None else round(face10, 1),
            "chat10": None if chat10 is None else round(chat10, 1),
            "motion_z": None if mz is None else round(mz, 2), "agree": int(fired), "bonus": bonus,
            "face_best": f"{best['emotion']} {best['intensity']}/10 ({best['look']})" if best else
            ("calm/neutral in all frames" if frames else "facecam not analysed")}


def evidence_text(c, ev, has_chat, chat_sample):
    s = c["sig"]
    L = [f"- Streamer voice loudness: {s['voice']:.1f} sigma above his normal level (0 = normal, 2+ = shouting)",
         f"- Streamer speech speed: {s['rate']:.1f} sigma",
         f"- Laughter heard from the streamer: {'yes' if s['laugh'] >= 2.5 else 'no'}",
         f"- Game sound spike while he was silent: {s['game']:.1f} sigma",
         f"- Facecam: {ev['face_best']}; face movement {ev['motion_z'] if ev['motion_z'] is not None else 'n/a'} sigma"]
    if has_chat:
        L.append(f"- Chat activity: {s['chat']:.1f} sigma above normal. Sample messages: " +
                 (" | ".join(chat_sample) if chat_sample else "(none)"))
    else:
        L.append("- Chat: not available")
    return "\n".join(L)


def chat_around(chat, t0, t1, n=8):
    msgs = [m[1] for m in chat if t0 <= m[0] <= t1 and len(m[1]) > 1]
    hot = [m for m in msgs if re.search(r"(lmao|lol|omg|wtf|clip|dead|cooked|bruh|no ?way|😂|🤣|💀|😭)", m, re.I)]
    pick = (hot + [m for m in msgs if m not in hot])[:n]
    return [m[:50] for m in pick]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def make_window(j, c, wl, total):
    def tl(key, default, edge):
        try:
            i = int(num(j.get(key), -1))
            if 0 <= i < len(wl):
                return wl[i][edge]
        except Exception:
            pass
        return default
    pk = tl("peak_id", c["peak"], "s")
    s = tl("start_id", c["start"], "s") - 0.3
    e = tl("end_id", c["end"], "e") + 1.0
    s = min(s, pk - 4)
    e = max(e, pk + 3)
    if e - s > 58:
        s = max(s, pk - 25)
        e = min(e, s + 58)
    if e - s < 15:
        s, e = pk - 10, pk + 8
    return round(max(0, s), 1), round(min(total, e), 1), pk


def overlap(a, b):
    o = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return o / max(1, min(a["end"] - a["start"], b["end"] - b["start"])) if o > 0 else 0


def valid_title(t, existing):
    t = str(t).strip().strip('"').strip("'").strip()
    t = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", t)
    t = t.replace("<", "").replace(">", "").strip()
    if not (12 <= len(t) <= 75) or " " not in t:
        return None
    if t.lower() in [x.lower() for x in existing]:
        return None
    return t


def make_titles(c, game, debug):
    prompt = (f"Game: {game or 'unknown'}\nType of moment: {c['tag']}\nWhat happens: {c['reason']}\n"
              f"Streamer's reaction: {c['evidence']['face_best']}\nTranscript snippet: {c['snippet']}\n\nWrite the 3 titles now.")
    for attempt in range(2):
        try:
            res = chat_text(TEXT_MODEL, TITLE_SYSTEM, prompt, 200, 0.7 + 0.1 * attempt, True, 4096, debug)
            titles = []
            for t in res.get("titles", []):
                v = valid_title(t, titles)
                if v:
                    titles.append(v)
            if len(titles) >= 2:
                return titles[:3]
        except Exception as e:
            print("  title error:", e)
    try:
        raw = chat_text(TEXT_MODEL, TITLE_SYSTEM_TEXT, prompt, 150, 0.8, False, 4096, debug)
        titles = []
        for line in raw.splitlines():
            v = valid_title(line, titles)
            if v:
                titles.append(v)
        if titles:
            return titles[:3]
    except Exception as e:
        print("  title error:", e)
    base = c["reason"].rstrip(".")
    return [((game + ": ") if game else "") + base[:55]]


def main():
    global TEXT_MODEL, VISION_MODEL
    args = sys.argv[1:]
    if not args:
        print('Usage: python judge.py stream.mp4 [--game "GTA 5"] [--top 24] [--keep 8] [--cam x,y,w,h] '
              '[--min-score 50] [--gate 2.5] [--text-model qwen2.5:7b] [--no-vision] [--fresh] [--debug]')
        return
    video = args[0]
    base = os.path.splitext(video)[0]
    game = opt(args, "--game")
    top = int(opt(args, "--top", 24))
    keep = int(opt(args, "--keep", 8))
    min_score = float(opt(args, "--min-score", 50))
    gate = float(opt(args, "--gate", 2.5))
    debug = "--debug" in args
    TEXT_MODEL = opt(args, "--text-model", TEXT_MODEL)
    VISION_MODEL = opt(args, "--vision-model", VISION_MODEL)
    box = [float(v) for v in opt(args, "--cam").split(",")] if opt(args, "--cam") else DEFAULT_CAM

    cands = load_json(base + "_cands.json")
    lines = load_json(base + "_lines.json")
    if not cands or lines is None:
        print("Run analyze.py first.")
        return
    cands = cands[:top]
    chat = load_json(base + "_chat.json", [])
    meta = load_json(base + "_meta.json", {})
    has_chat = bool(meta.get("has_chat"))
    total = probe_duration(video)
    face_file = base + "_face.json"
    face = {} if "--fresh" in args else load_json(face_file, {})
    print("Facecam box (x,y,w,h):", box)

    unload(TEXT_MODEL)
    unload(VISION_MODEL)

    # Stage 1: facecam motion + expression
    bm = None
    if "--no-vision" not in args:
        print(f"\nStage 1/3: reading the streamer's face for {len(cands)} moments ...", flush=True)
        if face.get("baseline") is None:
            face["baseline"] = motion_baseline(video, total, box)
        bm = face["baseline"]
        todo = [c for c in cands if not face.get(str(c["peak"]), {}).get("frames")]
        if todo:
            try:
                warm(VISION_MODEL)
            except Exception as ex:
                print("  could not load vision model:", ex, f"(check: ollama list, expected {VISION_MODEL})")
        for k, c in enumerate(cands, 1):
            key = str(c["peak"])
            e = face.setdefault(key, {})
            if e.get("frames"):
                continue
            t0 = time.time()
            e["motion"] = cam_motion(video, c["peak"] - 10, 20, box)
            e["frames"] = []
            for j, off in enumerate((-8, -4, 0, 4, 8)):
                t = min(max(c["peak"] + off, 0), total - 1)
                img = f"work/cam_{key}_{j}.jpg"
                if grab_cam(video, t, box, img):
                    try:
                        r = look(img)
                        r["t"] = round(t, 1)
                        e["frames"].append(r)
                    except Exception as ex:
                        print("\n  vision error:", ex)
            ev = compute_evidence(c, e, bm, has_chat)
            print(f"  {k}/{len(cands)} {mmss(c['peak'])}: {ev['face_best']} ({time.time() - t0:.0f}s)", flush=True)
            save_json(face_file, face)
        unload(VISION_MODEL)
        if all(not face.get(str(c["peak"]), {}).get("frames") for c in cands):
            print("\nSTOPPING: no facecam frames could be read, so face evidence would be missing.")
            print("Open camcheck.jpg, fix the box with --cam, or use --no-vision to skip faces.")
            return

    # Stage 2: evidence gate + judge
    print("\nStage 2/3: judging moments ...", flush=True)
    warm(TEXT_MODEL)
    judged, rejected = [], {"no reaction": 0, "not a moment": 0}
    for k, c in enumerate(cands, 1):
        f = face.get(str(c["peak"]), {})
        ev = compute_evidence(c, f, bm, has_chat)
        if ev["reaction"] < gate:
            rejected["no reaction"] += 1
            print(f"  {k}/{len(cands)} {mmss(c['peak'])}: skipped, reaction evidence {ev['reaction']} < {gate}")
            continue
        ctx_s, ctx_e = c["start"] - 35, c["end"] + 20
        wl = [l for l in lines if ctx_s <= l["s"] <= ctx_e][:60]
        tx = "\n".join(f"[{i}] {int(l['s'])}s {'S' if l['label'] == 'streamer' else 'G'}: "
                       f"{(l['text'] or '(laughs)')[:140]}" for i, l in enumerate(wl))[:3800] or "(no speech)"
        sample = chat_around(chat, c["peak"] - 10, c["peak"] + 15) if has_chat else []
        prompt = (f"Game: {game or 'unknown'}\nCandidate peak at {int(c['peak'])}s.\n\n"
                  f"Measured evidence:\n{evidence_text(c, ev, has_chat, sample)}\n\n"
                  f"Transcript (line numbers in brackets):\n{tx}")
        print(f"  judging {k}/{len(cands)} {mmss(c['peak'])} ", end="", flush=True)
        t0 = time.time()
        try:
            j = chat_text(TEXT_MODEL, JUDGE_SYSTEM, prompt, 300, 0.2, True, 6144, debug)
        except Exception as ex:
            print("error:", ex)
            j = {}
        print(f" {time.time() - t0:.0f}s", flush=True)
        if not j:
            continue
        typ = str(j.get("type", "Hype")).strip()
        typ = next((t for t in TYPES if t.lower() == typ.lower()), "Hype")
        if typ == "Not a moment":
            rejected["not a moment"] += 1
            continue
        hook, story, share, selfc = [max(0, min(10, num(j.get(n), 4))) for n in
                                     ("hook", "story", "share", "selfcontained")]
        quote = str(j.get("quote", "")).strip()
        s_text = norm(" ".join(l["text"] for l in wl if l["label"] == "streamer"))
        quote_ok = bool(quote) and norm(quote) in s_text
        final = 100 * (0.35 * ev["reaction"] / 10 + 0.20 * hook / 10 + 0.15 * story / 10 +
                       0.10 * share / 10 + 0.10 * selfc / 10) + ev["bonus"]
        if quote and not quote_ok:
            final -= 5
        s, e, pk = make_window(j, c, wl, total)
        reason = str(j.get("reason", "")).strip()
        if len(reason) < 12:
            reason = "a high-energy moment picked from voice, chat and facecam signals"
        snippet = " ".join(l["text"] for l in wl if l["label"] == "streamer" and s - 1 <= l["s"] <= e)[:700]
        judged.append({"start": s, "end": e, "peak": pk, "final": round(final, 1), "tag": typ,
                       "reason": reason, "quote": quote, "quote_ok": quote_ok, "evidence": ev,
                       "scores": {"hook": hook, "story": story, "share": share, "selfcontained": selfc},
                       "signals": c["sig"], "snippet": snippet})

    judged.sort(key=lambda c: -c["final"])
    final_list = []
    for c in judged:
        if c["final"] >= min_score and all(overlap(c, f) < 0.4 for f in final_list):
            final_list.append(c)
    final_list = final_list[:keep]
    below = len([c for c in judged if c["final"] < min_score])
    print(f"\nKept {len(final_list)} clips. Rejected: {rejected['no reaction']} with no measurable reaction, "
          f"{rejected['not a moment']} judged 'not a moment', {below} scored under {min_score:g}.")

    print("\nStage 3/3: writing titles ...", flush=True)
    for i, c in enumerate(final_list, 1):
        print(f"  clip {i}/{len(final_list)} ", end="", flush=True)
        c["titles"] = make_titles(c, game, debug)
        print(" ok", flush=True)

    save_json(base + "_clips.json", final_list)
    for i, c in enumerate(final_list, 1):
        ev = c["evidence"]
        print(f"\n#{i}  {c['final']}/100  [{c['tag']}]  {mmss(c['start'])} -> {mmss(c['end'])} ({c['end'] - c['start']:.0f}s)")
        print(f"  evidence: reaction {ev['reaction']}/10 | voice {ev['voice10']} | face {ev['face10']} | "
              f"chat {ev['chat10']} | signals agreeing: {ev['agree']}")
        print(f"  facecam: {ev['face_best']}")
        print(f"  why: {c['reason']}" + ("" if c["quote_ok"] or not c["quote"] else "  (quote unverified)"))
        for t in c["titles"]:
            print("  title:", t)
    print(f"\nSaved {base}_clips.json. Now run: python cut_clips.py {video}")


main()
