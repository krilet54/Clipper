import sys, re, json, subprocess, statistics, urllib.request, os, time

MODEL = "qwen2.5:3b"
CHUNK_SEC = 300
CHUNK_TIMEOUT = 300

PICK_SYSTEM = """You are a YouTube Shorts editor. You choose the best MOMENTS from a gaming stream transcript.
A moment is a whole scene of 15 to 45 seconds, NOT a single line, so "end" must be at least 15 seconds after "start".
Start a few seconds before the peak so the viewer gets the hook immediately, and end right after the payoff.
The clip must make sense to someone who has seen nothing else.
Good moments: funny lines, rage, shock, big reactions, clutch plays, chaotic action, strong banter.
Use ONLY times that appear in the transcript. Do not invent times.
For "reason", write one NEW sentence describing what actually happens in that moment.
Return ONLY JSON: {"clips":[{"start":<number>,"end":<number>,"score":<1-10>,"reason":"<sentence>"}]}"""

TITLE_SYSTEM = """You write YouTube Shorts titles for gaming clips.
Rules: write exactly 3 different titles, each 40-60 characters.
Describe what happens or the stakes, create curiosity, include the game name if given.
Do NOT quote the dialogue. Do NOT promise anything the clip does not show. No ALL CAPS.
Return ONLY JSON: {"titles":["<title 1>","<title 2>","<title 3>"]}"""


def parse_srt(path):
    text = open(path, encoding="utf-8").read().replace("\r", "")
    lines = []
    for block in re.split(r"\n\s*\n", text):
        rows = [r for r in block.strip().split("\n") if r.strip()]
        if len(rows) < 2:
            continue
        m = re.search(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)",
                      rows[1] if "-->" in rows[1] else rows[0])
        if not m:
            continue
        g = list(map(int, m.groups()))
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        body = " ".join(rows[2:] if "-->" in rows[1] else rows[1:])
        body = re.sub(r"\(.*?\)|\[.*?\]|\*.*?\*", "", body).strip(" -")
        if len(body) > 1:
            lines.append((start, end, body))
    lines.sort()
    return lines


def loudness(video, limit):
    cmd = ["ffmpeg", "-nostats", "-loglevel", "error", "-i", video]
    if limit:
        cmd += ["-t", str(limit)]
    cmd += ["-vn", "-ac", "1", "-ar", "8000", "-af",
            "asetnsamples=n=8000:p=0,astats=metadata=1:reset=1,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    vals, t = [], None
    for line in out.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
        m = re.search(r"RMS_level=(-?[\d.]+)", line)
        if m and t is not None:
            vals.append((t, float(m.group(1))))
    if len(vals) < 10:
        print("Loudness analysis returned nothing; continuing without it.")
        return []
    levels = [v for _, v in vals]
    thr = statistics.mean(levels) + 1.5 * statistics.pstdev(levels)
    spikes, cur = [], None
    for t, v in vals:
        if v >= thr:
            if cur and t - cur[1] <= 2:
                cur[1] = t
                cur[2] = max(cur[2], v)
            else:
                cur = [t, t, v]
                spikes.append(cur)
    spikes.sort(key=lambda s: -s[2])
    return sorted(spikes[:40])


def warm_up():
    print(f"Loading {MODEL} into memory...", flush=True)
    t0 = time.time()
    body = json.dumps({"model": MODEL, "prompt": "hi", "stream": False,
                       "keep_alive": "30m", "options": {"num_predict": 1}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        print(f"  model ready in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:
        print("  Could not reach Ollama:", e)
        print("  Make sure Ollama is running (llama icon in the system tray) and try again.")
        sys.exit(1)


def extract_json(raw):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(raw[a:b + 1])
        except Exception:
            pass
    clips = []
    for m in re.finditer(r'\{[^{}]*"start"[^{}]*\}', raw):
        try:
            clips.append(json.loads(m.group(0)))
        except Exception:
            pass
    return {"clips": clips}


def ask_llm(system, prompt, max_tokens, temperature=0.3, debug=False):
    body = json.dumps({
        "model": MODEL, "stream": True, "format": "json", "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": max_tokens},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    raw, n, t0 = "", 0, time.time()
    with urllib.request.urlopen(req, timeout=CHUNK_TIMEOUT) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            raw += d.get("message", {}).get("content", "")
            n += 1
            if n % 20 == 0:
                print(".", end="", flush=True)
            if time.time() - t0 > CHUNK_TIMEOUT:
                print(" (time limit)", end="")
                break
            if d.get("done"):
                break
    if debug:
        print("\n  RAW:", raw[:1200])
    return extract_json(raw)


def overlap(a, b):
    o = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return o / max(1, min(a["end"] - a["start"], b["end"] - b["start"])) if o > 0 else 0


def snap(s, e, lines):
    starts = [l[0] for l in lines if l[0] <= s]
    s2 = max(starts) if starts else s
    ends = [l[1] for l in lines if l[1] >= e and l[1] <= s2 + 60]
    e2 = min(ends) if ends else e
    e2 = min(e2 + 0.5, s2 + 60)
    return max(0, s2 - 0.3), e2


def good_reason(r):
    r = (r or "").strip().lower()
    return len(r) > 12 and r not in ("why it works", "<sentence>", "one new sentence")


def make_titles(clip, lines, game, debug):
    tx = "\n".join(t for s, e, t in lines if clip["start"] - 1 <= s <= clip["end"])
    prompt = (f"Game: {game or 'unknown'}\nWhat happens: {clip['reason']}\n\n"
              f"Transcript of the clip:\n{tx[:1800]}\n\nWrite the 3 titles now.")
    for attempt in range(2):
        try:
            res = ask_llm(TITLE_SYSTEM, prompt, 200, 0.7 + 0.1 * attempt, debug)
            titles = []
            for t in res.get("titles", []):
                t = str(t).strip().strip('"')
                if len(t) >= 12 and t.lower() not in [x.lower() for x in titles] and "<title" not in t.lower():
                    titles.append(t)
            if len(titles) >= 2:
                return titles[:3]
        except Exception as e:
            print("  title error:", e)
    fallback = (f"{game}: " if game else "") + clip["reason"].rstrip(".")
    return [fallback[:70]]


def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: python pick_clips.py test.mp4 [--limit 300] [--game "GTA 5"] '
              '[--target "clutch plays"] [--top 8] [--debug]')
        return
    video = args[0]

    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    limit = int(opt("--limit", 0))
    target = opt("--target")
    game = opt("--game")
    top = int(opt("--top", 8))
    debug = "--debug" in args

    srt = os.path.splitext(video)[0] + ".srt"
    lines = parse_srt(srt)
    if limit:
        lines = [l for l in lines if l[0] < limit]
    if not lines:
        print("No transcript lines found in", srt)
        return
    total = lines[-1][1]
    print(f"{len(lines)} transcript lines, {total:.0f}s covered. Measuring loudness...", flush=True)
    spikes = loudness(video, limit)
    print(f"{len(spikes)} loud spikes found.", flush=True)
    warm_up()

    mode = (f'TARGETED MODE: only pick moments matching this request: "{target}". '
            "If nothing matches in this chunk, return an empty list."
            if target else
            "AUTO MODE: pick the 3 best moments in this chunk, even if none is perfect. Always return at least 1.")
    if game:
        mode += f"\nThe game is: {game}."

    found = []
    for c0 in range(0, int(total) + 1, CHUNK_SEC):
        c1 = c0 + CHUNK_SEC
        tx = "\n".join(f"[{int(s)}s] {t}" for s, e, t in lines if c0 <= s < c1)
        if not tx:
            continue
        cs = sorted([s for s in spikes if c0 <= s[0] < c1], key=lambda s: -s[2])[:10]
        sp = "\n".join(f"{int(a)}s-{int(b)+1}s" for a, b, _ in sorted(cs)) or "none"
        prompt = f"{mode}\n\nLoud audio spikes (screaming, big reactions, explosions):\n{sp}\n\nTranscript:\n{tx}"
        print(f"Picking moments in {c0}s-{c1}s ", end="", flush=True)
        t0 = time.time()
        try:
            res = ask_llm(PICK_SYSTEM, prompt, 500, 0.3, debug).get("clips", [])
        except Exception as e:
            print("\n  LLM error:", e)
            continue
        print(f" done in {time.time() - t0:.0f}s, {len(res)} returned", flush=True)
        for c in res:
            try:
                s, e = float(c["start"]), float(c["end"])
                if s < 0 or s > total:
                    continue
                if e - s < 15:
                    mid = (s + e) / 2
                    s, e = mid - 10, mid + 10
                s = max(0, s)
                e = min(e, s + 60, total)
                s, e = snap(s, e, lines)
                if e - s < 8:
                    continue
                reason = str(c.get("reason", "")).strip()
                found.append({"start": round(s, 1), "end": round(e, 1),
                              "score": int(c.get("score", 5)),
                              "reason": reason if good_reason(reason) else "a standout moment in the stream"})
            except Exception:
                continue

    found.sort(key=lambda c: -c["score"])
    final = []
    for c in found:
        if all(overlap(c, f) < 0.4 for f in final):
            final.append(c)
    final = final[:top]

    for i, c in enumerate(final, 1):
        print(f"Writing titles for clip {i}/{len(final)} ", end="", flush=True)
        c["titles"] = make_titles(c, lines, game, debug)
        print(" ok", flush=True)

    out = os.path.splitext(video)[0] + "_clips.json"
    json.dump(final, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if not final:
        print("\nNo clips passed the filters. Run again with --debug to see the model's raw answer.")
    for i, c in enumerate(final, 1):
        print(f"\n#{i} score {c['score']}  {c['start']}s -> {c['end']}s ({c['end']-c['start']:.0f}s)")
        print("  why:", c["reason"])
        for t in c["titles"]:
            print("  title:", t)
    print("\nSaved", out)


main()