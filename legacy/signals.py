import sys, os, re, json, subprocess, statistics


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def probe_duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return float(out)


def run_audio(video, limit):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    if limit:
        cmd += ["-t", str(limit)]
    cmd += ["-i", video, "-vn", "-af",
            "aformat=sample_rates=8000:channel_layouts=mono,asetnsamples=n=8000:p=0,"
            "astats=metadata=1:reset=1,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=sig_audio.txt",
            "-f", "null", os.devnull]
    subprocess.run(cmd)
    print()


def run_motion(video, start, dur, box, outfile):
    x, y, w, h = box
    crop = (f"crop=trunc(iw*{w}/2)*2:trunc(ih*{h}/2)*2:"
            f"trunc(iw*{x}/2)*2:trunc(ih*{y}/2)*2")
    vf = (f"fps=4,{crop},scale=160:-2,format=yuv420p,tblend=all_mode=difference,"
          f"signalstats,metadata=mode=print:key=lavfi.signalstats.YAVG:file={outfile}")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
                    "-ss", str(start), "-t", str(dur), "-i", video,
                    "-an", "-vf", vf, "-f", "null", os.devnull])
    print()


def read_series(path, key):
    pat = re.compile(re.escape(key) + r"=(\S+)")
    out, t = [], None
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = pat.search(line)
        if m and t is not None:
            try:
                v = float(m.group(1))
                if v != v or abs(v) == float("inf"):
                    v = -90.0
            except ValueError:
                v = -90.0
            out.append((t, v))
    return out


def per_second(series, n, how, default):
    bins = [[] for _ in range(n)]
    for t, v in series:
        i = int(t)
        if 0 <= i < n:
            bins[i].append(v)
    res, last = [], default
    for b in bins:
        if b:
            last = max(b) if how == "max" else sum(b) / len(b)
        res.append(last)
    return res


def robust_z(vals, block=600, min_scale=1.0, rel=0.0):
    z = []
    for b in range(0, len(vals), block):
        seg = vals[b:b + block]
        med = statistics.median(seg)
        mad = statistics.median([abs(v - med) for v in seg])
        scale = max(1.4826 * mad, min_scale, rel * abs(med))
        z += [max(0.0, min(6.0, (v - med) / scale)) for v in seg]
    return z


def smooth(v, k=1):
    n = len(v)
    return [sum(v[max(0, i - k):i + k + 1]) / len(v[max(0, i - k):i + k + 1]) for i in range(n)]


def mmss(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def box_at(layout, t):
    for seg in layout:
        if seg["start"] <= t < seg["end"]:
            return seg["box"]
    return None


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python signals.py test.mp4 [--limit 1200] [--top 25] [--cam x,y,w,h]")
        return
    video = args[0]
    limit = int(opt(args, "--limit", 0))
    top = int(opt(args, "--top", 25))
    base = os.path.splitext(video)[0]

    dur = probe_duration(video)
    if limit:
        dur = min(dur, limit)
    n = int(dur) + 1

    manual = opt(args, "--cam")
    if manual:
        layout = [{"start": 0, "end": dur, "box": [float(v) for v in manual.split(",")]}]
        print("Using manual facecam box:", layout[0]["box"])
    elif os.path.exists(base + "_cam.json"):
        layout = json.load(open(base + "_cam.json"))
        layout = [l for l in layout if l["start"] < dur]
        if layout:
            layout[-1]["end"] = min(layout[-1]["end"], dur)
        print(f"Using facecam layout from {base}_cam.json ({len(layout)} region(s))")
    else:
        layout = [{"start": 0, "end": dur, "box": None}]
        print(f"No {base}_cam.json found. Run findcam.py first or pass --cam. Using audio only.")

    print("Pass 1: audio loudness...")
    run_audio(video, limit)
    aud = read_series("sig_audio.txt", "lavfi.astats.Overall.RMS_level")
    if not aud:
        print("No audio data. Open sig_audio.txt and send me the first 10 lines.")
        return
    a = robust_z(per_second(aud, n, "mean", -60.0), min_scale=2.0)

    m = [0.0] * n
    valid = [False] * n
    for k, seg in enumerate(layout):
        if not seg["box"]:
            continue
        s, e = int(seg["start"]), min(n, int(seg["end"]))
        if e - s < 5:
            continue
        print(f"Pass 2: facecam motion {mmss(s)} -> {mmss(e)} ...")
        f = f"sig_motion_{k}.txt"
        run_motion(video, s, e - s, seg["box"], f)
        ser = [(t + s, v) for t, v in read_series(f, "lavfi.signalstats.YAVG")]
        if not ser:
            print("  no motion data for this region")
            continue
        raw = per_second(ser, n, "max", 0.0)[s:e]
        z = robust_z(raw, min_scale=0.05, rel=0.25)
        for i, v in enumerate(z):
            m[s + i] = v
            valid[s + i] = True

    if not any(valid):
        print("Note: no facecam motion available, scoring on audio only.")

    comb = smooth([(a[i] + m[i] + 0.5 * min(a[i], m[i])) if valid[i] else 1.6 * a[i]
                   for i in range(n)])

    order = sorted(range(n), key=lambda i: -comb[i])
    picks = []
    for i in order:
        if comb[i] <= 0.5:
            break
        if all(abs(i - p) >= 30 for p in picks):
            picks.append(i)
        if len(picks) >= top:
            break
    if not picks:
        print("No candidates found.")
        return
    best = comb[picks[0]]
    cands = []
    for i in picks:
        s, e = max(0, i - 14), min(n, i + 12)
        ap = max(a[s:e])
        has_cam = any(valid[s:e])
        mp = max(m[s:e]) if has_cam else None
        cands.append({
            "start": float(s), "end": float(e), "peak": i,
            "score": max(1, min(10, round(10 * comb[i] / best))),
            "audio_z": round(ap, 1),
            "motion_z": round(mp, 1) if mp is not None else None,
            "cam_box": box_at(layout, i),
            "reason": (f"audio spike {ap:.1f} sigma, facecam motion {mp:.1f} sigma" if has_cam
                       else f"audio spike {ap:.1f} sigma (no facecam in this part)"),
            "titles": [],
        })
    json.dump(cands, open(base + "_cands.json", "w", encoding="utf-8"), indent=1)
    json.dump({"audio_z": [round(v, 2) for v in a], "motion_z": [round(v, 2) for v in m]},
              open(base + "_signals.json", "w"))
    print(f"\n{n}s analysed. Top candidates:\n")
    for k, c in enumerate(cands, 1):
        mo = c["motion_z"] if c["motion_z"] is not None else "n/a"
        print(f"#{k:2d} score {c['score']:2d}  {mmss(c['start'])} -> {mmss(c['end'])}   "
              f"audio {c['audio_z']}  motion {mo}")
    print(f"\nSaved {base}_cands.json and {base}_signals.json")


main()