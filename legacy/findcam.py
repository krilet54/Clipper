import sys, os, json, subprocess
import cv2

N_SAMPLES = 8
SEG = 300
CASCADE_FILES = ["haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml"]


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def probe_duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout.strip()
    return float(out)


def grab(video, t, path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video,
                    "-frames:v", "1", "-vf", "scale=960:-2", path])
    return cv2.imread(path)


def load_cascades():
    cs = []
    for f in CASCADE_FILES:
        c = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, f))
        if not c.empty():
            cs.append(c)
    return cs


def detect(img, cascades):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for c in cascades:
        faces = c.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
        if len(faces):
            return [tuple(int(v) for v in f) for f in faces]
    return []


def best_cluster(dets, W, H):
    clusters = []
    for fi, (x, y, w, h) in dets:
        cx, cy = x + w / 2, y + h / 2
        for c in clusters:
            mx = sum(i[0] for i in c["items"]) / len(c["items"])
            my = sum(i[1] for i in c["items"]) / len(c["items"])
            if abs(cx - mx) < 0.04 * W and abs(cy - my) < 0.06 * H:
                c["items"].append((cx, cy, w, h))
                c["frames"].append(fi)
                break
        else:
            clusters.append({"items": [(cx, cy, w, h)], "frames": [fi]})
    best, best_score = None, 0
    for c in clusters:
        n = len(set(c["frames"]))
        cx = sum(i[0] for i in c["items"]) / len(c["items"])
        score = n + (0.5 if (cx < 0.3 * W or cx > 0.7 * W) else 0)
        if n >= 3 and score > best_score:
            best, best_score = c, score
    return best


def make_box(c, W, H):
    k = len(c["items"])
    cx = sum(i[0] for i in c["items"]) / k
    cy = sum(i[1] for i in c["items"]) / k
    fw = sum(i[2] for i in c["items"]) / k
    bw, bh = fw * 3.6, fw * 3.0
    x0 = min(max(0, cx - bw / 2), W - bw)
    y0 = min(max(0, cy - bh * 0.4), H - bh)
    return [round(x0 / W, 3), round(y0 / H, 3), round(bw / W, 3), round(bh / H, 3)]


def similar(a, b):
    return (abs(a[0] - b[0]) < 0.03 and abs(a[1] - b[1]) < 0.04 and abs(a[2] - b[2]) < 0.04)


def mmss(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python findcam.py test.mp4 [--limit 1200]")
        return
    video = args[0]
    limit = int(opt(args, "--limit", 0))
    base = os.path.splitext(video)[0]
    dur = probe_duration(video)
    if limit:
        dur = min(dur, limit)

    cascades = load_cascades()
    if not cascades:
        print("Could not load OpenCV face cascades.")
        return

    segs = []
    start = 0.0
    while start < dur - 5:
        segs.append([start, min(start + SEG, dur)])
        start += SEG

    results = []
    for si, (s, e) in enumerate(segs):
        dets, imgs, W, H = [], {}, 960, 540
        for k in range(N_SAMPLES):
            t = s + (e - s) * (k + 0.5) / N_SAMPLES
            img = grab(video, t, "tmp_cam.jpg")
            if img is None:
                continue
            H, W = img.shape[:2]
            imgs[k] = img
            for f in detect(img, cascades):
                dets.append((k, f))
        c = best_cluster(dets, W, H)
        box, dbg = None, None
        if c:
            box = make_box(c, W, H)
            dbg = imgs[c["frames"][0]].copy()
        results.append({"start": s, "end": e, "box": box, "dbg": dbg})
        print(f"  {mmss(s)}-{mmss(e)}: {'facecam ' + str(box) if box else 'no facecam found'}", flush=True)

    # fill single-segment gaps between two matching neighbours
    for i in range(1, len(results) - 1):
        a, b, c = results[i - 1], results[i], results[i + 1]
        if b["box"] is None and a["box"] and c["box"] and similar(a["box"], c["box"]):
            b["box"], b["dbg"] = a["box"], a["dbg"]

    # merge neighbours with the same box
    merged = []
    for r in results:
        if merged and ((merged[-1]["box"] is None and r["box"] is None) or
                       (merged[-1]["box"] and r["box"] and similar(merged[-1]["box"], r["box"]))):
            merged[-1]["end"] = r["end"]
        else:
            merged.append(dict(r))

    out = []
    print("\nFacecam layout found:")
    for k, r in enumerate(merged, 1):
        out.append({"start": r["start"], "end": r["end"], "box": r["box"]})
        print(f"  {mmss(r['start'])} -> {mmss(r['end'])}: {r['box'] if r['box'] else 'none (audio-only)'}")
        if r["box"] and r["dbg"] is not None:
            H, W = r["dbg"].shape[:2]
            x, y, w, h = r["box"]
            img = r["dbg"]
            cv2.rectangle(img, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (0, 255, 0), 3)
            cv2.imwrite(f"cam_debug_{k}.jpg", img)
    json.dump(out, open(base + "_cam.json", "w"), indent=1)
    print(f"\nSaved {base}_cam.json. Open cam_debug_*.jpg to check the green box sits on the streamer.")


main()