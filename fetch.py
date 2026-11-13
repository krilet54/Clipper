import sys, os, glob, subprocess
from common import *


def clean_old(name):
    pats = [name + ".mp4", name + ".mkv", name + ".webm", name + ".f*", name + ".part", name + ".*.part",
            name + ".live_chat.json", name + ".heatmap.json", name + "_*.json",
            os.path.join("work", name + "_*"), os.path.join("work", "cam_*.jpg"), "camcheck.jpg"]
    n = 0
    for pat in pats:
        for f in glob.glob(pat):
            try:
                os.remove(f)
                n += 1
            except Exception:
                pass
    if n:
        print(f"Removed {n} files left over from the previous stream.")


def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: python fetch.py "<url>" [--hours 2] [--max-gb 5] [--name stream] '
              '[--no-chat] [--cookies-from chrome]')
        return
    url = args[0]
    hours = float(opt(args, "--hours", 2))
    cap = float(opt(args, "--max-gb", 5))
    name = opt(args, "--name", "stream")
    cookies = opt(args, "--cookies-from")
    base = ["yt-dlp", "--no-playlist"] + (["--cookies-from-browser", cookies] if cookies else [])
    video = name + ".mp4"
    clean_old(name)

    ok = False
    for h in (720, 480, 360):
        fmt = f"bv*[height<={h}]+ba/b[height<={h}]"
        print(f"\nDownloading the first {hours:g}h at up to {h}p ...")
        r = subprocess.run(base + ["-f", fmt, "--download-sections", f"*0-{int(hours * 3600)}",
                                   "--merge-output-format", "mp4", "-o", name + ".%(ext)s", url])
        found = sorted(glob.glob(name + "*.mp4"), key=os.path.getsize, reverse=True)
        if r.returncode != 0 or not found:
            print("\nDownload failed. Read the yt-dlp error above.")
            print("Try: python -m pip install -U yt-dlp   and/or add: --cookies-from chrome")
            return
        if found[0] != video:
            os.replace(found[0], video)
        gb = os.path.getsize(video) / 1e9
        print(f"Video size: {gb:.2f} GB (cap {cap:g} GB)")
        if gb <= cap:
            ok = True
            break
        print("Over the cap, retrying at a lower resolution ...")
        os.remove(video)
    if not ok:
        print("Could not get under the size cap, even at 360p. Try --hours 1.")
        return
    print(f"Saved {video}, length {mmss(probe_duration(video))}")

    x, y, w, h = DEFAULT_CAM
    vf = f"crop=trunc(iw*{w}/2)*2:trunc(ih*{h}/2)*2:trunc(iw*{x}/2)*2:trunc(ih*{y}/2)*2"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "600", "-i", video,
                    "-frames:v", "1", "-update", "1", "-vf", vf, "camcheck.jpg"])
    print("Open camcheck.jpg: it should show the streamer's face. If not, use --cam x,y,w,h in judge.py.")

    # "Most replayed" data, saved now for the next upgrade (says NA if the video has none)
    subprocess.run(base + ["--skip-download", "--print-to-file", "%(heatmap)j", name + ".heatmap.json", url])

    if "--no-chat" in args:
        return
    chat = name + ".live_chat.json"
    print(f"\nDownloading chat for the first {hours:g}h ...")
    r = subprocess.run([sys.executable, "getchat.py", url, "--hours", str(hours), "--name", name])
    if r.returncode == 0 and os.path.exists(chat) and os.path.getsize(chat) > 1000:
        return
    print("getchat.py failed. Falling back to yt-dlp, which downloads the WHOLE stream's chat (slow) ...")
    subprocess.run(base + ["--skip-download", "--write-subs", "--sub-langs", "live_chat",
                           "-o", name + ".%(ext)s", url])
    print(f"Chat saved: {chat}" if os.path.exists(chat) else "No chat saved. Analysis continues without chat.")


main()
