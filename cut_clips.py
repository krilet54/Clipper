import sys, os, re, subprocess
from common import *


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:20] or "clip"


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python cut_clips.py stream.mp4 [--n 8] [--copy]")
        return
    video = args[0]
    base = os.path.splitext(video)[0]
    clips = load_json(base + "_clips.json")
    if not clips:
        print(f"No clips in {base}_clips.json. Run judge.py first.")
        return
    n = int(opt(args, "--n", len(clips)))
    os.makedirs("clips", exist_ok=True)
    for f in os.listdir("clips"):
        if f.endswith(".mp4"):
            os.remove(os.path.join("clips", f))
    notes = []
    for i, c in enumerate(clips[:n], 1):
        dur = c["end"] - c["start"]
        out = f"clips/{i:02d}_{slug(c.get('tag', 'clip'))}_{mmss(c['start']).replace(':', '-')}.mp4"
        print(f"Cutting #{i}: {mmss(c['start'])} -> {mmss(c['end'])} ({dur:.0f}s)")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{c['start']:.2f}",
               "-i", video, "-t", f"{dur:.2f}"]
        if "--copy" in args:
            cmd += ["-c", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart"]
        subprocess.run(cmd + [out])
        notes.append(f"{out}\n  score {c.get('final')}  [{c.get('tag')}]\n  {c.get('reason', '')}\n  " +
                     "\n  ".join("title: " + t for t in c.get("titles", [])) + "\n")
    open("clips/titles.txt", "w", encoding="utf-8").write("\n".join(notes))
    print("\nDone. Clips and titles.txt are in", os.path.abspath("clips"))


main()
