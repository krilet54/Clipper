import sys, subprocess
from common import opt

args = sys.argv[1:]
if not args:
    print('Usage: python run.py "<youtube url>" [--game "GTA 5"] [--hours 2] [--keep 8] [--skip-fetch]')
    sys.exit()
py = sys.executable


def step(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        print("Step failed, stopping.")
        sys.exit(1)


if "--skip-fetch" not in args:
    step([py, "fetch.py", args[0], "--hours", opt(args, "--hours", "2")])
step([py, "analyze.py", "stream.mp4"])
step([py, "judge.py", "stream.mp4", "--game", opt(args, "--game", ""), "--keep", opt(args, "--keep", "8")])
step([py, "cut_clips.py", "stream.mp4"])
