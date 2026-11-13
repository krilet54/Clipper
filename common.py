import os, re, json, time, subprocess, urllib.request

OLLAMA = "http://localhost:11434"
WHISPER_FILE = "ggml-small.bin"
SPK_MODEL = "wespeaker_en_voxceleb_resnet34.onnx"
SPK_URL = "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/" + SPK_MODEL
DEFAULT_CAM = [0.0, 0.245, 0.208, 0.207]  # x, y, w, h as fractions of the frame
os.makedirs("work", exist_ok=True)


def opt(args, name, default=None):
    if name in args and args.index(name) + 1 < len(args):
        return args[args.index(name) + 1]
    return default


def mmss(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def num(x, default):
    try:
        return float(x)
    except Exception:
        return default


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(out)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def post(path, payload, timeout=300):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def unload(model):
    try:
        post("/api/generate", {"model": model, "keep_alive": 0}, 60)
    except Exception:
        pass


def warm(model):
    post("/api/generate", {"model": model, "prompt": "hi", "stream": False, "keep_alive": "20m",
                           "options": {"num_predict": 1}}, 600)


def extract_json(raw):
    raw = (raw or "").strip()
    pieces = [raw]
    if "{" in raw and "}" in raw:
        pieces.append(raw[raw.find("{"):raw.rfind("}") + 1])
    for p in pieces:
        try:
            v = json.loads(p)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return {}


def chat_text(model, system, prompt, max_tokens, temperature=0.3, as_json=True, ctx=6144, debug=False):
    payload = {"model": model, "stream": True, "keep_alive": "20m",
               "options": {"temperature": temperature, "num_ctx": ctx, "num_predict": max_tokens},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}]}
    if as_json:
        payload["format"] = "json"
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    raw, n, t0 = "", 0, time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            raw += d.get("message", {}).get("content", "")
            n += 1
            if n % 20 == 0:
                print(".", end="", flush=True)
            if time.time() - t0 > 240 or d.get("done"):
                break
    if debug:
        print("\n  RAW:", raw[:1500])
    return extract_json(raw) if as_json else raw
