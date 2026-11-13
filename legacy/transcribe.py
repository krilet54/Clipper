import os, sys, json, glob, site

# Point Windows at the CUDA DLLs that pip installed
for sp in site.getsitepackages():
    for dll_dir in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel

video = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
out = os.path.splitext(video)[0] + "_transcript.json"

try:
    model = WhisperModel("small", device="cuda", compute_type="int8_float16")
    print("Running on GPU")
except Exception as e:
    print("GPU failed, falling back to CPU:", e)
    model = WhisperModel("small", device="cpu", compute_type="int8")

segments, info = model.transcribe(video, vad_filter=True, word_timestamps=True)
print(f"Language: {info.language}, duration: {info.duration:.0f}s")

result = []
for seg in segments:
    result.append({
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip(),
        "words": [{"w": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                  for w in (seg.words or [])],
    })
    print(f"[{seg.start:7.1f}s] {seg.text.strip()}")

with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
print("Saved", out)