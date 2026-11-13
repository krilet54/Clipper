import sys, os, re, json, time, wave, subprocess, urllib.request
import numpy as np
from common import *

SR = 16000
HALLU = re.compile(r"^\W*(thank you( so much)?( for watching)?|thanks for watching|you|bye|subtitles by.*)\W*$", re.I)
LAUGH_TAG = re.compile(r"laugh|chuckl|giggl|cackl", re.I)
HYPE_VOICE = re.compile(r"\b(oh my god|omg|holy|what the|wtf|no way|nah|bro what|let'?s go+|are you serious|"
                        r"i can'?t|stop it|get out|shut up|hell no|oh no|help|dead|crazy|insane|yo+)\b", re.I)
CHAT_HYPE = re.compile(r"(lmao|lmfao|\blol\b|omg|wtf|no ?way|\bclip\b|insane|crazy|cooked|dead|bruh|nooo|"
                       r"😂|🤣|💀|😭|🔥|kekw|pog|\bw\b|\bl\b)", re.I)


def ensure_spk_model():
    if os.path.exists(SPK_MODEL):
        return
    print("Downloading speaker model (26 MB) ...")
    req = urllib.request.Request(SPK_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(SPK_MODEL, "wb") as f:
        f.write(r.read())


def ensure_audio(video, wav):
    if os.path.exists(wav) and os.path.getsize(wav) > 10000:
        return
    print("Extracting audio ...")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
                    "-ar", str(SR), "-c:a", "pcm_s16le", wav], check=True)


def load_wav(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def parse_srt(path, offset):
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8", errors="ignore").read().replace("\r", "")
    out = []
    for block in re.split(r"\n\s*\n", text):
        rows = [r for r in block.strip().split("\n") if r.strip()]
        if len(rows) < 2:
            continue
        idx = 1 if "-->" in rows[1] else 0
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", rows[idx])
        if not m:
            continue
        g = list(map(int, m.groups()))
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000 + offset
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000 + offset
        raw = " ".join(rows[idx + 1:])
        tags = " ".join(re.findall(r"\(.*?\)|\[.*?\]|\*.*?\*", raw))
        body = re.sub(r"\(.*?\)|\[.*?\]|\*.*?\*", "", raw).strip(" -")
        laugh = bool(LAUGH_TAG.search(tags))
        if len(body) <= 1 and not laugh:
            continue
        if HALLU.match(body) and not laugh:
            continue
        out.append({"s": round(s, 2), "e": round(e, 2), "text": body, "laugh": laugh})
    out.sort(key=lambda d: d["s"])
    return out


def transcribe(wav, total, lang, base, chunk=600):
    segs = []
    n = int(np.ceil(total / chunk))
    for i in range(n):
        s0 = i * chunk
        d = min(chunk, total - s0)
        srt = f"work/{base}_tx_{i:03d}.srt"
        done = f"work/{base}_tx_{i:03d}.done"
        if not os.path.exists(done):
            t0 = time.time()
            if os.path.exists(srt):
                os.remove(srt)
            af = f"whisper=model={WHISPER_FILE}:language={lang}:destination={srt}:format=srt"
            r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{s0:.2f}",
                                "-t", f"{d:.2f}", "-i", wav, "-vn", "-af", af, "-f", "null", "-"])
            if r.returncode == 0:
                open(done, "w").write("ok")
            print(f"  chunk {i + 1}/{n} ({mmss(s0)}) done in {time.time() - t0:.0f}s", flush=True)
        segs += parse_srt(srt, s0)
    return segs


def embed(ex, samples):
    st = ex.create_stream()
    st.accept_waveform(sample_rate=SR, waveform=samples)
    st.input_finished()
    return np.array(ex.compute(st), dtype=np.float32)


def build_profile(En, seed_thr=0.5):
    sub = En
    if len(En) > 4000:
        sub = En[np.random.RandomState(0).choice(len(En), 4000, replace=False)]
    S = sub @ sub.T
    thr = seed_thr
    cnt = (S > thr).sum(1)
    if cnt.max() < 10:
        thr = 0.35
        cnt = (S > thr).sum(1)
    seed = int(np.argmax(cnt))
    prof = sub[S[seed] > thr].mean(0)
    for _ in range(3):
        prof = prof / np.linalg.norm(prof)
        m = (sub @ prof) > thr
        if m.sum() < 5:
            break
        prof = sub[m].mean(0)
    return prof / np.linalg.norm(prof)


def auto_threshold(v):
    lo, hi = np.percentile(v, 20), np.percentile(v, 80)
    for _ in range(30):
        m = (lo + hi) / 2
        a, b = v[v < m], v[v >= m]
        if len(a) == 0 or len(b) == 0:
            break
        lo, hi = a.mean(), b.mean()
    return float(np.clip((lo + hi) / 2, 0.25, 0.60))


def speaker_labels(segs, x, base, args):
    try:
        import sherpa_onnx
    except ImportError:
        print("Missing package. Run: pip install sherpa-onnx")
        sys.exit(1)
    ensure_spk_model()
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=SPK_MODEL, num_threads=4, debug=False, provider="cpu")
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    cache = f"work/{base}_emb.npz"
    sig = f"{len(segs)}:{round(sum(s['s'] for s in segs), 1)}"
    idx, E = None, None
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["sig"]) == sig:
            idx, E = z["idx"], z["E"]
            print("  using cached voice embeddings")
    if idx is None:
        ii, vecs, t0 = [], [], time.time()
        for k, sg in enumerate(segs):
            if sg["e"] - sg["s"] < 1.0:
                continue
            a = int(sg["s"] * SR)
            b = int(min(sg["e"], sg["s"] + 8) * SR)
            w = x[a:b].astype(np.float32) / 32768.0
            if len(w) < SR:
                continue
            vecs.append(embed(ex, w))
            ii.append(k)
            if len(ii) % 300 == 0:
                print(f"  embedded {len(ii)} segments ({time.time() - t0:.0f}s)", flush=True)
        idx, E = np.array(ii), np.array(vecs)
        np.savez(cache, idx=idx, E=E, sig=np.array(sig))
    if len(E) < 10:
        print("Too little speech found to learn the streamer's voice.")
        sys.exit(1)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    ref = opt(args, "--ref")
    if ref:
        a, b = [float(v) for v in ref.split(",")]
        vs = [embed(ex, x[int(t * SR):int((t + 4) * SR)].astype(np.float32) / 32768.0)
              for t in np.arange(a, b - 3, 4)]
        prof = np.mean(vs, axis=0)
        prof = prof / np.linalg.norm(prof)
        print(f"  voice profile taken from your reference {a:.0f}-{b:.0f}s")
    else:
        prof = build_profile(En)
    sims = En @ prof
    thr = num(opt(args, "--voice-thr"), 0) or auto_threshold(sims)

    row = {int(i): k for k, i in enumerate(idx)}
    for i, sg in enumerate(segs):
        if i in row:
            sg["sim"] = round(float(sims[row[i]]), 3)
            sg["label"] = "streamer" if sims[row[i]] >= thr else "other"
    for i, sg in enumerate(segs):
        if "label" in sg:
            continue
        if not sg["text"] and sg["laugh"]:
            sg["label"] = "streamer"
            continue
        lab = None
        for j in (i - 1, i + 1, i - 2, i + 2):
            if 0 <= j < len(segs) and "sim" in segs[j] and abs(segs[j]["s"] - sg["s"]) < 2.5:
                lab = segs[j]["label"]
                break
        sg["label"] = lab or "other"

    st = [s for s in segs if s["label"] == "streamer"]
    print(f"  voice threshold {thr:.2f}: {len(st)} of {len(segs)} segments are the streamer "
          f"({100 * len(st) / max(1, len(segs)):.0f}%)")
    print("  CHECK these. STREAMER lines should be him talking:")
    for s in st[:: max(1, len(st) // 4)][:4]:
        print(f"    {mmss(s['s'])} S: {s['text'][:80]}")
    ot = [s for s in segs if s["label"] == "other"]
    print("  GAME/OTHER lines should be game dialogue, NPCs, guests or noise:")
    for s in ot[:: max(1, len(ot) // 4)][:4]:
        print(f"    {mmss(s['s'])} G: {s['text'][:80]}")
    return thr


def band_db(x, T):
    voice = np.full(T, -90.0, np.float32)
    full = np.full(T, -90.0, np.float32)
    win = np.hanning(SR).astype(np.float32)
    for a in range(0, T, 300):
        b = min(T, a + 300)
        blk = x[a * SR:b * SR]
        n = len(blk) // SR
        if n == 0:
            continue
        blk = blk[:n * SR].astype(np.float32).reshape(n, SR) / 32768.0
        full[a:a + n] = 20 * np.log10(np.sqrt((blk ** 2).mean(1)) + 1e-6)
        P = np.abs(np.fft.rfft(blk * win, axis=1)) ** 2
        voice[a:a + n] = 10 * np.log10(P[:, 300:3400].sum(1) + 1e-6)
    return voice, full


def rz(v, mask=None):
    ref = v[mask] if mask is not None and mask.sum() > 20 else v
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) * 1.4826
    return (v - med) / max(mad, 0.5 * ref.std(), 1e-6)


def local_z(series, half=300):
    T = len(series)
    z = np.zeros(T, np.float32)
    for t0 in range(0, T, 10):
        w = series[max(0, t0 - half):min(T, t0 + half):5]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        z[t0:t0 + 10] = (series[t0:t0 + 10] - med) / max(mad, 0.5 * np.sqrt(med + 1), 1.0)
    return z


def boxsum(v, w):
    return np.convolve(v, np.ones(w, np.float32), mode="same")


def load_chat(path, total):
    msgs = []
    if not os.path.exists(path):
        return msgs
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            r = d.get("replayChatItemAction") or {}
            ms = r.get("videoOffsetTimeMsec") or d.get("videoOffsetTimeMsec")
            if ms is None:
                continue
            t = num(ms, -1) / 1000.0
            if t < 0 or t >= total:
                continue
            for a in r.get("actions", []):
                item = (a.get("addChatItemAction") or {}).get("item") or {}
                for kind, rend in item.items():
                    if not isinstance(rend, dict):
                        continue
                    text = ""
                    for run in (rend.get("message") or {}).get("runs") or []:
                        if "text" in run:
                            text += run["text"]
                        elif "emoji" in run:
                            eid = run["emoji"].get("emojiId", "")
                            text += eid if eid and len(eid) <= 4 else " "
                    special = ("Paid" in kind) or ("Sponsorship" in kind) or ("Membership" in kind)
                    if text.strip() or special:
                        msgs.append([round(t, 1), text.strip()[:120], 1 if special else 0])
    return msgs


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python analyze.py stream.mp4 [--lang en] [--chat stream.live_chat.json] "
              "[--ref 300,330] [--voice-thr 0.4] [--topn 40]")
        return
    video = args[0]
    base = os.path.splitext(video)[0]
    lang = opt(args, "--lang", "en")
    topn = int(opt(args, "--topn", 40))
    chat_path = opt(args, "--chat", base + ".live_chat.json")
    if not os.path.exists(WHISPER_FILE):
        print(f"{WHISPER_FILE} not found in this folder.")
        return
    total = probe_duration(video)
    T = int(total)
    print(f"Video length: {mmss(total)}")

    wav = f"work/{base}_audio.wav"
    ensure_audio(video, wav)
    x = load_wav(wav)

    print("\nStage 1/4: transcribing the whole video ...", flush=True)
    segs = transcribe(wav, total, lang, base)
    print(f"  {len(segs)} segments")
    if not segs:
        print("No speech found. Check that ggml-small.bin works with your ffmpeg.")
        return

    print("\nStage 2/4: telling the streamer's voice apart from game audio ...", flush=True)
    thr = speaker_labels(segs, x, base, args)

    print("\nStage 3/4: measuring signals ...", flush=True)
    vdb, fdb = band_db(x, T)
    cov = np.zeros(T, np.float32)
    wps = np.zeros(T, np.float32)
    laugh = np.zeros(T, np.float32)
    hype = np.zeros(T, np.float32)
    for sg in segs:
        if sg["label"] != "streamer":
            continue
        dur = max(0.5, sg["e"] - sg["s"])
        words = len(sg["text"].split())
        for t in range(int(sg["s"]), min(T, int(sg["e"]) + 1)):
            ov = min(sg["e"], t + 1) - max(sg["s"], t)
            if ov > 0:
                cov[t] = max(cov[t], ov)
                wps[t] += words * ov / dur
        mid = int((sg["s"] + sg["e"]) / 2)
        if 0 <= mid < T:
            if sg["laugh"]:
                laugh[mid] += 1
            hype[mid] += len(HYPE_VOICE.findall(sg["text"])) + 0.5 * sg["text"].count("!")
    vmask = cov >= 0.6
    voice_z = np.zeros(T, np.float32)
    if vmask.sum() > 20:
        voice_z[vmask] = rz(vdb, vmask)[vmask]
    gmask = cov < 0.2
    game_z = np.zeros(T, np.float32)
    if gmask.sum() > 20:
        game_z[gmask] = rz(fdb, gmask)[gmask]
    wps_s = np.convolve(wps, np.ones(8, np.float32) / 8, mode="same")
    rate_z = np.zeros(T, np.float32)
    if vmask.sum() > 20:
        rate_z[vmask] = rz(wps_s, vmask)[vmask]
    laugh_s = 2.5 * np.minimum(boxsum(laugh, 8), 2)
    hype_s = 0.8 * np.minimum(boxsum(hype, 8), 5)

    print("\nStage 4/4: chat replay ...", flush=True)
    msgs = load_chat(chat_path, total)
    has_chat = len(msgs) > 50
    chat_z = np.zeros(T, np.float32)
    if has_chat:
        cnt = np.zeros(T, np.float32)
        hyp = np.zeros(T, np.float32)
        for t, text, sp in msgs:
            i = int(t)
            if 0 <= i < T:
                cnt[i] += 1 + 4 * sp
                if sp or CHAT_HYPE.search(text):
                    hyp[i] += 1
        chat_z = np.clip(np.maximum(local_z(boxsum(cnt, 10)), local_z(boxsum(hyp, 10))), 0, 8)
        print(f"  {len(msgs)} chat messages in this range")
    else:
        print("  no usable chat found: scoring without chat (clips will be less reliable)")

    parts = {"voice": voice_z, "chat": chat_z, "rate": rate_z, "game": game_z, "laugh": laugh_s, "hype": hype_s}
    W = {"voice": 1.0, "chat": 1.0 if has_chat else 0.0, "rate": 0.4, "game": 0.4, "laugh": 0.8, "hype": 0.6}
    comp = sum(W[k] * np.clip(parts[k], 0, 6) for k in parts)
    comp = np.convolve(comp, np.ones(5, np.float32) / 5, mode="same")
    comp[:15] = 0

    min_comp = float(opt(args, "--min-comp", 2.5))
    cs = comp.copy()
    cands = []
    while len(cands) < topn:
        t = int(np.argmax(cs))
        if cs[t] < min_comp:
            break
        sig = {k: round(float(parts[k][max(0, t - 6):t + 7].max()), 2) for k in parts}
        fired = [k for k, th in (("voice", 2), ("chat", 2), ("laugh", 2.5), ("game", 2.5), ("hype", 1.5))
                 if sig[k] >= th]
        cands.append({"peak": t, "start": max(0, t - 14), "end": min(T, t + 14),
                      "score": round(float(comp[t]), 2), "sig": sig, "fired": fired})
        cs[max(0, t - 45):t + 45] = -1

    save_json(f"{base}_lines.json", [{"s": s["s"], "e": s["e"], "text": s["text"], "label": s["label"],
                                      "laugh": s["laugh"], "sim": s.get("sim")} for s in segs])
    save_json(f"{base}_chat.json", msgs)
    save_json(f"{base}_cands.json", cands)
    save_json(f"{base}_meta.json", {"total": total, "has_chat": has_chat, "voice_thr": thr})

    print(f"\n{len(cands)} candidates. Top 15:")
    for i, c in enumerate(cands[:15], 1):
        print(f"#{i:<2} {mmss(c['peak'])}  score {c['score']:<5}  fired: {', '.join(c['fired']) or '-'}")
    print(f"\nSaved {base}_cands.json, {base}_lines.json, {base}_chat.json. Now run: python judge.py {video}")


main()
