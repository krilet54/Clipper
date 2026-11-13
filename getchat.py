import sys, json
from common import opt
from chat_downloader import ChatDownloader

args = sys.argv[1:]
if not args:
    print('Usage: python getchat.py "<youtube url>" [--hours 2] [--name stream]')
    sys.exit(1)
url = args[0]
hours = float(opt(args, "--hours", 2))
name = opt(args, "--name", "stream")
out = name + ".live_chat.json"
n = 0
with open(out, "w", encoding="utf-8") as f:
    chat = ChatDownloader().get_chat(url, end_time=int(hours * 3600),
                                     message_groups=["messages", "superchat"])
    for m in chat:
        t = m.get("time_in_seconds")
        if t is None or t < 0:
            continue
        paid = str(m.get("message_type", "")).startswith("paid")
        kind = "liveChatPaidMessageRenderer" if paid else "liveChatTextMessageRenderer"
        line = {"replayChatItemAction": {"videoOffsetTimeMsec": str(int(t * 1000)), "actions": [
            {"addChatItemAction": {"item": {kind: {"message": {"runs": [{"text": m.get("message") or ""}]}}}}}]}}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
        n += 1
        if n % 5000 == 0:
            print(f"  {n} messages, up to minute {int(t) // 60}", flush=True)
print(f"Saved {out}: {n} messages")
if n == 0:
    sys.exit(1)
