"""Exercise real Gemini HTTP requests against a local server; never spend a real API allowance."""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, ".")


def check(name, condition):
    assert condition, name
    print(f"PASS {name}")


def main(folder):
    os.environ.update(PEAKCUT_DATA=folder, DATABASE_URL="", PEAKCUT_ACCOUNTS="0",
                      PEAKCUT_AI_PROVIDER="gemini", PEAKCUT_AI_MODELS="first,second", PEAKCUT_AI_KEY="test-only",
                      PEAKCUT_AI_KEY_FILE="", PEAKCUT_CAPTION_DAILY_REQUESTS="7")
    from studio import ai, auth, brain, ctx, db, usage
    import numpy as np

    clip = {"start": 0, "end": 15.9, "score": 95, "hook": "A huge comeback", "title": "The winning play",
            "description": "A streamer wins the round.", "hashtags": ["gaming"], "reason": "A comeback"}
    requests = []

    class Gemini(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            schema = body["generationConfig"]["responseSchema"]
            prompt = body["contents"][0]["parts"][0]["text"]
            caption = schema["items"]["type"] == "STRING" or "i" in schema["items"].get("properties", {})
            if schema["items"]["type"] == "STRING":
                slots = json.loads(prompt.splitlines()[-1])["words"]
                answer = (["This", "is", "very", "good."] * len(slots))[:len(slots)]
            else:
                answer = [] if caption else [clip]
            sent, wrote = (320, 180) if caption else (500, 100)
            requests.append({"path": self.path, "key": self.headers.get("x-goog-api-key"),
                             "caption": caption, "tokens": sent + wrote})
            response = json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps(answer)}]}}],
                                   "usageMetadata": {"promptTokenCount": sent, "totalTokenCount": sent + wrote}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Gemini)
    thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
    thread.start()
    db.init()
    user = auth.local_user().id
    source = "We were losing the round but Ninja made an incredible comeback.".split()
    words = [{"w": source[i % len(source)], "s": i * .25, "e": i * .25 + .2, "lang": "en"} for i in range(160)]
    signals = np.zeros(40)
    signals[10:16] = 3

    def video():
        clips, info = brain.find_clips(words, signals, 40, "Ninja tournament", min_len=5, max_len=30, max_clips=1)
        check("video selection uses the AI answer, not the loudness fallback", len(clips) == 1 and info["model"] == "first"
              and info["tokens"] == 600 and clips[0]["title"] == clip["title"] and not info["note"])

    def other_tasks():
        clips, tokens = brain.find_live_clips(words, signals, 40, min_len=5, max_len=30)
        check("live clipping still uses AI", bool(clips) and tokens == 600)
        result = brain.describe("Ninja wins the round after a huge comeback")
        check("manual clip hooks and descriptions still use AI", result["title"] == clip["title"] and bool(result["hook"]))
        clips, tokens = brain.judge_transcript(words, 0, 40, min_len=5, max_len=30)
        check("live speech judging still uses AI", bool(clips) and tokens == 600)
        moments, tokens = brain.judge_signals(signals, [1] * 40, 0, 40)
        check("reaction and chat judging still use AI", bool(moments) and tokens == 600)

    try:
        with ctx.as_user(user), patch.object(ai, "base_url", return_value=f"http://127.0.0.1:{endpoint.server_port}"):
            for _ in range(9):
                video()
            other_tasks()
            for i in range(7):
                lang = "hi" if i % 2 == 0 else "en"
                timed = [{"w": w, "s": j * .5, "e": j * .5 + .4, "lang": lang}
                         for j, w in enumerate(["यह", "बहुत", "अच्छा", "है"] if lang == "hi" else ["This", "is", "very", "good."])]
                fixed = brain.english_captions(timed, lang, "A good play")
                check("HTTP caption answers preserve original timing slots", [dict(w, w="") for w in fixed]
                      == [dict(w, w="") for w in timed] and " ".join(w["w"] for w in fixed) == "This is very good.")
            check("20 actual local HTTP requests allocate 13 to clipping and other work, 7 to captions",
                  len(requests) == 20 and sum(r["caption"] for r in requests) == 7)
            check("the real Gemini request builder sends the configured model, header and schema",
                  all(r["path"] == "/first:generateContent" and r["key"] == "test-only" for r in requests))
            summary = usage.summary({})
            purposes = {row[0]: (row[1], row[2]) for row in summary["by_purpose"]}
            check("the usage page records real returned counts under the right tasks",
                  summary["calls_total"] == 20 and summary["tokens_total"] == 11300
                  and purposes["Video: finding clips"] == (5400, 9)
                  and sum(count for purpose, (_tokens, count) in purposes.items() if purpose.startswith("Caption")) == 7)
            caption_tokens = sum(tokens for purpose, (tokens, _count) in purposes.items() if purpose.startswith("Caption"))
            check("the simulated workload accounts for 69 percent non-caption tokens and 31 percent caption tokens",
                  round(100 * caption_tokens / summary["tokens_total"]) == 31)
            try:
                brain.ask_ai("test", brain.FIX_SCHEMA, purpose="Caption translation")
                raise AssertionError("eighth caption request was allowed")
            except RuntimeError as error:
                check("the eighth caption attempt is stopped before HTTP and does not create fake usage",
                      "Caption budget" in str(error) and len(requests) == 20 and usage.summary({})["calls_total"] == 20)
            other_tasks()
            check("all four other AI tasks remain available after the caption cap", len(requests) == 24)
            video()
            check("video selection remains available after the caption cap", len(requests) == 25)
            second, _ = auth._create("quota-test@example.invalid", "Quota test", password_hash="unused")
            with ctx.as_user(second.id):
                try:
                    brain.ask_ai("test", brain.FIX_SCHEMA, purpose="Caption translation")
                    raise AssertionError("another user bypassed the shared caption budget")
                except RuntimeError:
                    check("another account cannot bypass the shared caption allowance", len(requests) == 25)
                video()
                check("another account can still clip and sees only its own usage", usage.summary({})["calls_total"] == 1)
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        thread.join(timeout=5)
        db.engine.dispose()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pc-quota-allocation-") as data:
        main(data)
