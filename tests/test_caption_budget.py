"""Caption cost, timing and request limits, with every network call replaced."""
import copy
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".")


def check(name, condition):
    assert condition, name
    print(f"PASS {name}")


def words(n=30, offset=0, lang="en"):
    return [{"w": "hello", "s": offset + i * .25, "e": offset + i * .25 + .2,
             "lang": lang, "confidence": .8} for i in range(n)]


def rejected(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return True
    return False


def main(data):
    os.environ.update(PEAKCUT_DATA=data, DATABASE_URL="", PEAKCUT_ACCOUNTS="0",
                      PEAKCUT_AI_PROVIDER="gemini", PEAKCUT_AI_MODELS="first,second", PEAKCUT_AI_KEY="test-only",
                      PEAKCUT_AI_KEY_FILE="", PEAKCUT_CAPTION_DAILY_REQUESTS="7")
    from studio import ai, auth, brain, ctx, db, jobs, media, render, transcribe, usage
    import numpy as np

    db.init()
    user = auth.local_user().id
    groups = [(words(offset=i * 10), "en") for i in range(15)]
    original = copy.deepcopy(groups)
    with patch.object(brain, "ask_ai", return_value=([{"i": 2, "w": "benching"}], "test", 1)) as ask:
        fixed = brain.fix_caption_batches(groups, "Ninja gets benched")
    check("15 English clips share one request, with sparse corrections", ask.call_count == 1
          and fixed[0][2]["w"] == "benching" and groups == original)
    check("batching preserves every clip's timing and metadata",
          [[dict(w, w="") for w in group] for group in fixed] == [[dict(w, w="") for w in group] for group, _ in groups])
    payload = json.loads(ask.call_args.args[0].splitlines()[-1])
    check("each word is sent once, without timestamps or a repeated sentence transcript",
          len(payload["words"]) == 450 and isinstance(payload["words"][0], str) and "<sentences>" not in ask.call_args.args[0])
    with patch.object(brain, "ask_ai", return_value=([{"i": 2, "w": "xQc"}], "test", 1)) as ask:
        overlap = brain.fix_caption_batches([(groups[0][0], "en"), (groups[0][0][2:], "en")])
    check("overlapping clips share corrected words without duplicating input",
          len(json.loads(ask.call_args.args[0].splitlines()[-1])["words"]) == 30
          and overlap[0][2]["w"] == overlap[1][0]["w"] == "xQc")
    foreign = [(words(4, i * 10, "hi"), "hi") for i in range(15)]
    with patch.object(brain, "ask_ai", return_value=(["This", "is", "very", "good."] * 15, "test", 1)) as ask:
        fixed = brain.fix_caption_batches(foreign)
    check("15 translated clips also share one request and retain all slots", ask.call_count == 1
          and len(fixed) == 15 and all(len(group) == 4 and group[0]["w"] == "This" for group in fixed))
    with patch.object(brain, "ask_ai", side_effect=RuntimeError("HTTP 429")) as ask:
        failed = brain.fix_caption_batches(foreign)
    check("a failed batch does not fan out into 15 paid retries", ask.call_count == 1
          and all(isinstance(result, RuntimeError) for result in failed))
    for invalid in ([{"i": True, "w": "bad"}], [{"i": -1, "w": "bad"}], [{"i": 30, "w": "bad"}],
                    [{"i": 1, "w": "ok"}, {"i": 1, "w": "bad"}], [{"i": 1, "w": "हिन्दी"}]):
        with patch.object(brain, "ask_ai", return_value=(invalid, "test", 1)):
            check("invalid sparse edits never reach the renderer", rejected(lambda: brain._fix_caption_batch(groups[:1])))
    with patch.object(brain, "CAPTION_BATCH_WORDS", 40), patch.object(brain, "ask_ai", return_value=([], "test", 1)) as ask:
        long = words(95)
        fixed = brain.fix_caption_batches([(long, "en"), ([], "hi")])
    check("even a single oversized clip is bounded and assembled in order", ask.call_count == 3
          and fixed == [long, []] and all(len(json.loads(call.args[0].splitlines()[-1])["words"]) <= 40 for call in ask.call_args_list))

    with ctx.as_user(user):
        job = jobs.create({"type": "file", "name": "Batch test", "path": str(Path(data) / "source.mp4")}, {}, enqueue=False)
        job.update(lang="en", title="Batch test")
        clips = [{"id": f"clip{i}", "start": i * 10, "end": i * 10 + 6, "status": "queued"} for i in range(15)]
        job["clips"] = clips
        transcript = [w for group, _ in groups for w in group]
        with patch.object(brain, "ask_ai", return_value=([], "test", 1)) as ask:
            jobs._prepare_captions(job, [(clip, transcript) for clip in clips])
            jobs._prepare_captions(job, [(clip, transcript) for clip in clips])
        check("prepared captions are saved and reused before any clip is encoded", ask.call_count == 1
              and all(clip.get("fixed_words") for clip in clips)
              and "fixed_words" in json.loads((Path(job["dir"]) / "job.json").read_text())["clips"][0])
        clips[0].update(start=1, end=5)
        with patch.object(brain, "english_captions", side_effect=AssertionError("cache missed")):
            cached = jobs._caption_words(job, clips[0], transcript, 1)
        check("trimming inside a cached clip needs no AI and keeps offsets", cached and cached[0]["s"] == -1)
        with patch.object(jobs, "_english_part", side_effect=OSError("unavailable")):
            check("optional YouTube English lookup failures keep the source available", jobs._caption_source(
                {"lang": "hi"}, {"start": 0, "end": 10}, foreign[0][0]) == foreign[0])

        def clear():
            with db.engine.begin() as conn:
                conn.execute(db.usage.delete())
            usage.forget_all()

        def seed():
            clear()
            usage.record(100000, 0, "Video: finding clips")

        seed()
        with patch.object(ai, "call", return_value=([], 100, 10)) as call:
            for _ in range(7):
                brain.ask_ai("test", purpose="Caption wording fix")
            check("the eighth Gemini caption attempt is blocked", rejected(lambda: brain.ask_ai("test", purpose="Caption translation"))
                  and call.call_count == 7)
            db.engine.dispose()
            check("the allowance survives database reconnection", rejected(lambda: brain.ask_ai("test", purpose="Caption translation"))
                  and call.call_count == 7)
            brain.ask_ai("test", purpose="Video: finding clips")
            check("caption limits leave clipping available", call.call_count == 8)
        rows = usage.summary({})
        check("usage records actual caption tokens once per request", ["Caption wording fix", 770, 7] in rows["by_purpose"])
        seed()
        with patch.object(ai, "call", side_effect=RuntimeError("HTTP 429")) as call:
            check("failed captions make exactly one primary-model attempt", rejected(lambda: brain.ask_ai("test", purpose="Caption translation"))
                  and call.call_count == 1 and call.call_args.args[0] == "first")
        check("failed attempts persist in the usage ledger", usage.summary({})["calls_total"] == 2)
        clear()
        usage.record(1000, 0, "Video: finding clips")
        usage.record(700, 0, "Caption translation")
        with patch.object(ai, "call", return_value=([], 1, 1)) as call:
            check("caption spending above 40 percent pauses further requests", rejected(lambda: brain.ask_ai("test", purpose="Caption translation")) and not call.called)
            with patch.dict(os.environ, PEAKCUT_CAPTION_DAILY_REQUESTS="0"):
                brain.ask_ai("test", purpose="Caption translation")
            with patch.object(ai, "provider", return_value="ollama"):
                brain.ask_ai("test", purpose="Caption translation")
            check("the allowance is configurable and does not cap other providers", call.call_count == 2)
        seed()
        with db.engine.begin() as conn:
            conn.execute(db.usage.insert().values(user_id=user, t=time.time() - 86401, tokens_in=999999,
                                                tokens_out=0, purpose="Caption translation"))
        with patch.object(ai, "call", return_value=([], 1, 1)) as call:
            brain.ask_ai("test", purpose="Caption translation")
        check("old history does not consume the rolling allowance", call.call_count == 1)
        seed()
        with patch.dict(os.environ, PEAKCUT_CAPTION_DAILY_REQUESTS="2"), patch.object(ai, "call", return_value=([], 1, 1)) as call:
            def attempt():
                return not rejected(lambda: brain.ask_ai("test", purpose="Caption translation"))
            with ThreadPoolExecutor(8) as pool:
                accepted = list(pool.map(ctx.bind(lambda _: attempt()), range(8)))
            check("concurrent jobs cannot overspend the caption request allowance", sum(accepted) == 2 and call.call_count == 2)

        # Exercise the real video pipeline, including preparation, batch caching and parallel rendering.
        clear()
        job = jobs.create({"type": "file", "name": "Pipeline", "path": str(Path(data) / "source.mp4")}, {}, enqueue=False)
        found = [{"start": i * 6, "end": i * 6 + 4, "score": 90, "hook": "Wow", "title": "Clip",
                  "description": "", "hashtags": []} for i in range(15)]
        transcript = words(400)
        with (patch.object(transcribe, "model_ready", return_value=True), patch.object(transcribe, "transcribe", return_value=transcript),
              patch.object(media, "probe", return_value={"duration": 100, "width": 1920, "height": 1080}),
              patch.object(media, "save_thumb", return_value=False), patch.object(media, "extract_audio"),
              patch.object(media, "load_audio", return_value=np.ones(16000 * 100, dtype=np.int16)),
              patch.object(brain, "find_clips", return_value=(found, {"model": "test", "tokens": 1000})),
              patch.object(brain, "ask_ai", return_value=([], "test", 1)) as ask,
              patch.object(jobs, "_facecam", return_value=None), patch.object(jobs.quality, "static_ranges", return_value=[]),
              patch.object(render, "_render_in", return_value={"file": "test.mp4"})):
            jobs._run_job(job)
            check("the complete 15-clip pipeline uses one caption request", ask.call_count == 1
                  and len(job["clips"]) == 15 and all(c["status"] == "done" for c in job["clips"]))
            ask.reset_mock()
            job["resume"] = True
            for clip in job["clips"]:
                clip["status"] = "queued"
            job["clips"][0]["captions"] = False
            jobs._run_job(job)
            check("restart recovery reuses paid captions and honors captions off", not ask.called
                  and all(c["status"] == "done" for c in job["clips"]))
            hung = jobs.create(job["source"], {}, enqueue=False)
            release = threading.Event()
            cached_captions = jobs._cached_captions
            def slow_preparation(video, clip):
                if clip["id"] == "clip01" and threading.current_thread() is not threading.main_thread():
                    release.wait(5)
                return cached_captions(video, clip)
            selected = [{k: c[k] for k in ("start", "end", "score", "hook", "title", "description", "hashtags")}
                        for c in found[:2]]
            try:
                with (patch.object(jobs, "CLIP_MINUTES", .01), patch.object(jobs, "_cached_captions", side_effect=slow_preparation),
                      patch.object(brain, "find_clips", return_value=(selected, {"model": "test", "tokens": 1000}))):
                    started = time.monotonic()
                    jobs._run_job(hung)
                check("a hung preparation cannot hold up another clip or the video", time.monotonic() - started < 4
                      and hung["status"] == "done" and [c["status"] for c in hung["clips"]] == ["error", "done"])
            finally:
                release.set()
    db.engine.dispose()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pc-caption-budget-") as folder:
        main(folder)
