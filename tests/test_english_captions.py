"""English output, contextual corrections and unchanged timing slots. No service calls or model downloads."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, ".")


def check(name, condition):
    assert condition, name
    print(f"PASS {name}")


def ws(text):
    return [{"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4, "confidence": 0.8}
            for i, w in enumerate(text.split())]


def text(words):
    return " ".join(w["w"] for w in words if w["w"])


def rejected(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return True
    return False


def main(data):
    os.environ.update(PEAKCUT_DATA=data, DATABASE_URL="", PEAKCUT_ACCOUNTS="0", PEAKCUT_ASR_ENGINE="whisper")
    from studio import brain, config, fetch_model, jobs, render, transcribe
    import numpy as np

    check("only known English skips a translation lookup",
          [brain.needs_translation(x) for x in ("en", "en-US", "en_GB", "hi", "ur-PK", "es", "", "hinglish")]
          == [False, False, False, True, True, True, True, True])
    for source, lang, answer in [
        ("यह बहुत अच्छा है", "hi", ["This", "is", "very", "good."]),
        ("یہ بہت اچھا ہے", "ur", ["This", "is", "very", "good."]),
        ("yaar ye kya hai", "", ["What", "is", "this,", "dude?"]),
        ("yaar ye kya hai", "en", ["What", "is", "this,", "dude?"]),
        ("hola amigo", "es", ["Hello,", "friend!"]),
        ("जो उसको", "hi", ["That", "person"]),
        ("हाँ", "hi", ["Yes."]),
    ]:
        words = ws(source)
        original = copy.deepcopy(words)
        with patch.object(brain, "ask_ai", return_value=(answer, "test", 1)) as ask:
            out = brain.english_captions(words, lang, "Ninja's gaming stream")
        check(f"{lang or 'unknown'} speech becomes English: {source}", text(out) == " ".join(answer))
        check("all timing slots, metadata and the input survive",
              len(out) == len(words) and words == original
              and [dict(w, w="") for w in out] == [dict(w, w="") for w in words])
        prompt = ask.call_args.args[0]
        check("translation includes title, surrounding sentences and untrusted-data rules",
              all(x in prompt for x in ("Ninja", "surrounding sentences", "untrusted data", "Romanised Hinglish"))
              and json.loads(prompt.splitlines()[-1])["words"] == source.split())

    words = ws("they are bitching nicks again")
    answer = ["They", "are", "benching", "xQc", "again."]
    with patch.object(brain, "ask_ai", return_value=(answer, "test", 1)) as ask:
        out = brain.fix_caption_words(words, "xQc gets benched during the tournament", "en")
    check("contextual fixes may differ phonetically without changing timing", text(out) == " ".join(answer)
          and [(w["s"], w["e"]) for w in out] == [(w["s"], w["e"]) for w in words])
    check("false profanity is distinguished from real profanity in context",
          all(x in ask.call_args.args[0] for x in ("benching", "surrounding sentences", "genuine profanity", "Do not invent")))

    words = ws("twenty five kills")
    with patch.object(brain, "ask_ai", return_value=(["25", "", "kills!"], "test", 1)):
        out = brain.fix_caption_words(words, lang="en")
    check("merged words retain their empty timing slot", len(out) == 3 and out[1] == dict(words[1], w=""))
    ass = render.build_ass(out, 0, 3, "", "bounce", [], 300, 1380)
    check("the subtitle renderer handles empty slots", "25" in ass and "KILLS!" in ass)

    for bad in (["short"], ["", "", ""], [None, "is", "good"], ["यह", "is", "good"],
                ["こんにちは", "is", "good"], ["x" * 500, "is", "good"], {"text": "English"}):
        with patch.object(brain, "ask_ai", return_value=(bad, "test", 1)):
            check("invalid translations cannot reach subtitles", rejected(lambda: brain.english_captions(ws("यह अच्छा है"), "hi")))
    with patch.object(brain, "ask_ai", side_effect=RuntimeError("offline")):
        for lang in ("hi", "ur", "es", ""):
            check("failed translations never fall back to original speech",
                  rejected(lambda: brain.english_captions(ws("yaar ye kya hai"), lang)))
        english = ws("hello there friend")
        check("English still works without a chat service", brain.fix_caption_words(english, lang="en") == english)
        yt = ws("hello my friend")
        check("YouTube English works without a chat service",
              brain.english_captions(ws("hola mi amigo"), "es", youtube_english=lambda: yt) == yt)
    with patch.object(brain, "ask_ai", return_value=(["Hello", "my", "friend"], "test", 1)):
        def unavailable():
            raise RuntimeError("YouTube refused")
        check("YouTube failures still use the transcript", text(brain.english_captions(ws("hola mi amigo"), "es", youtube_english=unavailable)) == "Hello my friend")
    check("silence needs no caption service", brain.english_captions([], "hi") == [])

    clip = {"start": 10.0, "end": 12.0, "fixed_for": [10.0, 12.0, "en2"], "fixed_words": ws("yaar ye kya hai")}
    job = {"lang": "hi", "title": "Tournament"}
    timed = [{"w": "यह", "s": 10.0, "e": 10.4}, {"w": "अच्छा", "s": 10.5, "e": 10.9}]
    fixed = [dict(w, w=t) for w, t in zip(timed, ("Very", "good!"))]
    with patch.object(brain, "english_captions", return_value=fixed) as prepare:
        out = jobs._caption_words(job, clip, timed, 10.0)
        jobs._caption_words(job, clip, timed, 10.0)
        check("old Hinglish caches are replaced and new captions are reused", prepare.call_count == 1 and out[0]["s"] == 0.0 and abs(out[-1]["e"] - 0.9) < 1e-9)
        job["title"] = "New tournament"
        jobs._caption_words(job, clip, timed, 10.0)
        job["lang"] = "ur"
        jobs._caption_words(job, clip, timed, 10.0)
        check("title and language changes invalidate caption cleanup", prepare.call_count == 3)
    clip.pop("fixed_for")
    with patch.object(brain, "english_captions", side_effect=RuntimeError("translation failed")):
        check("the pipeline does not swallow translation failures", rejected(lambda: jobs._caption_words(job, clip, timed, 0)))

    # A stand-in checks the audio contract and the exact word times, not recognition quality.
    heard = [SimpleNamespace(word=" Hello", start=0.13, end=0.67), SimpleNamespace(word=" friend.", start=0.82, end=1.23)]
    segment = SimpleNamespace(text="Hello friend.", words=heard, end=1.23)
    progress = []
    with patch.object(transcribe, "_whisper_model") as model:
        model.transcribe.return_value = (iter([segment]), SimpleNamespace(language="en"))
        out = transcribe.transcribe(np.ones(32000, dtype=np.int16) * 16384, progress.append)
        args, kwargs = model.transcribe.call_args
        check("multilingual speech keeps the original words and timestamps",
              kwargs["task"] == "transcribe" and kwargs["word_timestamps"] and kwargs["vad_filter"]
              and float(args[0][0]) == 0.5)
    check("speech word times are not redistributed", out == [{"w": "Hello", "s": 0.13, "e": 0.67, "lang": "en"}, {"w": "friend.", "s": 0.82, "e": 1.23, "lang": "en"}] and progress[-1] == 1.0)
    check("empty audio does not load a model", transcribe.transcribe(np.array([], dtype=np.int16)) == [])

    hindi = [dict(w, lang="hi") for w in ws("यह अच्छा है")]
    with patch.object(brain, "ask_ai", side_effect=RuntimeError("offline")):
        check("detected speech language overrides a stale English job label",
              rejected(lambda: brain.english_captions(hindi, "en")))
    with patch.object(brain, "ask_ai", return_value=(["This", "is", "good!"], "test", 1)):
        check("detected Hindi is translated with its original timing slots",
              text(brain.english_captions(hindi, "en")) == "This is good!")

    def download(_name, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"):
            (path / name).write_bytes(b"model")
    with patch("faster_whisper.utils.download_model", side_effect=download):
        fetch_model.ensure()
    check("the multilingual model installs wholly inside the data folder", fetch_model.ready() and config.WHISPER_MODEL_DIR.parent == config.MODELS_DIR)
    with patch.object(config, "WHISPER_MODEL_DIR", Path(data) / "missing"), patch("faster_whisper.utils.download_model", side_effect=OSError("offline")):
        check("a failed model download remains retryable", rejected(fetch_model.ensure) and not fetch_model.ready() and not (config.MODELS_DIR / "speech-model.unpacking").exists())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="pc-english-") as data:
        main(data)
