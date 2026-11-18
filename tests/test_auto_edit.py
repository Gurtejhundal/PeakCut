"""Timing, failure cleanup and real FFmpeg checks; no cloud TTS or model files required."""
import asyncio
import contextlib
import io
import itertools
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import auto_edit as edit


@contextlib.contextmanager
def working_folder(folder):
    before = Path.cwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


class TimingTests(unittest.TestCase):
    def test_all_durations_and_optional_tracks(self):
        rng = random.Random(31)
        for count in (1, 2, 3):
            for roles in itertools.combinations(("intro", "mid", "outro"), count):
                for duration in (0.04, 0.1, 0.9, 2.5, 5.0, 30, 10000):
                    for _ in range(20):
                        lengths = [rng.uniform(0.01, 1) for _ in roles]
                        scale = duration * rng.uniform(0.01, 0.999) / sum(lengths)
                        tracks = [(role, f"{role}.mp3", length * scale) for role, length in zip(roles, lengths)]
                        scheduled = edit.schedule_voiceovers(duration, tracks)
                        previous = 0
                        for (role, filename, length), (got_file, start, got_length) in zip(tracks, scheduled):
                            self.assertEqual((filename, length), (got_file, got_length))
                            self.assertGreaterEqual(start, previous - 1e-8)
                            self.assertLessEqual(start + length, duration + 1e-8)
                            previous = start + length

    def test_midpoint_is_preserved_when_it_fits(self):
        for duration in (1, 5, 30, 100):
            tracks = [(role, role, duration * 0.1) for role in ("intro", "mid", "outro")]
            scheduled = edit.schedule_voiceovers(duration, tracks)
            self.assertAlmostEqual(scheduled[1][1], duration * 0.47)
            self.assertAlmostEqual(scheduled[-1][1] + scheduled[-1][2], duration - min(0.25, duration * 0.02))

    def test_outro_without_middle_has_no_four_second_minimum(self):
        scheduled = edit.schedule_voiceovers(1.0, [("outro", "outro.wav", 0.8)])
        self.assertGreaterEqual(scheduled[0][1], 0)
        self.assertLessEqual(scheduled[0][1] + scheduled[0][2], 1)

    def test_exact_fit_drops_margins(self):
        scheduled = edit.schedule_voiceovers(3, [(role, role, 1) for role in ("intro", "mid", "outro")])
        self.assertEqual([start for _, start, _ in scheduled], [0, 1, 2])

    def test_impossible_narration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Shorten the narration"):
            edit.schedule_voiceovers(1, [("intro", "a", 0.8), ("outro", "b", 0.8)])

    def test_invalid_durations(self):
        for value in (None, "N/A", "nan", "inf", 0, -1):
            with self.assertRaises(ValueError):
                edit.positive_duration(value)

    def test_retry_failure_cleans_partial_audio(self):
        async def failure(*args):
            Path(args[1]).write_bytes(b"partial")
            raise RuntimeError("service unavailable")
        with tempfile.TemporaryDirectory() as folder, working_folder(folder), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(edit, "stream_edge_tts", side_effect=failure) as generate, patch.object(edit.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "Offline fallback"):
                    edit.generate_voice_with_progress("text", "tts.mp3", "voice", "+0%", "+0Hz", "am_michael", "Intro")
                self.assertEqual(generate.call_count, 3)
                self.assertFalse(Path("tts.mp3").exists())

    def test_offline_fallback_keeps_directory_and_british_language(self):
        from types import SimpleNamespace
        kokoro = unittest.mock.Mock()
        kokoro.create.return_value = ([0.1], 24000)
        def write(filename, *_):
            Path(filename).write_bytes(b"audio")
        with tempfile.TemporaryDirectory() as folder, working_folder(folder), contextlib.redirect_stdout(io.StringIO()):
            for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
                Path(name).touch()
            Path("folder.mp3-name").mkdir()
            with (patch.object(edit, "stream_edge_tts", new=AsyncMock(side_effect=OSError("offline"))),
                  patch.object(edit.time, "sleep"),
                  patch.dict(sys.modules, {"kokoro_onnx": SimpleNamespace(Kokoro=lambda *a: kokoro),
                                           "soundfile": SimpleNamespace(write=write)})):
                output = edit.generate_voice_with_progress("text", "folder.mp3-name/tts.mp3", "voice", "+0%", "+0Hz", "bm_george", "Intro")
            self.assertEqual(output, "folder.mp3-name/tts.wav")
            self.assertTrue(Path(output).is_file())
            self.assertEqual(kokoro.create.call_args.kwargs["lang"], "en-gb")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class RenderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="auto-edit-test-")
        self.cwd = working_folder(self.folder.name)
        self.cwd.__enter__()
        self.real_run = subprocess.run
        self.commands = []
        self.sound = Path("voice.wav")
        self.run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=1600:duration=0.12", "-ar", "48000", str(self.sound)])

    def tearDown(self):
        self.cwd.__exit__(None, None, None)
        self.folder.cleanup()

    def run_ffmpeg(self, arguments):
        return self.real_run(["ffmpeg", "-y", "-v", "error", *arguments], check=True, capture_output=True)

    def generate(self, text, filename, *_):
        result = str(Path(filename).with_suffix(".wav"))
        shutil.copyfile(self.sound, result)
        return result

    def encode(self, command, **kwargs):
        if command[0] != "ffmpeg":
            return self.real_run(command, **kwargs)
        self.commands.append(command)
        if getattr(self, "fail_render", False):
            raise subprocess.CalledProcessError(1, "ffmpeg")
        return self.real_run(command, **kwargs, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def edit(self, source, texts):
        with (patch.object(sys, "argv", ["auto_edit.py", str(source)]),
              patch("builtins.input", side_effect=["", *texts]),
              patch.object(edit, "generate_voice_with_progress", side_effect=self.generate),
              patch.object(edit.subprocess, "run", side_effect=self.encode),
              contextlib.redirect_stdout(io.StringIO())):
            edit.main()
        return Path("Edited_Shorts") / f"{source.stem}_monetizable.mp4"

    def test_short_silent_landscape_video(self):
        source = Path("silent clip.mp4")
        self.run_ffmpeg(["-f", "lavfi", "-i", "color=c=blue:size=320x180:rate=20:duration=0.8", "-c:v", "libx264", str(source)])
        output = self.edit(source, ["intro", "mid", "outro"])
        info = edit.probe_media(output)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((video["width"], video["height"]), (1080, 1920))
        self.assertAlmostEqual(float(video["duration"]), 0.8, delta=0.06)
        self.assertTrue(any(s["codec_type"] == "audio" for s in info["streams"]))
        graph = self.commands[0][self.commands[0].index("-filter_complex") + 1]
        self.assertIn("force_original_aspect_ratio=increase", graph)
        self.assertIn(":all=1", graph)
        self.assertFalse(list(Path("Edited_Shorts").glob(".auto-edit-*")))

    def test_audio_start_offset_and_short_source_audio(self):
        source = Path("delayed.mp4")
        self.run_ffmpeg(["-copyts", "-itsoffset", "2", "-f", "lavfi", "-i", "color=c=blue:size=160x90:rate=20:duration=2",
                         "-itsoffset", "2.5", "-f", "lavfi", "-i", "sine=frequency=400:duration=1",
                         "-c:v", "libx264", "-fps_mode", "passthrough", "-c:a", "pcm_s16le", "-f", "matroska", str(source)])
        source_info = edit.probe_media(source)
        video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
        audio = next(s for s in source_info["streams"] if s["codec_type"] == "audio")
        self.assertGreater(float(audio["start_time"]) - float(video["start_time"]), 0.4)
        output = self.edit(source, ["", "", "outro"])
        info = edit.probe_media(output)
        self.assertAlmostEqual(float(info["format"]["duration"]), 2, delta=0.08)
        pcm = self.real_run(["ffmpeg", "-v", "error", "-i", str(output), "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"], check=True, capture_output=True).stdout
        import array
        samples = array.array("f")
        samples.frombytes(pcm)
        def rms(start, end):
            block = samples[int(start * 48000):int(end * 48000)]
            return math.sqrt(sum(x*x for x in block) / len(block))
        self.assertLess(rms(0.1, 0.3), 0.001)
        self.assertGreater(rms(0.7, 0.9), 0.005)
        self.assertGreater(rms(1.87, 1.92), 0.005)  # outro survives the early end of the source audio

    def test_failed_render_preserves_previous_export_and_cleans_files(self):
        source = Path("source.mp4")
        self.run_ffmpeg(["-f", "lavfi", "-i", "color=size=160x90:duration=1", "-c:v", "libx264", str(source)])
        output = Path("Edited_Shorts/source_monetizable.mp4")
        output.parent.mkdir()
        output.write_bytes(b"previous export")
        self.fail_render = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.edit(source, ["intro", "", ""])
        self.assertEqual(output.read_bytes(), b"previous export")
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_narration_too_long_cleans_up_before_encoding(self):
        source = Path("tiny.mp4")
        self.run_ffmpeg(["-f", "lavfi", "-i", "color=size=160x90:duration=0.2:rate=20", "-c:v", "libx264", str(source)])
        with self.assertRaisesRegex(ValueError, "Shorten the narration"):
            self.edit(source, ["intro", "mid", "outro"])
        self.assertFalse(self.commands)
        self.assertFalse(list(Path("Edited_Shorts").iterdir()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
