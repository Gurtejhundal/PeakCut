"""Add intro, midpoint and outro narration to a vertical short.

Run from the project folder: python auto_edit.py path/to/clip.mp4
Outputs go into ./Edited_Shorts. Optional Kokoro model files live in the working folder.
"""
import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUTPUT_DIR = Path("Edited_Shorts")

VOICES = {
    "1": ("en-US-AndrewNeural", "+8%", "+0Hz", "am_michael", "Warm & Natural Storyteller (Best All-Round)"),
    "2": ("en-US-BrianNeural", "+10%", "-3Hz", "am_fenrir", "Deep & Punchy Gaming Breakdown (Hard)"),
    "3": ("en-US-ChristopherNeural", "+12%", "+2Hz", "am_puck", "Energetic & Fast Hype Commentary (Gaming)"),
    "4": ("en-GB-RyanNeural", "+8%", "+0Hz", "bm_george", "Crisp British Narrator (High-Contrast)"),
    "5": ("en-US-AvaNeural", "+8%", "+0Hz", "af_bella", "Expressive & Modern Female (Engaging)"),
    "6": ("en-US-GuyNeural", "+10%", "-2Hz", "am_michael", "Classic YouTube Shorts Narrator (Punchy)"),
}

def render_progress(label, bytes_got, elapsed, done=False):
    bar_len = 22
    filled = bar_len if done else int((elapsed * 3) % bar_len) + 1
    bar = "█" * filled + "░" * (bar_len - filled)
    status = "DONE" if done else "FETCHING"
    print(f"\r  [{bar}] {status:<8} | {elapsed:4.1f}s | {bytes_got / 1024:5.1f} KB - {label}   ",
          end="\n" if done else "", flush=True)


async def stream_edge_tts(text, out_file, voice, rate, pitch, label):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    start = time.monotonic()
    bytes_got = 0
    render_progress(label, 0, 0.0)
    with open(out_file, "wb") as output:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                output.write(chunk["data"])
                bytes_got += len(chunk["data"])
            render_progress(label, bytes_got, time.monotonic() - start)
    if not bytes_got:
        raise RuntimeError("Empty audio stream received")
    render_progress(label, bytes_got, time.monotonic() - start, done=True)


def generate_voice_with_progress(text, out_mp3, voice, rate, pitch, fallback_voice, label):
    last_error = None
    for attempt in range(1, 4):
        try:
            asyncio.run(asyncio.wait_for(
                stream_edge_tts(text, out_mp3, voice, rate, pitch, f"{label} (try {attempt}/3)"),
                timeout=30.0,
            ))
            return out_mp3
        except Exception as error:
            last_error = error
            print(f"\n  [!] Attempt {attempt}/3 failed ({type(error).__name__}).")
            Path(out_mp3).unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(0.8)

    if Path("kokoro-v1.0.onnx").is_file() and Path("voices-v1.0.bin").is_file():
        print(f"  [->] Switching to local offline Kokoro ({fallback_voice})...")
        import soundfile as sf
        from kokoro_onnx import Kokoro
        out_wav = str(Path(out_mp3).with_suffix(".wav"))
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        samples, sr = kokoro.create(text, voice=fallback_voice, speed=1.08,
                                    lang="en-gb" if fallback_voice.startswith("b") else "en-us")
        sf.write(out_wav, samples, sr)
        print(f"  [██████████████████████] OFFLINE  | Saved {out_wav}")
        return out_wav
    raise RuntimeError("Online TTS failed after 3 attempts. Offline fallback needs kokoro-onnx, "
                       "kokoro-v1.0.onnx and voices-v1.0.bin in the working folder.") from last_error


def probe_media(file_path):
    return json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(file_path)], timeout=30))


def positive_duration(value):
    try:
        duration = float(value)
    except (TypeError, ValueError):
        raise ValueError("The file has no readable duration.") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The file duration must be finite and greater than zero.")
    return duration


def get_duration(file_path):
    return positive_duration(probe_media(file_path).get("format", {}).get("duration"))


def schedule_voiceovers(clip_dur, tracks):
    """Keep full narration in order; shrink pauses before moving a voice away from its preferred position."""
    clip_dur = positive_duration(clip_dur)
    durations = [positive_duration(duration) for _, _, duration in tracks]
    total = sum(durations)
    if total > clip_dur:
        raise ValueError(f"The narration needs {sum(durations):.2f}s but the video is only {clip_dur:.2f}s. "
                         "Shorten the narration or use a longer clip; no speech has been cut.")
    margin = min(0.25, clip_dur * 0.02, (clip_dur - total) / 2)
    preferred = [{"intro": margin, "mid": clip_dur * 0.47, "outro": clip_dur - margin - duration}[role]
                 for (role, _, _), duration in zip(tracks, durations)]
    gaps = [preferred[i + 1] - preferred[i] - durations[i] for i in range(len(tracks) - 1)]
    spare = max(0.0, clip_dur - 2 * margin - total)
    gap = min(1.0, spare / max(1, len(tracks) - 1), max(0.0, min(gaps, default=1.0)))
    result = []
    earliest = margin
    for index, ((role, audio_file, _), duration) in enumerate(zip(tracks, durations)):
        latest = clip_dur - margin - sum(durations[index:]) - gap * (len(tracks) - index - 1)
        start = min(max(preferred[index], earliest), max(earliest, latest))
        result.append((audio_file, start, duration))
        earliest = start + duration + gap
    return result


def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: python auto_edit.py "path/to/clip.mp4"')
    clip_path = Path(sys.argv[1].strip('"')).expanduser()
    if not clip_path.is_file():
        raise ValueError(f"Video file not found: {clip_path}")
    for program in ("ffmpeg", "ffprobe"):
        if not shutil.which(program):
            raise RuntimeError(f"Install {program} and make sure it is on PATH.")
    metadata = probe_media(clip_path)
    video = next((stream for stream in metadata.get("streams", [])
                  if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError("The input file has no video stream.")
    video_start = float(video.get("start_time") or 0)
    if not math.isfinite(video_start):
        raise ValueError("The video has an invalid start timestamp.")
    duration = video.get("duration")
    if duration in (None, "N/A"):
        duration_tag = video.get("tags", {}).get("DURATION")
        if duration_tag:
            hours, minutes, seconds = map(float, duration_tag.split(":"))
            duration = hours * 3600 + minutes * 60 + seconds - video_start
        else:
            duration = metadata.get("format", {}).get("duration")
            if "matroska" in metadata.get("format", {}).get("format_name", ""):
                duration = float(duration) - video_start
    clip_dur = positive_duration(duration)
    has_audio = any(stream.get("codec_type") == "audio" for stream in metadata["streams"])

    print("\n=== Select AI Narrator Voice ===")
    for key, value in VOICES.items():
        print(f"  {key}. {value[4]} [{value[0]}]")
    choice = input("Choose voice [1-6, default 1]: ").strip() or "1"
    voice, rate, pitch, fallback_voice, desc = VOICES.get(choice, VOICES["1"])
    print(f"Selected: {desc}\n")
    texts = [input("1. Enter INTRO hook text (plays at the start): ").strip(),
             input("2. Enter MID-CLIP commentary (around halfway, or Enter to skip): ").strip(),
             input("3. Enter OUTRO takeaway line (plays near the end): ").strip()]
    if not any(texts):
        raise ValueError("No text entered. Exiting.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUTPUT_DIR) / f"{clip_path.stem}_monetizable.mp4"
    if out_file.resolve() == clip_path.resolve():
        raise ValueError("The output must not replace the input video.")
    # A private work folder prevents collisions and protects a previous export if rendering fails.
    with tempfile.TemporaryDirectory(prefix=".auto-edit-", dir=OUTPUT_DIR) as work:
        print("\nFetching AI Voiceovers:")
        tracks = []
        roles = ("intro", "mid", "outro")
        labels = ("Intro Hook", "Mid Commentary", "Outro Takeaway")
        for role, text, label in zip(roles, texts, labels):
            if text:
                audio_file = generate_voice_with_progress(text, str(Path(work) / f"tts_{role}.mp3"),
                                                          voice, rate, pitch, fallback_voice, label)
                tracks.append((role, audio_file, get_duration(audio_file)))
        tts_tracks = schedule_voiceovers(clip_dur, tracks)
        starts = {role: track[1] for (role, _, _), track in zip(tracks, tts_tracks)}
        mid_start = starts.get("mid", clip_dur * 0.47)
        outro_start = starts.get("outro", max(0.0, clip_dur - 3.5))
        for (role, _, _), (_, start, duration) in zip(tracks, tts_tracks):
            print(f"  {role.title()}: {start:.2f}s - {start + duration:.2f}s")

        inputs = ["-copyts", "-i", str(clip_path)]
        punch_expr = f"between(t,{mid_start},{min(mid_start + 3.0, clip_dur)})+between(t,{outro_start},{clip_dur})"
        filter_parts = [
            f"[0:{video['index']}]setpts=PTS-STARTPTS,"
            "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,split=2[v_norm][v_zoom_in];"
            "[v_zoom_in]scale=1232:2190,crop=1080:1920:(iw-1080)/2:(ih-1920)/2[v_punched];"
            f"[v_norm][v_punched]overlay=0:0:enable='{punch_expr}'[vout]"
        ]
        duck_conditions, voice_labels = [], []
        for index, (audio_file, start, duration) in enumerate(tts_tracks, start=1):
            inputs.extend(["-i", str(audio_file)])
            delay_samples = round(start * 48000)
            filter_parts.append(f"[{index}:a:0]aresample=48000,asetpts=PTS-STARTPTS,"
                                f"adelay={delay_samples}S:all=1,volume=1.35[v{index}]")
            voice_labels.append(f"[v{index}]")
            duck_conditions.append(f"between(t,{start},{start + duration})")
        # Keep source audio aligned with the video, including a late start or a short audio track.
        if has_audio:
            background = f"[0:a:0]asetpts=PTS-({video_start})/TB,aresample=48000:async=1:first_pts=0,apad"
        else:
            background = "anullsrc=r=48000:cl=stereo"
        duck_expr = "+".join(duck_conditions)
        filter_parts.append(f"{background},atrim=duration={clip_dur},"
                            f"volume='if({duck_expr},0.38,1.08)':eval=frame[bg]")
        filter_parts.append("[bg]" + "".join(voice_labels)
                            + f"amix=inputs={len(tts_tracks) + 1}:duration=first:dropout_transition=0:normalize=0,"
                            "loudnorm=I=-14:TP=-1.0:LRA=11[aout]")
        rendered = Path(work) / "rendered.mp4"
        cmd = ["ffmpeg", "-y", "-nostdin", *inputs, "-filter_complex", ";".join(filter_parts),
               "-map", "[vout]", "-map", "[aout]", "-t", str(clip_dur),
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(rendered)]
        print("\nRendering transformed 1080p Short...")
        subprocess.run(cmd, check=True)
        os.replace(rendered, out_file)
    print(f"\nDone! Saved ready-to-upload file to:\n{out_file}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, EOFError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)
