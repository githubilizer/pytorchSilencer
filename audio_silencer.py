#!/usr/bin/env python3
"""Utility to trim video/audio using processed transcript with silence markers.

Cuts out portions of the video that are marked with ``[SILENCE-CUT]`` in the
transcript. The number after ``[SILENCE-CUT]`` now represents the amount of
silence to remove. Any remaining portion of the gap is used as the crossfade
length when splicing the video.

This version relies on ``ffmpeg`` via ``subprocess`` instead of ``moviepy``.
"""
import argparse
import re
import subprocess
from typing import List, Tuple, Optional

def parse_cut_segments(path: str) -> List[Tuple[float, float, Optional[float]]]:
    """Return list of (start, end, cut) tuples for ``[SILENCE-CUT]`` markers.

    ``cut`` is the amount of silence to remove. If not specified in the
    transcript, ``cut`` will be ``None``.
    """
    segments: List[Tuple[float, float, Optional[float]]] = []
    timestamp = re.compile(r"\[(\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)\]")
    cut_re = re.compile(r"\[SILENCE-CUT(?: ([\d\.]+)s)?\]")
    with open(path, "r") as f:
        for line in f:
            if "[SILENCE-CUT" not in line:
                continue
            m = timestamp.search(line)
            if not m:
                continue
            cut_m = cut_re.search(line)
            remain = float(cut_m.group(1)) if cut_m and cut_m.group(1) else None
            start = float(m.group(1))
            end = float(m.group(2))
            segments.append((start, end, remain))
    return segments

def get_video_duration(path: str) -> float:
    """Return duration of the video in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def get_video_fps(path: str) -> float:
    """Return average frames per second for the video using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    fps_str = result.stdout.strip()
    if "/" in fps_str:
        num, denom = fps_str.split("/")
        return float(num) / float(denom)
    return float(fps_str)


def cut_video(video_path: str, transcript_path: str, output_path: str) -> None:
    segments = parse_cut_segments(transcript_path)
    video_duration = get_video_duration(video_path)
    fps = int(round(get_video_fps(video_path)))

    if not segments:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                output_path,
            ],
            check=True,
        )
        return

    keep_segments: List[Tuple[float, float]] = []
    crossfades: List[float] = []
    current = 0.0
    for start, end, cut in segments:
        seg_dur = end - start
        if cut is not None and cut < seg_dur:
            keep_start = start + cut / 2
            keep_end = end - cut / 2
            remain = seg_dur - cut
            # Use the remaining duration as the crossfade length.
            cf = max(remain, 0.0)
        else:
            keep_start = start + 0.1 * seg_dur
            keep_end = end - 0.1 * seg_dur
            cf = min(seg_dur * 0.1, 1.0)

        if keep_start > current:
            keep_segments.append((current, keep_start))
            crossfades.append(cf)
        current = keep_end
    if current < video_duration:
        keep_segments.append((current, video_duration))

    filter_parts = []
    for idx, (s, e) in enumerate(keep_segments):
        filter_parts.append(
            f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS,fps={fps}[v{idx}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{idx}]"
        )

    vf = "v0"
    af = "a0"
    out_dur = keep_segments[0][1] - keep_segments[0][0]
    for idx in range(1, len(keep_segments)):
        cf = crossfades[idx - 1] if idx - 1 < len(crossfades) else 0
        offset = max(out_dur - cf, 0)
        filter_parts.append(
            f"[{vf}][v{idx}]xfade=transition=fade:duration={cf}:offset={offset}[vx{idx}]"
        )
        filter_parts.append(f"[{af}][a{idx}]acrossfade=d={cf}[ax{idx}]")
        vf = f"vx{idx}"
        af = f"ax{idx}"
        out_dur = out_dur - cf + (keep_segments[idx][1] - keep_segments[idx][0])

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{vf}]",
        "-map",
        f"[{af}]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Trim video using processed transcript (uses ffmpeg)"
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--transcript", required=True, help="Processed transcript with SILENCE-CUT markers")
    parser.add_argument("--output", required=True, help="Output video file")
    args = parser.parse_args()

    cut_video(args.video, args.transcript, args.output)


if __name__ == "__main__":
    main()
