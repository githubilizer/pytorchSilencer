#!/usr/bin/env python3
"""Utility to trim video/audio using processed transcript with silence markers.

Cuts out portions of the video that are marked with [SILENCE-CUT] in the
transcript. Each cut removes 80% of the silence from the middle of the
segment and crossfades the remaining 10%% at each edge.
"""
import argparse
import re
from typing import List, Tuple

from moviepy.editor import VideoFileClip, CompositeVideoClip

def parse_cut_segments(path: str) -> List[Tuple[float, float]]:
    """Return list of (start, end) tuples for segments marked as SILENCE-CUT."""
    segments = []
    timestamp = re.compile(r"\[(\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)\]")
    with open(path, "r") as f:
        for line in f:
            if "[SILENCE-CUT]" not in line:
                continue
            m = timestamp.search(line)
            if m:
                start = float(m.group(1))
                end = float(m.group(2))
                segments.append((start, end))
    return segments

def cut_video(video_path: str, transcript_path: str, output_path: str) -> None:
    video = VideoFileClip(video_path)
    segments = parse_cut_segments(transcript_path)

    if not segments:
        video.write_videofile(output_path, codec="libx264")
        return

    clips = []
    crossfades = []
    current = 0.0
    for start, end in segments:
        duration = end - start
        keep_start = start + 0.1 * duration
        keep_end = end - 0.1 * duration
        if keep_start > current:
            clips.append(video.subclip(current, keep_start))
            crossfades.append(min(duration * 0.1, 1.0))
        current = keep_end
    if current < video.duration:
        clips.append(video.subclip(current, video.duration))

    final = clips[0]
    for idx, clip in enumerate(clips[1:]):
        cf = crossfades[idx] if idx < len(crossfades) else 0
        clip = clip.set_start(final.duration - cf).crossfadein(cf)
        final = CompositeVideoClip([final, clip])
    final.write_videofile(output_path, codec="libx264")


def main():
    parser = argparse.ArgumentParser(description="Trim video using processed transcript")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--transcript", required=True, help="Processed transcript with SILENCE-CUT markers")
    parser.add_argument("--output", required=True, help="Output video file")
    args = parser.parse_args()

    cut_video(args.video, args.transcript, args.output)


if __name__ == "__main__":
    main()
