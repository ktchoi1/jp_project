import json
import os

import numpy as np
import pandas as pd


# this script is for the whisperx scene clips, not the whole-film srt pass.

SCENE_FILES = {
    "movie_one__old": "(insert scene file path here)",
    "movie_one_new": "(insert scene file path here)",
    "movie_two_old": "(insert scene file path here)",
    "movie_two_new": "(insert scene file path here)",
    "movie_three_old": "(insert scene file path here)",
    "movie_three_new": "(insert scene file path here)",
}

# opens and parses a json file
# returns as dict
def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

# pulls the speaker label out of a segment
# then returns "unknown" if cant find
def pick_speaker(segment):
    if segment.get("speaker"):
        return segment["speaker"]

    word_speakers = [word.get("speaker") for word in segment.get("words", []) if word.get("speaker")]
    if word_speakers:
        return word_speakers[0]

    return "unknown"


# takes raw whisperx output and returns a clean list of segments 
# skips anything missing a start or end time
def normalize_segments(data):
    raw_segments = data.get("segments", [])
    cleaned = []
    for segment in raw_segments:
        if "start" not in segment or "end" not in segment:
            continue
        cleaned.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment.get("text", "").strip(),
                "speaker": pick_speaker(segment),
            }
        )
    return cleaned

# computes all the temporal features for a single scene
def analyze_scene(scene_name, segments):
    # returns None if there are fewer than 2 segments
    if len(segments) < 2:
        return None

    for segment in segments:
        segment["n_words"] = len(segment["text"].split())
        segment["duration"] = max(segment["end"] - segment["start"], 1e-9)
        segment["wps"] = segment["n_words"] / segment["duration"]

    gaps = []
    pauses = []
    switch_gaps = []
    overlap_count = 0
    for index in range(1, len(segments)):
        gap = segments[index]["start"] - segments[index - 1]["end"]
        gaps.append(gap)
        if gap < 0:
            overlap_count += 1
        else:
            pauses.append(gap)

        # switch_gaps are only gaps where the speaker actually changed

        if segments[index]["speaker"] != segments[index - 1]["speaker"]:
            switch_gaps.append(gap)

    duration = segments[-1]["end"] - segments[0]["start"]
    turns_per_minute = len(segments) / max(duration / 60, 1e-9)

    return {
        "scene": scene_name,
        "n_turns": len(segments),
        "scene_duration_sec": round(duration, 3),
        "turns_per_minute": round(turns_per_minute, 3),
        "total_words": sum(segment["n_words"] for segment in segments),
        "speech_rate_mean_wps": round(np.mean([segment["wps"] for segment in segments]), 3),
        "speech_rate_std_wps": round(np.std([segment["wps"] for segment in segments]), 3),
        "turn_len_mean_words": round(np.mean([segment["n_words"] for segment in segments]), 3),
        "turn_len_median_words": round(np.median([segment["n_words"] for segment in segments]), 3),
        "pause_mean_sec": round(np.mean(pauses), 3) if pauses else 0,
        "pause_median_sec": round(np.median(pauses), 3) if pauses else 0,
        "response_latency_mean_sec": round(np.mean(switch_gaps), 3) if switch_gaps else 0,
        "response_latency_median_sec": round(np.median(switch_gaps), 3) if switch_gaps else 0,
        "overlap_count": overlap_count,
        "overlap_rate": round(overlap_count / max(1, len(segments) - 1), 3),
        "speech_rate_median_wps": round(np.median([s["wps"] for s in segments]), 3),
        "pause_std_sec": round(np.std(pauses), 3) if pauses else 0,
        "pause_count": len(pauses),
    }

# loops through all the scene files, skips any that are missing
# runs analysis on each
def main():
    rows = []

    for scene_name, path in SCENE_FILES.items():
        if not os.path.exists(path):
            print(f"missing scene file, skipping: {path}")
            continue

        scene_data = load_json(path)
        scene_segments = normalize_segments(scene_data)
        scene_row = analyze_scene(scene_name, scene_segments)
        if scene_row is not None:
            rows.append(scene_row)

    scene_df = pd.DataFrame(rows)
    scene_df.to_csv("scene_comparison.csv", index=False)
    print(scene_df.to_string(index=False))
    print("\nsaved scene_comparison.csv")


if __name__ == "__main__":
    main()
