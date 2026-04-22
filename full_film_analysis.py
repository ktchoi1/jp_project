import json
import os
 
import numpy as np
import pandas as pd
 
 
FILM_FILES = {
    "full_movie_old": "(insert whisper output json path here)",
    "full_movie_new": "(insert whisper output json path here)",
}
 
 
# opens and parses a json file, returns contents as a dict
def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
 
 
# takes film name and whisperx json and computes temporal features
def analyze_film(film_name, path):
    data = load_json(path)
    segments = data.get("segments", [])
    if not segments:
        return None
 
    for segment in segments:
        words = segment.get("words", [])
        segment["n_words"] = len(segment["text"].split())
        segment["duration"] = max(segment["end"] - segment["start"], 1e-9)
        segment["wps"] = segment["n_words"] / segment["duration"]
 
    # segments must have >2 words
    valid_segments = [segment for segment in segments if segment["n_words"] >= 2]
 
    gaps = []
    switch_gaps = []
    overlaps = 0
    within_pauses = []
 
    for index in range(1, len(segments)):
        gap = segments[index]["start"] - segments[index - 1]["end"]
        if gap < 0:
            overlaps += 1
        elif gap < 10:
            gaps.append(gap)
 
        prev_speaker = segments[index - 1].get("speaker", "")
        next_speaker = segments[index].get("speaker", "")
        if prev_speaker != next_speaker and 0 <= gap < 10:
            switch_gaps.append(gap)
 
    # pauses between words during a turn 
    for segment in segments:
        words = segment.get("words", [])
        for index in range(1, len(words)):
            if "start" not in words[index] or "end" not in words[index - 1]:
                continue
            gap = words[index]["start"] - words[index - 1]["end"]
            if 0 < gap < 2:
                within_pauses.append(gap)
 
    film_duration = segments[-1]["end"] - segments[0]["start"]
    total_words = sum(segment["n_words"] for segment in segments)
 
    return {
        "film": film_name,
        "film_duration_min": round(film_duration / 60, 1),
        "n_segments": len(segments),
        "seg_rate_mean_wps": round(np.mean([segment["wps"] for segment in valid_segments]), 3),
        "seg_rate_median_wps": round(np.median([segment["wps"] for segment in valid_segments]), 3),
        "seg_rate_std_wps": round(np.std([segment["wps"] for segment in valid_segments]), 3),
        "turn_len_mean_words": round(np.mean([segment["n_words"] for segment in valid_segments]), 3),
        "turn_len_median_words": round(np.median([segment["n_words"] for segment in valid_segments]), 3),
        "pause_mean_sec": round(np.mean(gaps), 3) if gaps else 0,
        "pause_median_sec": round(np.median(gaps), 3) if gaps else 0,
        "pause_std_sec": round(np.std(gaps), 3) if gaps else 0,
        "pause_count": len(gaps),
        "response_latency_mean_sec": round(np.mean(switch_gaps), 3) if switch_gaps else 0,
        "response_latency_median_sec": round(np.median(switch_gaps), 3) if switch_gaps else 0,
        "within_pause_mean_sec": round(np.mean(within_pauses), 3) if within_pauses else 0,
        "within_pause_median_sec": round(np.median(within_pauses), 3) if within_pauses else 0,
        "overlap_count": overlaps,
        "overlap_rate": round(overlaps / max(1, len(segments) - 1), 3),
    }
 
 
 
def main():
    rows = []
    for film_name, path in FILM_FILES.items():
        if not os.path.exists(path):
            print(f"missing film file, skipping: {path}")
            continue
        row = analyze_film(film_name, path)
        if row is not None:
            rows.append(row)
 
    film_df = pd.DataFrame(rows)
    film_df.to_csv("full_film_comparison.csv", index=False)
    print(film_df.to_string(index=False))
    print("\nsaved full_film_comparison.csv")
 
 
if __name__ == "__main__":
    main()
