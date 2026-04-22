import os
import re

import pandas as pd


# these are the film files i actually used in the main srt pass.
# labels are the period split i used later in the classifier.

SRT_FILES = {
    "novel_old": ("(insert srt file path here)", 0),
    "novel_new": ("(insert srt file path here)", 1),
}


# srt timestamp string to float (seconds)
def srt_time_to_seconds(time_text):
    clean = time_text.strip().replace(",", ".")
    hours, minutes, seconds = clean.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# strips html tags 
def clean_subtitle_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    return text.strip()


# parses and returns a list w start, end, and text.
def parse_srt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    subtitles = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        try:
            start_text, end_text = lines[1].split(" --> ")
        except ValueError:
            continue

        text = clean_subtitle_text(" ".join(lines[2:]))
        subtitles.append(
            {
                "start": srt_time_to_seconds(start_text),
                "end": srt_time_to_seconds(end_text),
                "text": text,
            }
        )

    return subtitles


# takes subtitle dicts and gives back the temporal features
def extract_temporal_features(subtitles):
    if len(subtitles) < 2:
        return {}

    pauses = []
    speech_rates = []
    turn_lengths = []
    overlaps = 0

    for index, subtitle in enumerate(subtitles):
        words = subtitle["text"].split()
        word_count = len(words)
        duration = subtitle["end"] - subtitle["start"]

        turn_lengths.append(word_count)
        if duration > 0:
            speech_rates.append(word_count / duration)

        if index == 0:
            continue

        gap = subtitle["start"] - subtitles[index - 1]["end"]
        if gap < 0:
            overlaps += 1
        else:
            pauses.append(gap)

    pause_series = pd.Series(pauses, dtype="float64")
    speech_series = pd.Series(speech_rates, dtype="float64")
    turn_series = pd.Series(turn_lengths, dtype="float64")

    return {
        "pause_mean_sec": pause_series.mean(),
        "pause_median_sec": pause_series.median(),
        "pause_std_sec": pause_series.std(),
        "pause_count": len(pauses),
        "overlap_count": overlaps,
        "overlap_rate": overlaps / max(1, len(subtitles) - 1),
        "speech_rate_mean_wps": speech_series.mean(),
        "speech_rate_median_wps": speech_series.median(),
        "turn_len_mean_words": turn_series.mean(),
        "turn_len_median_words": turn_series.median(),
        "n_turns": len(subtitles),
        "total_words": int(turn_series.sum()),
    }


# loops and parses srt
def main():
    rows = []

    for film_name, (relative_path, label) in SRT_FILES.items():
        if not os.path.exists(relative_path):
            print(f"missing srt, skipping: {relative_path}")
            continue

        print(f"parsing {film_name}...")
        subtitles = parse_srt(relative_path)
        features = extract_temporal_features(subtitles)
        if not features:
            print(f"not enough subtitle blocks for {film_name}")
            continue

        features["film"] = film_name
        features["label"] = label
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    ordered_cols = ["film", "label"] + [col for col in feature_df.columns if col not in {"film", "label"}]
    feature_df = feature_df[ordered_cols]
    feature_df.to_csv("srt_temporal_features.csv", index=False)
    print("\nsaved srt_temporal_features.csv")


if __name__ == "__main__":
    main()