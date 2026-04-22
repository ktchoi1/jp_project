import json
import os
import re

import pandas as pd


SCENE_FILES = {
    "(insert scene name here)": "(insert whisper output json path here)",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ourselves"}


# loads whisperx json 
def load_scene_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return " ".join(segment.get("text", "").strip() for segment in data.get("segments", []))

# splits the string 
def split_sentences(text):
    return [piece.strip() for piece in re.split(r"[.!?]+", text) if piece.strip()]

# syllable counting it isnt that good tbh
def count_syllables(word):
    word = word.lower()
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in "aeiouy"
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

# computes flesch reading ease score 
def flesch_reading_ease(text):
    words = re.findall(r"[a-z]+", text.lower())
    sentences = split_sentences(text)
    if not words or not sentences:
        return 0.0
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables = sum(count_syllables(word) for word in words) / len(words)
    return round(206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables, 1)


# producing the dict of lexical features from scene name
#  word length, sentence length, type token ratio, flesch score,
# contraction rate, first person rate, and sentence length variance
def lexical_features(scene_name, text):
    alpha_words = re.findall(r"[a-z]+", text.lower())
    raw_tokens = re.findall(r"[a-z']+", text.lower())
    sentences = split_sentences(text)
    sentence_lengths = [len(re.findall(r"[a-z]+", sentence.lower())) for sentence in sentences]

    word_count = len(alpha_words)
    sentence_count = max(1, len(sentences))

    return {
        "scene": scene_name,
        "n_words": word_count,
        "n_sentences": sentence_count,
        "avg_word_length": round(sum(len(word) for word in alpha_words) / max(1, word_count), 2),
        "avg_sent_length": round(word_count / sentence_count, 2),
        "type_token_ratio": round(len(set(alpha_words)) / max(1, word_count), 3),
        "flesch_reading_ease": flesch_reading_ease(text),
        "contraction_rate": round(sum(1 for token in raw_tokens if "'" in token) / max(1, word_count), 3),
        "first_person_rate": round(sum(1 for word in alpha_words if word in FIRST_PERSON) / max(1, word_count), 3),
        "sent_len_variance": round(pd.Series(sentence_lengths, dtype="float64").var(), 3),
    }


# computes lexical features for scene files
def main():
    rows = []
    for scene_name, path in SCENE_FILES.items():
        if not os.path.exists(path):
            print(f"missing scene file, skipping: {path}")
            continue
        text = load_scene_text(path)
        rows.append(lexical_features(scene_name, text))

    lexical_df = pd.DataFrame(rows)
    lexical_df.to_csv("scene_lexical_comparison.csv", index=False)
    print(lexical_df.to_string(index=False))
    print("\nsaved scene_lexical_comparison.csv")


if __name__ == "__main__":
    main()