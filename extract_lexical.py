import os
import re

import pandas as pd


NOVEL_FILES = {
    "first_novel": ("(insert dialogue csv path here)", 0, "novel"),
    "second_novel": ("(insert dialogue csv path here)", 0, "novel"),
    "third_novel": ("(insert dialogue csv path here)", 0, "novel"),
    "fourth_novel": ("(insert dialogue csv path here)", 0, "novel"),
}

FILM_FILES = {
    "first_movie_old": ("(insert srt file path here)", 0, "film"),
    "first_movie_new": ("(insert srt file path here)", 1, "film"),
    "second_movie_old": ("(insert srt file path here)", 0, "film"),
    "second_movie_new": ("(insert srt file path here)", 1, "film"),
    "third_movie_old": ("(insert srt file path here)", 0, "film"),
    "third_movie_new": ("(insert srt file path here)", 1, "film"),
    "fourth_movie_old": ("(insert srt file path here)", 0, "film"),
    "fourth_movie_new": ("(insert srt file path here)", 1, "film"),
}


# strips html tags 
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    return text.strip()


# walks up the dependency tree from a token and returns how deep it is ayo
# not used inf inal
def dependency_depth(token):
    depth = 0
    current = token
    while current.head != current:
        current = current.head
        depth += 1
    return depth


# reads an srt file and returns a list of cleaned subtitle strings
def collect_srt_texts(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    texts = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            text = clean_text(" ".join(lines[2:]))
            if text:
                texts.append(text)
    return texts


# loads a novel dialogue csv and returns the dialogue column as a list of strings
def collect_novel_texts(path):
    dialogue_df = pd.read_csv(path)
    for possible_col in ["quote", "text", "dialogue"]:
        if possible_col in dialogue_df.columns:
            return dialogue_df[possible_col].dropna().astype(str).tolist()
    raise ValueError(f"no dialogue text column found in {path}")


# splits a string into sentences
def split_sentences(text):
    return [piece.strip() for piece in re.split(r"[.!?]+", text) if piece.strip()]


# rough syllable count for a single word
# lowkey not perfect but consistent enough for flesch scoring
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
    return 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables


# takes a list of text strings and a spacy nlp object and returns a dict of
# lexical features
def lexical_features_from_texts(texts, nlp):
    full_text = " ".join(texts)
    doc = nlp(full_text[:1_000_000])

    tokens = [token for token in doc if not token.is_space and not token.is_punct]
    words = [token.text.lower() for token in tokens]
    sentences = list(doc.sents)

    word_lengths = [len(word) for word in words]
    sentence_lengths = [
        len([token for token in sentence if not token.is_space and not token.is_punct])
        for sentence in sentences
    ]

    pos_counts = {}
    dep_depths = []
    for token in tokens:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        dep_depths.append(dependency_depth(token))

    # i used ai to help sanity check this part because spacy parses can get messy idk it was breaking
    formality_score = (pos_counts.get("NOUN", 0) + pos_counts.get("ADJ", 0)) / max(1, len(tokens))

    return {
        "avg_word_length": pd.Series(word_lengths, dtype="float64").mean(),
        "avg_sent_length": pd.Series(sentence_lengths, dtype="float64").mean(),
        "type_token_ratio": len(set(words)) / max(1, len(words)),
        "flesch_reading_ease": flesch_reading_ease(full_text[:500_000]),
        "formality_score": formality_score,
        "avg_dep_depth": pd.Series(dep_depths, dtype="float64").mean(),
        "total_tokens": len(tokens),
    }


# loads spacy, then loops through novel and film files in order, computes lexical
# features for each, and saves everything to lexical_features.csv
def main():
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "spacy is not installed in this python environment. install spacy and the en_core_web_sm model first."
        ) from exc

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    rows = []

    for source_name, (path, label, source_type) in NOVEL_FILES.items():
        if not os.path.exists(path):
            print(f"missing file, skipping: {path}")
            continue

        print(f"doing lexical features for {source_name}...")
        texts = collect_novel_texts(path)
        features = lexical_features_from_texts(texts, nlp)
        features["source"] = source_name
        features["type"] = source_type
        features["label"] = label
        rows.append(features)

    for source_name, (path, label, source_type) in FILM_FILES.items():
        if not os.path.exists(path):
            print(f"missing file, skipping: {path}")
            continue

        print(f"doing lexical features for {source_name}...")
        texts = collect_srt_texts(path)
        features = lexical_features_from_texts(texts, nlp)
        features["source"] = source_name
        features["type"] = source_type
        features["label"] = label
        rows.append(features)

    lexical_df = pd.DataFrame(rows)
    ordered_cols = ["source", "type", "label"] + [
        col for col in lexical_df.columns if col not in {"source", "type", "label"}
    ]
    lexical_df = lexical_df[ordered_cols]
    lexical_df.to_csv("lexical_features.csv", index=False)
    print("\nsaved lexical_features.csv")


if __name__ == "__main__":
    main()