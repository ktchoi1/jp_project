import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FILM_TO_SOURCE = {
    "old_one": ("(insert srt file path here)", "(insert dialogue csv path here)", 0),
    "new_one": ("(insert srt file path here)", "(insert dialogue csv path here)", 1),
}


# strips html tags (stolen from extract_lexical)
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    return text.strip()


# reads an srt file and returns a list of cleaned subtitle strings
def load_srt_text(path):
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
    return " ".join(texts)


# loads novel dialogue csv and returns the dialogue column as a list of strings
def load_novel_text(path):
    dialogue_df = pd.read_csv(path)
    for possible_col in ["quote", "text", "dialogue"]:
        if possible_col in dialogue_df.columns:
            return " ".join(dialogue_df[possible_col].dropna().astype(str).tolist())
    raise ValueError(f"no dialogue text column found in {path}")


# computes tfidf cosine similarity between the film, 
# dialogue and the source novel dialogue, 
# computation assisted by ai
def main():
    rows = []

    for film_name, (srt_path, novel_csv_path, label) in FILM_TO_SOURCE.items():
        if not os.path.exists(srt_path) or not os.path.exists(novel_csv_path):
            print(f"missing files for {film_name}, skipping")
            continue

        print(f"doing lexical drift for {film_name}...")
        film_text = load_srt_text(srt_path)
        novel_text = load_novel_text(novel_csv_path)

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([novel_text, film_text])
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

        rows.append(
            {
                "film": film_name,
                "label": label,
                "lexical_similarity": round(similarity, 4),
                "lexical_drift": round(1 - similarity, 4),
            }
        )

    drift_df = pd.DataFrame(rows)
    drift_df.to_csv("lexical_drift.csv", index=False)
    print("\nsaved lexical_drift.csv")


if __name__ == "__main__":
    main()