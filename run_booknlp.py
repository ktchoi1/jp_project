import os

import pandas as pd

# just the booknlp pass


NOVELS = {
    "novel": "(insert novel txt path here)",

}

BOOKNLP_PARAMS = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "small",
}


# runs  booknlp pipeline on each novel 
# pulls just the dialogue rows out of the quotes files 
# saves to dialogue_csvs

def main():
    try:
        from booknlp.booknlp import BookNLP
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "booknlp is not installed in this python environment. install it first, then rerun this script."
        ) from exc

    os.makedirs("booknlp_output", exist_ok=True)
    os.makedirs("dialogue_csvs", exist_ok=True)

    model = BookNLP("en", BOOKNLP_PARAMS)

    for novel_name, novel_path in NOVELS.items():
        if not os.path.exists(novel_path):
            print(f"missing novel txt, skipping: {novel_path}")
            continue
        print(f"running booknlp for {novel_name}...")
        output_dir = os.path.join("booknlp_output", novel_name)
        os.makedirs(output_dir, exist_ok=True)
        model.process(novel_path, output_dir, novel_name)

    print("\nnow pulling just the dialogue rows into csv files...")

    for novel_name in NOVELS:
        quotes_path = os.path.join("booknlp_output", novel_name, f"{novel_name}.quotes")
        if not os.path.exists(quotes_path):
            print(f"missing quotes file for {novel_name}: {quotes_path}")
            continue

        quotes_df = pd.read_csv(quotes_path, sep="\t")
        keep_cols = [
            col
            for col in ["quote_start", "quote_end", "quote", "char_id", "speaker"]
            if col in quotes_df.columns
        ]
        quotes_df = quotes_df[keep_cols].copy()
        quotes_df["novel"] = novel_name

        out_path = os.path.join("dialogue_csvs", f"{novel_name}_dialogue.csv")
        quotes_df.to_csv(out_path, index=False)
        print(f"saved {len(quotes_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
