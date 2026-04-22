import pandas as pd

lexical_df = pd.read_csv("lexical_features.csv")
temporal_df = pd.read_csv("srt_temporal_features.csv")
drift_df = pd.read_csv("lexical_drift.csv")

# lexical_features.csv has both novels and films, so keep only films here
film_lexical_df = lexical_df[lexical_df["type"] == "film"].copy()

# lexical uses source, temporal use film
film_lexical_df = film_lexical_df.rename(columns={"source": "film"})
film_lexical_df = film_lexical_df.drop(columns=["type"])

# merge on film + label
unified_df = (
    film_lexical_df
    .merge(temporal_df, on=["film", "label"], how="inner")
    .merge(drift_df[["film", "lexical_drift"]], on="film", how="left")
)
unified_df.to_csv("unified_features.csv", index=False)
print("saved unified_features.csv")