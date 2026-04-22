import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler



warnings.filterwarnings("ignore")


# lexical features from stamatatos
# keeping separate to test each on its own 
TEMPORAL_FEATURES = [
    "pause_mean_sec",
    "pause_median_sec",
    "pause_std_sec",
    "pause_count",
    "overlap_count",
    "overlap_rate",
    "speech_rate_mean_wps",
    "speech_rate_median_wps",
    "turn_len_mean_words",
    "turn_len_median_words",
    "n_turns",
    "total_words",
]

LEXICAL_FEATURES = [
    "avg_word_length",
    "avg_sent_length",
    "type_token_ratio",
    "flesch_reading_ease",
    "formality_score",
    "avg_dep_depth",
]


# runs leave-one-out cross validation on a given feature set and returns
# accuracy, precision, recall, and f1. 
# ai assisted: refits on the full dataset at the end 
def leave_one_out_metrics(feature_df, labels, run_name):
    # train on everything else and test on one

    x_values = feature_df.fillna(0).to_numpy()
    y_values = labels.to_numpy()
    predictions = np.zeros(len(x_values), dtype=int)

    for test_index in range(len(x_values)):
        x_train = np.delete(x_values, test_index, axis=0)
        y_train = np.delete(y_values, test_index)
        x_test = x_values[test_index : test_index + 1]

        # scaling inside the loop debug
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        classifier = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        classifier.fit(x_train_scaled, y_train)
        predictions[test_index] = classifier.predict(x_test_scaled)[0]

    result = {
        "classifier": run_name,
        "accuracy": accuracy_score(y_values, predictions),
        "precision": precision_score(y_values, predictions, zero_division=0),
        "recall": recall_score(y_values, predictions, zero_division=0),
        "f1": f1_score(y_values, predictions, zero_division=0),
    }

    # refit on full data 
    scaler = StandardScaler()
    full_x_scaled = scaler.fit_transform(feature_df.fillna(0).to_numpy())
    classifier = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    classifier.fit(full_x_scaled, y_values)

    return result, classifier, scaler


# loads the feature csv, runs the three classifiers
# saves the results table
def main():
    data_df = pd.read_csv("unified_features.csv")

    # only keep columns that actually exist in the csv
    temporal_features = [col for col in TEMPORAL_FEATURES if col in data_df.columns]
    lexical_features = [col for col in LEXICAL_FEATURES if col in data_df.columns]
    combined_cols = temporal_features + lexical_features
    labels = data_df["label"]

    # run all three and compare
    lexical_result, _, _ = leave_one_out_metrics(data_df[lexical_features], labels, "lexical only")
    temporal_result, _, _ = leave_one_out_metrics(data_df[temporal_features], labels, "temporal only")
    combined_result, combined_model, combined_scaler = leave_one_out_metrics(
        data_df[combined_cols],
        labels,
        "combined",
    )

    results_df = pd.DataFrame([lexical_result, temporal_result, combined_result])
    results_df.to_csv("results_table.csv", index=False)
    print(results_df.to_string(index=False))
    print("\nsaved results_table.csv")

    try:
        import shap
    except ModuleNotFoundError:
        print("shap is not installed here, so i skipped regenerating the shap outputs.")
        return

    # i used ai here to help with the shap call because shap's api changes ODE
    # i did not want to debug forever.
    combined_x_scaled = combined_scaler.transform(data_df[combined_cols].fillna(0).to_numpy())
    explainer = shap.LinearExplainer(combined_model, combined_x_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(combined_x_scaled)

    # bar chart of mean absolute shap values
    plt.figure()
    shap.summary_plot(shap_values, combined_x_scaled, feature_names=combined_cols, plot_type="bar", show=False)
    plt.title("shap feature importance - combined classifier")
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)

    # saving rankings as csv 
    shap_df = pd.DataFrame(
        {
            "feature": combined_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "type": ["Temporal" if col in temporal_features else "Lexical" for col in combined_cols],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv("shap_rankings.csv", index=False)
    print("saved shap_rankings.csv")
    print("saved shap_summary.png")


if __name__ == "__main__":
    main()