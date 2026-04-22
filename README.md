# welcome to my readme

this folder is meant to show how to recreate the pipeline, but does not
give you everys source text / file (sorry). you have to get those yourselv.es

edit the file dicts at the top of each with your own files\
without these files, empty csvs will be produced and the pipeline will not
succeed for u


## main pipeline order

1. `run_booknlp.py`
   input: plain-text novels in `novels/`
   output: dialogue csvs in `dialogue_csvs/`

2. `parse_srts.py`
   input: subtitle files in `srt_files/`
   output: `srt_temporal_features.csv`

3. `extract_lexical.py`
   input: novel dialogue csvs and subtitle files
   output: `lexical_features.csv`

4. `lexical_drift.py`
   input: film subtitle files plus their matching novel dialogue csvs
   output: `lexical_drift.csv`

5. `merge_features.py`
   input: `lexical_features.csv`, `srt_temporal_features.csv`, `lexical_drift.csv`
   output: `unified_features.csv`

6. `classifiers.py`
   input: `unified_features.csv`
   outputs:
   `results_table.csv`

   `shap_summary.png` if `shap` is installed
   su
## scene-level scripts

- `compare_scenes.py`
  input: whisperx scene json files
  output: `scene_comparison.csv`

- `scene_lexical_comparison.py`
  input: whisperx scene json files
  output: `scene_lexical_comparison.csv`

- `full_film_analysis.py`
  input: whisperx full-film json files
  output: `full_film_comparison.csv`

## notes

- thank you for stopping by!
- you need to install booknlp, spacy, whisperx, and shap for the full experience

