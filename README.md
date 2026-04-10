# Rejectors in the Wild: Deployment Barriers for LLM Rejectors

This repo contains the code (and the data setup) needed to reproduce the results from our paper. The datasets have to be downloaded first, but everything else is included here.
The paper -- and this repository -- are organized into three barriers. For each barrier, the files should be run in order, as listed below.
Most of the workflow lives in Jupyter notebooks under `./notebooks`. Longer-running parts are implemented as Python scripts in `./src` and called from the notebooks.

---

## Rebuttal

We have added the following files that contain the additional experiments requested for the rebuttal:

* **`./notebooks/Rebuttal_Barrier1_analyzeHumanLabels.ipynb`**
  Computes accuracy for F1 and BLEU-4 on 1,000 random balanced subsets and splits the F1 results into the individual datasets.

* **`./notebooks/Rebuttal_Barrier1_entailment_capability.ipynb`**
  Applies NLI-based approaches to the MATH and OpenbookQA datasets.

## Barrier I — Sensitivity of Rejectors

* **`./notebooks/Barrier1_downloadHelm.ipynb`**
  Downloads the HELM Lite dataset → stores raw data in `./helm_lite_v1.13.0/`

* **`./notebooks/Barrier1_parseHelm.ipynb`**
  Parses HELM Lite outputs and creates per-model files → stores parsed dataset in `./helmBenchmark/<model>.pkl`

* **`./notebooks/Barrier1_addEntailmentModelDecision.ipynb`**
  Apply entailment-based labeling rules and add correctness labels to each model file → updates `./helmBenchmark/<model>.pkl`

* **`./src/Barrier1_rejectors.py`**
  Trains one rejector per LLM across labeling rules, epochs, and runs → stores logits and labels in
  `./results/HelmLiteRejectors/<model>/<labeling_rule>/<run>/`

* **`./src/Barrier1_parseResults.py`**
  Aggregates predictions and computes evaluation metrics (AUROC, AUPR, Brier, Accuracy@0.5, coverage–accuracy) → stores
  `results.pkl` and `results_compressed.pkl` in `./results/HelmLiteRejectors/`

* **`./notebooks/Barrier1_visualizer.ipynb`**
  Analyzes rejector performance (run variability, epoch impact, labeling rules, model comparison) → exports figures to `./results/figures/`

* **`./notebooks/Barrier1_selectHELMSamples.ipynb`**
  Samples HELM instances for manual annotation → stores `./results/HELMSamples.csv`

* **Human labeling (external step)**
  Annotates sampled predictions → produces `./results/HELMSamplesLabeled.csv`

* **`./notebooks/Barrier1_analyzeHumanLabels.ipynb`**
  Evaluates alignment between labeling rules and human judgments (Spearman, AUROC, accuracy, thresholds) → reads
  `HELMSamplesLabeled.csv` and outputs analysis figures/tables

---

## Barrier II — Redundancy Across Rejectors

* **`./src/Barrier2_trainRejectors.py`**
  Trains rejectors for each model and tests them on all other models to build transfer matrices → saves logits and labels per model pair (i, j) in
  `./results/HelmLiteHeatmap/`

* **`./notebooks/Barrier2_loader.ipynb`**
  Loads labels and logits, computes metrics, and saves them in a large and a compressed transfer matrix → stores
  `heatmap_data.pkl` and `heatmap_data_compressed.pkl`

* **`./notebooks/Barrier2_visualizer.ipynb`**
  Generates transfer-matrix figures → exports figures to `./results/figures/`

* **`./notebooks/Barrier2_analyzer.ipynb`**
  Analyzes redundancy structure (e.g., clusters, strong/weak targets, symmetry)

* **Appendix variants**
  Repeats the workflow with alternative labeling rules → stores outputs in
  `./results/HelmLiteHeatmap_0.1_0.4/` and `./results/HelmLiteHeatmap_0.9_0.1/`

---

## Barrier III — Abstention produces no Outputs

### Dataset Preparation

* **`./src/Barrier3_downloadOpenLLMLeaderboard.py`**
  Downloads OpenLLMLeaderboard data from Hugging Face → stores raw files locally in the defined `/path/to/hf` Hugging Face folder

* **`./notebooks/Barrier3_parseOpenLLMLeaderboard.ipynb`**
  Parses benchmark files into per-model dumps → stores `./openLLMLeaderboard/<model>.pkl`

* **`./notebooks/Barrier3_prepareOpenLLMLeaderboard.ipynb`**
  Extracts prompts and correctness labels depending on task → updates model files

* **`./notebooks/Barrier3_compressOpenLLMLeaderboard.ipynb`**
  Reduces file size for faster loading → stores compressed versions in `./openLLMLeaderboard/compressed/<model>.pkl`

* **`./notebooks/Barrier3_createData_OpenLLMLeaderboard.ipynb`**
  Builds routing datasets → stores `train.pkl` and `test.pkl` in `./openLLMLeaderboard/data/`

* **`./notebooks/Barrier3_createData_RouterBench.ipynb`**
  Downloads, parses, and prepares RouterBench for routing experiments → stores `train.pkl` and `test.pkl` in `./routerBench/data/`

### Router Preparation

* **`./src/Barrier3_train_router.py`**
  Trains rejectors for each LLM (multiple runs) → stores models and held-out bank in
  `./results/<dataset>/<run>/rejectors/`

* **`./src/Barrier3_collect_rejector_predictions.py`**
  Applies rejectors to test and held-out bank → stores predictions in
  `./results/<dataset>/<run>/predictions/` and bank outputs in `./results/<dataset>/<run>/bank/`

* **`./notebooks/Barrier3_createHeatmaps.ipynb`**
  Constructs transfer matrices from held-out bank predictions → stores `./results/<dataset>/<run>/heatmap_compressed.pkl`

### Router Evaluation

* **RouterBench dependency**  
  Download RouterBench from the official repo and place it under `./src/` (so it can be imported by the baseline scripts):  
  https://github.com/withmartian/routerbench

* **MLP router settings**  
  For better and more stable results, set the MLP’s `max_iter` to **1000**.  
  We also expose `random_state` as an argument in the `mlp_router` `__init__` to run the script multiple times with different seeds.

* **`./src/Barrier3_evaluate_baselines.py`**
  Evaluates baseline routers (e.g., KNN, MLP) → stores results in
  `./results/baselines/<dataset>/router_results_<run>_<mlp_layer>.npz`

* **`./notebooks/Barrier3_evaluateRouters.ipynb`**
  Collects baselines and routing strategies and compares performance

---

## Directory Structure (after executing all files)

```
root/
    ├─ README.md
    ├─ notebooks/          # Jupyter notebooks (fast-to-run code)
    │  ├─ Barrier1_downloadHelm.ipynb
    │  ├─ Barrier1_parseHelm.ipynb
    │  ├─ Barrier1_addEntailmentModelDecision.ipynb
    │  ├─ Barrier1_visualizer.ipynb
    │  ├─ Barrier1_selectHELMSamples.ipynb
    │  ├─ Barrier1_analyzeHumanLabels.ipynb
    │  │
    │  ├─ Barrier2_loader.ipynb
    │  ├─ Barrier2_visualizer.ipynb
    │  ├─ Barrier2_analyzer.ipynb
    │  │
    │  ├─ Barrier3_parseOpenLLMLeaderboard.ipynb
    │  ├─ Barrier3_prepareOpenLLMLeaderboard.ipynb
    │  ├─ Barrier3_compressOpenLLMLeaderboard.ipynb
    │  ├─ Barrier3_createData_OpenLLMLeaderboard.ipynb
    │  ├─ Barrier3_createData_RouterBench.ipynb
    │  ├─ Barrier3_createHeatmaps.ipynb
    │  └─ Barrier3_evaluateRouters.ipynb
    │
    ├─ src/          # Python scripts (long-running code: training / large eval)
    │  ├─ Barrier1_rejectors.py
    │  ├─ Barrier1_parseResults.py
    │  │
    │  ├─ Barrier2_training.py
    │  │
    │  ├─ Barrier3_downloadOpenLLMLeaderboard.py
    │  ├─ Barrier3_train_router.py
    │  ├─ Barrier3_collect_rejector_predictions.py
    │  │─ Barrier3_evaluate_baselines.py
    │  └─ helper.py          # helper functions used throughout the project
    │
    ├─ helm_lite_v1.13.0/          # raw HELM Lite download
    │  └─ (downloaded HELM Lite data)
    │
    ├─ helmBenchmark/          # parsed HELM Lite dataset
    │  ├─ <model_name1>.pkl                
    │  ├─ <model_name2>.pkl 
    │  └─ ...
    │
    ├─ openLLMLeaderboard/          # parsed OpenLLMLeaderboard dataset
    │  ├─ <model_name1>.pkl                      
    │  ├─ <model_name2>.pkl 
    │  ├─ ...
    │  ├─ compressed/
    │  │  ├─ <model_name1>.pkl
    │  │  ├─ <model_name2>.pkl
    │  │  └─ ...
    │  └─ data/
    │     ├─ train.pkl
    │     └─ test.pkl
    │
    ├─ routerBench/          # parsed RouterBench dataset (+ train/test)
    │  └─ data/
    │     ├─ train.pkl
    │     └─ test.pkl
    │
    └─ results/
       ├─ figures/
       │  ├─ <figure1>.pdf
       │  ├─ <figure2>.pdf
       │  └─ ...
       │
       ├─ HELMSamples.csv
       ├─ HELMSamplesLabeled.csv
       │
       ├─ HelmLiteRejectors/          # Barrier 1 output
       │  ├─ results.pkl
       │  ├─ results_compressed.pkl
       │  └─ <model_x>/
       │     └─ <labeling_rule_x>/
       │        ├─ labels.npy
       │        └─ <run_x>/
       │           └─ <logits_epoch_x>.npy
       │
       ├─ HelmLiteHeatmap/          # Barrier 2 output
       │  ├─ heatmap_data.pkl
       │  ├─ heatmap_data_compressed.pkl
       │  └─ <source_model_x>/
       │      └─ <target_model_x>/
       │           ├─ Shuffle_labels.npy/
       │           └─ Shuffle_logits.npy/
       │
       ├─ HelmLiteHeatmap_0.1_0.4/          # Barrier 2 appendix output
       │  ├─ heatmap_data.pkl
       │  ├─ heatmap_data_compressed.pkl
       │  └─ <source_model_x>/
       │      └─ <target_model_x>/
       │           ├─ Shuffle_labels.npy/
       │           └─ Shuffle_logits.npy/
       │
       ├─ HelmLiteHeatmap_0.9_0.1/          # Barrier 2 appendix output
       │  ├─ heatmap_data.pkl
       │  ├─ heatmap_data_compressed.pkl
       │  └─ <source_model_x>/
       │      └─ <target_model_x>/
       │           ├─ Shuffle_labels.npy/
       │           └─ Shuffle_logits.npy/
       │
       ├─ openLLMLeaderboard/          # Barrier 3 output
       │  └─ <run_x>/
       │     ├─ rejectors/
       │     │  ├─ <model>/
       │     │  └─ bank.pkl
       │     ├─ predictions/
       │     │  └─ <model>.npy
       │     ├─ bank/
       │     │  └─ <model>.npy
       │     └─ heatmap_compressed.pkl
       │
       ├─ routerBench/          # Barrier 3 output
       │  └─ <run_x>/
       │     ├─ rejectors/
       │     │  ├─ <model>/
       │     │  └─ bank.pkl
       │     ├─ predictions/
       │     │  └─ <model>.npy
       │     ├─ bank/
       │     │  └─ <model>.npy
       │     └─ heatmap_compressed.pkl
       │
       └─ baselines/          # Barrier 3 baselines
          └─ <dataset>/
             └─ router_results_<run>_<mlp_layer>.npz
```
