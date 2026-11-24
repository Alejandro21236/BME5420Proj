# Predicting extracellular matrix gene-set scores from H&E slides

This project uses H&E whole-slide images and RNA expression data from lung adenocarcinoma to explore extracellular matrix (ECM) composition in a mechanobiology classroom. A graph attention model takes UNI2-h patch embeddings from a slide and predicts nine ECM-related gene-set scores. Students can compare the model's outputs across slides and discuss the roles of ECM components in tissue structure and cell–matrix interactions.

The targets are derived from gene-expression signatures that include ECM genes and integrin genes. They are **normalized ssGSEA scores**, not direct measurements of protein concentration. The model is intended for education and research, not clinical use.

## Repository contents

| File | Purpose |
| --- | --- |
| `ssgseascoring.py` | Computes single-sample gene-set enrichment scores for nine ECM-related gene sets. |
| `normalizessgsea.py` | Divides each sample's nine scores by their row sum. |
| `luadfolds.py` | Assigns slides to folds and train/validation/test subsets, then merges them with the scores. |
| `TrainModel` | Trains and evaluates one selected fold of the graph attention model. This file has no `.py` extension. |
| `infmech.py` | Extracts patches and UNI2-h embeddings from an SVS slide, loads the fold 1 model, and prints nine predictions. |
| `groundtruth.csv` | Prepared slide IDs, case IDs, split assignments, and normalized targets. |
| `fold_1_best_model.pt` | Saved fold 1 model weights. |

`infmech.py` contains whole-slide patch extraction, UNI2-h embedding extraction, and inference in one script. Update its paths and authentication as described below before running it. To train using `TrainModel`, supply embeddings in the format described below.

## How the model works

1. The label-generation scripts calculate enrichment scores from the gene-expression matrix for nine ECM-related gene sets and normalize the scores for each sample.
2. A slide is represented by up to 2,048 UNI2-h patch embeddings, each with 1,536 features.
3. Patches are connected when their embedding cosine similarity is greater than 0.5.
4. Two graph attention layers and a multi-head attention layer process the patch representations. Mean pooling produces one slide representation.
5. A decoder predicts nine values, one for each target listed below. Training minimizes mean squared error.

The targets, in model-output order, are collagen IV, collagen VI, fibrillar collagen, fibronectin, laminin, osteopontin, tenascin C, thrombospondin, and vitronectin.

## Requirements

- Python 3.10 is a suitable starting point.
- An NVIDIA GPU and a CUDA-enabled PyTorch installation are required by the current `TrainModel` script, which sets `DEVICE = "cuda"`.
- Python packages: PyTorch, PyTorch Geometric, pandas, NumPy, SciPy, scikit-learn, and GSEApy. Inference also needs `timm`, Pillow, `huggingface_hub`, OpenCV, and OpenSlide's Python package and native library.
- To generate new labels: a log2 expression matrix with gene symbols and sample identifiers, plus slide embeddings.
- To train from the included `groundtruth.csv`: the matching UNI2-h embedding files. These embeddings are not included in the repository.

Install compatible PyTorch and PyTorch Geometric versions for your GPU and CUDA driver. For example, the following installs the PyTorch 2.8 CUDA 12.6 build used during development; your GPU driver must support it:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install torch-geometric pandas numpy scipy scikit-learn gseapy
python -m pip install timm pillow huggingface_hub opencv-python openslide-python
```

For SVS inference, also install the native OpenSlide library, for example with `conda install -c conda-forge openslide`, or use an OpenSlide module supplied by your cluster. Installing `openslide-python` alone does not provide the native library. Access to the gated [UNI2-h model](https://huggingface.co/MahmoodLab/UNI2-h) and Hugging Face authentication are required to download the weights.

Check your environment before training:

```bash
python -c "import torch, torch_geometric, pandas, numpy, scipy, sklearn, gseapy; print('CUDA available:', torch.cuda.is_available())"
```

Run this check on an allocated GPU node. `CUDA available` must be `True` for the training script as written. Package versions are not pinned in this repository; resolve compatibility issues in your own environment before training.

## Input files

### Expression matrix

To regenerate scores, place `TCGA-LUAD_expression_matrix_log2.tsv` in the working directory. The first column must be named `gene` and contain gene symbols; the remaining columns contain sample expression values. The identifiers must match the case IDs used in the merge, or be converted to matching case IDs before running `luadfolds.py`.

### Patch embeddings

The training script expects one file named `<slide_id>.npy` per slide in the embedding directory. Each file must contain a dictionary of patch coordinates and embeddings:

```python
{
    (x, y): {"patch_embeddings": embedding},
    # Additional patch coordinates and embeddings
}
```

Each `embedding` must be a one-dimensional NumPy-compatible vector with 1,536 values. The script reads files with `allow_pickle=True`; use only files from a trusted source. The first 2,048 entries are used, so the ordering of patches in the dictionary matters when there are more than 2,048.

### Ground truth and splits

`groundtruth.csv` must include `slide_id`, `case_id`, `fold`, `split`, and the nine target columns. The split values are `train`, `val`, and `test`. The included CSV already has these columns; to train with it, begin at **Train a model** below.

## Generate labels and splits

Run these steps from the repository directory if you want to regenerate the targets. Each script reads and writes files relative to the current working directory.

### 1. Calculate ssGSEA scores

```bash
python ssgseascoring.py
```

The script reads `TCGA-LUAD_expression_matrix_log2.tsv`, calculates scores using the nine gene sets defined in the script, and writes:

- `ssgsea_long.csv`: one sample–gene-set result per row.
- `ssgsea_scores.csv`: one sample per row and one gene set per column, using the GSEApy `NES` result.

### 2. Normalize the scores

```bash
python normalizessgsea.py
```

This reads `ssgsea_scores.csv` and writes `ssgseanomrscore.csv` (the filename spelling is intentional). For each sample, each score is divided by the sum of its nine scores. These relative gene-set scores are the targets used for training; they are not measured protein quantities.

### 3. Assign slides to folds

Open `luadfolds.py` and replace its `path1 = "..."` value with the directory containing the slide `.npy` files. Then run:

```bash
python luadfolds.py
```

The script writes `slide_fold_assignments.csv` and merges the normalized scores by case ID into `groundtruth.csv`. It derives the case ID from the first 12 characters of each embedding filename and keeps one slide per case. Inspect the merged output to confirm that the sample identifiers matched and that the intended cases are present.

**How the folds are assigned:** the current script divides the cases into five disjoint groups and then divides each group into approximately 60% training, 15% validation, and 25% test cases. Each model is trained only on its selected group's training subset. This is not standard five-fold cross-validation in which each model trains on the other four groups. The script uses an unordered set to build the case list and does not fix a random seed; regenerated assignments may differ between runs.

## Train a model

1. Open `TrainModel` and replace its `path1 = "..."` value with the directory containing the slide embeddings.
2. Place `groundtruth.csv` in the working directory. Missing slide embedding files are reported and skipped; each selected fold still needs nonempty train, validation, and test sets.
3. Run a fold:

```bash
python TrainModel --fold 1
```

Choose `--fold 1` through `--fold 5`. Each invocation trains one new model using the specified fold's assignments. The defaults are 20 epochs, learning rate `1e-4`, batch size 1, a maximum of 2,048 patches, and seed 42. The model with the lowest validation loss is evaluated on that fold's test split.

To submit five independent jobs on a SLURM cluster, use a job array and adapt the account, partition, environment, and time to your cluster:

```bash
#!/bin/bash
#SBATCH --array=1-5
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=fold_%A_%a.out
#SBATCH --error=fold_%A_%a.err

# Activate the environment before running this command.
python TrainModel --fold "$SLURM_ARRAY_TASK_ID"
```

Submit the completed job script with `sbatch`. Each task requests its own GPU; tasks run concurrently when resources are available.

### Outputs

Training writes the following files under `results/`, replacing `<fold>` with the selected fold number:

| File | Contents |
| --- | --- |
| `fold_<fold>_best_model.pt` | Weights from the epoch with the lowest validation loss. |
| `fold_<fold>_history.csv` | Training and validation loss for each epoch. |
| `fold_<fold>_predictions.csv` | Slide IDs and true and predicted values for all nine targets. |
| `fold_<fold>_testicles.csv` | Per-target test MSE, MAE, R², Pearson correlation, and Spearman correlation. The filename matches the current code. |

The supplied `fold_1_best_model.pt` is at the repository root; newly trained weights are saved under `results/`.

## Run inference on an SVS slide

`infmech.py` reads one H&E slide, extracts non-overlapping 256 × 256 pixel patches from level 0, and uses Otsu thresholding to retain patches with at least 20% tissue. It saves the selected patches as PNG files, extracts UNI2-h embeddings, creates the patch graph, and prints nine predictions from the fold 1 model. The input slide is specified **in the script**; there is currently no command-line image argument or graphical upload interface.

### 1. Prepare UNI2-h access

Request access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) if needed, then authenticate with Hugging Face in the environment that will run the script:

```bash
hf auth login
```

The repository currently contains `login("")` near the top of `infmech.py`. Remove that line to use the cached authentication; do not place an access token in the script or commit it to the repository.

### 2. Set the input and output paths

Open `infmech.py`. Its paths are currently hard-coded and the paths for patch writing, patch reading, and saved-embedding loading do **not** match. Change them together:

1. Set the slide path in `openslide.OpenSlide(...)` to your `.svs` file.
2. Set `pathh` to the directory where you want the patch PNGs saved. The script creates that directory.
3. Replace the later `root = Path(...)` assignment with `root = pathh`, so the embedding loop reads those same patches.
4. Replace the path inside the later `np.load(...)` call with `output_path`, the variable set when the embedding dictionary is saved:

   ```python
   data = np.load(output_path, allow_pickle=True).item()
   ```

The script creates `pathh / "embeddings"` and writes `registered.npy` there when the patch directory is named `registered`. The saved dictionary contains one `(x, y)` key per patch, with its 1,536-value vector under `"patch_embeddings"`. Use a separate or empty patch directory for each slide; the script reads matching images already present in that directory.

### 3. Set the model-weights path

The repository provides `fold_1_best_model.pt` at its root, but `infmech.py` currently looks for `results/fold_1_best_model.pt`. When running from the repository root, change the `t.load(...)` path to:

```python
"fold_1_best_model.pt"
```

Alternatively, use the path to a compatible checkpoint you have trained and saved under `results/`.

### 4. Run the script

From the repository root, on a CUDA GPU node:

```bash
python infmech.py
```

The script prints the nine predicted normalized gene-set scores to the terminal. It does not currently write a predictions CSV for inference. It uses up to the first 2,048 saved embeddings. The patch grid uses the slide's native level-0 resolution; the physical area represented by a patch depends on the scan resolution. Full-slide patching and UNI2-h extraction may take longer than the classroom activity, so prepare embeddings or results in advance for a short demo.

## Classroom use

Students can examine the nine target signatures, compare predicted values for different slides, and discuss how ECM proteins and integrins participate in tissue organization and cell–matrix interactions. A classroom presentation can show the progression from a slide to patch embeddings, a patch graph, and nine predicted gene-set scores.

For a short classroom session, prepare slides, embeddings, and predictions in advance. `infmech.py` provides a scripted slide-to-prediction workflow. Interpret results against measured labels and report the evaluation metrics alongside the predictions.
