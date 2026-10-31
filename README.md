# CVSM-Net

Official implementation of **CVSM-Net**, a cross-view spatial alignment and prototype-guided semantic memory network for radiology report generation.

> **Paper:** *CVSM-Net: Cross-View Spatial Alignment and Prototype-guided Semantic Memory Network for Radiology Report Generation*

## Overview

CVSM-Net generates radiology reports from chest X-ray images by integrating complementary visual evidence and adaptive semantic guidance.

The framework consists of two main components:

- **Cross-View Spatial Alignment (CVSA):** models spatial relationships between frontal and lateral representations through cross-view attention. Lateral features query frontal features to aggregate complementary evidence, while residual integration preserves the original lateral information.
- **Prototype-Guided Semantic Memory (PGSM):** retrieves relevant learnable clinical prototypes according to the evolving decoding context. Cascaded multimodal binding integrates the retrieved semantic information with visual and textual representations, and the resulting evidence guides report generation through vocabulary-logit intervention.

Together, these components address ambiguous cross-view spatial correspondence and insufficient adaptive visual-semantic coupling.

## Main Features

- Cross-view attention for spatial information exchange between frontal and lateral chest X-ray representations.
- Residual integration to preserve original visual evidence during cross-view alignment.
- Learnable clinical prototypes optimized jointly with the report generation objective.
- Context-dependent sparse prototype retrieval during decoding.
- Cascaded multimodal binding and semantic intervention at the vocabulary-logit level.
- Evaluation on the IU X-Ray and MIMIC-CXR benchmarks.

## Project Structure

```text
CVSM-Net/
├── models/
├── modules/
├── pycocoevalcap/
├── LICENSE
├── README.md
├── compute_ce.py
├── main_plot.py
├── main_test.py
├── main_train.py
├── plot_mimic_cxr.sh
├── test_iu_xray.sh
├── test_mimic_cxr.sh
├── train_iu_xray.sh
└── train_mimic_cxr.sh
```

The main components are organized as follows:

| Path | Description |
|---|---|
| `models/` | Overall report generation model |
| `modules/` | Visual encoding, cross-view alignment, semantic memory, decoding, data loading, and training utilities |
| `pycocoevalcap/` | Report generation evaluation utilities |
| `main_train.py` | Training entry point |
| `main_test.py` | Checkpoint evaluation entry point |
| `main_plot.py` | Visualization entry point |
| `compute_ce.py` | Clinical metric computation from labeled prediction and reference files |
| `train_*.sh` | Dataset-specific training configurations |
| `test_*.sh` | Dataset-specific experiment scripts; verify their entry points before use |
| `plot_mimic_cxr.sh` | MIMIC-CXR visualization configuration |

Some implementation names differ from the terminology used in the paper. Cross-view alignment is controlled by `--use_crft`, while the semantic memory pathway is controlled by `--use_hd_memory`.

## Data

Experiments are conducted on **IU X-Ray** and **MIMIC-CXR**.

Images and processed annotation files are not included in this repository. Prepare the datasets separately and configure `--image_dir` and `--ann_path` accordingly.

The provided training scripts expect the following layout:

```text
CVSM-Net/
├── data/
│   ├── iu_xray/
│   │   ├── images/
│   │   └── annotation.json
│   └── mimic_cxr/
│       ├── images/
│       └── EVAP_clean.json
├── models/
├── modules/
├── main_train.py
└── main_test.py
```

`EVAP_clean.json` is the annotation filename referenced by the current MIMIC-CXR scripts. If your annotation file has a different name, update `--ann_path`.

Annotation files should contain `train`, `val`, and `test` splits. Each sample should provide:

- `id`: sample identifier.
- `report`: reference report text.
- `image_path`: a list of image paths relative to `--image_dir`.

### Dataset Statistics

The manuscript reports the following numbers of reports:

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| IU X-Ray | 2,770 | 395 | 790 |
| MIMIC-CXR | 222,758 | 1,808 | 3,269 |

Use the same annotations, preprocessing, and dataset partitions when comparing with the reported results.

> Follow the access requirements, licenses, and terms of use of the original datasets. Do not redistribute restricted medical images or annotations through this repository.

## Environment

The implementation uses Python and PyTorch. Major dependencies identified in the source code include:

- PyTorch
- torchvision
- NumPy
- SciPy
- pandas
- scikit-learn
- Pillow
- OpenCV
- tqdm
- spaCy

The repository also includes COCO-style caption evaluation utilities under `pycocoevalcap/`.

Install mutually compatible PyTorch, torchvision, and CUDA versions. Exact Python and dependency versions are not pinned in the current repository.

The experiments described in the manuscript were conducted on an NVIDIA vGPU with 48 GB of memory.

## Running the Code

Run all commands from the project root.

Before starting:

1. Prepare the images and annotation files.
2. Update dataset paths and output directories in the relevant scripts.
3. Replace machine-specific `PYTHONPATH` entries with the local project path, or remove them when running from the project root.
4. Check the implementation notes below before attempting reproduction.

### Training

For IU X-Ray:

```bash
bash train_iu_xray.sh
```

For MIMIC-CXR:

```bash
bash train_mimic_cxr.sh
```

Both scripts invoke `main_train.py`.

Model configuration, optimization parameters, decoding settings, and output paths are specified through command-line arguments.

To inspect the available arguments:

```bash
python main_train.py --help
```

### Evaluation

Use `main_test.py` and provide the trained checkpoint through `--load`.

For the configuration in `train_iu_xray.sh`, an evaluation command is:

```bash
python main_test.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --num_layers 3 \
    --d_model 512 \
    --topk 40 \
    --cmm_size 512 \
    --cmm_dim 256 \
    --hd_dim 256 \
    --use_hd_memory True \
    --hd_memory_weight 0.8 \
    --update_hd_memory True \
    --hd_topk 10 \
    --use_crft True \
    --seed 7580 \
    --beam_size 3 \
    --save_dir results/iu_xray_crft_hd/test/ \
    --load results/iu_xray_crft_hd/model_best.pth
```

For MIMIC-CXR:

```bash
bash test_mimic_cxr.sh
```

Update the checkpoint path in the script before running evaluation.

Always match the model architecture, tokenizer configuration, memory settings, and preprocessing to the selected checkpoint. The evaluation entry point can proceed without a checkpoint, so explicitly verify that `--load` points to the intended trained model.

> The current `test_iu_xray.sh` invokes `main_train.py` and disables the semantic memory pathway. Use the evaluation command above for checkpoint testing instead.

### Clinical Evaluation

`compute_ce.py` computes clinical metrics from separately labeled generated and reference reports.

Before running it:

1. Generate reports using the selected checkpoint.
2. Apply the same clinical label extraction procedure to generated and reference reports.
3. Ensure that samples and label columns are aligned.
4. Update `res_path` and `gts_path` in `compute_ce.py`.

Then run:

```bash
python compute_ce.py
```

This script evaluates existing label files; it does not perform clinical label extraction.

## Experimental Settings

The following shared architectural settings are described in the manuscript:

| Setting | Value |
|---|---|
| Visual backbone | ImageNet-pretrained ResNet-101 |
| Input image size | 224 × 224 |
| Visual feature dimension | 2,048 |
| Hidden dimension | 512 |
| Transformer layers | 3 |
| Attention heads | 8 |
| Optimizer | Adam |

The current training scripts use the following dataset-specific configurations:

| Parameter | IU X-Ray | MIMIC-CXR |
|---|---:|---:|
| Batch size | 16 | 16 |
| Maximum epochs | 100 | 60 |
| Maximum report length | 60 | 120 |
| Word frequency threshold | 3 | 5 |
| Visual encoder learning rate | 1e-4 | 1e-5 |
| Encoder-decoder learning rate | 5e-4 | 6e-5 |
| Beam size | 3 | 5 |
| Random seed | 7580 | 7580 |
| `use_crft` | True | False |
| `use_hd_memory` | True | True |

These are the configurations currently present in the released scripts. They should not be treated as a verified reproduction configuration for every result in the manuscript.

## Main Results

Natural language generation results reported in the manuscript are summarized below:

| Dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|
| IU X-Ray | 0.526 | 0.367 | 0.267 | 0.202 | 0.249 | 0.436 | 0.495 |
| MIMIC-CXR | 0.396 | 0.245 | 0.170 | 0.117 | 0.163 | 0.293 | 0.178 |

The manuscript additionally evaluates clinical effectiveness using micro-averaged precision, recall, and F1-score derived from clinical labels.

## Reproducibility Notes

- Preserve the dataset partitions, report preprocessing, and image-view ordering used for each experiment.
- Record the software environment, random seed, training configuration, and selected checkpoint.
- Keep vocabulary construction and token indices consistent between training and evaluation.
- Match all model and memory parameters when loading a checkpoint.
- Keep datasets, checkpoints, logs, and local environment directories outside version control.
- Preserve third-party license notices and attribution.

### Current Implementation Notes

The current release contains several items that need to be reconciled before reproducing the complete manuscript results:

- **Vocabulary saving:** `main_train.py` calls `tokenizer.save_vocab(...)`, but the current tokenizer implementation does not define this method. Implement consistent vocabulary serialization and loading before training and evaluation.
- **Evaluation metrics:** `modules/metrics.py` currently enables BLEU and ROUGE-L. METEOR and CIDEr must be configured separately to reproduce the complete results table.
- **Missing-view handling:** the current dataset loaders duplicate the available image for single-view samples, whereas the manuscript describes zero-image padding.
- **Experiment configuration:** some script settings differ from the manuscript, including learning rates, memory settings, and the MIMIC-CXR maximum report length. The released MIMIC-CXR training script also disables cross-view alignment.

## Citation

Citation information will be added after publication. During double-blind review, author-identifying information is omitted from this repository.

## License

The repository includes an Apache License 2.0 license file. See [LICENSE](LICENSE) for details.

Third-party components, including the bundled caption evaluation utilities, retain their own licenses and attribution requirements. Datasets remain subject to their original access conditions and terms of use.
