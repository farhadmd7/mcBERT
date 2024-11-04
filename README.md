# mcBERT
### mcBERT: Patient-Level Single-cell Transcriptomics Data Representation
Single-cell RNA sequencing (scRNA-seq) transcriptomics improves our understanding of cellular heterogeneity in healthy and pathological states.
However, most scRNA-seq analyses remain confined to single cells or distinct cell populations, limiting their clinical applicability.
Addressing the need to translate single-cell insights into a patient-level disease understanding, we introduce mcBERT, a new method that leverages scRNA-seq data and a transformer-based model to generate integrative patient representations using a self-supervised learning phase followed by contrastive learning to refine these representations.

## Installation

To use mcBERT, follow the installation steps with conda:

```
conda install --name mcBERT python=3.9
conda activate mcBERT
pip install -r requirements.txt
```

## Data Pre-Processing
Once the tissue-specific datasets have been downloaded and stored in a common folder, two pre-processing steps are required to prepare the data for use by mcBERT:

1. Extract the highly variable genes (HVGs) of the selected datasets (see [Determine HVG](mcBERT/extract_highly_variable_genes.py))
2. Using the HVGs, separately save each donor to a file and apply data normalization. (see [Save Donors](mcBERT/preprocess_data.py))

## Training
It is recommended to follow the two-stage training process as outlined in the paper.
In the first step, mcBERTis trained in a self-supervised training setting by running ``python pretrain.py --config <config_file>``.
Afterwards, proceed with the fine-tuning ``python train.py --config <config_file>``

All configurations of the training process, the mcBERT architecture, and the file log and dataset paths are done in a config file.
We provide three example files in the [configs folder](configs), one for pre-training, one for fine-tuning, and one for inference, each containing the training and architecture setup used for the experiments in the paper.

Training progress for both pre-training and fine-tuning is logged via Tensorboard.
UMAPs of the patient's embedding space are generated during fine-tuning and also stored in Tensorboard.

## Inference
An example script to infer patients, using the output of mcBERT for further downstream analysis, such as creating similar patient-level UMAPs as provided in the paper can be found in the [inference.py](inference.py) script.

## Citation

TBA