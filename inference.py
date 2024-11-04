from argparse import ArgumentParser
from glob import glob

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from mcBERT.utils.clustering_utils import apply_UMAP
from mcBERT.utils.patient_level_dataset import Patient_level_dataset
from mcBERT.utils.utils import get_scRNA_model, prepare_dataset, set_seeds
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

set_seeds(42)


"""Script to plot the T-SNE plots and calculate cosine similarity across the patients
"""

# Config file for inference
parser = ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="path to yaml config file for inference of donors",
)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

model = get_scRNA_model(cfg)
model.load_state_dict(torch.load(cfg.model.model_ckpt))
model.cuda()
model.eval()

files_all = glob(cfg.H5AD_FILES)
df = prepare_dataset(files_all, multiprocess=True)
print(
    f"Using {len(df['donor_id'].unique())} patients representing {df['disease'].unique()} diseases"
)

dataset = Patient_level_dataset(
    df,
    cfg.HIGHLY_VAR_GENES_PATH,
    inference=True,
    random_cell_stratification=0,
)
dataloder = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
)

patient_embeddings = np.empty((len(dataset), 288))

with torch.no_grad():
    # Embedding all donors using mcBERT
    for i, batch in enumerate(tqdm(dataloder)):
        outputs = model(batch.to("cuda"))
        encoder_out = outputs.cpu()
        patient_embeddings[
            i * dataloder.batch_size : i * dataloder.batch_size + len(batch)
        ] = np.array(encoder_out)

# Calculate cosine similarity between all patients, could be used for further testing
cosine_sim = cosine_similarity(patient_embeddings)

# Create the UMAP plot
X_umap_embeddings = apply_UMAP(patient_embeddings)
fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=X_umap_embeddings[:, 0], y=X_umap_embeddings[:, 1], hue=df["disease"])
plt.title("UMAP plot of patients")
plt.legend(loc="upper left")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("umap_plot.png")
