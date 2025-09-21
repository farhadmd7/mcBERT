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
from sklearn.metrics import pairwise_distances

from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


def main():

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
    model.load_state_dict(torch.load(cfg.model.model_ckpt, map_location=cfg.device))
    model.to(cfg.device)
    model.eval()

    files_all = glob(cfg.H5AD_FILES+ '/*.h5ad')
    df = prepare_dataset(files_all, sample_key=cfg.dataset.sample_key,condition_key=cfg.dataset.condition_key,multiprocess=True)
    print(
        f"Using {len(df['donor_id'].unique())} patients representing {df['disease'].unique()} diseases"
    )

    dataset = Patient_level_dataset(
        df,
        cfg.HIGHLY_VAR_GENES_PATH,
        inference=True,
        random_cell_stratification=0,
        cell_type_key=cfg.dataset.cell_type_key,
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
            outputs = model(batch.to(cfg.device))
            encoder_out = outputs.cpu()
            patient_embeddings[
                i * dataloder.batch_size : i * dataloder.batch_size + len(batch)
            ] = np.array(encoder_out)

    # Calculate cosine similarity between all patients, could be used for further testing
    # cosine_sim = cosine_similarity(patient_embeddings)


    # Cosine distance:
    cosine_dist = pairwise_distances(patient_embeddings, metric="cosine")
    pd.DataFrame(cosine_dist, index=df["donor_id"], columns=df["donor_id"]).to_csv("mcbert_cosine_distance.csv")


    # # Create the UMAP plot
    # X_umap_embeddings = apply_UMAP(patient_embeddings)
    # fig = plt.figure(figsize=(10, 10))
    # sns.scatterplot(x=X_umap_embeddings[:, 0], y=X_umap_embeddings[:, 1], hue=df["disease"])
    # plt.title("UMAP plot of patients")
    # plt.legend(loc="upper left")
    # plt.xlabel("UMAP 1")
    # plt.ylabel("UMAP 2")
    # plt.tight_layout()
    # plt.savefig("umap_plot.png")

if __name__ == "__main__":
    # Required on macOS/Windows; safe elsewhere.
    import multiprocessing as mp
    mp.freeze_support()
    # Keep spawn (recommended on macOS); avoid forcing "fork"
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # start method already set in this interpreter
        pass
    main()