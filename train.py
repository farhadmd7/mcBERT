import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
from mcBERT.utils.clustering_utils import get_plot_as_img
from mcBERT.utils.metrics import (
    calc_silhouette_score,
    cosine_similarity_patient_embeddings,
)
from mcBERT.utils.patient_level_dataset import Patient_level_dataset
from mcBERT.utils.utils import get_scRNA_model, prepare_dataset, set_seeds
from omegaconf import OmegaConf
from pytorch_metric_learning import losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

set_seeds(42)

"""
After pre-processing all datasets and saving each donor individually, the model can be trained.
Previous Pre-Training is recommended.
"""


# Config file for training
parser = ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="path to yaml config file for fine-tuning training",
)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

if not os.path.exists(cfg.train.checkpoints_dir):
    os.mkdir(cfg.train.checkpoints_dir)
if not os.path.exists(cfg.train.log_dir):
    os.mkdir(cfg.train.log_dir)

# Load files and prepare dataset
files = glob(cfg.H5AD_FILES)
if cfg.train.exclude_dataset != "":
    files = [file for file in files if cfg.train.exclude_dataset not in file]
df = prepare_dataset(files, multiprocess=True)
if "exclude_diseases" in cfg.train:
    df = df[~df["disease"].isin(cfg.train.exclude_diseases)]


# Drop all patients which disease only has one patient
df = df.groupby("disease").filter(lambda x: len(x) > 1)

if cfg.train.no_test_dataset:
    df_use, df_test = train_test_split(
        df, test_size=0.2, stratify=df["disease"], random_state=42
    )  # Note: df_test not used during fine-tuning! Only for later testing
    df_train, df_val = train_test_split(
        df_use, test_size=0.125, stratify=df_use["disease"], random_state=42
    )
else:
    df_train, df_val = train_test_split(
        df, test_size=0.2, stratify=df["disease"], random_state=42
    )

df_train.reset_index(inplace=True)
df_val.reset_index(inplace=True)
print(
    f"Using {len(df_train)} patients for training and {len(df_val)} patients for validation representing {len(df['disease'].unique())} unique disease"
)
print("Training diseases: ", df_train["disease"].unique())

ds_train = Patient_level_dataset(
    df_train,
    select_gene_path=cfg.HIGHLY_VAR_GENES_PATH,
    inference=False,
    n_cells=1023,
    oversampling=cfg.train.oversampling,
)
ds_val = Patient_level_dataset(
    df_val,
    select_gene_path=cfg.HIGHLY_VAR_GENES_PATH,
    inference=False,
    n_cells=1023,
    oversampling=cfg.val.oversampling,
)

dataloader_train = DataLoader(
    ds_train,
    batch_size=cfg.train.batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    shuffle=True,
)
dataloader_val = DataLoader(
    ds_val,
    batch_size=cfg.train.eval_batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    shuffle=False,
)

model = get_scRNA_model(cfg).cuda()
train_loss = losses.SupConLoss(temperature=0.1)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
)
writer = SummaryWriter(cfg.train.log_dir)

best_loss = np.inf
best_mean_cos_dist = 2
best_mean_cos_same_dist = 2
best_mean_cos_diff_dist = 2
best_silhouette_score_all = -1
best_silhouette_score_val = -1


##################
# START OF TRAINING LOOP
##################
for epoch in range(0, cfg.train.num_epochs + 1):
    running_loss = 0

    model.train()
    tqdm_loader_train = tqdm(dataloader_train, total=len(dataloader_train))

    # training loop
    for i, batch in enumerate(tqdm_loader_train):
        # prof.step()
        tqdm_loader_train.set_description(f"Epoch {epoch}, loss: {running_loss:.4f}")
        optimizer.zero_grad()
        x = batch[0].cuda()
        labels = batch[1].cuda()
        x = model(x)
        loss = train_loss(x, labels.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() / len(tqdm_loader_train)

    model.eval()

    # Calculate embeddings for all training samples again for later cosine similarity calculation
    tqdm_loader_train = tqdm(dataloader_train, total=len(dataloader_train))
    train_embeddings = torch.zeros((len(ds_train), cfg.model["embed_dim"])).cuda()
    train_diseases = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm_loader_train):
            x = batch[0].cuda()
            labels = batch[1].cuda()
            label_names = batch[2]
            x = model(x)

            train_embeddings[
                i * dataloader_train.batch_size : i * dataloader_train.batch_size
                + len(labels),
                :,
            ] = x.detach().cpu()
            train_diseases += label_names

    # Calculate embeddings for all validation samples
    # validation loop
    val_running_loss = 0
    tqdm_loader_val = tqdm(dataloader_val, total=len(dataloader_val))
    val_embeddings = torch.zeros((len(ds_val), cfg.model["embed_dim"])).cuda()
    val_diseases = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm_loader_val):
            tqdm_loader_val.set_description(
                f"Epoch {epoch}, val_loss: {val_running_loss:.4f}"
            )
            x = batch[0].cuda()
            labels = batch[1].cuda()
            label_names = batch[2]
            x = model(x)

            val_embeddings[
                i * dataloader_val.batch_size : i * dataloader_val.batch_size
                + len(labels),
                :,
            ] = x.detach().cpu()
            val_diseases += label_names

            loss = train_loss(x, labels.float())

            val_running_loss += loss.item() / len(tqdm_loader_val)

    # calculate cosine similarity of all validation vs training embeddings
    labels_train = np.array(train_diseases)
    labels_val = np.array(val_diseases)
    mean_same_cosine_dist, mean_diff_cosine_dist = cosine_similarity_patient_embeddings(
        train_embeddings, val_embeddings, labels_train, labels_val
    )
    mean_cosine_dist = 0.5 * mean_same_cosine_dist + 0.5 * mean_diff_cosine_dist

    # calculate Silhouette Scores
    silhouette_score_val = calc_silhouette_score(val_embeddings, labels_val)
    silhouette_score_train = calc_silhouette_score(train_embeddings, labels_train)
    silhouette_score_all = calc_silhouette_score(
        torch.cat([train_embeddings, val_embeddings], dim=0),
        np.concatenate([labels_train, labels_val]),
    )

    # Tensorboard logging
    writer.add_scalar("Loss/train", running_loss, epoch)
    writer.add_scalar("Loss/val", val_running_loss, epoch)
    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("Weight_decay", optimizer.param_groups[0]["weight_decay"], epoch)
    writer.add_scalar(
        "Mean Cosine Distance between val and train samples", mean_cosine_dist, epoch
    )
    writer.add_scalar("mCosDist same classes", mean_same_cosine_dist, epoch)
    writer.add_scalar("mCosDist diff classes", mean_diff_cosine_dist, epoch)
    writer.add_scalar("Silhouette Score Validation", silhouette_score_val, epoch)
    writer.add_scalar("Silhouette Score Train", silhouette_score_train, epoch)
    writer.add_scalar("Silhouette Score All", silhouette_score_all, epoch)

    # UMAP plot for Tensorboard
    if epoch % cfg.train.umap_frequency == 0:
        scatter_image = get_plot_as_img(
            np.array(train_embeddings.cpu()),
            np.array(val_embeddings.cpu()),
            labels_train,
            labels_val,
        )
        writer.add_figure("UMAP Plot", scatter_image, epoch)

    # Save model checkpoint based on different criteria
    if epoch % cfg.train.save_ckpt_freq == 0:
        torch.save(model.state_dict(), cfg.train.checkpoints_dir + f"/{epoch}.pt")

    if val_running_loss < best_loss:
        best_loss = val_running_loss
        torch.save(model.state_dict(), cfg.train.checkpoints_dir + "/val_best_loss.pt")

    if mean_cosine_dist < best_mean_cos_dist:
        best_mean_cos_dist = mean_cosine_dist
        torch.save(model.state_dict(), cfg.train.checkpoints_dir + "/best.pt")

    if mean_same_cosine_dist < best_mean_cos_same_dist:
        best_mean_cos_same_dist = mean_same_cosine_dist
        torch.save(model.state_dict(), cfg.train.checkpoints_dir + "/best_same.pt")

    if mean_diff_cosine_dist < best_mean_cos_diff_dist:
        best_mean_cos_diff_dist = mean_diff_cosine_dist
        torch.save(model.state_dict(), cfg.train.checkpoints_dir + "/best_diff.pt")

    if silhouette_score_all > best_silhouette_score_all:
        best_silhouette_score_all = silhouette_score_all
        torch.save(
            model.state_dict(),
            cfg.train.checkpoints_dir + "/best_silhouette_all.pt",
        )

    if silhouette_score_val > best_silhouette_score_val:
        best_silhouette_score_val = silhouette_score_val
        torch.save(
            model.state_dict(),
            cfg.train.checkpoints_dir + "/best_silhouette_val.pt",
        )
