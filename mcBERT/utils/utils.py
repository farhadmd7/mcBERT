import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from mcBERT.encoder import McBERT_Encoder


def get_diseases(file: str) -> list:
    """Returns the disease, patient_identifier, number of genes and file name of a h5ad file that only contains one patient.

    Args:
        file (str): Path to the h5ad file

    Returns:
        list: [disease, patient_identifier, number of genes, file name]
    """

    h5ad_file = sc.read_h5ad(file, backed="r")

    if "mypatients" in h5ad_file.obs.columns:
        patient_identifier = "mypatients"
    elif "donor_id" in h5ad_file.obs.columns:
        patient_identifier = "donor_id"
    elif "Donor" in h5ad_file.obs.columns:
        patient_identifier = "Donor"

    if "myconditions" in h5ad_file.obs.columns:
        condition_identifier = "myconditions"
    elif "Condition" in h5ad_file.obs.columns:
        condition_identifier = "Condition"
    elif "disease" in h5ad_file.obs.columns:
        condition_identifier = "disease"

    disease = h5ad_file.obs[condition_identifier].unique()[0]
    donor_id_file = (
        "_".join(file.split("_")[:-1])
        + "_"
        + h5ad_file.obs[patient_identifier].unique()[0]
    )
    h5ad_file_name = Path(file).stem.split(".h5ad")[0]

    return (
        disease,
        donor_id_file,
        len(h5ad_file.var.index),
        h5ad_file_name,
    )


def get_file_info(
    files: list, multiprocess: bool = False, n_jobs: int = 20
) -> list[list]:
    """Load file information:
    (disease, patient_identifier, number of genes, file name)

    Args:
        files (list): _description_
        multiprocess (bool, optional): Use multiprocessing Pool to faster load files. Defaults to False.
        n_jobs (int, optional): Num of Processes for multiprocessing. Defaults to 20.

    Returns:
        list[list]: List of file information
    """
    file_info = []

    if multiprocess:
        with Pool(n_jobs) as p:
            file_info = p.map(get_diseases, files, chunksize=5)
    else:
        for file in files:
            file_info.append(get_diseases(file))

    return file_info


def prepare_dataset(files, **kwargs):
    file_info = get_file_info(files, **kwargs)
    diseases = [info[0] for info in file_info]
    donor_ids = [info[1] for info in file_info]
    n_genes = [info[2] for info in file_info]
    dataset = [info[3] for info in file_info]

    df = pd.DataFrame(
        {
            "file_path": files,
            "disease": diseases,
            "donor_id": donor_ids,
            "n_genes": n_genes,
            "dataset": dataset,
        }
    )

    # Correct healthy labels
    df.loc[df["disease"] == "healthy", "disease"] = "Healthy"
    df.loc[df["disease"] == "normal", "disease"] = "Healthy"
    df.loc[df["disease"] == "control", "disease"] = "Healthy"
    df.loc[df["disease"] == "Ref", "disease"] = "Healthy"

    return df


def get_scRNA_model(cfg):
    model = McBERT_Encoder(cfg, out_as_dict=False)

    if "train" in cfg and cfg.train.pretrained:
        # Load Data2Vec pre-trained model. Need to rename the keys to match the scRNA model w/o pretraining
        state_dict_data2vec = torch.load(cfg.model.pre_train_ckpt)["data2vec"]
        for key in list(state_dict_data2vec.keys()):
            if "encoder." in key:
                state_dict_data2vec[key.replace("encoder.", "", 1)] = (
                    state_dict_data2vec.pop(key)
                )
            elif "regression_head" in key:
                del state_dict_data2vec[key]
        model.load_state_dict(state_dict_data2vec)

    return model


def set_seeds(seed: int):
    """Set a seed for reproducibility purposes.

    Args:
        seed (int): seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
