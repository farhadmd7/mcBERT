import numpy as np
import torch
from sklearn.metrics import silhouette_score


def cosine_similarity_patient_embeddings(
    train_embeddings: torch.Tensor,
    test_embeddings: torch.Tensor,
    train_diseases: np.ndarray,
    test_diseases: np.ndarray,
) -> tuple[float, float]:
    """Implementation of the mean cosine similarity metric for patient embeddings.
    Returns both the mean cosine distance of same diseased patients as well as differently diseased patients.


    Args:
        train_embeddings (torch.Tensor): Training embeddings of patients
        test_embeddings (torch.Tensor): Test embeddings of patients to be compared with training embeddings
        train_diseases (np.ndarray): List of disease labels of training patients
        test_diseases (np.ndarray): List of disease labels of test patients

    Returns:
        tuple[float, float]: Mean cosine distance of same diseased patients and differently diseased patients
    """

    # calculate cosine similarity of all validation vs training embeddings
    train_embeddings = torch.divide(train_embeddings.T, train_embeddings.norm(dim=1)).T
    test_embeddings = torch.divide(test_embeddings.T, test_embeddings.norm(dim=1)).T
    cosine_sim = torch.mm(test_embeddings, train_embeddings.T)

    # Prepare labels and create boolean matrix for same and different labels
    label_sim = (
        np.expand_dims(test_diseases, axis=1) == np.expand_dims(train_diseases, axis=0)
    ).astype(int)

    # calculate mean cosine distance
    cosine_dist = np.array(1 - cosine_sim.cpu())

    # calculate mean cosine distance of label_sim=1 and label_sim=0
    mean_same_cosine_dist = np.abs(
        (cosine_dist - (1 - label_sim))[label_sim == 1]
    ).mean()
    mean_diff_cosine_dist = np.abs(
        (cosine_dist - (1 - label_sim))[label_sim < 0.5]
    ).mean()

    return mean_same_cosine_dist, mean_diff_cosine_dist


def calc_silhouette_score(embeddings: torch.Tensor, labels: np.ndarray) -> float:
    """Calculates the silhouette score of the embeddings based on the labels.

    Args:
        embeddings (torch.Tensor): Embeddings of the patients
        labels (np.ndarray): Labels of the patients

    Returns:
        float: Silhouette score of the embeddings
    """
    embeddings = torch.divide(embeddings.T, embeddings.norm(dim=1)).T
    cosine_sim = torch.mm(embeddings, embeddings.T)
    cosine_dist = np.array(1 - cosine_sim.cpu())
    cosine_dist[cosine_dist < 0] = 0
    silhouette_score_res = silhouette_score(cosine_dist, labels, metric="precomputed")
    return silhouette_score_res
