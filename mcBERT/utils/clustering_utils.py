import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


def get_plot_as_img(train_emd, val_emd, labels_train, labels_val):
    sort_train = np.argsort(labels_train)
    hues = [list(labels_train[sort_train]) + list(labels_val)]
    styles = ["Train"] * len(labels_train) + ["Val"] * len(labels_val)

    # Plot mean embeddings
    X = np.concatenate([train_emd[sort_train], val_emd])
    X_cosine_sim = cosine_similarity(X)
    cosine_dist = 1 - X_cosine_sim
    cosine_dist = np.abs(cosine_dist)
    # X_embedded = TSNE(metric="precomputed", init="random").fit_transform(cosine_dist)
    X_embedded = UMAP(
        n_components=2, init="random", random_state=0, metric="cosine", n_jobs=1
    ).fit_transform(cosine_dist)
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=hues[0], style=styles)
    return fig


def apply_UMAP(X):
    # use UMAP to project the data onto a 2D plane
    umap_2d = UMAP(n_components=2, init="random", random_state=0, metric="cosine")
    X_embedded = umap_2d.fit_transform(X)

    return X_embedded
