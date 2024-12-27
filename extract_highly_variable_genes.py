from glob import glob

import pandas as pd
import scanpy as sc

"""Script to extract the most highly variable genes across multiple dataset. Currently, to save memory each h5ad file is processed individually and
are not concatenated. This means that the highly variable genes are not calculated across all datasets but rather within each dataset.

NOTE: Since we process them individually, we cannot gurantee to get x (e.g., 1000) most highly variable genes.
Therefore, greedily increases top_genes_per_file until we get 1,000 genes.
"""

DATASETS_PATH = "..."  # Path to the folder containing all h5ad files
TOP_GENES_PER_FILE = 1000

highly_variable_genes = set()
genes_per_set = []

# Loop through all h5ad files and extract highly variable genes
for file in glob(DATASETS_PATH + "*.h5ad"):
    print(file)
    adata = sc.read_h5ad(file, chunk_size=20000)
    if adata.raw is not None:
        adata.X = adata.raw.X
    adata = sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=TOP_GENES_PER_FILE, inplace=False
    )  # BASED ON seurat_v3: https://www.sciencedirect.com/science/article/pii/S0092867419305598

    highly_variable_genes = highly_variable_genes.union(
        set(adata.index[adata.highly_variable])
    )
    genes_per_set.append(set(adata.index[adata.highly_variable]))

highly_variable_genes = highly_variable_genes.intersection(*genes_per_set)
df = pd.DataFrame({"genes": list(highly_variable_genes)})

# SAVE GENES Here
df.to_csv("...", index=False)
