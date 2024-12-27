from glob import glob

import numpy as np
import pandas as pd
import scanpy as sc

"""
After getting the most highly variable genes, we need to process the data to only include these genes and normalize each cell to sum up to 1.
Further, to not overload the RAM and assure scalability to multiple large datasets, each patient is extracted and saved in an individual file. 
"""

# The most highly variable genes
SELECTED_GENES = np.array(pd.read_csv(".csv")["genes"])
SAVE_PATH = r"..."
DONOR_COLUMN = "donor_id"


def process_donor(donor):
    global data, file

    print(f"Processing: {donor}")
    donor_h5 = data[data.obs[DONOR_COLUMN] == donor]
    save_path = SAVE_PATH + f"/{donor_h5.obs['Dataset'].unique()[0]}_DONOR_{donor}.h5ad"
    sc.write(save_path, donor_h5)
    print(f"Saved to {save_path}")


# Specify the datasets to be processed here:
for file in glob("...*.h5ad"):
    print(f"Processing: {file}")
    data = sc.read_h5ad(file, chunk_size=20000)
    data.X = data.raw.X
    file = file.raw.to_adata().to_memory()[:, SELECTED_GENES]
    data.obs[DONOR_COLUMN] = data.obs["Patient"].astype(str)

    sc.pp.normalize_total(data, target_sum=1, inplace=True)
    donors = data.obs[DONOR_COLUMN].unique()

    for donor in donors:
        process_donor(donor)
