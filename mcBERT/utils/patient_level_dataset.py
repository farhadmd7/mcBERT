import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset


class Patient_level_dataset(Dataset):
    """Class for the patient level dataset, used for training the data2vec and fine-tuning model."""

    def __init__(
        self,
        df,
        select_gene_path,
        inference=False,
        n_cells=1023,
        oversampling=False,
        mlm_probability=0.15,
        random_cell_stratification=0.1,
    ):
        super(Patient_level_dataset, self).__init__()

        self.select_genes = np.array(pd.read_csv(select_gene_path)["genes"])
        self.num_genes = len(self.select_genes)
        self.inference = inference
        self.n_cells = n_cells
        self.df = df

        # Automatically detect the column name of the cell identifier from the given set of columns
        self.cell_id_name = list(
            set(sc.read_h5ad(df["file_path"][0], backed="r").obs.columns).union(
                set(["mycelltypes", "Cell_Type", "cell_type"])
            )
        )[0]

        if oversampling:
            self.donor_oversampling()

        self.donors = self.df["donor_id"]
        self.donor_disease = self.df["disease"].values
        self.donor_disease_num = pd.Categorical(self.donor_disease).codes

        # [CLS] and [MASK] token are one-hot encoded using the first two indices
        self.mask_token, self.cls_token = torch.zeros(self.num_genes + 2), torch.zeros(
            self.num_genes + 2
        )
        self.cls_token[0] = 1
        self.mask_token[1] = 1
        self.mlm_probability = mlm_probability
        self.random_cell_stratification = random_cell_stratification

    def __len__(self):
        return len(self.donors)

    def __getitem__(self, index: int):
        """Method for enumerator of the dataset.
        NOTE: During inference, only the selected genes are returned, without any additional information.
        During training, the selected genes as well as the disease code and disease is returned.
        """

        # Randomly select n_cells embedded genes of idx file
        selected_X_1 = torch.zeros(self.n_cells + 1, self.num_genes + 2)
        idx_file = sc.read_h5ad(self.df["file_path"][index], backed="r")[
            :, self.select_genes
        ]

        cell_selection_idx = self.stratified_cell_selection(idx_file)
        gene_selection = torch.tensor(idx_file.chunk_X(cell_selection_idx))
        idx_file.file.close()

        selected_X_1[1:, 2:] = gene_selection

        # CLS Token
        selected_X_1[0, 0] = 1

        del idx_file

        if self.inference:
            return selected_X_1

        else:
            return (
                selected_X_1,
                self.donor_disease_num[index],
                self.donor_disease[index],
            )

    def stratified_cell_selection(self, h5ad_file: sc.AnnData) -> list:
        """Stratified cell selection, to balance the dataset by selecting the same number of cells from each cell type.
        Using self.random_cell_stratification to add some randomness to the selection (10% by default)

        Args:
            h5ad_file (sc.AnnData): h5ad file to select cells from

        Returns:
            list: list of indices of startified selected cells
        """

        cell_selection_idx = []

        if "cell_id" in h5ad_file.obs.columns:
            cell_identifier = "cell_id"
        elif "Cell_type" in h5ad_file.obs.columns:
            cell_identifier = "Cell_type"
        elif "mycelltypes" in h5ad_file.obs.columns:
            cell_identifier = "mycelltypes"
        elif "cell_type" in h5ad_file.obs.columns:
            cell_identifier = "cell_type"
        elif "Celltype" in h5ad_file.obs.columns:
            cell_identifier = "Celltype"

        cell_count = h5ad_file.obs[cell_identifier].value_counts()
        cells_added = 0
        for cell_type, counts in zip(cell_count.index, cell_count):
            if cells_added >= self.n_cells:
                break
            num_cells_select = int(
                (
                    (self.n_cells / len(h5ad_file))
                    + np.random.uniform(
                        -self.random_cell_stratification,
                        self.random_cell_stratification,
                    )
                )
                * counts
            )
            num_cells_select = min(max(num_cells_select, 0), self.n_cells - cells_added)
            cell_selection_idx.extend(
                np.random.choice(
                    np.where(h5ad_file.obs[cell_identifier] == cell_type)[0],
                    num_cells_select,
                )
            )
            cells_added += num_cells_select
        cell_selection_idx.extend(
            np.random.choice(len(h5ad_file), self.n_cells - cells_added)
        )

        return cell_selection_idx

    def donor_oversampling(self):
        """Balance Dataset through oversampling, by duplicating the donors with less represented diseases in the self.donors list"""

        disease_count = self.df["disease"].value_counts()
        max_count = disease_count.max()
        for disease, count in disease_count.items():
            if count == max_count:
                continue
            donors_to_duplicate = self.df[self.df["disease"] == disease]
            donors_to_duplicate = donors_to_duplicate.sample(
                n=max_count - count, replace=True
            )
            self.df = pd.concat([self.df, donors_to_duplicate])
        self.df.reset_index(drop=True, inplace=True)

    def _mask_tokens(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Masking function for BERT masking, needed for D2V training

        Args:
            inputs (torch.Tensor): unmasked input tensor of shape [batch_size, num_cells, num_genes + 2]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked input, unmasked input and masked indices
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape[:2], self.mlm_probability)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[:, 0] = False

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape[:2], 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.mask_token

        return inputs, labels, masked_indices

    def collate_fn(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate the batch of data using BERT masking strategy. carefully ported from
        transformers.data.DataCollatorForLanguageModeling

        Args:
            batch (torch.Tensor): unmaksed input tensor of multiple batched donors

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masekd input, unmasked input and masked indices
        """

        batch = torch.stack(batch)
        # If special token mask has been preprocessed, pop it from the dict.
        src, trg, masked_indices = self._mask_tokens(batch)

        # Prepend the "masked_indices"
        return src, trg, masked_indices
