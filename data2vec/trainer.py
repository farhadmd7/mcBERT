"""
Train Data2Vec for mcRNA-seq data. Adapted from the original data2vec text trainer.
"""

from glob import glob

import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data2vec import Data2Vec
from data2vec.utils import AverageMeter, maybe_save_checkpoint
from mcBERT.encoder import McBERT_Encoder
from mcBERT.utils.patient_level_dataset import Patient_level_dataset
from mcBERT.utils.utils import prepare_dataset


class mcRNA_Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_epochs = self.cfg.train.num_epochs
        self.device = self.cfg.device
        self.ckpt_dir = cfg.train.checkpoints_dir
        self.save_ckpt_freq = cfg.train.save_ckpt_freq
        # Model, Optim, Criterion
        self.encoder = McBERT_Encoder(cfg, out_as_dict=True)
        self.model = Data2Vec(encoder=self.encoder, cfg=cfg)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.optimizer.lr)
        self.criterion = nn.SmoothL1Loss(reduction="none", beta=cfg.criterion.loss_beta)
        self.criterion.to(self.device)

        # Datasets & Data Loaders
        all_patients_files = glob(cfg.H5AD_FILES + '/*.h5ad')

        train_files, test_files = train_test_split(all_patients_files, test_size=0.2)
        
        df_train = prepare_dataset(
            train_files,
            multiprocess=True,
            sample_key=self.cfg.dataset.sample_key,
            condition_key=self.cfg.dataset.condition_key,
        )
        df_test = prepare_dataset(
            test_files,
            multiprocess=True,
            sample_key=self.cfg.dataset.sample_key,
            condition_key=self.cfg.dataset.condition_key,
        )
        
        self.train_dataset = Patient_level_dataset(
            df_train,
            select_gene_path=cfg.HIGHLY_VAR_GENES_PATH,
            n_cells=1023,
            inference=True,
            cell_type_key=self.cfg.dataset.cell_type_key
        )
        self.test_dataset = Patient_level_dataset(
            df_test,
            select_gene_path=cfg.HIGHLY_VAR_GENES_PATH,
            n_cells=1023,
            inference=True,
            cell_type_key=self.cfg.dataset.cell_type_key
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            collate_fn=self.test_dataset.collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=cfg.train.eval_batch_size,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.train_dataset.collate_fn,
        )

        # Tensorboard
        self.tensorboard = SummaryWriter(log_dir=self.cfg.train.log_dir)

        # Trackers
        self.loss_tracker = AverageMeter("loss")
        self.var_tracker_x = AverageMeter("variance_x")
        self.var_tracker_y = AverageMeter("variance_y")

    def train_step(self, batch):
        """
        Train one batch of data and return loss.

        Args:
            batch: A batch of data, inputs, labels and mask with shape [batch_size, seq_len]

        Returns:
            Loss value
        """
        src, trg, mask = batch
        src, trg, mask = src.to(self.device), trg.to(self.device), mask.to(self.device)

        x, y = self.model(src, trg, mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        var_pred = self.compute_var(x)
        var_target = self.compute_var(y)

        return loss.item(), var_pred, var_target

    def test_step(self, batch):
        """
        Test a model on one batch of data and return loss.

        Args:
            batch: A batch of data, inputs, labels and mask with shape [batch_size, seq_len]

        Returns:
            Loss value
        """
        src, trg, mask = batch
        src, trg, mask = src.to(self.device), trg.to(self.device), mask.to(self.device)

        x, y = self.model(src, trg, mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))

        var_pred = self.compute_var(x)
        var_target = self.compute_var(y)

        return loss.item(), var_pred, var_target

    def train_epoch(self, epoch_num):
        """
        Train the model for one epoch and verbose using the progress bar.

        Args:
            epoch_num: number of the current epoch

        Returns:
            The average loss through the whole epoch
        """
        self.model.train()
        self.loss_tracker.reset()
        self.var_tracker_x.reset()
        self.var_tracker_y.reset()
        with tqdm(
            self.train_loader,
            unit="batch",
            desc=f"Epoch: {epoch_num}/{self.num_epochs} ",
            bar_format="{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}",
            ascii=" #",
        ) as iterator:
            for batch in iterator:
                loss, var_pred, var_target = self.train_step(batch)
                self.model.ema_step()
                self.loss_tracker.update(loss)
                self.var_tracker_x.update(var_pred)
                self.var_tracker_y.update(var_target)

                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)

        return avg_loss, self.var_tracker_x.avg, self.var_tracker_y.avg

    def evaluate(self):
        """
        Evaluate the model on the test set

        Returns:
            The average loss through the whole test dataset
        """
        self.model.eval()
        self.loss_tracker.reset()
        self.loss_tracker.reset()
        self.var_tracker_x.reset()
        self.var_tracker_y.reset()
        with tqdm(
            self.test_loader,
            unit="batch",
            desc=f"Evaluating... ",
            bar_format="{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}",
            ascii=" #",
        ) as iterator:
            with torch.no_grad():
                for batch in iterator:
                    loss, var_pred, var_target = self.test_step(batch)
                    self.loss_tracker.update(loss)

                    self.var_tracker_x.update(var_pred)
                    self.var_tracker_y.update(var_target)

                    avg_loss = self.loss_tracker.avg
                    iterator.set_postfix(loss=avg_loss)

        return avg_loss, self.var_tracker_x.avg, self.var_tracker_y.avg

    def train(self):
        """
        Train and evaluate the model on the datasets and save checkpoints and write summaries to TensorBoard.

        """
        for epoch in range(1, self.num_epochs + 1):
            print()
            train_loss, train_pred_var, train_target_var = self.train_epoch(epoch)
            val_loss, val_pred_var, val_target_var = self.evaluate()

            # tensorboard
            self.tensorboard.add_scalar("train_loss", train_loss, epoch)
            self.tensorboard.add_scalar("train_pred_var", train_pred_var, epoch)
            self.tensorboard.add_scalar("train_target_var", train_target_var, epoch)

            self.tensorboard.add_scalar("val_loss", val_loss, epoch)
            self.tensorboard.add_scalar("val_pred_var", val_pred_var, epoch)
            self.tensorboard.add_scalar("val_target_var", val_target_var, epoch)

            # save checkpoint
            maybe_save_checkpoint(
                self.model, self.optimizer, self.ckpt_dir, epoch, self.save_ckpt_freq
            )

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        return torch.sqrt(y.var(dim=0) + 1e-6).mean()
