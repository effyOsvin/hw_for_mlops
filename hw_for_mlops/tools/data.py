from typing import Optional

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .load_data import load_data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        X, y = load_data(self.cfg.X_path, self.cfg.y_path)
        X_train, X_val, y_train, y_val = train_test_split(
            X[: self.cfg.count_data],
            y[: self.cfg.count_data],
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )
        self.train_dataset = MyDataset(X_train, y_train)
        self.val_dataset = MyDataset(X_train, y_train)

    def train_dataloader(self, state="train") -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
        )
