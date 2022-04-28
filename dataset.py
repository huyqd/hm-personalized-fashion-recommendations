import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sparse
import torch
from torch.utils.data import DataLoader

from utils import Column, IOPath

CURRENT_PATH = Path(__file__).cwd()


def get_csr_matrix(df):
    csr = sparse.csr_matrix(
        (
            df[Column.N_INTERACTIONS].tolist(),
            (df[Column.CUSTOMER_ID].values, df[Column.ARTICLE_ID].values),
        ),
    )

    return csr


def get_mapping():
    articles = pd.read_csv(
        IOPath.ARTICLES, dtype={Column.ARTICLE_ID: str}
    )
    all_articles = articles[Column.ARTICLE_ID].unique().tolist()
    articles_mapping = dict(list(enumerate(all_articles)))
    articles_mapping = {v: k for k, v in articles_mapping.items()}
    with open(IOPath.ARTICLES_MAPPING, "w") as f:
        json.dump(articles_mapping, f)

    customers = pd.read_csv(
        IOPath.CUSTOMERS
    )
    all_customers = customers[Column.CUSTOMER_ID].unique().tolist()

    customers_mapping = dict(list(enumerate(all_customers)))
    customers_mapping = {v: k for k, v in customers_mapping.items()}
    with open(IOPath.CUSTOMERS_MAPPING, "w") as f:
        json.dump(customers_mapping, f)

    return articles_mapping, customers_mapping


def interaction_matrix_split(reduce=True):
    interaction_df = pd.read_csv(IOPath.TRANSACTIONS_TRAIN,
                                 dtype={Column.ARTICLE_ID: str},
                                 index_col=[0],
                                 parse_dates=True)
    if reduce:
        interaction_df = interaction_df.loc["2020-08-21":]

    interaction_df[Column.N_INTERACTIONS] = 1

    with open(IOPath.ARTICLES_MAPPING, "r") as f:
        articles_mapping = json.load(f)

    with open(IOPath.CUSTOMERS_MAPPING, "r") as f:
        customers_mapping = json.load(f)

    interaction_df[Column.ARTICLE_ID] = interaction_df[Column.ARTICLE_ID].map(articles_mapping)
    interaction_df[Column.CUSTOMER_ID] = interaction_df[Column.CUSTOMER_ID].map(customers_mapping)

    last_date = interaction_df.index.max()
    train = interaction_df.loc[: (last_date - pd.offsets.Day(14))]
    val = interaction_df.loc[(last_date - pd.offsets.Day(13)): (last_date - pd.offsets.Day(7))]
    test = interaction_df.loc[(last_date - pd.offsets.Day(6)):]

    assert val.index.nunique() == test.index.nunique() == 7
    assert train.index.max() + pd.offsets.Day(1) == val.index.min()
    assert val.index.max() + pd.offsets.Day(1) == test.index.min()

    train = (
        train.groupby([Column.CUSTOMER_ID, Column.ARTICLE_ID])[Column.N_INTERACTIONS]
            .sum()
            .reset_index()
    )
    train = train[[Column.CUSTOMER_ID, Column.ARTICLE_ID, Column.N_INTERACTIONS]]
    val = val.drop_duplicates(subset=[Column.CUSTOMER_ID, Column.ARTICLE_ID])[
        [Column.CUSTOMER_ID, Column.ARTICLE_ID, Column.N_INTERACTIONS]
    ]
    test = test.drop_duplicates(subset=[Column.CUSTOMER_ID, Column.ARTICLE_ID])[
        [Column.CUSTOMER_ID, Column.ARTICLE_ID, Column.N_INTERACTIONS]
    ]

    IOPath.INTERACTION_MATRIX.mkdir(parents=True, exist_ok=True)
    train_csr = get_csr_matrix(train)
    test_csr = get_csr_matrix(test)
    val_csr = get_csr_matrix(val)

    np.savez_compressed(
        IOPath.INTERACTION_MATRIX / "data.npz",
        train_csr=train_csr,
        val_csr=val_csr,
        test_csr=test_csr,
    )

    return


class HMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, n_negative_samples=4, n_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.n_negative_samples = n_negative_samples
        self.n_workers = n_workers
        data = np.load(IOPath.INTERACTION_MATRIX / "data.npz", allow_pickle=True)
        self.train_csr = data['train_csr'].item()[:1000]
        self.val_csr = data['val_csr'].item()[:1000]
        self.test_csr = data['test_csr'].item()[:1000]

        self.train_sparse = None
        self.train_score = None
        self.test_users = None
        self.test_items = None
        self.test_labels = None

        self.train_ds = None
        self.test_ds = None

        self.train_steps = None
        self.test_steps = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.val_users, self.val_items, self.val_labels = self._negative_sampling(self.val_csr)

        if stage == "test" or stage is None:
            self.test_users, self.test_items, self.test_labels = self._negative_sampling(self.test_csr)

    def _negative_sampling(self, csr):
        user_idx, item_idx, values = sparse.find(csr)
        user_ids = np.unique(user_idx)
        n_items = csr.shape[1]
        item_range = np.arange(n_items)

        users = []
        items = []
        labels = []
        for user in user_ids:
            pos_items = item_idx[user_idx == user]
            n_pos_items = pos_items.shape[0]

            # Sample negative items
            n_neg_items = self.n_negative_samples * n_pos_items
            sampling_prob = np.ones(n_items)
            sampling_prob[pos_items] = 0  # Don't sample positive items
            sampling_prob /= sampling_prob.sum()
            neg_items = np.random.choice(item_range, size=n_neg_items, replace=True, p=sampling_prob)

            users.append(np.repeat(user, repeats=n_pos_items + n_neg_items))
            items.append(np.concatenate([pos_items, neg_items]))
            labels.append(np.concatenate([np.ones(n_pos_items), np.zeros(n_neg_items)]))

        users = np.concatenate(users)
        items = np.concatenate(items)
        labels = np.concatenate(labels)

        return users, items, labels

    def train_dataloader(self):
        self.train_users, self.train_items, self.train_labels = self._negative_sampling(self.train_csr)
        self.train_ds = HMDataset(self.train_users, self.train_items, self.train_labels)

        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=True)

    def val_dataloader(self):
        self.val_ds = HMDataset(self.val_users, self.val_items, self.val_labels)

        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=True)

    def test_dataloader(self):
        self.test_ds = HMDataset(self.test_users, self.test_items, self.test_labels)

        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=True)


class HMDataset(torch.utils.data.Dataset):
    def __init__(self, users, items, labels):
        self.users = torch.from_numpy(users)
        self.items = torch.from_numpy(items)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def __len__(self):
        return self.users.size()[0]


if __name__ == '__main__':
    interaction_matrix_split()
