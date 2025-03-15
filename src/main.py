import os

import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from columns import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from dataset import AliExpressDataset
from esmm import ESMM

TRAIN_CSV_PATH = "data/train_sampled_0.1.csv"


def train(model, train_dataloader, optimizer, criterion):
    model.train()
    train_loss_ctr = 0.0
    train_loss_ctcvr = 0.0

    for batch in tqdm(train_dataloader):
        categorical_feat, numerical_feats, click, cv = batch

        click = click.to(torch.float32)
        cv = cv.to(torch.float32)

        categorical_feat = categorical_feat.squeeze(1)
        numerical_feats = numerical_feats.squeeze(1)

        p_ctr, p_cvr = model(categorical_feat, numerical_feats)

        p_ctr = p_ctr.squeeze(1)
        p_cvr = p_cvr.squeeze(1)
        p_ctcvr = p_ctr * p_cvr

        loss_ctr = criterion(p_ctr, click)
        loss_ctcvr = criterion(p_ctcvr, cv)

        loss_total = loss_ctr + loss_ctcvr

        train_loss_ctr += loss_ctr.item()
        train_loss_ctcvr += loss_ctcvr.item()

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
    return {
        "train_loss_ctr": train_loss_ctr,
        "train_loss_ctcvr": train_loss_ctcvr,
    }


def evaluate(model, valid_dataloader, criterion):
    model.eval()
    total_ctr_preds = []
    total_ctcvr_preds = []
    total_click_labels = []
    total_ctcvr_labels = []

    with torch.no_grad():
        valid_loss_ctr = 0.0
        valid_loss_ctcvr = 0.0
        for batch in tqdm(valid_dataloader):
            categorical_feat, numerical_feats, click, cv = batch

            click = click.to(torch.float32)
            cv = cv.to(torch.float32)

            categorical_feat = categorical_feat.squeeze(1)
            numerical_feats = numerical_feats.squeeze(1)

            p_ctr, p_cvr = model(categorical_feat, numerical_feats)

            p_ctr = p_ctr.squeeze(1)
            p_cvr = p_cvr.squeeze(1)
            p_ctcvr = p_ctr * p_cvr

            loss_ctr = criterion(p_ctr, click)
            loss_ctcvr = criterion(p_ctcvr, cv)

            # loss_total = loss_ctr + loss_ctcvr

            valid_loss_ctr += loss_ctr
            valid_loss_ctcvr += loss_ctcvr

            p_ctr = p_ctr.cpu().numpy()
            p_ctcvr = p_ctcvr.cpu().numpy()

            total_ctr_preds.extend(p_ctr.flatten())
            total_ctcvr_preds.extend(p_ctcvr.flatten())
            total_click_labels.extend(click.flatten())
            total_ctcvr_labels.extend(cv.flatten())

    # auc of ctr task
    ctr_auc = roc_auc_score(total_click_labels, total_ctr_preds)

    # auc of ctcvr task
    ctcvr_auc = roc_auc_score(total_ctcvr_labels, total_ctcvr_preds)

    return {
        "ctr_auc": ctr_auc,
        "ctcvr_auc": ctcvr_auc,
        "ctr_preds": total_ctr_preds,
        "ctcvr_preds": total_ctcvr_preds,
        "click_labels": total_click_labels,
        "ctcvr_labels": total_ctcvr_labels,
    }


def main():

    # preprocess dataset
    df = pl.read_csv(TRAIN_CSV_PATH)

    for col_name in CATEGORICAL_COLUMNS:
        unique_values = df[col_name].unique()
        unique_values.sort()
        mapping_dict = {value: i for i, value in enumerate(unique_values)}
        df = df.with_columns(pl.col(col_name).replace(mapping_dict))

    df = df.with_columns(
        cs.starts_with("numerical_").cast(pl.Float32),
        pl.col("click").cast(pl.Float32),
        pl.col("conversion").cast(pl.Float32),
    )
    train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42)

    train_dataset = AliExpressDataset(train_df, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS)
    valid_dataset = AliExpressDataset(valid_df, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS)

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    embedding_sizes = [
        (len(df[col_name].unique()), 4) for col_name in CATEGORICAL_COLUMNS
    ]
    hidden_dims = [64, 32]
    task_tower_hidden_dims = [16]

    model = ESMM(
        embedding_sizes=embedding_sizes,
        num_numerical=63,
        hidden_dims=hidden_dims,
        task_tower_hidden_dims=task_tower_hidden_dims,
    )

    lr = 1e-3
    epoch = 5
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    for e in range(epoch):
        print(f"epoch : {e} / {epoch}")
        train_results = train(model, train_dataloader, optimizer, criterion)
        print(f"Train CTR loss : {train_results["train_loss_ctr"]}")
        print(f"Train CTCVR loss : {train_results["train_loss_ctcvr"]}")
        valid_results = evaluate(model, valid_dataloader, criterion)
        # print(valid_results)
        print(f"Valid CTR AUC : {valid_results["ctr_auc"]}")
        print(f"Valid CTCVR AUC : {valid_results["ctcvr_auc"]}")
        print()


if __name__ == "__main__":
    main()
