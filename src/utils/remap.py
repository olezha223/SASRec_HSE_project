from typing import Any

import polars as pl


def remap_histories_and_targets(train_df, test_df):
    histories = {
        row["uid"]: list(row["item_id"])
        for row in train_df.group_by("uid", maintain_order=True)
        .agg(pl.col("item_id").sort_by("timestamp"))
        .iter_rows(named=True)
    }

    raw_targets = dict(test_df.group_by("uid").agg(pl.col("item_id")).iter_rows())
    targets = {
        uid: t
        for uid, t in raw_targets.items()
        if uid in histories and len(histories[uid]) > 0
    }
    return histories, targets


def remap_item_ids(df: pl.DataFrame, item_to_idx: dict[Any, int]) -> pl.DataFrame:
    return df.with_columns(
        pl.col("item_id").replace(item_to_idx).alias("item_id")
    )

def remap_train_test_df(train_df, test_df, item_to_idx):
    train_df = remap_item_ids(train_df, item_to_idx)
    test_df = remap_item_ids(test_df, item_to_idx)
    return train_df, test_df
