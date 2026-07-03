import os
from typing import Any

import numpy as np
import polars as pl

from src.config import train_config

PATH_INTERACTIONS = os.path.join(train_config.DATA_DIR, "interactions.parquet")
PATH_EMBEDDINGS = os.path.join(train_config.DATA_DIR, "embeddings.parquet")
PATH_ARTISTS = os.path.join(train_config.DATA_DIR, "artists.parquet")

np.random.seed(42)


def get_data() -> tuple[pl.DataFrame, pl.DataFrame, dict[Any, int]]:
    interactions = pl.read_parquet(PATH_INTERACTIONS)
    embeddings = pl.read_parquet(PATH_EMBEDDINGS)
    artists = pl.read_parquet(PATH_ARTISTS)

    embeddings_items = embeddings.select(pl.col("item_id").unique())
    data = interactions.join(embeddings_items, on="item_id", how="semi")
    item_counts = data["item_id"].value_counts()
    popular_items = item_counts.filter(pl.col("count") >= train_config.CORE_MIN_INTERACTIONS_PER_ITEM).select("item_id")
    data = data.join(popular_items, on="item_id", how="semi")
    data = data.join(artists, on="item_id", how="left")
    max_ts = data["timestamp"].max()
    test_start_ts = max_ts - train_config.TEST_INTERVAL_SECONDS
    train_df = data.filter(pl.col("timestamp") < test_start_ts)
    test_df = data.filter(pl.col("timestamp") >= test_start_ts)

    all_items = pl.concat([train_df.select("item_id"), test_df.select("item_id")]).unique()
    item_to_idx = {old: idx for idx, old in enumerate(all_items["item_id"].to_list())}
    return train_df, test_df, item_to_idx