import gc

import torch
from torch.utils.data import DataLoader

from src.config import train_config
from src.run.eval_loop import eval_nip
from src.run.eval_module import EvalNIPModel
from src.run.train_loop import train
from src.run.train_module import TrainNIPModel
from src.utils.catalog_size import get_catalog_size
from src.utils.log_q import get_q_counts
from src.utils.remap import remap_histories_and_targets
from src.yambda.collate_fn import collate_fn
from src.yambda.dataset import get_data
from src.yambda.eval_dataset import YambdaEvalDataset
from src.yambda.train_dataset import YambdaTrainDataset

train_df, test_df, item_to_idx = get_data()

catalog_size = get_catalog_size(item_to_idx=item_to_idx)

histories, targets = remap_histories_and_targets(train_df, test_df)

yambda_train_dataset = YambdaTrainDataset(histories=histories)
yambda_eval_dataset = YambdaEvalDataset(histories=histories, targets=targets)

yambda_train_dataloader = DataLoader(
    dataset=yambda_train_dataset,
    batch_size=train_config.TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
)

yambda_eval_dataloader = DataLoader(
    dataset=yambda_eval_dataset,
    batch_size=train_config.EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
)

q = get_q_counts(
    yambda_train_dataset=yambda_train_dataset,
    catalog_size=catalog_size,
    yambda_train_len=len(yambda_train_dataset)
)

gc.collect()
torch.cuda.empty_cache()

train_graph = TrainNIPModel(
    num_items=catalog_size,
    embedding_dim=64,
    num_negatives=512,
    q_counts=q,
).to(train_config.DEVICE)
optimizer = torch.optim.Adam(params=train_graph.parameters(), lr=train_config.LEARNING_RATE)

checkpoint, epoch_losses = train(
    dataloader=yambda_train_dataloader,
    model=train_graph,
    optimizer=optimizer,
    num_epochs=train_config.NUM_EPOCHS,
    device=train_config.DEVICE,
)

test_dataset = yambda_eval_dataset
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=train_config.EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
)

test_graph = EvalNIPModel(
    num_items=catalog_size,
    embedding_dim=64,
).to(train_config.DEVICE)


final_metrics_nip = eval_nip(
    dataloader=test_dataloader,
    model=test_graph,
    catalog_size=catalog_size,
    topk=train_config.TOPK,
    device=train_config.DEVICE,
)
print(final_metrics_nip)
