import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame.data.loader import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.nn import ExcelFormer
from torch_frame.transforms import CatToNumTransform, MutualInformationSort
from torch_frame import TaskType

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--mixup', type=str, default = None,
                    choices=[None, 'feature', 'hidden'])
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--compile', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)


print(f'The device we will use is {device}')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

print(f'The path is {path}')

dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                            scale=args.scale, idx=args.idx)

dataset.materialize()
print(f'Data has been already downloaded')
train_dataset, val_dataset, test_dataset = dataset.split()
train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame
# CategoricalCatBoostEncoder encodes the categorical features
# into numerical features with CatBoostEncoder.
categorical_transform = CatToNumTransform()
categorical_transform.fit(train_dataset.tensor_frame, train_dataset.col_stats)

train_tensor_frame = categorical_transform(train_tensor_frame)
val_tensor_frame = categorical_transform(val_tensor_frame)
test_tensor_frame = categorical_transform(test_tensor_frame)
col_stats = categorical_transform.transformed_stats

# MutualInformationSort sorts the features based on mutual
# information.
mutual_info_sort = MutualInformationSort(task_type=dataset.task_type)

mutual_info_sort.fit(train_tensor_frame, col_stats)
train_tensor_frame = mutual_info_sort(train_tensor_frame)
val_tensor_frame = mutual_info_sort(val_tensor_frame)
test_tensor_frame = mutual_info_sort(test_tensor_frame)

train_loader = DataLoader(train_tensor_frame, batch_size=args.batch_size,
                        shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=args.batch_size)
test_loader = DataLoader(test_tensor_frame, batch_size=args.batch_size)

is_classification = dataset.task_type.is_classification

if is_classification:
    out_channels = dataset.num_classes
else:
    out_channels = 1

is_binary_class = is_classification and out_channels == 2

if is_binary_class:
    metric_computer = AUROC(task='binary')
elif is_classification:
    metric_computer = Accuracy(task='multiclass', num_classes=out_channels)
else:
    metric_computer = MeanSquaredError()
metric_computer = metric_computer.to(device)

model = ExcelFormer(
    in_channels=args.channels,
    out_channels=out_channels,
    num_layers=args.num_layers,
    num_cols=train_tensor_frame.num_cols,
    num_heads=args.num_heads,
    residual_dropout=0.,
    diam_dropout=0.3,
    aium_dropout=0.,
    mixup=args.mixup,
    col_stats=mutual_info_sort.transformed_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)
model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma)


def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(train_loader, desc=f'Epoch: {epoch}', leave=False):
        tf = tf.to(device)
        # Train with FEAT-MIX or HIDDEN-MIX
        pred_mixedup, y_mixedup = model(tf, mixup_encoded=True)
        if is_classification:
            # Softly mixed one-hot labels
            loss = F.cross_entropy(pred_mixedup, y_mixedup)
        else:
            loss = F.mse_loss(pred_mixedup.view(-1), y_mixedup.view(-1))
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(y_mixedup)
        total_count += len(y_mixedup)
        optimizer.step()
    return loss_accum / total_count


@torch.no_grad()
def test(loader: DataLoader) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if is_binary_class:
            metric_computer.update(pred[:, 1], tf.y)
        elif is_classification:
            pred_class = pred.argmax(dim=-1)
            metric_computer.update(pred_class, tf.y)
        else:
            metric_computer.update(pred.view(-1), tf.y.view(-1))

    if is_classification:
        return metric_computer.compute().item()
    else:
        return metric_computer.compute().item()**0.5


if is_classification:
    metric = 'Acc' if not is_binary_class else 'AUC'
    best_val_metric = 0
    best_test_metric = 0
else:
    metric = 'RMSE'
    best_val_metric = float('inf')
    best_test_metric = float('inf')

# import time
# start_time = time.time()

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric = test(train_loader)
    val_metric = test(val_loader)
    test_metric = test(test_loader)

    if is_classification and val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
        f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
    lr_scheduler.step()

# end_time = time.time()
# elapsed_time = end_time - start_time


print(f'The ExcelFormer_state -> Task Type: {args.task_type}; Idx: {args.idx}; Seed: {args.seed}')
print(f'Best Test {metric}: {best_test_metric:.4f}')
# print(f'Total training time: {elapsed_time:.2f} sec')
