import argparse
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch_frame import stype
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame.datasets import DataFrameBenchmark
from torch_frame.transforms import CatToNumTransform
from torch_frame import TaskType
import torch_geometric.nn as pyg_nn


parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, choices=['binary_classification', 'multiclass_classification', 'regression'], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'], default='small')
parser.add_argument('--idx', type=int, default=0, help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--mixup', type=str, default=None, choices=[None, 'feature', 'hidden'])
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--graph_channels', type=int, default=128)
parser.add_argument('--batch', type=int, default=2500)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--compile', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The device we will use is {device}')

path = osp.join(os.getcwd(), 'data')

dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                            scale=args.scale, idx=args.idx)
dataset.materialize()
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

is_classification = dataset.task_type.is_classification

if is_classification:
    out_channels = dataset.num_classes
else:
    out_channels = 1

is_binary_class = is_classification and out_channels == 2

if is_binary_class:
    metric_train = AUROC(task='binary')
    metric_val   = AUROC(task='binary')
    metric_test  = AUROC(task='binary')
elif is_classification:
    metric_train = Accuracy(task='multiclass', num_classes=out_channels)
    metric_val   = Accuracy(task='multiclass', num_classes=out_channels)
    metric_test  = Accuracy(task='multiclass', num_classes=out_channels)
else:
    metric_train = MeanSquaredError()
    metric_val   = MeanSquaredError()
    metric_test  = MeanSquaredError()
metric_train = metric_train.to(device)
metric_val   = metric_val.to(device)
metric_test  = metric_test.to(device)

num_rows = dataset.num_rows

x = torch.cat([
    train_tensor_frame.feat_dict[stype.numerical], 
    val_tensor_frame.feat_dict[stype.numerical],
    test_tensor_frame.feat_dict[stype.numerical]
], dim=0)
assert x.shape[0] == num_rows, f"Expected {num_rows} rows, but got {x.shape[0]}"

y = torch.cat([
    train_tensor_frame.y, 
    val_tensor_frame.y, 
    test_tensor_frame.y
], dim=0)
assert y.shape[0] == num_rows, f"Expected {num_rows} rows, but got {y.shape[0]}"


train_size = train_tensor_frame.y.shape[0]
val_size = val_tensor_frame.y.shape[0]
test_size = test_tensor_frame.y.shape[0]

train_mask = torch.zeros(num_rows, dtype=torch.bool)
val_mask = torch.zeros(num_rows, dtype=torch.bool)
test_mask = torch.zeros(num_rows, dtype=torch.bool)

train_mask[:train_size] = True
val_mask[train_size:train_size + val_size] = True
test_mask[train_size + val_size:] = True

from graph import MTAM_KNN
edge_builder = MTAM_KNN(metric='cosine', num_threads=8)
edge_index = edge_builder.fit(X=x, base_k=3)
print(f'Total # of edges:  {edge_index.shape[1]}.')

from torch_geometric.data import Data
from torch_geometric.utils import degree
full_data = Data(x=x, edge_index=edge_index, y=y).contiguous()
row, col = full_data.edge_index
degree_row = degree(row, full_data.num_nodes)
degree_col = degree(col, full_data.num_nodes)
full_data.edge_weight = 1.0 / torch.sqrt(degree_row[row] * degree_col[col])
full_data.train_mask = train_mask
full_data.val_mask = val_mask
full_data.test_mask = test_mask
full_data.node_id = torch.arange(full_data.num_nodes)

from torch_geometric.loader import GraphSAINTRandomWalkSampler
train_loader = GraphSAINTRandomWalkSampler(full_data, batch_size=args.batch, walk_length=2, num_steps=args.num_steps, sample_coverage=100)

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=args.graph_channels):
        super().__init__()
        self.conv1 = pyg_nn.GraphConv(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = pyg_nn.GraphConv(in_channels=hidden_channels, out_channels=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x

import math
from torch import Tensor
from torch.nn.init import _calculate_correct_fan, calculate_gain

def attenuated_kaiming_uniform_(
    tensor: Tensor,
    scale: float = 0.1,
    a: float = math.sqrt(5),
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu',
) -> Tensor:
    r"""Attenuated Kaiming Uniform Initialization.

    Args:
        tensor (tensor): Input tensor to be initialized
        scale (float): Positive rescaling constant to the variance.
        a (float): Negative slope of the rectifier used after this layer
        mode (str): Either 'fan_in' (default) or 'fan_out'. Choosing
        'fan_in' preserves the magnitude of the variance of the weights
        in the forward pass. Choosing 'fan_out' preserves the magnitudes
        in the backwards pass.
        nonlinearity (str) : the non-linear function (nn.functional name),
                    recommended to use only with 'relu' or 'leaky_relu'.
    """
    with torch.no_grad():
        fan = _calculate_correct_fan(tensor, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain * scale / math.sqrt(fan)
        # Calculate uniform bounds from standard deviation
        bound = math.sqrt(3.0) * std
        return tensor.uniform_(-bound, bound)

from typing import Any

from torch.nn import ModuleDict

import torch_frame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import FeatureEncoder
from torch_frame.nn.encoder.stype_encoder import StypeEncoder


class StypeWiseFeatureEncoder(FeatureEncoder):
    r"""Feature encoder that transforms each stype tensor into embeddings and
    performs the final concatenation.

    Args:
        out_channels (int): Output dimensionality.
        col_stats
            (dict[str, dict[:class:`torch_frame.data.stats.StatType`, Any]]):
            A dictionary that maps column name into stats. Available as
            :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            A dictionary that maps :class:`torch_frame.stype` into
            :class:`torch_frame.nn.encoder.StypeEncoder` class. Only
            parent :class:`stypes <torch_frame.stype>` are supported
            as keys.
    """
    def __init__(
        self,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder],
    ) -> None:
        super().__init__()

        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.encoder_dict = ModuleDict()
        for stype, stype_encoder in stype_encoder_dict.items():
            if stype != stype.parent:
                if stype.parent in stype_encoder_dict:
                    msg = (
                        f"You can delete this {stype} directly since encoder "
                        f"for parent stype {stype.parent} is already declared."
                    )
                else:
                    msg = (f"To resolve the issue, you can change the key from"
                            f" {stype} to {stype.parent}.")
                raise ValueError(f"{stype} is an invalid stype to use in the "
                                    f"stype_encoder_dcit. {msg}")
            if stype not in stype_encoder.supported_stypes:
                raise ValueError(
                    f"{stype_encoder} does not support encoding {stype}.")

            if stype in col_names_dict:
                stats_list = [
                    self.col_stats[col_name]
                    for col_name in self.col_names_dict[stype]
                ]
                # Set lazy attributes
                stype_encoder.stype = stype
                stype_encoder.out_channels = out_channels
                stype_encoder.stats_list = stats_list
                self.encoder_dict[stype.value] = stype_encoder

    def forward(
        self,
        data: Data,
        mixup_encoded: bool = False,
    ) -> Tensor:
        # Here only left for numerical
        x = self.encoder_dict[torch_frame.numerical.value](data, mixup_encoded)
        return x

from abc import ABC, abstractmethod

from torch.nn import (
    Embedding,
    EmbeddingBag,
    ModuleList,
    Parameter,
    ParameterList,
    Sequential,
)
from torch.nn.init import kaiming_uniform_
from torch_frame import NAStrategy, stype
from torch_frame.nn.encoder.stypewise_encoder import StypeEncoder


class ExcelFormerEncoder(StypeEncoder):
    r"""An attention based encoder that transforms input numerical features
    to a 3-dimensional tensor.

    Before being fed to the embedding layer, numerical features are normalized
    and categorical features are transformed into numerical features by the
    CatBoost Encoder implemented with the Sklearn Python package. The features
    are then ranked based on mutual information.
    The original encoding is described in
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.


    Args:
        out_channels (int): The output channel dimensionality.
        stats_list (list[dict[StatType, Any]]): The list of stats for each
            column within the same stype.
    """

    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: list[dict[StatType, Any]] | None = None,
        stype: stype | None = None,
        post_module: torch.nn.Module | None = None,
        na_strategy: NAStrategy | None = None,
    ):
        super().__init__(out_channels, stats_list, stype, post_module,
                        na_strategy)

    def init_modules(self) -> None:
        super().init_modules()
        mean = torch.tensor(
            [stats[StatType.MEAN] for stats in self.stats_list])
        self.register_buffer("mean", mean)
        std = (torch.tensor([stats[StatType.STD]
                            for stats in self.stats_list]) + 1e-6)
        self.register_buffer("std", std)
        num_cols = len(self.stats_list)
        
        # GNNEncoder
        self.My_Encoder = GNNEncoder(in_channels=num_cols, out_channels=num_cols)
        
        self.W_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.W_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_1 = Parameter(Tensor(num_cols, self.out_channels))
        self.b_2 = Parameter(Tensor(num_cols, self.out_channels))
        self.reset_parameters()

    def forward(
        self,
        data: Data,
        mixup_encoded: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        feat = data.x
        feat = self.na_forward(feat)
        feat = (feat - self.mean) / self.std
        
        feat_gnn = self.My_Encoder(x=feat, edge_index=data.edge_index, edge_weight=data.edge_weight)
        
        x1 = self.W_1[None] * feat_gnn[:, :, None] + self.b_1[None]
        x2 = self.W_2[None] * feat_gnn[:, :, None] + self.b_2[None]
        x = torch.tanh(x1) * x2
        
        x = torch.nan_to_num(x, nan=0)

        return self.post_forward(x)

    def encode_forward(self) -> None:
        return None


    def reset_parameters(self) -> None:
        super().reset_parameters()
        attenuated_kaiming_uniform_(self.W_1)
        attenuated_kaiming_uniform_(self.W_2)
        kaiming_uniform_(self.b_1, a=math.sqrt(5))
        kaiming_uniform_(self.b_2, a=math.sqrt(5))

from torch.nn import Module, ModuleList
from torch_frame.nn.decoder import ExcelFormerDecoder
from torch_frame.typing import NAStrategy

def feature_mixup(
    x: Tensor,
    y: Tensor,
    num_classes: int,
    beta: float | Tensor = 0.5,
    mixup_type: str | None = None,
    mi_scores: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    assert num_classes > 0
    assert mixup_type in [None, 'feature', 'hidden']

    beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
    beta_distribution = torch.distributions.beta.Beta(beta, beta)
    shuffle_rates = beta_distribution.sample(torch.Size((len(x), 1)))
    shuffled_idx = torch.randperm(len(x), device=x.device)
    assert x.ndim == 3, """
    FEAT-MIX or HIDDEN-MIX is for encoded numerical features
    of size [batch_size, num_cols, in_channels]."""
    b, f, d = x.shape
    if mixup_type == 'feature':
        assert mi_scores is not None
        mi_scores = mi_scores.to(x.device)
        # Hard mask (feature dimension)
        mixup_mask = torch.rand(torch.Size((b, f)),
                                device=x.device) < shuffle_rates
        # L1 normalized mutual information scores
        norm_mi_scores = mi_scores / mi_scores.sum()
        # Mixup weights
        lam = torch.sum(
            norm_mi_scores.unsqueeze(0) * mixup_mask, dim=1, keepdim=True)
        mixup_mask = mixup_mask.unsqueeze(2)
    elif mixup_type == 'hidden':
        # Hard mask (hidden dimension)
        mixup_mask = torch.rand(torch.Size((b, d)),
                                device=x.device) < shuffle_rates
        mixup_mask = mixup_mask.unsqueeze(1)
        # Mixup weights
        lam = shuffle_rates
    else:
        # No mixup
        mixup_mask = torch.ones_like(x, dtype=torch.bool)
        # Fake mixup weights
        lam = torch.ones_like(shuffle_rates)
    x_mixedup = mixup_mask * x + ~mixup_mask * x[shuffled_idx]    
    y_shuffled = y[shuffled_idx]
    if num_classes == 1:
        # Regression task or binary classification
        lam = lam.squeeze(1)
        y_mixedup = lam * y + (1 - lam) * y_shuffled
    else:
        # Classification task
        one_hot_y = F.one_hot(y, num_classes=num_classes)
        one_hot_y_shuffled = F.one_hot(y_shuffled, num_classes=num_classes)
        y_mixedup = (lam * one_hot_y + (1 - lam) * one_hot_y_shuffled)
    return x_mixedup, y_mixedup

from torch.nn import Dropout, LayerNorm, Linear, Module
from torch.nn.init import zeros_

from torch_frame.nn.conv import TableConv


def init_attenuated(linear: Linear) -> None:
    attenuated_kaiming_uniform_(linear.weight)
    zeros_(linear.bias)


class AiuM(Module):
    r"""Attentive Intra-feature Update Module.

    Args:
        channels (int): Input channel dimensionality
        dropout (float): Percentage of random deactivation in the AiuM module
    """
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.lin_1 = Linear(channels, channels)
        self.lin_2 = Linear(channels, channels)
        self.dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_attenuated(self.lin_1)
        init_attenuated(self.lin_2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(torch.tanh(self.lin_1(x)) * (self.lin_2(x)))
        return x


class DiaM(Module):
    r"""Directed Inter-feature Attention Module.

    Args:
        channels (int): Input channel dimensionality
        num_cols (int): Number of columns
        num_heads (int): Number of heads in the attention module
        dropout (float): Percentage of random deactivation in the DiaM module
    """
    def __init__(self, channels: int, num_cols: int, num_heads: int,
                    dropout: float) -> None:
        if num_heads > 1:
            assert channels % num_heads == 0
        super().__init__()
        self.lin_q = Linear(channels, channels)
        self.lin_k = Linear(channels, channels)
        self.lin_v = Linear(channels, channels)
        self.lin_out = Linear(channels, channels) if num_heads > 1 else None
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        self.seq_ids: Tensor
        self.register_buffer('seq_ids', torch.arange(num_cols))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in [self.lin_q, self.lin_k, self.lin_v]:
            init_attenuated(lin)
        if self.lin_out is not None:
            init_attenuated(self.lin_out)

    def _reshape(self, x: Tensor) -> Tensor:
        B, num_cols, channels = x.shape
        d_head = channels // self.num_heads
        x = x.reshape(B, num_cols, self.num_heads, d_head)
        x = x.transpose(1, 2)
        x = x.reshape(B * self.num_heads, num_cols, d_head)
        return x

    def forward(
        self,
        x: Tensor,
        mixup_encoded: bool = False,
    ) -> Tensor:
        B, num_cols, _ = x.shape
        Q, K, V = self.lin_q(x), self.lin_k(x), self.lin_v(x)
        d_heads = V.shape[-1] // self.num_heads
        Q = self._reshape(Q)
        K = self._reshape(K)
        attention_score = torch.einsum('ijk, ilk->ijl', Q, K)
        scaled_attention_score = attention_score / math.sqrt(d_heads)
        attention_probs = F.softmax(scaled_attention_score, dim=-1)
        attention = self.dropout(attention_probs)
        x = torch.einsum('ijk, ikl->ijl', attention, self._reshape(V))
        x = x.reshape(B, self.num_heads, num_cols, d_heads)
        x = x.transpose(1, 2)
        x = x.reshape(B, num_cols, self.num_heads * d_heads)
        if self.lin_out is not None:
            x = self.lin_out(x)
        return x

class ExcelFormerConv(TableConv):
    r"""The ExcelFormer Layer introduced in the
    `"ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
    <https://arxiv.org/abs/2301.02819>`_ paper.

    Args:
        channels (int): Input/output channel dimensionality.
        num_cols (int): Number of columns.
        num_heads (int): Number of attention heads.
        diam_dropout (float): diam_dropout. (default: 0)
        aium_dropout (float): aium_dropout. (default: 0)
        residual_dropout (float): residual dropout. (default: 0)
    """
    def __init__(
        self,
        channels: int,
        num_cols: int,
        num_heads: int,
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ) -> None:

        super().__init__()
        self.norm_1 = LayerNorm(channels)
        self.DiaM = DiaM(channels, num_cols, num_heads, diam_dropout)
        self.norm_2 = LayerNorm(channels)
        self.AiuM = AiuM(channels, aium_dropout)
        self.residual_dropout = residual_dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.DiaM.reset_parameters()
        self.AiuM.reset_parameters()


    def forward(
        self,
        x: Tensor,
        mixup_encoded: bool = False,
    ) -> Tensor:
        x = self.norm_1(x)
        x_residual = self.DiaM(x=x, mixup_encoded=mixup_encoded)
        x = F.dropout(x_residual, self.residual_dropout, self.training) + x
        x_residual = self.norm_2(x)
        x_residual = self.AiuM(x_residual)
        x = F.dropout(x_residual, self.residual_dropout, self.training) + x
        return x

class ExcelFormer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cols: int,
        num_layers: int,
        num_heads: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        diam_dropout: float = 0.0,
        aium_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        mixup: str | None = None,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        assert mixup in [None, 'feature', 'hidden']

        self.in_channels = in_channels
        self.out_channels = out_channels
        if col_names_dict.keys() != {stype.numerical}:
            raise ValueError("ExcelFormer only accepts numerical "
                            "features.")

# The space we want to develop:
# Do not modify this block
############################################################################################
        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.numerical:
                ExcelFormerEncoder(out_channels, na_strategy=NAStrategy.MEAN)
            }

        assert set(stype_encoder_dict.keys()) == {stype.numerical}, f'Expected a only numerical, but got else.'

        self.excelformer_encoder = StypeWiseFeatureEncoder(
            out_channels=self.in_channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        
############################################################################################
        self.excelformer_convs = ModuleList([
            ExcelFormerConv(in_channels, num_cols, num_heads, diam_dropout,
                            aium_dropout, residual_dropout)
            for _ in range(num_layers)
        ])
        self.excelformer_decoder = ExcelFormerDecoder(in_channels,
                                                    out_channels, num_cols)
        
        self.reset_parameters()
        self.mixup = mixup
        self.beta = beta

    def reset_parameters(self) -> None:
        self.excelformer_encoder.reset_parameters()
        
        for excelformer_conv in self.excelformer_convs:
            excelformer_conv.reset_parameters()
        self.excelformer_decoder.reset_parameters()

    def forward(
        self,
        data: Data,
        mixup_encoded: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Remove col_names
        x = self.excelformer_encoder(data, mixup_encoded)
        # FEAT-MIX or HIDDEN-MIX is compatible with `torch.compile`
        if mixup_encoded:
            assert data.y is not None
            x, y_mixedup = feature_mixup(
                x,
                data.y,
                num_classes=self.out_channels,
                beta=self.beta,
                mixup_type=self.mixup,
            )
        for excelformer_conv in self.excelformer_convs:
            x = excelformer_conv(x=x, mixup_encoded=mixup_encoded)
        out = self.excelformer_decoder(x)
        
        if mixup_encoded:
            return out, y_mixedup
        return out

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
    # col_stats=mutual_info_sort.transformed_stats,
    col_stats=col_stats,
    col_names_dict=train_tensor_frame.col_names_dict,
).to(device)
model = torch.compile(model, dynamic=True) if args.compile else model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma)

def train(epoch: int) -> float:
    model.train()
    loss_accum = total_count = 0

    for data in tqdm(train_loader, desc=f'Epoch: {epoch}', leave=True):
        optimizer.zero_grad()
        
        data = data.to(device)
        mask = data.train_mask
        data.edge_weight = data.edge_norm * data.edge_weight
        # Train with FEAT-MIX or HIDDEN-MIX
        pred_mixedup, y_mixedup = model(data, mixup_encoded=True)
        if is_classification:
            # Softly mixed one-hot labels
            loss = F.cross_entropy(pred_mixedup[mask], y_mixedup[mask], reduction='none')
        else:
            loss = F.mse_loss(pred_mixedup[mask].view(-1), y_mixedup[mask].view(-1), reduction='none')
        loss = (loss * data.node_norm[mask]).sum()
        loss.backward()
        optimizer.step()
        
        loss_accum += float(loss) * len(data.y)
        total_count += len(data.y)
    return loss_accum / total_count

@torch.no_grad()
def test(data: Data) -> tuple[float, float, float]:
    model.eval()
    metric_train.reset()
    metric_val.reset()
    metric_test.reset()
    
    data = data.to(device)
    pred = model(data)
    
    if is_binary_class:
        metric_train.update(pred[data.train_mask, 1], data.y[data.train_mask])
        metric_val.update(pred[data.val_mask, 1], data.y[data.val_mask])
        metric_test.update(pred[data.test_mask, 1], data.y[data.test_mask])
    elif is_classification:
        pred_train = pred[data.train_mask].argmax(dim=-1)
        pred_val = pred[data.val_mask].argmax(dim=-1)
        pred_test = pred[data.test_mask].argmax(dim=-1)
        metric_train.update(pred_train, data.y[data.train_mask])
        metric_val.update(pred_val, data.y[data.val_mask])
        metric_test.update(pred_test, data.y[data.test_mask])
    else:
        metric_train.update(pred[data.train_mask].view(-1), data.y[data.train_mask].view(-1))
        metric_val.update(pred[data.val_mask].view(-1), data.y[data.val_mask].view(-1))
        metric_test.update(pred[data.test_mask].view(-1), data.y[data.test_mask].view(-1))

    if is_classification:
        train_metric = metric_train.compute().item()
        val_metric = metric_val.compute().item()
        test_metric = metric_test.compute().item()
    else:
        train_metric = metric_train.compute().item() ** 0.5
        val_metric = metric_val.compute().item() ** 0.5
        test_metric = metric_test.compute().item() ** 0.5

    return train_metric, val_metric, test_metric


if is_classification:
    metric = 'Acc' if not is_binary_class else 'AUC'
    best_val_metric = 0
    best_test_metric = 0
else:
    metric = 'RMSE'
    best_val_metric = float('inf')
    best_test_metric = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    train_metric, val_metric, test_metric = test(full_data)

    if is_classification and val_metric > best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric
    elif not is_classification and val_metric < best_val_metric:
        best_val_metric = val_metric
        best_test_metric = test_metric

    print(f'Train Loss: {train_loss:.4f}, Train {metric}: {train_metric:.4f}, '
        f'Val {metric}: {val_metric:.4f}, Test {metric}: {test_metric:.4f}')
    lr_scheduler.step()

print(f'The ENCODER_state -> Task Type: {args.task_type}; Idx: {args.idx}; Seed: {args.seed}')
print(f'Best Test {metric}: {best_test_metric:.4f}')
