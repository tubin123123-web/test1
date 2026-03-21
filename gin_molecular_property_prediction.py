"""
分子性质预测 GIN（Graph Isomorphism Network）模型
=====================================================
本文件实现了一个基于 GIN 的分子图神经网络，用于预测分子性质（如溶解度、毒性等）。

依赖：
    pip install torch torch-geometric rdkit-pypi

用法示例：
    python gin_molecular_property_prediction.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 原子与键的特征提取
# ---------------------------------------------------------------------------

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),          # 原子序数 1-118
    "degree": list(range(0, 11)),               # 成键数
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_hs": list(range(0, 9)),                # 氢原子数
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "is_aromatic": [True, False],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "is_conjugated": [True, False],
    "is_in_ring": [True, False],
}


def one_hot_encoding(value, choices: list) -> List[int]:
    """将值编码为 one-hot 向量；若不在列表中则归入最后一类。"""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def atom_features(atom) -> List[float]:
    """提取单个原子的特征向量。"""
    feats = (
        one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + one_hot_encoding(atom.GetDegree(), ATOM_FEATURES["degree"])
        + one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
        + one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
    )
    return feats


def bond_features(bond) -> List[float]:
    """提取单条键的特征向量。"""
    feats = (
        one_hot_encoding(bond.GetBondType(), BOND_FEATURES["bond_type"])
        + one_hot_encoding(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"])
        + one_hot_encoding(bond.IsInRing(), BOND_FEATURES["is_in_ring"])
    )
    return feats


def _compute_bond_feature_dim() -> int:
    dim = 0
    for choices in BOND_FEATURES.values():
        dim += len(choices) + 1  # +1 for "其他" 类别
    return dim


BOND_FEATURE_DIM = _compute_bond_feature_dim()


def mol_to_graph(smiles: str, label: Optional[float] = None) -> Optional[Data]:
    """
    将 SMILES 字符串转换为 PyTorch Geometric 的 Data 对象。

    Args:
        smiles: 分子的 SMILES 表示。
        label:  分子的性质标签（可选）。

    Returns:
        PyTorch Geometric Data 对象，若 SMILES 无效则返回 None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 原子特征
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float
    )

    # 边索引与键特征（无向图：每条键存两次）
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bf, bf]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if label is not None:
        data.y = torch.tensor([[label]], dtype=torch.float)

    return data


# ---------------------------------------------------------------------------
# GIN 卷积层
# ---------------------------------------------------------------------------

class GINConv(MessagePassing):
    """
    GIN（Graph Isomorphism Network）卷积层。

    消息传递公式：
        h_v^{(k)} = MLP^{(k)}( (1 + ε) · h_v^{(k-1)}
                               + Σ_{u ∈ N(v)} h_u^{(k-1)} )

    参考：
        Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019.
    """

    def __init__(self, in_channels: int, out_channels: int, eps: float = 0.0,
                 train_eps: bool = True):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=train_eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 消息传递 + 聚合
        agg = self.propagate(edge_index, x=x)
        # (1 + ε) · self + 邻居聚合
        out = self.mlp((1 + self.eps) * x + agg)
        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


# ---------------------------------------------------------------------------
# GIN 分子性质预测模型
# ---------------------------------------------------------------------------

class GINMoleculeNet(nn.Module):
    """
    基于 GIN 的分子性质预测模型。

    架构：
        输入原子特征
        → N 层 GINConv（每层均有 BN + ReLU）
        → 全局 readout（求和 or 平均）
        → MLP 分类/回归头

    Args:
        in_channels:   输入原子特征维度。
        hidden_channels: 隐藏层维度。
        out_channels:  输出维度（回归=1，分类=类别数）。
        num_layers:    GIN 层数。
        dropout:       Dropout 比例。
        pool:          图级 readout 方式，"sum" 或 "mean"。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 1,
        num_layers: int = 5,
        dropout: float = 0.5,
        pool: str = "mean",
    ):
        super().__init__()

        # 原子特征投影层（将不规则长度的 one-hot 映射到固定维度）
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GIN 卷积层堆叠
        self.convs = nn.ModuleList(
            [GINConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.pool = global_mean_pool if pool == "mean" else global_add_pool

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 投影到隐藏维度
        x = F.relu(self.input_proj(x))

        # 图卷积
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)

        # 图级 readout
        x = self.pool(x, batch)

        # 预测
        out = self.head(x)
        return out


# ---------------------------------------------------------------------------
# 训练与评估工具
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    评估模型，返回 (平均损失, RMSE)。
    适用于回归任务；分类任务请替换评估指标。
    """
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    rmse = torch.sqrt(F.mse_loss(preds, labels)).item()
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, rmse


# ---------------------------------------------------------------------------
# 计算原子特征维度（根据 ATOM_FEATURES 定义自动推算）
# ---------------------------------------------------------------------------

def _compute_atom_feature_dim() -> int:
    dim = 0
    for choices in ATOM_FEATURES.values():
        dim += len(choices) + 1  # +1 for "其他" 类别
    return dim


ATOM_FEATURE_DIM = _compute_atom_feature_dim()


# ---------------------------------------------------------------------------
# 演示：使用若干 SMILES 构造数据集并训练
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 示例分子（SMILES）及对应性质标签（此处为模拟数据）
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",   # 阿司匹林
        "c1ccccc1",                  # 苯
        "CCO",                       # 乙醇
        "CC(=O)O",                   # 乙酸
        "C1CCCCC1",                  # 环己烷
        "c1ccc(cc1)O",               # 苯酚
        "CC(C)O",                    # 异丙醇
        "CCCC",                      # 正丁烷
        "c1ccncc1",                  # 吡啶
        "CC(=O)N",                   # 乙酰胺
    ]
    # 模拟溶解度标签（logS）
    labels = [-1.2, -1.8, -0.3, -0.5, -2.1, -0.9, -0.4, -2.5, -0.7, -0.2]

    # 构建图数据集
    dataset = [mol_to_graph(smi, lab) for smi, lab in zip(smiles_list, labels)]
    dataset = [d for d in dataset if d is not None]
    print(f"有效分子数量：{len(dataset)}")

    # 划分训练集 / 测试集（8:2）
    split = int(0.8 * len(dataset))
    train_data, test_data = dataset[:split], dataset[split:]

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINMoleculeNet(
        in_channels=ATOM_FEATURE_DIM,
        hidden_channels=128,
        out_channels=1,
        num_layers=4,
        dropout=0.3,
        pool="mean",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"\n模型架构：\n{model}\n")
    print(f"原子特征维度：{ATOM_FEATURE_DIM}\n")

    # 训练循环
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % 10 == 0:
            test_loss, test_rmse = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:3d} | 训练损失: {train_loss:.4f} "
                f"| 测试损失: {test_loss:.4f} | 测试 RMSE: {test_rmse:.4f}"
            )

    print("\n训练完成！")
