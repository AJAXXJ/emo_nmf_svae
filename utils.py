import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from torchmetrics import F1Score, MetricCollection, Precision, Recall


@dataclass
class VAEAblationConfig:
    use_sparse: bool = True
    use_nmf_weights: bool = True
    use_align_loss: bool = True
    use_kl: bool = True
    finetune_bert_stage2: bool = True


def preprocess_data(normalization, data):
    if normalization == "vst_minmax":
        transformed = np.log(data + np.sqrt(data ** 2 + 1))
        return MinMaxScaler().fit_transform(transformed)
    if normalization == "minmax":
        return MinMaxScaler().fit_transform(data)
    if normalization == "zscore":
        return StandardScaler().fit_transform(data)
    if normalization == "l1":
        return normalize(data, norm="l1", axis=1)
    if normalization == "l2":
        return normalize(data, norm="l2", axis=1)
    raise ValueError(f"unsupported normalization: {normalization}")


def multi_classification_metrics(num_emotions):
    kwargs = {"num_labels": num_emotions, "task": "multilabel"}
    return MetricCollection({
        "precision_macro": Precision(average="macro", **kwargs),
        "recall_macro": Recall(average="macro", **kwargs),
        "f1_macro": F1Score(average="macro", **kwargs),
        "f1_micro": F1Score(average="micro", **kwargs),
    })


class Classifier(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, num_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.fc(self.drop(x))


class ZGatedBERT(nn.Module):
    def __init__(self, bert_dim, z_dim, num_heads=8, init_alpha=0.5):
        super().__init__()
        if not 0.0 < init_alpha < 1.0:
            raise ValueError(f"init_alpha must be in (0, 1), got {init_alpha}")
        self.num_heads = num_heads
        self.head_dim = bert_dim // num_heads
        self.z2gate = nn.Linear(z_dim, bert_dim)
        self.gate_activation = nn.Sigmoid()
        self.alpha_tilde = nn.Parameter(
            torch.full((num_heads,), math.log(init_alpha / (1.0 - init_alpha)))
        )

    def forward(self, bert_emb, z):
        gate = self.gate_activation(self.z2gate(z))
        bert_heads = bert_emb.view(-1, self.num_heads, self.head_dim)
        gate_heads = gate.view(-1, self.num_heads, self.head_dim)
        alpha = torch.sigmoid(self.alpha_tilde).view(1, self.num_heads, 1)
        gated_heads = alpha * bert_heads + (1 - alpha) * (bert_heads * gate_heads)
        return gated_heads.view(-1, bert_emb.size(-1))


def load_model(model_class, config):
    ckpt_path = os.path.join(config.checkpoints, "last.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = model_class(config)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(config.device)
    model.eval()
    return model
