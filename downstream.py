import re

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import MetricCollection

from utils import Classifier, ZGatedBERT, load_model, multi_classification_metrics


class EmotionMultiClassificationLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(vars(config))

        self.emo_vae = load_model(self.hparams.model, config)
        hidden_dim = self.emo_vae.bert.config.hidden_size

        self._freeze_stage1(finetune_bert=getattr(
            self.hparams.ablation_config, "finetune_bert_stage2", False
        ))

        self.layer_norm = nn.LayerNorm(self.hparams.n_modules, elementwise_affine=False)
        self.fusion = ZGatedBERT(hidden_dim, self.hparams.n_modules)
        self.classifier = Classifier(hidden_dim, self.hparams.dropout_rate, self.hparams.num_emotions)

        self.emotion_criterion = nn.BCEWithLogitsLoss()

        self.train_metrics = multi_classification_metrics(self.hparams.num_emotions).clone(prefix="train_")
        self.val_metrics = multi_classification_metrics(self.hparams.num_emotions).clone(prefix="val_")
        self.test_metrics = multi_classification_metrics(self.hparams.num_emotions).clone(prefix="test_")

    def _freeze_stage1(self, finetune_bert: bool):
        for p in self.emo_vae.vae.parameters():
            p.requires_grad = False
        for p in self.emo_vae.emo_embedding_head.parameters():
            p.requires_grad = False
        for p in self.emo_vae.bert.parameters():
            p.requires_grad = finetune_bert

    def train(self, mode=True):
        super().train(mode)
        self.emo_vae.vae.eval()
        self.emo_vae.emo_embedding_head.eval()
        if not getattr(self.hparams.ablation_config, "finetune_bert_stage2", False):
            self.emo_vae.bert.eval()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.emo_vae.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        c = out.last_hidden_state[:, 0, :]
        mu = self.emo_vae.mu_from_cls(c)
        z = self.layer_norm(mu) if re.search(r"EmoVAE", self.hparams.model_name) else mu
        return self.classifier(self.fusion(c, z))

    def shared_step(self, batch):
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids", None),
        )
        labels = (batch["labels"] > 0).float()
        loss = self.emotion_criterion(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).long()
        return loss, preds, labels

    def _shared_metric_step(self, batch, stage):
        loss, preds, labels = self.shared_step(batch)
        metrics: MetricCollection = getattr(self, f"{stage}_metrics")
        metrics.update(preds, labels)
        self.log_dict(metrics, on_epoch=(stage != "train"), prog_bar=(stage == "train"))
        self.log(f"{stage}_loss", loss, on_epoch=(stage != "train"), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_metric_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_metric_step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch)
        self.test_metrics.update(preds, labels)
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        return AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
            betas=(0.9, 0.999),
        )
