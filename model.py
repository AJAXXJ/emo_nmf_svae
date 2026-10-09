import pickle

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel


class EmoVAE(nn.Module):
    def __init__(self, input_dim, z_dim, emotion_dim, hidden_dim, dropout_rate,
                 module_weights, device, ablation_config):
        super().__init__()
        self.device = device
        self.ablation_config = ablation_config

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

        if ablation_config.use_nmf_weights and module_weights is not None:
            self.decoder = nn.Linear(z_dim, emotion_dim, bias=False)
            self.decoder.weight.data = torch.tensor(module_weights.T, dtype=torch.float32)
            self.decoder.weight.requires_grad = False
        else:
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim * 2, emotion_dim),
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class EmoVAELit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.lr = self.hparams.lr
        self.dropout_rate = self.hparams.dropout_rate

        self.bert = AutoModel.from_pretrained(self.hparams.pretrained_model)
        if self.hparams.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.emo_embedding_head = nn.Linear(
            self.bert.config.hidden_size, self.hparams.num_vae_emotions
        )

        self.vae = EmoVAE(
            input_dim=self.hparams.num_vae_emotions,
            z_dim=self.hparams.n_modules,
            emotion_dim=self.hparams.num_vae_emotions,
            hidden_dim=self.hparams.vae_hidden_dim,
            dropout_rate=self.dropout_rate,
            module_weights=self._load_nmf_weights(),
            device=self.hparams.device,
            ablation_config=self.hparams.ablation_config,
        )

        self._reset_buffers()

    def _reset_buffers(self):
        self.val_z_list, self.val_logvar_list, self.val_z_sample_list = [], [], []
        self.val_recon_list, self.val_labels_list = [], []
        self.test_z_list, self.test_logvar_list, self.test_z_sample_list = [], [], []
        self.test_recon_list, self.test_labels_list = [], []

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.set_grad_enabled(not self.hparams.freeze_bert):
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True,
                return_dict=True,
            )
        attentions = bert_outputs.attentions
        text_embeddings = bert_outputs.last_hidden_state[:, 0, :]

        emo_embedding = self.emo_embedding_head(text_embeddings)
        recon, mu, logvar, z = self.vae(emo_embedding)
        return recon, mu, logvar, z, emo_embedding, attentions

    def mu_from_cls(self, cls_embedding):
        e = self.emo_embedding_head(cls_embedding)
        mu, _ = self.vae.encode(e)
        return mu

    def shared_step(self, batch, module_scores):
        recon, mu, logvar, z, emo_embedding, attentions = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids", None),
        )
        losses = self.calculate_loss(
            emo_embedding, recon, batch["labels"], mu, logvar, module_scores
        )
        return (recon, mu, logvar, z), losses

    def calculate_loss(self, emo_embedding, recon, target, mu, logvar, module_scores=None):
        emotion_loss = ((emo_embedding - target) ** 2).sum(dim=1).mean()
        recon_loss = ((recon - emo_embedding.detach()) ** 2).sum(dim=1).mean()

        if self.hparams.ablation_config.use_kl:
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        else:
            kl_loss = torch.zeros((), device=recon.device)

        if self.hparams.ablation_config.use_sparse:
            sparse_loss = mu.abs().sum(dim=1).mean()
        else:
            sparse_loss = torch.zeros((), device=recon.device)

        if self.hparams.ablation_config.use_align_loss and module_scores is not None:
            mu_n = F.normalize(mu, p=2, dim=1, eps=1e-8)
            w_n = F.normalize(module_scores, p=2, dim=1, eps=1e-8)
            align_loss = ((mu_n - w_n) ** 2).sum(dim=1).mean()
        else:
            align_loss = torch.zeros((), device=recon.device)

        total_loss = (
            recon_loss
            + emotion_loss
            + self.hparams.beta * kl_loss
            + self.hparams.lambda_sparse * sparse_loss
            + self.hparams.lambda_align * align_loss
        )
        return total_loss, emotion_loss, recon_loss, kl_loss, sparse_loss, align_loss

    def training_step(self, batch, batch_idx):
        _, (loss, emotion_loss, recon_loss, kl_loss, sparse_loss, align_loss) = (
            self.shared_step(batch, batch["module_scores"])
        )
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_emotion_loss", emotion_loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss, prog_bar=True)
        self.log("train_sparse_loss", sparse_loss)
        self.log("train_align_loss", align_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (recon, mu, logvar, z), (loss, emotion_loss, *_rest) = self.shared_step(
            batch, batch["module_scores"]
        )
        self.val_z_list.append(mu.detach().cpu())
        self.val_logvar_list.append(logvar.detach().cpu())
        self.val_z_sample_list.append(z.detach().cpu())
        self.val_recon_list.append(recon.detach().cpu())
        self.val_labels_list.append(batch["labels"].detach().cpu())
        self.log("val_loss", loss)
        self.log("val_emotion_loss", emotion_loss)

    def test_step(self, batch, batch_idx):
        (recon, mu, logvar, z), (loss, *_rest) = self.shared_step(batch, None)
        self.test_z_list.append(mu.detach().cpu())
        self.test_logvar_list.append(logvar.detach().cpu())
        self.test_z_sample_list.append(z.detach().cpu())
        self.test_recon_list.append(recon.detach().cpu())
        self.test_labels_list.append(batch["labels"].detach().cpu())
        self.log("test_loss", loss)

    def on_validation_epoch_end(self):
        self._compute_metrics(
            self.val_z_list, self.val_logvar_list, self.val_z_sample_list,
            self.val_recon_list, self.val_labels_list, "val",
        )
        self._reset_buffers()

    def on_test_epoch_end(self):
        self._compute_metrics(
            self.test_z_list, self.test_logvar_list, self.test_z_sample_list,
            self.test_recon_list, self.test_labels_list, "test",
        )
        self._reset_buffers()

    def _compute_metrics(self, z_list, logvar_list, z_sample_list, recon_list, labels_list, prefix):
        mu = torch.cat(z_list).numpy()
        logvar = torch.cat(logvar_list).numpy()
        z_sample = torch.cat(z_sample_list).numpy()
        recon = torch.cat(recon_list).numpy()
        labels = torch.cat(labels_list).numpy()

        self.log(f"{prefix}_mu_sparsity",
                 float((np.abs(mu) < self.hparams.z_sparsity_threshold).mean()))

        high_mask = labels > self.hparams.label_positive_threshold
        if high_mask.sum() > 0:
            tp = np.logical_and(recon > self.hparams.threshold, high_mask).sum()
            self.log(f"{prefix}_high_recall", float(tp / high_mask.sum()))

        self.log(f"{prefix}_mae", float(np.abs(recon - labels).mean()))

        exp_logvar = np.exp(logvar)
        kl_per_dim = -0.5 * np.mean(1.0 + logvar - mu ** 2 - exp_logvar, axis=0)
        self.log(f"{prefix}_kl_mean", float(np.mean(kl_per_dim)))
        self.log(f"{prefix}_kl_active_dims",
                 int((kl_per_dim > self.hparams.kl_active_threshold).sum()))
        self.log(f"{prefix}_mean_sigma", float(np.exp(0.5 * logvar).mean()))

        mu_mean_abs = np.mean(np.abs(mu), axis=0)
        self.log(f"{prefix}_mu_mean_abs_mean", float(mu_mean_abs.mean()))
        self.log(f"{prefix}_mu_var_mean", float(np.var(mu, axis=0).mean()))
        self.log(f"{prefix}_z_mean_abs_mean", float(np.mean(np.abs(z_sample), axis=0).mean()))
        self.log(f"{prefix}_z_var_mean", float(np.var(z_sample, axis=0).mean()))
        self.log(f"{prefix}_mu_active_dims",
                 int((mu_mean_abs > self.hparams.mu_active_threshold).sum()))

        self.log(f"{prefix}_mse", float(np.mean((recon - labels) ** 2)))

    def _load_nmf_weights(self):
        if not self.hparams.ablation_config.use_nmf_weights:
            return None

        path = self.hparams.nmf_weights_path
        with open(path, "rb") as f:
            module_weights = pickle.load(f)["module_weights"]

        expected = (self.hparams.n_modules, self.hparams.num_vae_emotions)
        if tuple(module_weights.shape) != expected:
            raise ValueError(
                f"NMF module weights at {path} have shape "
                f"{tuple(module_weights.shape)}, but the model expects "
                f"(n_modules, num_vae_emotions) = {expected}. Regenerate the NMF "
                f"artifact with K={self.hparams.n_modules}."
            )
        return module_weights

    def configure_optimizers(self):
        return AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
            betas=(0.9, 0.999),
        )
