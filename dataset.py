import os
import pickle

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from utils import preprocess_data


class EmoNetBioDataset(Dataset):
    def __init__(self, data_path, feature_path, tokenizer, normalization):
        emotion_matrix = pd.read_csv(data_path)
        self.texts = emotion_matrix["text"].astype(str).tolist()
        self.labels = emotion_matrix.drop(columns=["text"]).values.astype(np.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.encodings = self.tokenizer(
            self.texts, max_length=128, padding="max_length", truncation=True,
            return_tensors="pt",
        )

        self.train_vae = os.path.exists(feature_path)
        if self.train_vae:
            with open(feature_path, "rb") as f:
                self.module_scores = pickle.load(f)["module_scores"]
            self.labels = preprocess_data(normalization, self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.FloatTensor(self.labels[idx]),
            "text": self.texts[idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        if self.train_vae:
            item["module_scores"] = torch.FloatTensor(self.module_scores[idx])
        return item


class EmoNetBioDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seed = config.seed
        self.batch_size = config.batch_size

    def setup(self, stage=None):
        self.full_dataset = EmoNetBioDataset(
            self.config.data_path,
            self.config.nmf_path,
            self.config.tokenizer,
            self.config.normalization,
        )

        total = len(self.full_dataset)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        test_size = total - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size], generator=generator
        )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.config, "num_workers", 0),
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
