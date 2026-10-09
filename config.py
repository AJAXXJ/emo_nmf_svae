import os
import random

import numpy as np
import pandas as pd
import torch

from utils import VAEAblationConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class EmoNetBioConfig:
    def __init__(self, dataset="go-emotion", seed=42, data_root=None):
        self.dataset = dataset
        self.seed = seed
        self.model_name = "EmoVAE"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_root = (
            data_root
            or os.environ.get("EMONETBIO_DATA_ROOT")
            or os.path.join(BASE_DIR, "data", "datasets")
        )
        self.data_dir = os.path.join(self.data_root, dataset)
        self.data_path = os.path.join(self.data_dir, "emotion_matrix.csv")
        self.nmf_path = os.path.join(self.data_dir, "nmf_features.pkl")
        self.nmf_weights_path = os.path.join(
            self.data_root, "go-emotion", "nmf_features.pkl"
        )
        self.normalization = "vst_minmax"

        self.pretrained_model = "/home/pretrained/roberta-large"
        self.tokenizer = self.pretrained_model
        self.freeze_bert = False

        self.n_modules = 6
        self.num_vae_emotions = 27
        self.vae_hidden_dim = 128
        self.hidden_dim = 128
        self.dropout_rate = 0.2

        self.beta = 0.01
        self.lambda_sparse = 0.05
        self.lambda_align = 0.02

        self.batch_size = 8
        self.lr = 5e-6
        self.epochs = 300
        self.patience = 10
        self.weight_decay = 1e-3
        self.eps = 1e-8
        self.num_workers = 0

        self.threshold = 0.025
        self.z_sparsity_threshold = 0.01
        self.label_positive_threshold = 0
        self.kl_active_threshold = 1e-3
        self.mu_active_threshold = 1e-3

        self.ablation_config = VAEAblationConfig()

        self.checkpoints = os.path.join(BASE_DIR, f"checkpoints/{dataset}")

        self._set_emotion_names()
        self._set_seed()

    def _set_emotion_names(self):
        emotion_matrix = pd.read_csv(self.data_path)
        self.emotion_names = emotion_matrix.drop(columns=["text"]).columns.tolist()
        self.num_emotions = len(self.emotion_names)

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
