# NMF-SVAE

Multi-label emotion classification with an NMF-guided sparse VAE.

BERT text encoder → emotion activity head → VAE with an NMF-initialized decoder →
gated fusion of the latent `z` and BERT features for classification.

Set the pretrained backbone path in `config.py` (`pretrained_model`).

## Data

This folder contains code only. Pass the data root with `--data_root` or the
`EMONETBIO_DATA_ROOT` environment variable (default `./data/datasets`).

```
<data_root>/<dataset>/
├── emotion_matrix.csv
└── nmf_features.pkl
```

## Train

```bash
python train.py                              # both stages
python train.py --data_root /path/to/data    # custom data root
python train.py --stage vae                  # stage 1 only
python train.py --stage downstream           # stage 2 only
```

- Stage 1 (`train_vae`): train EmoVAE, checkpoint saved to `checkpoints/<dataset>/`.
- Stage 2 (`train_downstream`): load the frozen backbone and train the classifier.
