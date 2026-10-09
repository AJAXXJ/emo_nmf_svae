# NMF-SVAE

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

- Stage 1 (`train_vae`): train NMF-SVAE, checkpoint saved to `checkpoints/<dataset>/`.
- Stage 2 (`train_downstream`): load the frozen backbone and train the classifier.
