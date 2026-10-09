import argparse
import os

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from config import EmoNetBioConfig
from dataset import EmoNetBioDataModule
from downstream import EmotionMultiClassificationLit
from model import EmoVAELit


def train_vae(config):
    config.batch_size = 32
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoints,
        filename="{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=config.patience, mode="min"
    )
    trainer = Trainer(
        precision="bf16-mixed",
        max_epochs=config.epochs,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=3,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    data_module = EmoNetBioDataModule(config)
    data_module.setup()
    trainer.fit(EmoVAELit(config), data_module)
    return trainer, data_module


def train_downstream(config):
    config.model = EmoVAELit
    config.batch_size = 8
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.checkpoints, "downstream"),
        filename="{epoch:02d}-{val_f1_macro:.2f}",
        monitor="val_f1_macro", mode="max", save_top_k=1, save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_f1_macro", patience=config.patience, mode="max"
    )
    trainer = Trainer(
        precision="bf16-mixed",
        max_epochs=config.epochs,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    data_module = EmoNetBioDataModule(config)
    data_module.setup()
    model = EmotionMultiClassificationLit(config)
    trainer.fit(model, data_module)
    trainer.test(model, data_module, ckpt_path="best")
    return trainer, data_module


def main():
    parser = argparse.ArgumentParser(description="Train EmoVAE (NMF-SVAE)")
    parser.add_argument("--dataset", default="go-emotion")
    parser.add_argument("--data_root", default=None,
                        help="dataset root directory (default: $EMONETBIO_DATA_ROOT or ./data/datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=["all", "vae", "downstream"], default="all")
    args = parser.parse_args()

    config = EmoNetBioConfig(dataset=args.dataset, seed=args.seed, data_root=args.data_root)

    if args.stage in ("all", "vae"):
        train_vae(config)
    if args.stage in ("all", "downstream"):
        train_downstream(config)


if __name__ == "__main__":
    main()
