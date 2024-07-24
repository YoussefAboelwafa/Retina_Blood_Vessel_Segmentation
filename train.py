from pytorch_lightning.loggers import CometLogger
from utils import *
from config import *
from model import LitUnet
from dataset import RetinaDataModule
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.callbacks import ModelCheckpoint

set_seed()

comet_logger = CometLogger(
    api_key="rwyMmTQC0QDIH0oF5XaSzgmh4",
    project_name="retina-blood-vessel-segmentation",
    workspace="youssefaboelwafa",
    experiment_name=str(args.job_id),
)

train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Sharpen(alpha=(0.5, 0.9), lightness=(0.5, 1.0), p=0.3),
        A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=0.3),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
    ]
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_iou",
    dirpath="checkpoints/",
    filename=f"lightning_{args.job_id}",
    mode="max",
)


if __name__ == "__main__":
    model = LitUnet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, learning_rate=LR
    )

    dm = RetinaDataModule(
        BASE_DIRECTORY,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
    )
    trainer = pl.Trainer(
        strategy="ddp",
        max_epochs=EPOCHS,
        accelerator=ACCELERATOR,
        devices=GPUS,
        enable_progress_bar=False,
        logger=comet_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dm)
