from pytorch_lightning.loggers import CometLogger
from utils import *
from config import *
from model import LitUnet
from dataset import RetinaDataModule
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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


class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        val_iou = trainer.callback_metrics.get("val_iou", "Metric not found")
        val_loss = trainer.callback_metrics.get("val_loss", "Metric not found")
        print(f"Checkpoint saved with val_iou: {val_iou}")
        print(f"Checkpoint saved with val_loss: {val_loss}")
        print("-" * 50)
        
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if trainer.interrupted:
            print("*" * 50)
            print("Training was interrupted by a callback")
            print("*" * 50)
        else:
            print("*" * 50)
            print("Training Ended Successfully")
            print("*" * 50)


checkpoint_callback = CustomModelCheckpoint(
    monitor="val_iou",
    dirpath="checkpoints/",
    filename=f"lightning_{args.job_id}",
    save_top_k=1,
    mode="max",
)

early_stopping = EarlyStopping(
    monitor="val_iou",
    min_delta=0.00,
    verbose=True,
    patience=100,
    mode="max",
)

if __name__ == "__main__":
    model = LitUnet(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, learning_rate=LR
    )

    dm = RetinaDataModule(
        BASE_DIRECTORY,
        train_transform=train_transform,
        test_transform=None,
        batch_size=BATCH_SIZE,
    )
    trainer = pl.Trainer(
        strategy="ddp",
        max_epochs=EPOCHS,
        accelerator=ACCELERATOR,
        devices=GPUS,
        enable_progress_bar=False,
        logger=comet_logger,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, dm)
