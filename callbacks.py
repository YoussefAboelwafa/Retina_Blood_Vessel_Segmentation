from config import *
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class CustomModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        val_iou = trainer.callback_metrics.get("val_iou", "Metric not found")
        val_loss = trainer.callback_metrics.get("val_loss", "Metric not found")
        print("Epoch:", trainer.current_epoch)
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
    filename=f"lightning_{JOB_ID}",
    save_top_k=1,
    mode="max",
)

early_stopping = EarlyStopping(
    monitor="val_iou",
    min_delta=0.00,
    patience=50,
    mode="max",
)