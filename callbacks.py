from pytorch_lightning.callbacks import ModelCheckpoint
from config import *

checkpoint_callback = ModelCheckpoint(
    monitor='val_iou',
    dirpath='checkpoints/',
    filename=f'lightning_{args.job_id}',
    mode='max',
)
