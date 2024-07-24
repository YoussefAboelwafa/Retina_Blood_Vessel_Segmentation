from pytorch_lightning import Trainer
from model import LitUnet
from dataset import RetinaDataModule
from config import *
from utils import *
import albumentations as A

set_seed()

EXP_ID = 19103

model = LitUnet.load_from_checkpoint(
    checkpoint_path= f"/scratch/y.aboelwafa/Retina/Retina_Blood_Vessel_Segmentation/checkpoints/lightning_{EXP_ID}.ckpt",
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    learning_rate=LR,
)

test_transform = A.Compose(
    [
        A.Resize(512, 512),
    ]
)

dm = RetinaDataModule(
    BASE_DIRECTORY,
    train_transform=None,
    test_transform=test_transform,
    batch_size=BATCH_SIZE,
)

trainer = Trainer(enable_progress_bar=False, logger=False)
trainer.test(model, dm)
