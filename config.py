import torch
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--job_id", type=int)
args = parser.parse_args()


EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size

IN_CHANNELS = 3
OUT_CHANNELS = 1

BASE_DIRECTORY = "dataset"

CHECKPOINT_PATH = "/scratch/y.aboelwafa/Retina/Retina_Blood_Vessel_Segmentation/checkpoints/pytorch"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
