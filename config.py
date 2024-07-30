import torch
import argparse
import warnings
import json
import os

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config.json")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = json.load(file)


EPOCHS = config["epochs"]
LR = config["learning_rate"]
BATCH_SIZE = config["batch_size"]
THRESHOLD = config["threshold"]
JOB_ID = os.getenv("SLURM_JOB_ID", "0")

GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

IN_CHANNELS = 3
OUT_CHANNELS = 1

BASE_DIRECTORY = "dataset"

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
