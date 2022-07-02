import os
import re
import sys
import csv
import json
import torch
import numpy as np
import random
import logging
import operator
import warnings
import torch.nn as nn
import torchmetrics

from tqdm import tqdm
from datasets import load_metric
from argparse import ArgumentParser
from collections import defaultdict
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from IPython.display import clear_output
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

MODEL_NAME = 'shibing624/macbert4csc-base-chinese'

def parse_args():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    return args

predict_output = None

def main(args):
    global predict_output
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    datamodule = EsunDataModule(data_dir=DATA_DIR,
                                  batch_size=BATCH_SIZE,
                                  tokenizer=tokenizer)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    model = EsunModel(tokenizer=tokenizer)
    trainer = Trainer.from_argparse_args(args=args,
                                         callbacks=[EarlyStopping(monitor='val_epoch_char_error_rate',
                                                                  patience=5,
                                                                  mode='min'),
                                                    ModelCheckpoint(dirpath=OUTPUT_DIR,
                                                                    filename='{epoch:02d}-{val_epoch_char_error_rate:.3f}',
                                                                    monitor='val_epoch_char_error_rate',
                                                                    save_top_k=1,  # save the best model
                                                                    mode='min',
                                                                    auto_insert_metric_name=True,
                                                                    save_weights_only=False,
                                                                    every_n_epochs=1)])
    trainer.tune(model=model,
                 datamodule=datamodule)
    trainer.fit(model=model,
                datamodule=datamodule)
    trainer.validate(model=model,
                     datamodule=datamodule)
