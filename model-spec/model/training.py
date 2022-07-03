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
from argparse import ArgumentParser
from data import EsunDataModule
from sub_model import EsunModel
from collections import defaultdict
from transformers import BertTokenizer
from transformers import BertForMaskedLM
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        help='huggingface model name')
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size')
    parser.add_argument('--split',
                        type=int,
                        help='what kfold split in used.')
    parser.add_argument('--checkpoint_output_dir',
                        type=str,
                        help='checkpoint output dir')
    parser = Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    return args

predict_output = None

def main(args):
    global predict_output
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name)
    datamodule = EsunDataModule(batch_size=args.batch_size,
                                tokenizer=tokenizer,
                                split=args.split)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    model = EsunModel(tokenizer=tokenizer)
    trainer = Trainer.from_argparse_args(args=args,
                                         callbacks=[EarlyStopping(monitor='val_epoch_char_error_rate',
                                                                  patience=5,
                                                                  mode='min'),
                                                    ModelCheckpoint(dirpath=os.path.join(args.checkpoint_output_dir, str(args.split)),
                                                                    filename='{epoch:02d}-{val_epoch_char_error_rate:.3f}',
                                                                    monitor='val_epoch_char_error_rate',
                                                                    save_top_k=1,
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

if __name__ == '__main__':
    sys.argv = [sys.argv[0]]
    sys.argv += ['--accelerator', 'cpu']
    sys.argv += ['--accumulate_grad_batches', '1']
    #sys.argv += ['--auto_lr_find']  # not working !!!
    #sys.argv += ['--auto_scale_batch_size', 'binsearch']
    sys.argv += ['--benchmark']  #  if deterministic is set to True, this will default to False.
    sys.argv += ['--device', '1']
    #sys.argv += ['--fast_dev_run', '1']  # for test purpose.
    #sys.argv += ['--deterministic']
    sys.argv += ['--log_every_n_steps', '1']
    sys.argv += ['--max_epochs', '100']
    #sys.argv += ['--num_sanity_val_steps', '-1']
    #sys.argv += ['--max_time', '00:00:10:00']
    #sys.argv += ['--precision', '16']
    #sys.argv += ['--max_steps', '1000']
    sys.argv += ['--model_name', 'shibing624/macbert4csc-base-chinese']
    sys.argv += ['--batch_size', '64']
    sys.argv += ['--split', '0']
    sys.argv += ['--checkpoint_output_dir', 'checkpoint']
    main(args=parse_args())