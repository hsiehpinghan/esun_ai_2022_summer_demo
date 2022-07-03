import os
import sys
import torch

from argparse import ArgumentParser
from data import EsunDataModule
from sub_model import EsunModel
from transformers import BertTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        help='huggingface model name')
    parser.add_argument('--data_path',
                        type=str,
                        help='data path')
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size')
    parser.add_argument('--max_length',
                        type=int,
                        help='max tokenized tokens length')
    parser.add_argument('--lr',
                        type=float,
                        help='learning rate')
    parser.add_argument('--split',
                        type=int,
                        help='what kfold split in used.')
    parser.add_argument('--checkpoint_output_dir',
                        type=str,
                        help='checkpoint output dir')
    parser = Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    return args

def main(args):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name)
    datamodule = EsunDataModule(data_path=args.data_path,
                                batch_size=args.batch_size,
                                tokenizer=tokenizer,
                                split=args.split)
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    model = EsunModel(tokenizer=tokenizer,
                      max_length=args.max_length,
                      model_name=args.model_name,
                      lr=args.lr)
    model_checkpoint = ModelCheckpoint(dirpath=os.path.join('/tmp', str(args.split)),
                                       filename='{epoch:02d}-{val_epoch_char_error_rate:.3f}',
                                       monitor='val_epoch_char_error_rate',
                                       save_top_k=1,
                                       mode='min',
                                       auto_insert_metric_name=True,
                                       save_weights_only=False,
                                       every_n_epochs=1)
    trainer = Trainer.from_argparse_args(args=args,
                                         callbacks=[EarlyStopping(monitor='val_epoch_char_error_rate',
                                                                  patience=5,
                                                                  mode='min'),
                                                    model_checkpoint])
    trainer.tune(model=model,
                 datamodule=datamodule)
    trainer.fit(model=model,
                datamodule=datamodule)
    trainer.validate(model=model,
                     datamodule=datamodule)
    best_model = EsunModel.load_from_checkpoint(checkpoint_path='/tmp/0/epoch=00-val_epoch_char_error_rate=0.002-v1.ckpt',
                                                map_location=torch.device('cuda' if args.accelerator=='gpu' else args.accelerator),
                                                tokenizer=tokenizer,
                                                max_length=args.max_length,
                                                model_name=args.model_name,
                                                lr=args.lr)
    best_model.tokenizer.save_pretrained(save_directory=os.path.join(args.checkpoint_output_dir, str(args.split)))
    best_model.bert.save_pretrained(save_directory=os.path.join(args.checkpoint_output_dir, str(args.split)))

if __name__ == '__main__':
    """
    sys.argv = [sys.argv[0]]
    sys.argv += ['--accelerator', 'cpu']
    sys.argv += ['--accumulate_grad_batches', '1']
    sys.argv += ['--benchmark']
    sys.argv += ['--device', '1']
    sys.argv += ['--log_every_n_steps', '1']
    sys.argv += ['--max_epochs', '100']
    sys.argv += ['--model_name', 'shibing624/macbert4csc-base-chinese']
    sys.argv += ['--data_path', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/esun_ai_2022_summer_20220415.txt']
    sys.argv += ['--batch_size', '64']
    sys.argv += ['--max_length', '128']
    sys.argv += ['--lr', '5e-6']
    sys.argv += ['--split', '0']
    sys.argv += ['--checkpoint_output_dir', '/tmp/checkpoint']
    """
    main(args=parse_args())