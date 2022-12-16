import argparse
import glob
import os
import json
import time
import logging
import random
import re
import sys
from sys import platform
from itertools import chain
from string import punctuation
from pprint import pprint
from torch.utils import data
from tqdm import tqdm

sys.path.append('../')
from utils import currentdate

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import textwrap

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from transformers import (
    AdamW,
    T5ForConditionalGeneration, 
    MT5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from models import T5FineTuner

# need this because of the following error:
# forrtl: error (200): program aborting due to control-C event
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

class QGDataModule(pl.LightningDataModule):

    def __init__(
        self,
        params,
        tokenizer: T5Tokenizer,
        train_list: list,
        val_list: list,
        test_list: list
        ): 
        super().__init__()
        self.tokenizer = tokenizer
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.batch_size = params.batch_size
        self.max_len_input = params.max_len_input
        self.max_len_output = params.max_len_output

    def setup(self):
        self.train_dataset = QGDataset(self.train_list, self.tokenizer, self.max_len_input, self.max_len_output)
        self.validation_dataset = QGDataset(self.val_list, self.tokenizer, self.max_len_input, self.max_len_output)
        self.test_dataset = QGDataset(self.test_list, self.tokenizer, self.max_len_input, self.max_len_output)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = 4) # why 4?

    def val_dataloader(self): 
        return DataLoader(self.validation_dataset, batch_size = self.batch_size, num_workers = 4) # why 4?,   batch_size = 1 ?

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 4, num_workers = 4) # why 4?,  batch_size = 1 ?

class QGDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer: T5Tokenizer,
        max_len_input: int,
        max_len_output: int
        ):

        self.tokenizer = tokenizer
        self.data = data
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        #data_row = self.data.iloc[index]
        data_row = self.data[index]

        input_concat = '<skill>' + data_row['attributes'][0] + '</skill>' + '<boa>' + data_row['answer1'] + '</boa>' + '<bot>' + ' '.join(data_row['sections_texts']) + '</bot>'

        # tokenize inputs
        source_encoding = self.tokenizer(
            input_concat,
            truncation = True,
            add_special_tokens=True,
            return_overflowing_tokens = True, # if (source_encoding.num_truncated_tokens.item() > 0): !!!!!!!! (future)
            max_length=self.max_len_input, 
            padding='max_length', 
            return_tensors="pt"
        )
        # tokenize targets
        target_encoding = self.tokenizer(
            data_row['question_reference'], 
            truncation = True,
            add_special_tokens=True,
            max_length=self.max_len_output, 
            padding='max_length',
            return_tensors="pt"
        )

        labels = target_encoding['input_ids']
        labels[labels==0] = -100

        return dict(
            # commented code because of this problem: https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
            #question=data_row['question'],
            #context=data_row['context'],
            #answer_text=data_row['answer'],

            source_ids = source_encoding["input_ids"].flatten(),
            target_ids = target_encoding["input_ids"].flatten(),

            source_mask = source_encoding['attention_mask'].flatten(),
            target_mask = target_encoding['attention_mask'].flatten(),

            labels=labels.flatten()
        )

def run(args):
    pl.seed_everything(args.seed_value)

    # Load Tokenizer
    t5_tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    t5_tokenizer.add_tokens(['<skill>','</skill>','<boa>','</boa>','<bot>','</bot>'], special_tokens=True)

    # Load model
    if "mt5" in args.model_name:
        t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        t5_model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Checkpoints and Logs Paths
    CHECKPOINTS_PATH = '../../checkpoints/' + args.dir_model_name
    TB_LOGS_PATH = CHECKPOINTS_PATH + "/tb_logs"
    CSV_LOGS_PATH = CHECKPOINTS_PATH + "/csv_logs"

    # Training...
    params_dict = dict(
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        train_path = args.train_path,
        val_path = args.val_path,
        test_path = args.test_path,
        max_len_input = args.max_len_input,
        max_len_output = args.max_len_output,
        max_epochs = args.max_epochs,
        batch_size = args.batch_size,
        patience = args.patience,
        optimizer = args.optimizer,
        learning_rate = args.learning_rate,
        epsilon = args.epsilon,
        num_gpus = args.num_gpus,
        seed_value = args.seed_value,
        checkpoints_path = CHECKPOINTS_PATH,
        tb_logs_path = TB_LOGS_PATH,
        csv_logs_path = CSV_LOGS_PATH,
        current_date = args.current_date
    )
    params = argparse.Namespace(**params_dict)

    model = T5FineTuner(params, t5_model, t5_tokenizer) # , train_dataset, validation_dataset, test_dataset

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_PATH, #save at this folder
        filename="model-{epoch:02d}-{val_loss:.2f}", #name for the checkpoint, before i was using "best-checkpoint"
        save_top_k=args.max_epochs, #save all epochs, before it was only the best (1)
        verbose=True, #output something when a model is saved
        monitor="val_loss", #monitor the validation loss
        mode="min" #save the model with minimum validation loss
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=TB_LOGS_PATH)
    csv_logger = pl_loggers.CSVLogger(save_dir=CSV_LOGS_PATH)
    csv_logger.log_hyperparams(params)

    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        max_epochs = args.max_epochs, 
        gpus = args.num_gpus,
        logger = [tb_logger, csv_logger]
    ) #progress_bar_refresh_rate=30

    # read data
    with open('../../data/FairyTaleQA_Dataset/processed/fairytaleqa_train.json', "r", encoding='utf-8') as read_file:
        train_list = json.load(read_file)
    with open('../../data/FairyTaleQA_Dataset/processed/fairytaleqa_val.json', "r", encoding='utf-8') as read_file:
        val_list = json.load(read_file)
    with open('../../data/FairyTaleQA_Dataset/processed/fairytaleqa_test.json', "r", encoding='utf-8') as read_file:
        test_list = json.load(read_file)

    # DELETE!!! JUST for FAST TESTING <----
    #####
    #train_df = pd.read_csv('../../data/squad_experiments/squad_v1_train.csv')
    #validation_df = pd.read_csv('../../data/squad_experiments/squad_v1_val.csv')
    #test_df = pd.read_csv('../../data/squad_experiments/squad_v1_val.csv')
    #####

    data_module = QGDataModule(params, t5_tokenizer, train_list, val_list, test_list)
    data_module.setup()

    start_time_train = time.time()

    trainer.fit(model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)
    #trainer.test(ckpt_path='best', dataloaders = data_module.test_dataloader)
    #trainer.test(ckpt_path=trainer.model_checkpoint.last_model_path)

    end_time_train = time.time()
    train_total_time = end_time_train - start_time_train
    print("Training time: ", train_total_time)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Fine tune T5 for Question Generation.')

    # Add arguments
    parser.add_argument('-dmn', '--dir_model_name', type=str, metavar='', default="qg_t5_small_512_64_8_3_skills_seed_42", required=False, help='Directory model name.')
    parser.add_argument('-mn','--model_name', type=str, metavar='', default="t5-small", required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default="t5-small", required=False, help='Tokenizer name.')

    parser.add_argument('-trp','--train_path', type=str, metavar='', default="../../data/FairytaleQA_Dataset/processed/fairytaleqa_train.json", required=False, help='Train path.')
    parser.add_argument('-vp','--val_path', type=str, metavar='', default="../../data/FairytaleQA_Dataset/processed/fairytaleqa_val.json", required=False, help='Validation path.')
    parser.add_argument('-tp','--test_path', type=str, metavar='', default="../../data/FairytaleQA_Dataset/processed/fairytaleqa_test.json", required=False, help='Test path.')

    parser.add_argument('-mli','--max_len_input', type=int, metavar='', default=512, required=False, help='Max len input for encoding.')
    parser.add_argument('-mlo','--max_len_output', type=int, metavar='', default=64, required=False, help='Max len output for encoding.')

    parser.add_argument('-me','--max_epochs', type=int, default=3, metavar='', required=False, help='Number of max Epochs')
    parser.add_argument('-bs','--batch_size', type=int, default=8, metavar='', required=False, help='Batch size.')
    parser.add_argument('-ptc','--patience', type=int, default=3, metavar='', required=False, help='Patience') # it is not being used for now
    parser.add_argument('-o','--optimizer', type=str, default='AdamW', metavar='', required=False, help='Optimizer')
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-4, metavar='', required=False, help='The learning rate to use.')
    parser.add_argument('-eps','--epsilon', type=float, default=1e-6, metavar='', required=False, help='Adam epsilon for numerical stability')

    parser.add_argument('-ng','--num_gpus', type=int, default=1, metavar='', required=False, help='Number of gpus.')
    parser.add_argument('-sv','--seed_value', type=int, default=42, metavar='', required=False, help='Seed value.')
    parser.add_argument('-cd', '--current_date', type=str, metavar='', default=currentdate(), required=False, help='Current date.')

    # Parse arguments
    args = parser.parse_args()

    # Start training
    run(args)