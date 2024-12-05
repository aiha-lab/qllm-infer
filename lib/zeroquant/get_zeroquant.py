import argparse
import copy
import logging
import math
import time
import os
import random
import json
import numpy as np
from pathlib import Path

import datasets
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import transformers
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.compression.helper import recursive_getattr
from lib.zeroquant.utils import *

def get_zeroquant_model(
        model,
        tokenizer,
        device,
        args
    ):

    from utils.data_utils import get_loaders
    train_dataloader = get_loaders('wikitext2', model=args.model_path)

    teacher_model = copy.deepcopy(model)

    args.deepspeed_config = 'test.json'
    args.weight_decay = 1e-5
    model = init_compression(model, args.deepspeed_config, teacher_model=teacher_model)  #<==========================================compression argument
    import pdb; pdb.set_trace()

    no_decay = ["bias", "RMSNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps =  math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    num_warmup_steps = int(args.num_warmup_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Prepare the model first eo enable compression feature
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    # Evaluate PPL
    ppls = eval_ppl(model.cuda(), tokenizer, args)

    # LKD
    model.eval()
    teacher_model.eval()
    start_time = time.time()

    for l in range(model.config.num_hidden_layers): # iterate across BERT layers
        logging.info(f"layer {l}")
        student_layer = recursive_getattr(model.module, f'model.decoder.layers.{l}')  # extract the lth layer of student

        optimizer_param = [
        {
            "params": [p for n, p in student_layer.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in student_layer.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]  

        optimizer = AdamW(optimizer_param, lr=args.learning_rate) 

        updated_steps = 0

        for _ in range(args.num_train_epochs):
            for _, batch in enumerate(train_dataloader):  # load each batch
                batch = to_device(batch, device)
                with torch.no_grad():
                    # for simplicity, we always run the full inference of the teacher model.
                    # To get the best performance, you can run the teacher model only for the first l layers,
                    # which requires some modifications to the modeling code.
                    teacher_out = teacher_model(**batch, output_hidden_states=True) # get the output of the teacher model
                layer_input = teacher_out.hidden_states[l] # extract the lth-layer's input of teacher
                teacher_o = teacher_out.hidden_states[l+1] # extract the lth-layer's output of teacher

                student_o = student_layer(layer_input)[0] # run inference for the student

                loss = torch.nn.functional.mse_loss(student_o, teacher_o) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                updated_steps += 1 
                if updated_steps >= args.max_train_steps :  # break when the number of steps is reached, typically in hundreds
                    break
            if updated_steps >= args.max_train_steps:
                break

    # Evaluate PPL
    ppls = eval_ppl(model.cuda(), tokenizer, args)
    return model
