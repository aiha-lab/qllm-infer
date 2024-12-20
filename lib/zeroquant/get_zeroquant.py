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
from utils.perplexity import eval_ppl

def get_zeroquant_model(
        model,
        tokenizer,
        device,
        args
    ):

    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    args.deepspeed_config = 'zeroquant_config.json'
    args.weight_decay = 1e-5
    args.learning_rate = 5e-6
    args.gradient_accumulation_steps = 1
    args.num_train_epochs = 1
    args.max_train_steps = 600
    args.lr_scheduler_type = SchedulerType.LINEAR
    args.batch_size = 4

    from utils.data_utils import get_loaders
    train_dataloader = DataLoader(get_loaders('wikitext2', model=args.model_path), batch_size=args.batch_size)

    model = init_compression(model, args.deepspeed_config)  #<==========================================compression argument

    # print(model)

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
    
    num_warmup_steps = 0
    # num_warmup_steps = int(args.num_warmup_epochs * num_update_steps_per_epoch)
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
        dist_init_required=False)

    # Evaluate PPL
    ppls = eval_ppl(model.cuda(), tokenizer, args, nsamples=10, datasets=['wikitext2'])
    torch.set_grad_enabled(True)
    model.eval()

    if args.zeroquant_lkd:
        print("Starting Layer-by-Layer Knowledge Distillation")
        from utils.import_model import model_from_hf_path
        teacher_model = model_from_hf_path(args.model_path,
                    args.use_cuda_graph,
                    device_map='auto',
                ).eval()
        teacher_model.eval()

        start_time = time.time()
        
        # Train only the last layer
        layer_list = [31]

        # for l in range(layer_list): # iterate across layers
        for l in layer_list: # iterate across layers
            logging.info(f"layer {l}")
            teacher_layer = recursive_getattr(teacher_model, f'model.layers.{l}')  # extract the lth layer of teacher
            student_layer = recursive_getattr(model.module, f'model.layers.{l}')  # extract the lth layer of student

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
                    inp_batch, tar_batch = batch
                    inp_batch = inp_batch.squeeze(1)
                    batch_dict = {"input_ids": inp_batch}

                    with torch.no_grad():
                        # for simplicity, we always run the full inference of the teacher model.
                        # To get the best performance, you can run the teacher model only for the first l layers,
                        # which requires some modifications to the modeling code.
                        teacher_out = teacher_model(**batch_dict, output_hidden_states=True) # get the output of the teacher model
                        
                    layer_input = teacher_out.hidden_states[l] # extract the lth-layer's input of teacher
                    # teacher_o = teacher_out.hidden_states[l+1] # extract the lth-layer's output of teacher

                    position_ids = torch.arange(layer_input.size(1), device=layer_input.device).unsqueeze(0).expand(layer_input.size(0), -1)
                    teacher_o = teacher_layer(layer_input, position_ids=position_ids)[0] # run inference for the teacher
                    student_o = student_layer(layer_input, position_ids=position_ids)[0] # run inference for the student
                    loss = torch.nn.functional.mse_loss(student_o, teacher_o) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if updated_steps % 10 == 0:
                        print(f"Layer {l}, Step {updated_steps}, Loss: {loss.item()}")

                    updated_steps += 1 
                    if updated_steps >= args.max_train_steps :  # break when the number of steps is reached, typically in hundreds
                        break
                if updated_steps >= args.max_train_steps:
                    end_time = time.time()
                    break
        
        print(f"Time taken for LKD: {end_time - start_time}")

    return model
