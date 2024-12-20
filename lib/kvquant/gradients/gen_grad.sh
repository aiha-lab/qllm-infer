#!/bin/bash


# model_name="llama2-7b"
# DEVICES=0

model_name="llama3.1-8b-instruct"
DEVICES=1


CUDA_VISIBLE_DEVICES=$DEVICES python run-fisher.py \
    --model_name_or_path "/raid/LLM/${model_name}" \
    --output_dir "outputs/${model_name}" \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen 2048 \
    --num_examples 16 