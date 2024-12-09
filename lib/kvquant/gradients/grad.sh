#!/bin/bash


DEVICES=0

# LLaMA-2 7B
# model_path="/raid/LLM/llama2-7b"
# output_dir="outputs/llama2-7b"

# LLaMA-3 8B Instruct
model_path="/raid/LLM/llama3.1-8b-instruct"
output_dir="outputs/llama3.1-8b-instruct"


CUDA_VISIBLE_DEVICES=$DEVICES python run-fisher.py \
    --model_name_or_path $model_path \
    --output_dir $output_dir \
    --dataset wikitext2 \
    --seqlen 2048 \
    --maxseqlen 2048 \
    --num_examples 16 