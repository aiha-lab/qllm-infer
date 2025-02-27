#!/bin/bash

export HF_HOME=/raid/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICES=$1
model_path=$2
cache_dir='./cache'
tasks=none
num_fewshot=none
limit=none
eval_ppl=true
eval_ppl_seqlen=2048
use_cuda_graph=true
seed=0
# Quantization
bits_a=8
sym_a=false
groupsize_a=-1
bits_w=8
sym_w=false
groupsize_w=-1
# SmoothQuant
smoothquant=true
smoothquant_alpha=0.55
smoothquant_dataset=pile
smoothquant_nsamples=512
smoothquant_seqlen=512
# GPTQ
gptq=false
gptq_dataset=c4
gptq_nsamples=128
gptq_seqlen=2048
gptq_true_sequential=false
gptq_percdamp=0.01
gptq_act_order=false
gptq_static_groups=false
# Chatbot Simulation
chat=false
# Log
logfile='logs/out.txt'
# Analysis Tools
analyze_stats=false
stats_csv_path='cache/llama3.1-8b-instruct-w8a8sq.csv'
get_layerwise_distance=false

CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
    --model_path $model_path \
    --cache_dir $cache_dir \
    --tasks $tasks \
    --num_fewshot $num_fewshot \
    --limit $limit \
    --eval_ppl $eval_ppl \
    --eval_ppl_seqlen $eval_ppl_seqlen \
    --use_cuda_graph $use_cuda_graph \
    --seed $seed \
    --bits_a $bits_a \
    --sym_a $sym_a \
    --groupsize_a $groupsize_a \
    --bits_w $bits_w \
    --sym_w $sym_w \
    --groupsize_w $groupsize_w \
    --smoothquant $smoothquant \
    --smoothquant_alpha $smoothquant_alpha \
    --smoothquant_dataset $smoothquant_dataset \
    --smoothquant_nsamples $smoothquant_nsamples \
    --smoothquant_seqlen $smoothquant_seqlen \
    --gptq $gptq \
    --gptq_dataset $gptq_dataset \
    --gptq_nsamples $gptq_nsamples \
    --gptq_seqlen $gptq_seqlen \
    --gptq_true_sequential $gptq_true_sequential \
    --gptq_percdamp $gptq_percdamp \
    --gptq_act_order $gptq_act_order \
    --gptq_static_groups $gptq_static_groups \
    --chat $chat \
    --logfile $logfile \
    --analyze_stats $analyze_stats \
    --stats_csv_path $stats_csv_path \
    --get_layerwise_distance $get_layerwise_distance
