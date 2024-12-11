#!/bin/bash

export HF_HOME=/raid/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICES=$1
model_path=$2
cache_dir='./cache'

# tasks='boolq,arc_challenge,arc_easy,hellaswag,piqa,winogrande,mmlu'
# tasks='gsm8k,truthfulqa'
tasks=none

num_fewshot=none
limit=none

# eval_ppl=true
eval_ppl=false

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
smoothquant=false
smoothquant_alpha=0.5
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
# KIVI
kivi=false
kivi_prefill_with_quant=false
kivi_k_bits=2 # 4, 2
kivi_v_bits=2 # 4, 2
kivi_group_size=32
kivi_residual_length=128
# KVQuant
kvquant=true
kvquant_prefill_with_quant=false
kvquant_kv_bits=2 # 4, 3, 2
kvquant_nuq=true
kvquant_include_sparse=true
kvquant_sparsity_threshold=0.99
kvquant_first_few_fp16=1

if [ "$kivi" = "true" ]
then
    echo -e "\n******************************"
    echo "KIVI"
    echo "Model: ${model_path}"
    echo "Tasks: ${tasks}"
    echo "Prefill with Quantization: ${kivi_prefill_with_quant}"
    echo "KV Bits: ${kivi_k_bits}, ${kivi_v_bits}"
    echo -e "******************************\n"
fi

if [ "$kvquant" = "true" ]
then
    echo -e "\n******************************"
    echo "KVQuant"
    echo "Model: ${model_path}"
    echo "Tasks: ${tasks}"
    echo "Prefill with Quantization: ${kvquant_prefill_with_quant}"
    echo "KV Bits: ${kvquant_kv_bits}"
    echo -e "******************************\n"
fi

# Chatbot Simulation
chat=false
# chat=true
# Needle-In-A-Haystack Task Example
# niah=false
niah=true
# Log
logfile='logs/out.txt'
# Analysis Tools
analyze_stats=false
stats_csv_path='cache/llama3.1-8b-instruct-w8a8sq.csv'
get_layerwise_distance=false

for bits_a in 16
do
for bits_w in 16
do
for smoothquant in false
do
for gptq in false
do
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
    --kivi $kivi \
    --kivi_prefill_with_quant $kivi_prefill_with_quant \
    --kivi_k_bits $kivi_k_bits \
    --kivi_v_bits $kivi_v_bits \
    --kivi_group_size $kivi_group_size \
    --kivi_residual_length $kivi_residual_length \
    --kvquant $kvquant \
    --kvquant_prefill_with_quant $kvquant_prefill_with_quant \
    --kvquant_kv_bits $kvquant_kv_bits \
    --kvquant_nuq $kvquant_nuq \
    --kvquant_include_sparse $kvquant_include_sparse \
    --kvquant_sparsity_threshold $kvquant_sparsity_threshold \
    --kvquant_first_few_fp16 $kvquant_first_few_fp16 \
    --chat $chat \
    --niah $niah \
    --logfile $logfile \
    --analyze_stats $analyze_stats \
    --stats_csv_path $stats_csv_path \
    --get_layerwise_distance $get_layerwise_distance
done
done
done
done