#/bin/bash

export HF_HOME=/raid/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

# llama3.1-8b-instruct, avg_bits: 3.99
DEVICES=$1
model_path=$2
bits_w=3
groupsize_w=16
bits_sz=3
groupsize_qq=16
ol_thre=0.25

# llama3.1-8b-instruct, avg_bits: 3.0

#bits_w=2
#groupsize_w=32
#bits_sz=3
#groupsize_qq=16
#ol_thre=0.5

# '/'로 split한 후 마지막 요소를 가져와 model_name에 저장
model_name=$(basename "$model_path")
use_cuda_graph=true
seed=0

echo "Model Name: $model_name"

cache_dir='./cache'
# lm_eval arguments
#tasks=none
num_fewshot=none
limit=none

# Quantization
bits_a=16
groupsize_a=-1
sym_a=false # const

sym_w=false # const

# SmoothQuant
smoothquant=false # const
smoothquant_alpha=0.5 # const
smoothquant_dataset=pile # const
smoothquant_nsamples=512 # const
smoothquant_seqlen=512 # const

# GPTQ
gptq_dataset=c4
gptq_nsamples=128 # const
gptq_seqlen=2048 # const
gptq_true_sequential=false
gptq_percdamp=0.01
gptq_act_order=true
gptq_static_groups=false

# SpQR
spqr_qq_scale_bits=$bits_sz
spqr_qq_zero_bits=$bits_sz
spqr_qq_zero_sym=false # const
spqr_qq_groupsize=$groupsize_qq
spqr_outlier_threshold=$ol_thre
spqr_simplified_outliers=false
spqr_offload_activation=false
spqr_load='cache/spqr-'$model_name'-w'$bits_w'g'$groupsize_w'-qw'$spqr_qq_scale_bits'qg'$spqr_qq_groupsize'-t'$spqr_outlier_threshold
spqr_save='cache/spqr-'$model_name'-w'$bits_w'g'$groupsize_w'-qw'$spqr_qq_scale_bits'qg'$spqr_qq_groupsize'-t'$spqr_outlier_threshold
spqr_skip_out_loss=true

# Log
logfile='logs/spqr-'$model_name'-w'$bits_w'g'$groupsize_w'th'$spqr_outlier_threshold'.txt'

eval_ppl=true
chat=false
tasks=none # pefill only
get_layerwise_distance=false
analyze_stats=false

eval_ppl_seqlen=2048
stats_csv_path='cache/'$model_name'-w'$bits_w'g'$groupsize_w'th'$spqr_outlier_threshold'-'spqr'.csv'

# diff bits_w, groupsize_w, algorithm
gptq=false
spqr=true

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
    --spqr $spqr\
    --spqr_qq_scale_bits $spqr_qq_scale_bits \
    --spqr_qq_zero_bits $spqr_qq_zero_bits \
    --spqr_qq_zero_sym $spqr_qq_zero_sym \
    --spqr_qq_groupsize $spqr_qq_groupsize \
    --spqr_outlier_threshold $spqr_outlier_threshold \
    --spqr_simplified_outliers $spqr_simplified_outliers \
    --spqr_offload_activation $spqr_offload_activation \
    --spqr_load $spqr_load \
    --spqr_save $spqr_save \
    --spqr_skip_out_loss $spqr_skip_out_loss \
    --chat $chat \
    --logfile $logfile \
    --analyze_stats $analyze_stats \
    --stats_csv_path $stats_csv_path \
    --get_layerwise_distance $get_layerwise_distance
