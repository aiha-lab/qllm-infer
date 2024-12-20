#!/bin/bash

export HF_HOME=/raid/hf_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1

DEVICES=$1
model_path=$2
cache_dir='./cache'
tasks=boolq,arc_challenge,piqa,winogrande,mmlu
#tasks=none
#tasks=big_bench
num_fewshot=none
limit=none
eval_ppl=true
eval_ppl_seqlen=2048
use_cuda_graph=true
seed=0

# Quantization
bits_a=16
sym_a=false
groupsize_a=-1
bits_w_list=(4)
sym_w=false
groupsize_w_list=(-1)

# SmoothQuant
smoothquant=false
smoothquant_alpha=0.5
smoothquant_dataset=pile
smoothquant_nsamples=512
smoothquant_seqlen=512

# GPTQ (Removed the loop over gptq)
gptq=false
gptq_dataset=c4
gptq_nsamples=128
gptq_seqlen=2048
gptq_true_sequential=false
gptq_percdamp=0.01
gptq_act_order=false
gptq_static_groups=false

# LUTGEMM
lutgemm=true
rtn=(false)  # if you want to use lutgemm with other quantization method (RTN). set this variable to true.
do_packing=false
round=$3 # alternating methods cycle

# Chatbot Simulation
chat=false
get_layerwise_distance=false

mkdir -p logs
mkdir -p cache

run_main_py() {
  local bits_a_local="$1"
  local bits_w_local="$2"
  local groupsize_w_local="$3"
  local rtn_local="$4"

  echo "#=====================================================================================================#"
  echo "Running with bits_a=$bits_a_local, bits_w=$bits_w_local, groupsize_w=$groupsize_w_local, rtn=$rtn_local"

  # Set the logfile, stats file path
  # Log
  local logfile="logs/out-rtn-${rtn_local}-w${bits_w_local}a${bits_a_local}-lutgemm-round${round}-group${groupsize_w_local}.txt"
  local logfile="logs/w3a16-g128-round30"
  # Analysis Tools
  local analyze_stats=false
  local stats_csv_path="cache/llama3.1-8b-instruct-w${bits_w_local}a${bits_a_local}-lutgemm-round${round}-group${groupsize_w_local}.csv"

  # Run main.py in a subshell to prevent variable modifications
  (
    CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
      --model_path "$model_path" \
      --cache_dir "$cache_dir" \
      --tasks "$tasks" \
      --num_fewshot "$num_fewshot" \
      --limit "$limit" \
      --eval_ppl "$eval_ppl" \
      --eval_ppl_seqlen "$eval_ppl_seqlen" \
      --use_cuda_graph "$use_cuda_graph" \
      --seed "$seed" \
      --bits_a "$bits_a_local" \
      --sym_a "$sym_a" \
      --groupsize_a "$groupsize_a" \
      --bits_w "$bits_w_local" \
      --sym_w "$sym_w" \
      --groupsize_w "$groupsize_w_local" \
      --smoothquant "$smoothquant" \
      --smoothquant_alpha "$smoothquant_alpha" \
      --smoothquant_dataset "$smoothquant_dataset" \
      --smoothquant_nsamples "$smoothquant_nsamples" \
      --smoothquant_seqlen "$smoothquant_seqlen" \
      --gptq "$gptq" \
      --gptq_dataset "$gptq_dataset" \
      --gptq_nsamples "$gptq_nsamples" \
      --gptq_seqlen "$gptq_seqlen" \
      --gptq_true_sequential "$gptq_true_sequential" \
      --gptq_percdamp "$gptq_percdamp" \
      --gptq_act_order "$gptq_act_order" \
      --gptq_static_groups "$gptq_static_groups" \
      --lutgemm "$lutgemm" --do_packing "$do_packing" --round "$round" --rtn "$rtn_local" \
      --chat "$chat" \
      --logfile "$logfile" \
      --analyze_stats "$analyze_stats" \
      --stats_csv_path "$stats_csv_path" \
      --get_layerwise_distance "$get_layerwise_distance"
  )
}

# Main loop
for bits_a_value in $bits_a; do
  for bits_w_value in "${bits_w_list[@]}"; do
    for groupsize_w_value in "${groupsize_w_list[@]}"; do
      for rtn_value in "${rtn[@]}"; do
        run_main_py "$bits_a_value" "$bits_w_value" "$groupsize_w_value" "$rtn_value"
      done
    done
  done
done
