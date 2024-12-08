#meta-llama/Llama-2-7b-hf
#huggyllama/llama-7b

# default
# gpuid=$1
# k_bits=$2
# v_bits=$3
# group_size=$4
# residual_length=$5
# tasks=$6
# model=$7


gpuid=0
k_bits=4
v_bits=4
group_size=32
residual_length=128
tasks="wikitext"
# model="/raid/LLM/llama2-7b"
model="/raid/LLM/llama3.1-8b-instruct"


model_name="${model#*/}"
echo "$model_name"
CUDA_VISIBLE_DEVICES=$gpuid python run_lm_eval_harness.py --model_name_or_path $model \
    --tasks $tasks \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length 
