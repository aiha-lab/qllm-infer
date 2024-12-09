#!/bin/bash


# model_name="llama2-7b"
# seqlen=2048
# DEVICES=0

model_name="llama3.1-8b-instruct"
seqlen=2048
DEVICES=1


dataset="wikitext2" # wikitext2 only
kv_bits_list=(4 3 2)

for kv_bits in "${kv_bits_list[@]}"
do
echo -e "\n******************************"
echo "Generate Quantizer"
echo "Model: ${model_name}"
echo "KV Bits: ${kv_bits}"
echo "Dataset: ${dataset}"
echo "Sequence Length: ${seqlen}"
echo -e "******************************\n"

CUDA_VISIBLE_DEVICES=$DEVICES python llama_simquant.py \
    "/raid/LLM/${model_name}" \
    --abits $kv_bits \
    --nsamples 16 \
    --seqlen $seqlen \
    --nuq \
    --include_sparse \
    --sparsity-threshold 0.99 \
    --quantizer-path "quantizers/quantizers_${model_name}_${kv_bits}bits.pickle" \
    --dataset $dataset \
    --fisher "../gradients/outputs/${model_name}/" \
    --quantize
done