# LLM Quantization Framework

## Getting Started

### Environment
```bash
# Docker
docker run -it --rm --gpus all --ipc=host -v /home/hwanii/workspace/qllm-infer:/root/qllm-infer -v /raid:/raid 166.104.35.43:5000/hwanii/pytorch2.1-cuda11.8:1.2 bash

cd /root/qllm-infer && pip install -r requirements.txt
cd /root/qllm-infer/lm-evaluation-harness && pip install -e . 
```

### Quantize and Evaluate
```bash
bash scripts/run.sh 1 /raid/LLM/llama2-7b
```

## Results

### Perplexity and Zero-shot CSQA Results
- Quantization Settings: Row-wise Asymmetric Uniform Quantization
- SmoothQuant Settings
  - Alpha: 0.85 (following https://github.com/mit-han-lab/smoothquant?tab=readme-ov-file#perplexity-results-on-llama-123-falcon-mistral-and-mixtral-with-w8a8-quantization)
  - Calibration Set: Pile Dataset
- GPTQ Settings
  - Calibration Set: C4 Dataset

**LLaMA2-7B Results**
|               | Quantization Method |      | Perplexity |        | CSQA Zero-shot Accuracy |        |            |
|---------------|---------------------|------|------------|--------|-------------------------|--------|------------|
|               | SmoothQuant         | GPTQ | Wikitext   | C4     | BoolQ                   | PIQA   | Winogrande |
| FP16 Baseline | -                   | -    | 5.472      | 6.973  | 77.737                  | 78.074 | 69.061     |
| W8A8          | x                   | x    | 5.560      | 7.074  | 77.431                  | 77.802 | 69.061     |
|               | o                   | x    | 5.514      | 7.029  | 77.737                  | 77.748 | 69.298     |
|               | x                   | o    | 5.569      | 7.073  | 78.043                  | 77.965 | 69.061     |
|               | o                   | o    | 5.515      | 7.026  | 78.104                  | 78.128 | 68.745     |
| W4A16         | x                   | x    | 6.116      | 7.719  | 73.609                  | 76.768 | 68.350     |
|               | o                   | x    | 23.612     | 25.831 | 59.786                  | 65.180 | 56.354     |
|               | x                   | o    | 6.058      | 7.415  | 74.465                  | 76.496 | 67.403     |
|               | o                   | o    | 7.029      | 8.174  | 71.774                  | 76.333 | 66.614     |
| W4A8          | x                   | x    | 6.225      | 7.859  | 73.028                  | 76.877 | 67.640     |
|               | o                   | x    | 22.349     | 24.183 | 60.887                  | 64.799 | 56.354     |
|               | x                   | o    | 6.866      | 7.667  | 73.914                  | 76.224 | 67.561     |
|               | o                   | o    | 7.184      | 8.274  | 71.713                  | 75.462 | 66.535     |
