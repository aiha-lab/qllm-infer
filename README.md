# LLM Quantization Framework
## Environment
```bash
# Clone the code
git clone https://github.com/aiha-lab/qllm-infer.git
QLLM_PATH=${PWD}/qllm-infer

# Requirements
cd /root/qllm-infer && pip install -r requirements.txt
cd /root/qllm-infer/lm-evaluation-harness && pip install -e . 
```

## Quick Links
- [Weight-Activation Quantization](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#weight-activation-quantization)
  - [SmoothQuant](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#smoothquant-accurate-and-efficient-post-training-quantization-for-large-language-models)
  - [ZeroQuant](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#zeroquant-efficient-and-affordable-post-training-quantization-for-large-scale-transformers-neurips-2022)
- [Weight-Only Quantization](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#weight-only-quantization)
  - [GPTQ](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#gptq-accurate-post-training-quantization-for-generative-pre-trained-transformers)
  - [LUT-GEMM](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#lut-gemm-quantized-matrix-multiplication-based-on-luts-for-efficiency-in-large-scale-generative-language-models-iclr-2024)
  - [SPQR](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#spqr---a-sparse-quantized-representation-for-near-lossless-llm-weight-compression)
- [KV Cache Quantization](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#kv-cache-quantization)
  - [KIVI](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#kivi-a-tuning-free-asymmetric-2bit-quantization-for-kv-cache-icml-2024)
  - [KVQuant](https://github.com/aiha-lab/qllm-infer?tab=readme-ov-file#kvquant-towards-10-million-context-length-llm-inference-with-kv-cache-quantization-neurips-2024)

***
# Weight-Activation Quantization
***
## SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

### About SmoothQuant
**SmoothQuant** is a method that enables faster and more memory-efficient inference for LLMs by performing both weight and activation computations in INT8, leveraging the high-throughput INT8 tensor cores. Due to the large magnitude of activation outliers in LLMs, reducing the bit precision of activations to 8-bit poses significant challenges. To address this, SmoothQuant *migrates* the quantization difficulty of activations into weights using channel-wise scaling.  
The scaling factor in SmoothQuant is applied per input channel, scaling down the magnitude of activation outliers in specific input channels while scaling up the corresponding weights in the same channels to maintain mathematical equivalence ($Y=(X\text{diag}(s)^{-1})(\text{diag}(s)W)=\hat{X}\hat{W}$). SmoothQuant enables W8A8 inference with just a single calibration step and introduces no additional inference overhead. This method has been widely integrated into various frameworks such as TensorRT-LLM and vLLM.

### Key Steps
- **Getting Activation Statistics via Calibration Dataset**:  
  The calibration dataset (e.g., the Pile dataset) is used for inference to determine the maximum magnitude of weights and activations.

- **Determining the Scaling Factor $s$**:  
  - The scaling factor $s$ is determined by the hyperparameter migration strength $\alpha$ and the channel-wise maximum magnitudes of weights and activations.  
  - The scaling factor for channel $j$ is computed as: $s_j=\text{max}(|X_j|)^\alpha/\text{max}(|W_j|)^{1-\alpha}$
  - Migration Strength $\alpha$: This controls the degree of quantization difficulty migration.  
    - Higher $\alpha$ values shift more of the quantization difficulty from activations to weights.
- **Smoothing the Model**:  
  - **For weights**: Scaling factors are pre-multiplied offline and stored as $\text{diag}(s)W$.  
  - **For activations**: To avoid runtime scaling, the scaling factor is pre-applied to the channel-wise weights and biases of the preceding LayerNorm ($X\text{diag}(s)^{-1}$).


### Getting Started
You can run SmoothQuant using the command ```bash scripts/run_smoothquant.sh $GPU_NUM $MODEL_PATH```.  
```sh
# Measuring Perplexity with SmoothQuant
eval_ppl=true
eval_ppl_seqlen=2048
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
```

If you want to analyze the statistics of weights and activations before and after quantization, you can use the analysis tool included in our script.
```sh
# Analysis Tools
analyze_stats=true # true false
get_layerwise_distance=false # true false
```

### Implementation Details
All implementation details are in ```lib/smoothquant```.  
**Getting Activation Statistics**
```py
# lib/smoothquant/calibration.py
def get_act_scales(model, tokenizer, smoothquant_dataset, num_samples=512, seq_len=512, args=None):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        # Getting Max Values
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max
...
```

**Determining Scaling Factor**
```py
# lib/smoothquant/smooth.py
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    ...
    # Activaiton max value from calibration dataset
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # Weight max value
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # Determining scaling factor
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # Scaling previous LayerNorm parameters to avoid runtime scaling for activations
    ln.weight.div_(scales)
    ln.bias.div_(scales)

    # Scaling weight parameters
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
```

### Evaluation Results
#### Determining migration strength for LLaMA3.1-8B-Instruct
We empirically find that $\alpha$=0.55 is best for LLaMA3.1-8B-Instruct.
| Migration Strength    |   0.35  |   0.40  |   0.45  |   0.50  |   **0.55**  |   0.60  |   0.65  |   0.70  |   0.75  |   0.80  |   0.85  |   0.90  |
|-----------------------|:-------:|:-------:|:-------:|:-------:|:-----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Wikitext-2 Perplexity |  7.2984 |  7.2936 |  7.2897 |  7.2867 |  **7.2840** |  7.2907 |  7.3198 |  7.3182 |  7.3189 |  7.3130 |  7.3159 |  7.3190 |
| C4 Perplexity         | 10.4271 | 10.4279 | 10.4163 | 10.4098 | **10.4091** | 10.4258 | 10.4438 | 10.4489 | 10.4452 | 10.4375 | 10.4408 | 10.4405 |

#### Perplexity and Zero-Shot CommonSense Question Answering
|         Model         |    Bit-width    | SmoothQuant | Perplexity ↓ |       | CSQA ↑ |       |       |           |       |            |              | MMLU Average Accuracy ↑ |
|:---------------------:|:---------------:|:-----------:|:------------:|:-----:|:-------------------------:|:-----:|:-----:|:---------:|:-----:|:----------:|:------------:|:-----------------------:|
|                       |                 |             |  Wikitext-2  |   C4  |           BoolQ           | ARC-C | ARC-E | HellaSwag |  PIQA | WinoGrande | CSQA Avearge |                         |
| LLaMA3.1- 8B-Instruct | 16-bit Baseline |             |     7.22     | 10.39 |           84.16           | 51.96 | 81.82 |   59.12   | 79.87 |    73.95   |     71.81    |          67.81          |
|                       |       W8A8      |      -      |     7.30     | 10.43 |           84.34           | 52.56 | 81.52 |   59.08   | 79.22 |    73.24   |     71.66    |          67.41          |
|                       |       W8A8      |      ✓      |     7.28     | 10.41 |           84.22           | 52.22 | 82.37 |   59.25   | 79.65 |    73.95   |     71.94    |          67.44          |
|                       |       W6A6      |      -      |     8.16     | 11.67 |           83.36           | 50.60 | 79.50 |   57.38   | 76.77 |    70.40   |     69.67    |          63.43          |
|                       |       W6A6      |      ✓      |     8.12     | 11.60 |           82.87           | 50.34 | 78.79 |   57.88   | 77.80 |    71.74   |     69.90    |          64.01          |


#### Conversational Ability
By setting `chat=true`, you can run a chatbot simulation using the quantized model.
|          Prompt from User         | Imagine you are participating in a race with a group of people. If you have just overtaken the third person, what's your current position? Where is the person you just overtook?                                                       |   |
|:---------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| 16-bit Baseline (Wiki2 ppl: 7.22) | If you have just overtaken the third person, that means you were behind them before, but now you are ahead of them.  You are now in the 3rd position, and the person you just overtook is now in the 4th position.                      |   |
|     W8A8 RTN (Wiki2 ppl: 7.30)    | If I have just overtaken the third person, that means I was behind them and have now moved ahead of them.  So, my current position is 3rd.  The person I just overtook is now behind me, in 4th position.                               |   |
|     W8A8 SQ (Wiki2 ppl: 7.28)     | If I have just overtaken the third person, that means I was behind them initially. After overtaking them, I am now in their position, which is the third position. The person I just overtook is now behind me, in the fourth position. |   |
|     W6A6 RTN (Wiki2 ppl: 8.16)    | If I've just overtaken the third person, that means I was behind them initially, but I've now moved ahead of them. So, my current position is 2nd, and the person I just overtook is in 3rd place.                                      |   |
|     W6A6 SQ (Wiki2 ppl: 8.12)     | If I have just overtaken the third person, that means I was behind them and have now passed them.  So, my current position is 3rd, and the person I just overtook is now behind me in 4th place.                                        |   |

***
## ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers (NeurIPS 2022)

### **Summary**
**ZeroQuant** is an end-to-end quantization and inference pipeline designed to optimize large-scale transformer models like BERT and GPT using Post-Training Quantization (PTQ) combined with Layer-by-Layer Knowledge Distillation (LKD). It achieves high efficiency and minimal accuracy loss without requiring significant retraining or fine-tuning. This repository’s implementation focuses on reproducing quantization and LKD rather than kernel optimization.

#### **Key Steps**
1.	**Weight Quantization**
- Quantize the weight tensor into 4-bit or 8-bit precision using symmetric group-wise quantization.
2.	**Activation Quantization**
- Apply token-wise dynamic quantization to activations, adapting to the range of each token dynamically.
3.	**Layer-by-Layer Knowledge Distillation (LKD)**
- Enhance the quantized model’s performance by distilling knowledge from a teacher model to the student model at each transformer layer.
- Optimize the alignment between the teacher and student model outputs to minimize performance loss.

---

### **Code Structure**

- **`main.py`**  
  - Execution script: `scripts/run_zeroquant.sh`

- **`zeroquant_setup.sh`**
  - Install the required packages and set up environment variables for running ZeroQuant.
    ``` 
    source ./zeroquant_setup.sh
    ```
    
- **`zeroquant_config.json`**
  - Config file for weight and activation quantization (bit precision, module, group size, etc.)
  - If you want to modify the quantization configuration, edit "weight_quantization" and "activation_quantization" field.

- **`lib/zeroquant/get_zeroquant.py`**  
  - Implements the core functionalities for **Post-Training Quantization (PTQ)** and **Layer-by-Layer Knowledge Distillation (LKD)**.  
  - **Quantization**
    - Performed using DeepSpeed's `init_compression` function.
  
  - **Layer-by-Layer Knowledge Distillation (LKD)**
     - A teacher model guides the student model for each transformer layer.
     - The process involves:
         1. Extracting the hidden states from the teacher model's corresponding layer.
         2. Passing the same inputs through the student model to obtain its outputs.
         3. Computing the MSE loss between the teacher’s and student’s hidden layer outputs.

- **`lib/zeroquant/utils.py`**  
  - Utility code for `get_zeroquant.py`.

---

### **Benchmark**
Performance evaluation of the **Llama-3.1-8B-Instruct** model with ZeroQuant applied to various benchmarks.

**RTN/SmoothQuant/GPTQ** results are based on per-channel quantization. To closely match per-channel quantization in the Llama-3.1-8B model structure, the "number_of_groups" in the ZeroQuant configuration was set to **4096**.

#### **Perplexity on Wikitext & C4**
- Perplexity after applying ZeroQuant quantization (without LKD)
- W8A8: Shows less than a 0.1 perplexity difference compared to SmoothQuant.
- W4/8A16: Applies 4-bit quantization only to FFN. Achieves higher perplexity than RTN’s W4A16 but lower than GPTQ’s W4A16.
- The paper does not mention W4A4, but performance evaluation shows a significant drop in perplexity and benchmark accuracy.
  
|         Model         |  Bits   |   Method    | Wikitext |   C4   |
| :-------------------: | :-----: | :---------: | :------: | :----: |
| LLaMA-3.1-8B-Instruct |   16    |      -      |   7.22   | 10.39  |
|                       |  W8A8   | SmoothQuant |   7.28   | 10.41  |
|                       |  W4A16  |     RTN     |   9.46   | 13.64  |
|                       |  W4A16  |    GPTQ     |   8.59   | 12.49  |
|                       |  W8A8   |  ZeroQuant  |   7.34   | 10.48  |
|                       | W4/8A16 |  ZeroQuant  |   8.97   | 12.63  |
|                       | W4/8A8  |  ZeroQuant  |   9.17   | 12.87  |
|                       |   W4A4  |  ZeroQuant  |  278.39  | 288.77 |


---

#### **Accuracy on Zero-shot CSQA, MMLU**

| Model                 | Bit-width       | Method      | BoolQ | ARC-C | ARC-E | HellaSwag | PIQA  | WinoGrande | CSQA Average | MMLU Average Accuracy ↑ |
| --------------------- | --------------- | ----------- | ----- | ----- | ----- | --------- | ----- | ---------- | ------------ | ----------------------- |
| LLaMA-3.1-8B-Instruct | 16-bit Baseline | -           | 84.16 | 51.96 | 81.82 | 59.12     | 79.87 | 73.95      | 71.81        | 67.81                   |
|                       | W8A8            | SmoothQuant | 84.22 | 52.22 | 82.37 | 59.25     | 79.65 | 73.95      | 71.94        | 67.44                   |
|                       | W4A16           | RTN         | 80.09 | 49.15 | 79.04 | 56.14     | 77.69 | 72.30      | 69.07        | 59.74                   |
|                       | W4A16           | GPTQ        | 81.53 | 42.92 | 74.66 | 57.08     | 77.20 | 70.09      | 67.25        | 62.78                   |
|                       | W8A8            | ZeroQuant   | 84.31 | 51.96 | 81.06 | 59.01     | 80.08 | 72.93      | 71.56        | 67.03                   |
|                       | W4/8A16         | ZeroQuant   | 83.18 | 50.51 | 80.09 | 56.95     | 78.67 | 71.67      | 70.18        | 62.56                   |
|                       | W4/8A8          | ZeroQuant   | 82.94 | 49.49 | 80.64 | 56.72     | 77.80 | 70.96      | 69.76        | 61.89                   |
|                       | W4A4            | ZeroQuant   | 44.86 | 19.88 | 30.26 | 28.05     | 55.01 | 50.12      | 38.03        | 23.81                   |

---
### Layer-by-Layer Knowledge Distillation (LKD)
- We previously conducted LKD experiments on the Llama-3.1-8B-Instruct model and observed that the magnitude of loss varies across layers. This indicates that ZeroQuant’s approach of applying KD to all layers with the same learning rate does not work effectively.
- When LKD was applied to only the last layer, a gradual improvement in perplexity was observed as the number of training steps increased.

| Model                     | Bit-width | LKD (only the last layer)| Learning Rate | Training Steps | Batch Size | Perplexity ↓ (Wikitext-2) | Perplexity ↓ (C4) |
| ------------------------- | --------- | ----------------------- | ------------- | -------------- | ---------- | ------------------------- | ----------------- |
| **LLaMA-3.1-8B-Instruct** | W4/8A16   | -                       | -             | -              | -          | 8.972                     | 12.633            |
|                           |           | ✓                       | 5e-6          | 100            | 4          | 8.969                     | -                 |
|                           |           | ✓                       | 5e-6          | 200            | 4          | 8.944                     | 12.608            |
|                           |           | ✓                       | 5e-6          | 400            | 4          | 8.909                     | 12.580            |
|                           |           | ✓                       | 5e-6          | 600            | 4          | **8.890**                 | **12.564**        |
|                           | W4A4      | -                       | -             | -              | -          | 278.39                    | 288.77            |
|                           |           | ✓                       | 1e-6          | 100            | 4          | 258.00                    | 269.39            |
|                           |           | ✓                       | 1e-6          | 200            | 4          | 251.18                    | 261.74            |
|                           |           | ✓                       | 1e-6          | 400            | 4          | **246.28**                | **256.30**        |

---

#### **Conversational Abilities**
- Excluding W4A4, it generates mostly accurate answers to the questions.

|      **Prompt from User** (Wiki2 ppl)      | "Imagine you are participating in a race with a group of people. If you have just overtaken the third person, what's your current position? Where is the person you just overtook?"                                  |
| :----------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   **16-bit Baseline (Wiki2 ppl: 7.22)**    | If you have just overtaken the third person, that means you were behind them before, but now you are ahead of them. You are now in the 3rd position, and the person you just overtook is now in the 4th position. |
|  **W8A8 ZeroQuant<br>(Wiki2 ppl: 7.34)**   | If I have just overtaken the third person, that means I was behind them and have now passed them. So, my current position is fourth. The person I just overtook is now in fourth place, and I am in third place.  |
| **W4/8A16 ZeroQuant<br>(Wiki2 ppl: 8.97)** | If you have just overtaken the third person, that means you are now in the third position. The person you just overtook is now in the fourth position.                                                               |
| **W4/8A8 ZeroQuant<br>(Wiki2 ppl: 9.17)**  | If you have just overtaken the third person, that means you are now in the third position. The person you just overtook is now in the fourth position.                                                               |
| **W4A4 ZeroQuant<br>(Wiki2 ppl: 278.39)**  | We're back! The next step-aide conversation starting location, to be in a group of people.                                                                                                                           |


***
# Weight-Only Quantization
***
## GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

### About GPTQ
**GPTQ** is a weight-only quantization method designed to reduce the memory overhead of large language models (LLMs). During weight quantization, it sequentially quantizes each column. To minimize the output quantization error, the weights of the remaining unquantized columns are updated iteratively. For memory efficiency, GPTQ performs optimization at the block level, where blocks are composed of grouped columns. Since GPTQ adjusts its own rounding values without modifying quantization parameters or other layer parameters (e.g., LayerNorm), it can be combined with other quantization techniques like SmoothQuant, making it a widely adopted method.

### Key Steps

- **Overall Procedure of GPTQ**:
  1. Group the columns of a weight matrix into blocks of *block_size (default=128)* columns each.
  2. Iteratively quantize the columns within a block, one at a time.
     - The update step is defined as: $\delta _F = - \frac{w_q - \text{quant}(w_q)}{[\textbf{H}_F^{-1}]_{qq}} \cdot (\textbf{H}_F^{-1})_{:,q}$.
     - The Hessian matrix is calculated as: $\textbf{H}_i = \frac{\partial^2E}{\partial \textbf{W}_{i,:}^2} = 2\textbf{XX}^T$.
  3. Update the remaining unquantized columns within the block.
  4. Once all columns in a block are quantized, proceed to quantize the next block.

### Getting Started
You can run GPTQ using the command ```bash scripts/run_gptq.sh $GPU_NUM $MODEL_PATH```.  
For LLaMA models, we have empirically observed that enabling the ```gptq_act_order=True``` option is essential.
```sh
# Measuring Perplexity with GPTQ
eval_ppl=true
eval_ppl_seqlen=2048
# Quantization
bits_a=16
sym_a=false
groupsize_a=-1
bits_w=4
sym_w=false
groupsize_w=-1
# GPTQ
gptq=true
gptq_dataset=c4
gptq_nsamples=128
gptq_seqlen=2048
gptq_true_sequential=false
gptq_percdamp=0.01
gptq_act_order=true
gptq_static_groups=false
```

If you want to analyze the statistics of weights and activations before and after quantization, you can use the analysis tool included in our script.
```sh
# Analysis Tools
analyze_stats=true # true false
get_layerwise_distance=false # true false
```

### Implementation Details
All implementation details are in ```lib/gptq/gptq.py```.  
**Weight updates in GPTQ**: $\delta _F = - \frac{w_q - \text{quant}(w_q)}{[\textbf{H}_F^{-1}]_{qq}} \cdot (\textbf{H}_F^{-1})_{:,q}$  
```py
# lib/gptq/gptq.py
def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
    ...
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                # Update weight via optimal update
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    ...
```
**Calculate Hessian Matrix**: $\textbf{H}_i = \frac{\partial^2E}{\partial \textbf{W}_{i,:}^2} = 2\textbf{XX}^T$
```py
# lib/gptq/gptq.py
def add_batch(self, inp, out):
    ...
    self.H += inp.matmul(inp.t())
```

### Evaluation Results
#### Perplexity
**OPT Models** (```gptq_actorder=False```)
|  Dataset|      | Wikitext-2 ↓ |       |       |       |       | C4 ↓    |       |       |       |       |
|---------|------|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|   OPT   | Bits |     125M    |  1.3B |  2.7B |  6.7B |  13B  |  125M |  1.3B |  2.7B |  6.7B |  13B  |
|   full  |  16  |    27.65    | 14.62 | 12.47 | 10.86 | 10.13 | 24.61 | 14.72 | 13.16 | 11.74 | 11.20 |
|   RTN   |   4  |    37.28    | 48.20 | 16.92 | 12.10 | 11.32 | 31.62 | 24.68 | 17.52 | 13.38 | 12.35 |
|   GPTQ  |   4  |    31.52    | 15.60 | 12.88 | 11.38 | 10.33 | 27.12 | 15.55 | 13.76 | 12.14 | 11.36 |
|   RTN   |   3  |    1.3e3    | 1.3e4 | 1.6e4 | 5.8e3 | 3.4e3 | 7.3e2 | 6.3e3 | 1.2e4 | 4.7e3 | 2.2e3 |
|   GPTQ  |   3  |    54.58    | 21.34 | 16.96 | 15.21 | 11.72 | 39.57 | 19.89 | 16.48 | 15.60 | 12.28 |

**LLaMA Models** (```gptq_actorder=False```)  
You may observe unexpected behavior in GPTQ that can even degrade performance.
| Dataset |      | Wikitext-2 ↓ |       |         |                 | C4 ↓ |       |        |                 |
|:-------:|:----:|:---------------------:|:-----:|:-------:|:---------------:|:---------------:|:-----:|:------:|:---------------:|
|  LLaMA  | Bits |          2-7B         | 2-13B |   3-8B  | 3.1-8B-Instruct |       2-7B      | 2-13B |  3-8B  | 3.1-8B-Instruct |
|   full  |  16  |          5.47         |  4.88 |   6.14  |       7.22      |       6.97      |  6.47 |  8.88  |      10.39      |
|   RTN   |   4  |          6.12         |  5.20 |   8.53  |       9.46      |       7.72      |  6.83 |  12.04 |      13.64      |
|   GPTQ  |   4  |          6.06         |  5.18 |  277.95 |      131.06     |       7.41      |  6.74 |  58.11 |      68.79      |
|   RTN   |   3  |         542.82        | 10.68 | 2193.32 |      897.85     |      404.47     | 12.50 | 476.54 |      466.50     |
|   GPTQ  |   3  |         10.34         |  6.75 |  133.18 |     1426.07     |      10.42      |  8.24 |  54.44 |      409.27     |

**LLaMA Models** **(```gptq_actorder=True```)**  
After applying `actorder`, GPTQ successfully reduces the perplexity of RTN.
| Dataset |      | Wikitext Perplexity ↓ |       |         |                 | C4 Perplexity ↓ |       |        |                 |
|:-------:|:----:|:---------------------:|:-----:|:-------:|:---------------:|:---------------:|:-----:|:------:|:---------------:|
|  LLaMA  | Bits |          2-7B         | 2-13B |   3-8B  | 3.1-8B-Instruct |       2-7B      | 2-13B |  3-8B  | 3.1-8B-Instruct |
|   full  |  16  |          5.47         |  4.88 |   6.14  |       7.22      |       6.97      |  6.47 |  8.88  |      10.39      |
|   RTN   |   4  |          6.12         |  5.20 |   8.53  |       9.46      |       7.72      |  6.83 |  12.04 |      13.64      |
|   GPTQ  |   4  |          5.84         |  5.15 |   7.29  |       8.59      |       7.36      |  6.70 |  10.46 |      12.49      |
|   RTN   |   3  |         542.82        | 10.68 | 2193.32 |      897.85     |      404.47     | 12.50 | 476.54 |      466.50     |
|   GPTQ  |   3  |          8.66         |  6.52 |  27.31  |      25.64      |      10.09      |  8.07 |  29.96 |      29.48      |


#### Zero-Shot CommonSense Question Answering
**OPT-6.7B**
| Method | Bits | Perplexity ↓ |       | Zero-shot CSQA Accuracy ↑ |       |       |           |       |            |         |
|:------:|:----:|:------------:|:-----:|:-------------------------:|:-----:|:-----:|:---------:|:-----:|:----------:|:-------:|
|        |      |     Wiki2    |   C4  |           BoolQ           | ARC-C | ARC-E | HellaSwag |  PIQA | WinoGrande | Average |
|  full  |  16  |     10.86    | 13.16 |           66.06           | 30.46 | 65.57 |   50.51   | 76.28 |    65.19   |  52.87  |
|   RTN  |   4  |     12.10    | 17.52 |           63.30           | 29.10 | 65.57 |   48.76   | 75.79 |    64.25   |  50.11  |
|  GPTQ  |   4  |     11.38    | 13.76 |           63.24           | 31.06 | 64.35 |   49.43   | 75.95 |    63.30   |  50.19  |
|   RTN  |   3  |     5.8e3    | 1.2e4 |           40.89           | 21.25 | 25.84 |   25.74   | 52.88 |    50.43   |  31.43  |
|  GPTQ  |   3  |     15.21    | 16.48 |           62.26           | 28.07 | 59.55 |   44.88   | 73.18 |    61.01   |  47.42  |

**LLaMA3.1-8B-Instruct (with ```gptq_actorder=True```)**
|      Method     | Bits | Perplexity ↓ |        | Zero-shot CSQA Accuracy ↑ |       |       |           |       |            |              | MMLU ↑ |
|:---------------:|:----:|:------------:|:------:|:-------------------------:|:-----:|:-----:|:---------:|:-----:|:----------:|:------------:|:------:|
|                 |      |  Wikitext-2  |   C4   |           BoolQ           | ARC-C | ARC-E | HellaSwag |  PIQA | WinoGrande | CSQA Avearge |        |
| 16-bit Baseline |  16  |     7.22     |  10.39 |           84.16           | 51.96 | 81.82 |   59.12   | 79.87 |    73.95   |     71.81    |  67.81 |
|       RTN       |   4  |     9.46     |  13.64 |           80.09           | 49.15 | 79.04 |   56.14   | 77.69 |    72.30   |     69.07    |  59.74 |
|       GPTQ      |   4  |     8.59     |  12.49 |           81.53           | 42.92 | 74.66 |   57.08   | 77.20 |    70.09   |     67.25    |  62.78 |
|       RTN       |   3  |    897.85    | 466.50 |           53.36           | 19.88 | 28.20 |   28.54   | 55.60 |    50.99   |     39.43    |  22.88 |
|       GPTQ      |   3  |     25.64    |  29.48 |           60.76           | 22.18 | 41.75 |   42.21   | 60.45 |    59.91   |     47.88    |  39.51 |

#### Generation Tasks
**LLaMA3.1-8B-Instruct inference results on BigBench and GSM8K**
|      Method     | BigBench | GSM8K |
|:---------------:|:--------:|:-----:|
| 16-bit Baseline |   69.19  | 75.97 |
|    W4A16 RTN    |   51.60  | 50.57 |
|    W4A16 GPTQ   |   60.54  | 57.54 |

#### Conversational Ability
By setting `chat=true`, you can run a chatbot simulation using the quantized model.
|          Prompt from User         | Give me a short introduction about the knowledge distillation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |   |
|:---------------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| 16-bit Baseline (Wiki2 ppl: 7.22) | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment in resource-constrained environments. The process involves training the student model on the output of the teacher model, rather than the original training data. This allows the student model to learn the patterns and relationships in the data that ... |   |
|    W4A16 RTN (Wiki2 ppl: 9.46)    | Knowledge Distillation is a machine learning technique that enables a smaller model to learn from a larger model. It was first introduced by Google in 2015. The main idea behind knowledge distillation is to transfer knowledge from a complex model to a simpler model, allowing the simpler model to learn from the complex model's outputs. The process involves training the complex model on the training data and then training the simpler model on the outputs of the complex model. The outputs of the complex model are used as soft labels for the simpler model, allowing it to learn from the complex model's predictions. ...                                                              |   |
|    W4A16 GPTQ (Wiki2 ppl: 8.59)   | Knowledge distillation is a machine learning technique that involves training a smaller model to mimic the behavior of a larger, more complex model. This is achieved by training the smaller model on the output of the larger model, allowing it to learn the most important features and relationships present in the data. The goal of knowledge distillation is to create a more efficient and scalable model that can be used in real-world applications, while also reducing the computational resources required to train and deploy the model.                                                                                                                                                    |   |
|   W3A16 RTN (Wiki2 ppl: 897.85)   | I have a short introduction, and I am a short introduction. I am a short introductionGuidIdI am a short introduction. ...                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |   |
|   W3A16 GPTQ (Wiki2 ppl: 25.64)   | I can help with that. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There is no knowledge distillation. I can be a helpful assistant. There ...                                                                                                                                                                |   |
***
## LUT-GEMM: Quantized Matrix Multiplication Based on LUTs for Efficiency in Large-Scale Generative Language Models (ICLR 2024)

### **Summary**
**LUT-GEMM** is an approach designed to accelerate General Matrix Multiplication (GEMM) by replacing costly multiplication operations with low-cost look-up table (LUT) accesses. Rather than introducing a new quantization scheme, it leverages the **BCQ (Binary-Coding Quantization)** format to represent quantized values, making GEMM operations LUT-based and more efficient.

#### **Key Steps**
1. **BCQ-Based Quantization**  
   - Quantize the weight tensor and represent it in the BCQ format.
2. **LUT Construction**  
   - Precompute all possible combinations of the full-precision activation sub-vector and binary patterns, storing them in a LUT.
3. **Matrix Multiplication via LUT**  
   - During inference, each multiplication is handled by a fast LUT lookup rather than a traditional multiply-accumulate operation.

---

### **Code Structure**

All functionalities related to LUT-GEMM can be found in the `lib/lutgemm/` directory.

- **`main.py`**  
  - Merged into the `main` branch  
  - Execution script: `scripts/run_lutgemm.sh`

- **`quantize_bcq.py`**  
  - Contains code for **BCQ format quantization**, including:
    - Quantization, packing, and searching algorithms
  - **Supported Searching Methods**:
    1. **greedy**: The most traditional BCQ method (`round=0`)
    2. **refined greedy**: Reduces quantization error by refining `scales (alpha)` after the greedy search (`round=1`)
    3. **alternating**: Alternately optimizes both `scales` and `binary matrices` (`round>=2`)
  - You can select a **searching method** using the third argument (`round`) in the script.
    - **Example**:
      ```bash
      bash scripts/run_lutgemm.sh 0 llama-3.1b-instruct 3
      ```

- **`rtn_parameter.py`**  
  - Implements **round-to-nearest (RTN)** quantization
  - Includes functionality to convert RTN-quantized values into the BCQ format

- **`setup_kernel.py`**  
  - Used to measure **kernel-level performance**

- **`utils.py`**  
  - Utility code, including classes such as `Quantizer` and `Packer`

---

### **Benchmark**
The BCQ searching algorithm used in LUT-GEMM is based on an alternating approach with 30 rounds.

#### **Perplexity on Wikitext & C4**
- 4-bit Quantization
  - Applying a large number of alternating cycles is somewhat beneficial for LUT-GEMM. However, unlike RTN and GPTQ, it shows significant performance degradation in the per-channel setting.
  - When fine-grained group-wise quantization is applied with a group size of 128, it achieves performance similar to other quantization methods, though a slight performance drop is observed.
- 3-bit Quantization
  - Similarly, even with a high number of cycles, performance in the per-channel setting is significantly subpar.

|      LLaMA     | Bits |  Groupsize  | Wikitext |    C4   |
|:-------------:|:----:|:-----------:|:--------:|:-------:|
| **full**      |  16  |      –      |   7.22   |  10.39  |
| **RTN**       |   4  | per-channel |   9.46   |  13.64  |
| **RTN**       |   4  |    128      |   7.75   |  11.07  |
| **GPTQ**      |   4  | per-channel |   8.59   |  12.49  |
| **GPTQ**      |   4  |    128      |   7.57   |  11.35  |
| **LUT-GEMM** |   4  | per-channel | 111.82  | 166.30  |
| **LUT-GEMM** |   4  |    128      |   8.25   |  11.88  |
| **RTN**       |   3  | per-channel | 897.85   | 466.50  |
| **RTN**       |   3  |    128      |  12.30   |  16.78  |
| **GPTQ**      |   3  | per-channel |  25.64   |  29.48  |
| **GPTQ**      |   3  |    128      |   9.14   |  12.86  |
| **LUT-GEMM** |   3  | per-channel | 941.25  | 664.89  |
| **LUT-GEMM** |   3  |    128      |  79.87  |  54.00  |

---

#### **Accuracy on Zero-shot CSQA (4 tasks), MMLU**
- Zero-shot CSQA inference results (group size = 128):
  - For W4A16, the C4 perplexity shows a similar trend. RTN achieves the best performance, while LUT-GEMM slightly lags behind.
  - For W3A16, the perplexity trend is similar. While there are tasks where it outperforms per-channel RTN and GPTQ, its performance is significantly inferior when compared group-wise.
- MMLU Results: Both W4A16 and W3A16 fall well short of the performance of RTN and GPTQ.

|      Method     | Bits |  Groupsize  | BoolQ | ARC-C |  PIQA | WinoGrande | CSQA Avg. | MMLU ↑ |
|:--------------:|:----:|:-----------:|:-----:|:-----:|:-----:|:----------:|:---------:|:------:|
| **16-bit Baseline** | 16 |     –      | 84.16 | 51.96 | 79.87 |   73.95    |   72.49   |  67.81 |
| **RTN**            |  4 | per-channel | 80.09 | 49.15 | 77.69 |   72.30    |   69.81   |  59.74 |
| **RTN**            |  4 |    128      | 82.97 | 50.34 | 79.71 |   73.56    |   71.65   |  65.24 |
| **GPTQ**           |  4 | per-channel | 81.53 | 42.92 | 77.20 |   70.09    |   67.94   |  62.78 |
| **GPTQ**           |  4 |    128      | 82.42 | 50.26 | 79.60 |   73.16    |   71.36   |  66.42 |
| **LUT-GEMM**       |  4 |    128      | 82.41 | 49.32 | 79.00 |   71.74    |   70.62   |  62.36 |
| **RTN**            |  3 | per-channel | 53.36 | 19.88 | 55.60 |   50.99    |   44.96   |  22.88 |
| **RTN**            |  3 |    128      | 74.10 | 42.41 | 75.68 |   67.17    |   64.84   |  52.46 |
| **GPTQ**           |  3 | per-channel | 60.76 | 22.18 | 60.45 |   59.91    |   50.83   |  39.51 |
| **GPTQ**           |  3 |    128      | 81.56 | 46.16 | 77.20 |   69.22    |   68.54   |  59.22 |
| **LUT-GEMM**       |  3 |    128      | 47.58 | 27.82 | 69.15 |   56.35    |   50.23   |  25.35 |

---

#### **GSM8K Accuracy**

|      Method     | GSM8K |
|:--------------:|:-----:|
| **16-bit Baseline** | 75.97 |
| **W4A16 RTN**       | 69.75 |
| **W4A16 GPTQ**      | 70.96 |
| **W4A16 LUT-GEMM**  | 58.38 |

---

#### **Kernel Latency (msec)**

|  m  |   n   |   k   |    cuBLAS FP16   |    GPTQ w3a16    | LUT-GEMM (w4a16) | LUT-GEMM (w3a16) |
|:---:|:-----:|:-----:|:----------------:|:----------------:|:----------------:|:----------------:|
|  1  |  512  |  2048 |  0.02314 (x1.00) |  0.02518 (x0.92) | 0.02419 (x0.96)  | 0.02316 (x1.00)  |
|  1  | 1024  |  4096 | 0.02598 (x1.00)  | 0.02457 (x1.06)  | 0.02547 (x1.02)  | 0.02381 (x1.09)  |
|  1  | 2048  |  8192 | 0.03563 (x1.00)  | 0.03300 (x1.08)  | 0.03870 (x0.92)  | 0.03539 (x1.01)  |
|  1  | 4096  | 16384 | 0.10519 (x1.00)  | 0.05907 (x1.78)  | 0.07507 (x1.40)  | 0.06074 (x1.73)  |
|  1  | 7168  | 28672 | 0.27271 (x1.00)  | 0.19261 (x1.42)  | 0.14639 (x1.86)  | 0.12277 (x2.22)  |
|  1  | 8192  | 32768 | 0.35090 (x1.00)  | 0.21446 (x1.64)  | 0.17513 (x2.00)  | 0.14435 (x2.43)  |
|  1  |12288  | 49152 | 0.75941 (x1.00)  | 0.44852 (x1.69)  | 0.36207 (x2.10)  | 0.29390 (x2.58)  |

---

#### **Conversational Abilities**
- It generates reasonably natural sentences and does not produce incorrect information.

| **Prompt from User** (Wiki2 ppl)     | "Give me a short introduction about the knowledge distillation"|
|:------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **16-bit Baseline (Wiki2 ppl: 7.22)** | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (the teacher model) to a smaller, simpler, and more efficient model (the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements. The student model is trained on the teacher model’s outputs, rather than the original dataset, allowing it to learn high-level patterns in the data... |
| **W4A16 RTN (Wiki2 ppl: 9.46)**      | Knowledge distillation is a machine learning technique that enables a smaller model to learn from a larger model. It was first introduced by Google in 2015. The main idea is to transfer knowledge from a complex model to a simpler model. The simpler model is trained on the complex model's outputs (soft labels), allowing it to learn from the complex model’s predictions. This approach can reduce model size while retaining performance...                                                  |
| **W4A16 GPTQ (Wiki2 ppl: 8.59)**     | Knowledge distillation involves training a smaller model to mimic the behavior of a larger, more complex model. It does this by using the larger model's predictions as training targets, allowing the smaller model to capture the most important features and relationships in the data. The result is a more efficient and scalable model that requires fewer computational resources.                                                                                                                |
| **W4A16 LUT-GEMM (Wiki2 ppl: 8.25)** | Knowledge distillation is a deep learning technique used for model compression and improvement. It was first proposed in 2014 by Google. The idea is to train a smaller (student) model to mimic the behavior of a larger (teacher) model, which is usually a large neural network trained on a vast dataset. The student model is thus more compact, easier to deploy, and less resource-intensive than the teacher model, yet retains much of its performance.     

***
## SpQR - A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

### Summary
SpQR is a hybrid sparse quantization form of compression that can compress an accurate pre-trained LLM to 3 to 4 bits per parameter with almost no loss. SpQR is able to achieve this compression ratio while achieving an end-to-end accuracy error of less than 1% over the density baseline. SpQR works by combining two innovations. First, it isolates outlier weights that appear to be causing disproportionately high errors, keeping these weights at high precision and storing other weights in a much lower format (e.g., 3 bits). Second, it implements group quantization with very small group sizes (e.g., 16 neighbors), but maintains compression by quantizing the quantization scale itself to a 3-bit representation.
### Key Steps
1. Bi-level quantization
	1. Row-wise weight quantization & outlier decomposition
	   - Perform weight quantization by groupsize ($\beta_1$) and identify per-weight outliers.
	2. Requantization & column-wise statistic quantization
	   - Isolate and re-quantize the identified outliers.
	   - At this point, we quantize the quantization statistics (scale, zero-point, etc.) by another groupsize ($\beta_2$).
2. Applying GPTQ
   - Perform GPTQ based on the weights generated by bi-level quantization.
   - Any outliers that occur during the GPTQ procedure are further identified and maintained at full-precision.
### Code Structure
All functionalities related to SpQR can be found in the `lib/spqr/` directory.

- `main.py`
	- Merged into the main branch
	- Execution script: `scripts/run_spqr.sh`

- `spqr/quant_group.py`
	- Includes a Quantizer with extended functionality for bilevel quantization.
		- `bits` ($b_w$) - number of bits per weight
		- `qq_scale_bits` ($b_s$) - number of bits per scale
		- `qq_zero_bits` ($b_z$) - number of bits per zero
		- `groupsize` ($\beta_1$) - groupsize for weight quantization
		- `qq_groupsize` ($\beta_2$) - groupsize for statistic quantization
	- It should be used separately from the Quantizer included in the GPTQ library.
- `spqr/spqr_engine.py`
	- Contains classes and functions that are key to performing SpQR quantization.
	- `get_leave_one_out_error` (function)
		- Measure the change in loss when excluding a specific weight from a quantization group
		- Quantify how much each weight contributes to quantization loss (outlier selection criteria)
	- `SPQRUtils` (class)
		- SPQRUtils class Like GPTQ, it generates the hessian matrix via the `add_batch` class method, and performs bilevel quantization and outlier decomposition in the `quantize` class method.
### Benchmark
- The SpQR quantization scheme can have different average number of bits per weight depending on the combination of parameters $b_w, b_s, b_z, r_o, \beta_1, \beta_2$. For a fair comparison with other quantization algorithms, we set the average number of bits per weight to be around 3 or 4 bits, see the [paper](https://arxiv.org/abs/2306.03078) for more details.

SpQR configuration

| Target Model | 3-bit  | 4-bit  |
| ------------ | ------ | ------ |
| $b_w$        | 2      | 3      |
| $b_s$        | 3      | 3      |
| $b_z$        | 3      | 3      |
| $r_o$        | 2.358% | 1.138% |
| $\beta_1$    | 32     | 16     |
| $\beta_2$    | 16     | 16     |
| avg bits     | 3.07   | 3.99   |
> [!note] Note
> The `spqr_engine` does not identify outliers by outlier rate ($r_o$). You need to tweak the `outlier_relative_threshold` parameter for each model to find the optimal avg_bits.

#### Perplexity on Wikitext & C4
- SpQR quantization is more accurate than GPTQ at the same quantization level. It is worth noting that since SpQR performs bilevel quantization, the number of weight bits is one bit less than GPTQ (e.g., $b_w$=2 in the 3.07 configuration and $b_w$=3 in the 3.99 configuration).

| Method | bits | wiki2  |   c4   |
| :----: | :--: | :----: | :----: |
|  full  |  16  |  7.24  | 10.41  |
|  RTN   |  4   |  9.51  | 13.66  |
|  GPTQ  |  4   |  9.03  |  12.6  |
|  SpQR  | 3.99 |  7.73  | 11.11  |
|  RTN   |  3   | 897.85 | 466.50 |
|  GPTQ  |  3   | 25.64  | 29.48  |
|  SpQR  | 3.07 | 11.77  | 15.33  |

#### Accuracy on Zero-shot CSQA (5 tasks)
 - Similar to the results for Perplexity, SpQR with the same bit level outperforms GPTQ in all tasks

| Method | bits | WinoGrande | PICA  | HellaSwag | ARC-E | ARC-C | Average |
| ------ | ---- | ---------- | ----- | --------- | ----- | ----- | ------- |
| full   | 16   | 74.11      | 80.03 | 59.1      | 82    | 52.05 | 69.46   |
| RTN    | 4    | 72.45      | 77.75 | 56.16     | 79.08 | 48.98 | 66.88   |
| GPTQ   | 4    | 71.51      | 79.76 | 56.35     | 78.28 | 47.35 | 66.65   |
| SpQR   | 3.99 | 73.56      | 80.09 | 58.51     | 80.77 | 50.68 | 68.72   |
| GPTQ   | 3    | 59.91      | 60.44 | 42.21     | 41.75 | 22.18 | 42.30   |
| SpQR   | 3.07 | 66.38      | 75.19 | 51.80     | 68.52 | 36.18 | 59.61   |

#### Conversational Abilities
- Despite being quantized at the 3-bit level, it shows good conversational ability (no repetition or incorrect explanations)

| Prompt from User                                     | Give me a short introduction about the knowledge distillation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 16-bit Baseline  <br>(Wiki2 ppl: 7.22)               | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment in resource-constrained environments.  <br>The process involves training the student model on the output of the teacher model, rather than the original training data. This allows the student model to learn the patterns and relationships in the data that ... |
| SpQR  <br>(avg_bits: 3.99)<br>(Wiki2 ppl: 7.73)  | Knowledge distillation is a machine learning technique used to reduce the complexity of a large, complex model (the "teacher") into a smaller, simpler model (the "student") while preserving its performance on a given task. This process involves transferring the knowledge from the teacher model to the student model, resulting in a more efficient and interpretable model that can be deployed in various applications.<br><br>The goal of knowledge distillation is to:<br><br>1.  Reduce model size: By compressing the teacher model into a smaller student model, knowledge distillation helps reduce the computational resources required for inference and deployment...                         |
| SpQR  <br>(avg bits: 3.07)<br>(Wiki2 ppl: 11.49) | Knowledge distillation is a technique in machine learning where a model, typically a neural network, is trained to mimic the behavior of a more complex or larger model. This process involves transferring knowledge from a teacher model to a student model, allowing the student model to learn from the teacher model's expertise.<br><br>Key Applications<br><br>Knowledge distillation is commonly used in:<br><br>1. Transfer learning: Transfer knowledge from a pre-trained model to a new, smaller model.<br><br>2. Model compression: Reduce the size of a model while maintaining its performance.<br><br>3. Knowledge transfer...                                                                  |


***
# KV Cache Quantization
***
## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (ICML 2024)

### **Summary**

**KIVI** is a tuning-free INT4 and INT2 KV cache quantization method for LLMs, enabling throughput increase. By leveraging asymmetric quantization, the K cache is quantized per-channel, and the V cache is quantized per-token, significantly reducing memory usage. Additionally, the full precision sliding window preserves recent tokens in full precision to minimize performance degradation.

---

### **Code Structure**

All functionalities related to KIVI can be found in the `lib/kivi` directory.

- **`main.py`**  
  - Merged into the `main` branch  
  - Execution script: `scripts/run_kvquant.sh`

- **`lib/kivi/models/llama_kivi_qllm.py`**  
  - Contains code for LLaMA model using KV cache quantization method (KIVI).

---

### **Benchmark**
- In all experiments below, the argument `group_size` and `residual_length` were set to `32` and `128`, respectively.
- The metrics for `MMLU`, `GSM8K`, and `TruthfulQA` shown in the table are `acc,none`, `exact_match,strict-match`, and `bleu_acc,none`.


#### **Perplexity on Wikitext & C4**
- The argument `prefill_with_quant` is set to `False`.
- KIVI-4 achieves comparable performance to the 16-bit Baseline in terms of perplexity, whereas KIVI-2 exhibits a noticeable increase.

|    Method    | Bits | Wikitext-2 |   C4   |
|:------------:|:----:|:----------:|:------:|
| **Baseline** |  16  |    7.24    | 10.34  |
|  **KIVI-4**  |   4  |    7.27    | 10.38  |
|  **KIVI-2**  |   2  |    8.83    | 12.40  |


#### **Accuracy on Zero-shot CSQA (4 tasks), MMLU**
- The argument `prefill_with_quant` is set to `False`.
- Both KIVI-4 and KIVI-2 achieve benchmark scores comparable to the 16-bit Baseline.

| **Method**   | **Bits** | **BoolQ** | **ARC-C** | **ARC-E** | **HellaSwag** | **PIQA** | **WinoGrande** | **CSQA Avg.** | **MMLU ↑** |
|:------------:|:--------:|:---------:|:---------:|:---------:|:-------------:|:-------:|:--------------:|:-------------:|:----------:|
| **Baseline** |    16    |   84.07   |   52.05   |   82.00   |     59.10     |  80.03  |      74.11     |     71.89     |    67.92   |
| **KIVI-4**   |     4    |   84.07   |   51.71   |   81.90   |     59.11     |  79.92  |      73.88     |     71.77     |    67.80   |
| **KIVI-2**   |     2    |   83.21   |   51.71   |   81.86   |     59.10     |  79.82  |      73.88     |     71.60     |    66.92   |



#### **Accuracy on GSM8K and TruthfulQA**
- The argument `prefill_with_quant` is set to `True`.
- KIVI-4 achieves benchmark scores comparable to the 16-bit Baseline on GSM8K. However, on TruthfulQA, KIVI-2 shows a slight improvement in benchmark scores, consistent with the trend reported in the paper.

|    Method    | Bits | GSM8K ↑ | TruthfulQA ↑ |
|:------------:|:----:|:-------:|:------------:|
| **Baseline** |  16  |  76.27  |     61.32    |
|  **KIVI-4**  |   4  |  76.19  |     61.69    |
|  **KIVI-2**  |   2  |  74.53  |     63.28    |


#### **Conversational Abilities**
- The argument `prefill_with_quant` is set to `True`.
- The argument `use_cache` of `model.generate()` in `utils/chatbot.py` is set to `True`.
- 16-bit Baseline, KIVI-4, and KIVI-2 generate reasonably natural sentences and do not produce incorrect information.

| **Prompt from User** | "Give me a short introduction about the knowledge distillation." |
|:------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **16-bit Baseline (TruthfulQA: 61.32)** | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment in resource-constrained environments. The process involves training the student model on the output of the teacher model, rather than the original training data. This allows the student model to learn the patterns and relationships in the data that ... |
| **KIVI-4 (TruthfulQA: 61.69)**      | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment in resource-constrained environments. The process involves training the student model on the output of the teacher model, rather than the original training data. This allows the student model to learn the patterns and relationships in the data that ... |
| **KIVI-2 (TruthfulQA: 63.28)**     | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment in real-world applications. \*\*Key Idea:\*\* The process involves training the student model to mimic the output of the teacher model, rather than directly learning from the original training data. This is achieved by minimizing the difference ... |


#### **Retrieval Abilities**
- The argument `prefill_with_quant` is set to `True`.
- The retrieval ability of each method is evaluated using five Needle-In-A-Haystack task examples.
- The code related to this task is located in `utils/needle_in_a_haystack` directory.
- 16-bit Baseline and KIVI-4 perform well on the Needle-In-A-Haystack task example, but the accuracy drops with KIVI-2.

| **Needle Word**     |   47x291   |  8a3605  |  1294k8  |    590p37   |   68v423   | Accuracy |
|:-------------------:|:----------:|:--------:|:--------:|:-----------:|:----------:|:--------:|
| **16-bit Baseline** |  "47x291." | "8a3605."| "1294k8."| "5940. The" |  "68v423." |    4/5   |
| **KIVI-4**          |  "47x291." | "8a3605."| "1294k8."| "5940. The" |  "68v423." |    4/5   |
| **KIVI-2**          | "47x. The" | "8a3605."|  "1294k."|  "594p37."  |  "68v423." |    2/5   |

***
## KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization (NeurIPS 2024)

### **Summary**

**KVQuant** is a 4/3/2-bit KV cache quantization method designed for LLMs, enabling efficient memory usage and faster inference. This method includes Per-Channel Quantization for the K cache, Pre-RoPE Quantization for the K cache, Non-Uniform Quantization for the KV cache, and Per-Vector Dense-and-Sparse Quantization for the KV cache.

---

### **Code Structure**

All functionalities related to KVQuant can be found in the `lib/kvquant` directory.

- **`main.py`**  
  - Merged into the `main` branch  
  - Execution script: `scripts/run_kvquant.sh`

- **`lib/kvquant/models/llama_kvquant_qllm.py`**  
  - Contains code for LLaMA model using KV cache quantization method (KVQuant).

- **`lib/kvquant/quant/llama_simquant.py`**  
  - Contains code to create quantizer files for each LLM. If you need to recreate the quantizer file for LLaMA-3.1-8B-Instruct or create one for another LLM, you can use this code.

- **`lib/kvquant/quant/quantizers`**  
  - Contain quantizer files (4/3/2-bit with 1% outlier) for LLaMA-3.1-8B-Instruct.

---

### **Benchmark**
- In all experiments below, the argument `nuq`, `include_sparse`, `sparsity_threshold`, and `first_few_fp16` were set to `true`, `true`, `0.99`, and `1`, respectively.
- The metrics for `MMLU`, `GSM8K`, and `TruthfulQA` shown in the table are `acc,none`, `exact_match,strict-match`, and `bleu_acc,none`.


#### **Perplexity on Wikitext & C4**
- The argument `prefill_with_quant` is set to `False`.
- KVQuant-4bit-1% and KVQuant-3bit-1% achieve comparable performance to the 16-bit Baseline in terms of perplexity, whereas KVQuant-2bit-1% exhibits a noticeable increase.

|       Method        | Bits | Wikitext-2 |   C4   |
|:-------------------:|:----:|:----------:|:------:|
|    **Baseline**     |  16  |    7.24    | 10.34  |
| **KVQuant-4bit-1%** |   4  |    7.29    | 10.41  |
| **KVQuant-3bit-1%** |   3  |    7.42    | 10.58  |
| **KVQuant-2bit-1%** |   2  |    8.27    | 11.69  |


#### **Accuracy on Zero-shot CSQA (4 tasks), MMLU**
- The argument `prefill_with_quant` is set to `False`.
- KVQuant-4bit-1% and KVQuant-3bit-1% achieve comparable performance to the 16-bit Baseline, whereas KVQuant-2bit-1% exhibits a noticeable decrease.

|       **Method**       | **Bits** | **BoolQ** | **ARC-C** | **ARC-E** | **HellaSwag** | **PIQA** | **WinoGrande** | **CSQA Avg.** | **MMLU ↑** |
|:-----------------------:|:--------:|:---------:|:---------:|:---------:|:-------------:|:-------:|:--------------:|:-------------:|:----------:|
|       **Baseline**      |    16    |   84.07   |   52.05   |   82.00   |     59.10     |  80.03  |      74.11     |     71.89     |    67.92   |
|    **KVQuant-4bit-1%**  |     4    |   84.00   |   52.73   |   81.81   |     59.18     |  79.92  |      73.24     |     71.81     |    67.13   |
|    **KVQuant-3bit-1%**  |     3    |   83.88   |   51.70   |   82.03   |     58.61     |  78.94  |      73.09     |     71.38     |    65.92   |
|    **KVQuant-2bit-1%**  |     2    |   80.80   |   48.72   |   79.25   |     56.81     |  78.45  |      71.03     |     69.18     |    59.63   |




#### **Accuracy on GSM8K and TruthfulQA**
- The argument `prefill_with_quant` is set to `True`.
- KVQuant-4bit-1% and KVQuant-3bit-1% achieve comparable performance to the 16-bit Baseline, whereas KVQuant-2bit-1% exhibits a noticeable decrease.

|      **Method**     | **Bits** | **GSM8K ↑** | **TruthfulQA ↑** |
|:-------------------:|:--------:|:-----------:|:----------------:|
|    **Baseline**     |    16    |    76.27    |       61.32      |
| **KVQuant-4bit-1%** |     4    |    76.65    |       60.95      |
| **KVQuant-3bit-1%** |     3    |    75.66    |       60.59      |
| **KVQuant-2bit-1%** |     2    |    56.63    |       57.41      |



#### **Conversational Abilities**
- The argument `prefill_with_quant` is set to `True`.
- The argument `use_cache` of `model.generate()` in `utils/chatbot.py` is set to `True`.
- 16-bit Baseline, KVQuant-4bit-1%, KVQuant-3bit-1%, and KVQuant-2bit-1% generate reasonably natural sentences and do not produce incorrect information.

| **Prompt from User** | "Give me a short introduction about the knowledge distillation." |
|:------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **16-bit Baseline (TruthfulQA: 61.32)** | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment in resource-constrained environments. The process involves training the student model on the output of the teacher model, rather than the original training data. This allows the student model to learn the patterns and relationships in the data that ... |
| **KVQuant-4bit-1% (TruthfulQA: 60.95)** | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (known as the teacher model) to a smaller, simpler, and more efficient model (known as the student model). The goal of knowledge distillation is to preserve the performance of the teacher model while reducing its size and computational requirements, making it more suitable for deployment on devices with limited resources. The process involves training the student model on the output of the teacher model, rather than the original training data. This allows the student model to learn the patterns and relationships in the data that ... |
| **KVQuant-3bit-1% (TruthfulQA: 60.59)** | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (the teacher) to a smaller, simpler, and more efficient model (the student). The goal of knowledge distillation is to preserve the essential information and patterns learned by the teacher model, while reducing its size and computational requirements. \*\*Key Concepts:\*\* 1. \*\*Teacher Model:\*\* A large, complex model that has been trained on a large dataset and has high accuracy. 2. \*\*Student Model:\*\* A smaller, simpler model that is trained to mimic the behavior of ... |
| **KVQuant-2bit-1% (TruthfulQA: 57.41)** | Knowledge distillation is a machine learning technique used to transfer knowledge from a complex, large, and computationally expensive model (the teacher) to a smaller, simpler, and more efficient model (the student). The goal of knowledge distillation is to compress the knowledge and capabilities of the teacher model into a more compact and deployable form, while maintaining its performance and accuracy. \*\*Key Concepts:\*\* 1. \*\*Teacher Model:\*\* The complex, large, and computationally expensive model that serves as the source of knowledge. 2. \*\*Student Model:\*\* The smaller, simpler, and more efficient ... |


#### **Retrieval Abilities**
- The argument `prefill_with_quant` is set to `True`.
- The retrieval ability of each method is evaluated using five Needle-In-A-Haystack task examples.
- The code related to this task is located in `utils/needle_in_a_haystack` directory.
- 16-bit Baseline, KVQuant-4bit-1%, and KVQuant-3bit-1% perform well on the Needle-In-A-Haystack task example, but the accuracy drops with KVQuant-2bit-1%.

|   **Needle Word**   |   47x291   |   8a3605   |  1294k8   |    590p37   |   68v423  | Accuracy |
|:-------------------:|:----------:|:----------:|:---------:|:-----------:|:---------:|:--------:|
| **16-bit Baseline** | "47x291."  | "8a3605."  | "1294k8." | "5940. The" | "68v423." |    4/5   |
| **KVQuant-4bit-1%** | "47x291."  | "8a3605."  | "1294k8." | "5940. The" | "68v423." |    4/5   |
| **KVQuant-3bit-1%** | "47x291."  | "8a3605."  | "1294k8." |  "594p37."  | "68v423." |    4/5   |
| **KVQuant-2bit-1%** |  "47x29."  | "8a3605."  | "1294k8." | "5940. The" |  "68v43." |    2/5   |

# References
```bib
@InProceedings{xiao2023smoothquant,
    title = {{S}mooth{Q}uant: Accurate and Efficient Post-Training Quantization for Large Language Models},
    author = {Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
    booktitle = {Proceedings of the 40th International Conference on Machine Learning},
    year = {2023}
}
```
```bib
@inproceedings{
frantar2023optq,
title={{OPTQ}: Accurate Quantization for Generative Pre-trained Transformers},
author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=tcbBPnfwxS}
}
```
```bib
@article{liu2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  journal={arXiv preprint arXiv:2402.02750},
  year={2024}
}
```
```bib
@article{hooper2024kvquant,
  title={KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization},
  author={Hooper, Coleman and Kim, Sehoon and Mohammadzadeh, Hiva and Mahoney, Michael W and Shao, Yakun Sophia and Keutzer, Kurt and Gholami, Amir},
  journal={arXiv preprint arXiv:2401.18079},
  year={2024}
}
```
```bib
@inproceedings{
dettmers2024spqr,
title={Sp{QR}: A Sparse-Quantized Representation for Near-Lossless {LLM} Weight Compression},
author={Tim Dettmers and Ruslan A. Svirschevski and Vage Egiazarian and Denis Kuznedelev and Elias Frantar and Saleh Ashkboos and Alexander Borzunov and Torsten Hoefler and Dan Alistarh},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Q1u25ahSuy}
}
```
```bib
@misc{park2023lutgemm,
      title={LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models}, 
      author={Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee and Dongsoo Lee},
      year={2023},
      eprint={2206.09557},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
```bib
@misc{yao2022zeroquantefficientaffordableposttraining,
      title={ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers}, 
      author={Zhewei Yao and Reza Yazdani Aminabadi and Minjia Zhang and Xiaoxia Wu and Conglong Li and Yuxiong He},
      year={2022},
      eprint={2206.01861},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2206.01861}, 
}
```
