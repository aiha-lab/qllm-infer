docker run -it --name hmk_docker --gpus all --ipc=host \
-v /home/hmk/workspace/study-dl-llm-quant/qllm-infer:/root/qllm-infer \
-v /raid:/raid \
166.104.35.43:5000/hwanii/pytorch2.1-cuda11.8:1.2 bash

