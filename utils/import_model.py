import transformers
import logging
from . import graph_wrapper

def model_from_hf_path(path,
                       use_cuda_graph=True,
                       device_map='auto',
                       quantization_config=None,
                       ):

    def maybe_wrap(use_cuda_graph):
        return (lambda x: graph_wrapper.get_graph_wrapper(x)
                ) if use_cuda_graph else (lambda x: x)

    model_cls = transformers.AutoModelForCausalLM
    model_str = path

    attn_implementation='eager'

    if attn_implementation=='eager':
        logging.warning("Using attn_implmentation='eager' disables SdpaAttention and FlashAttention")

    model = maybe_wrap(use_cuda_graph)(model_cls).from_pretrained(
        path,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
        device_map=device_map)

    return model
