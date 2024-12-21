"""
HYU-AIHA-LAB
Quantization Framework for LLM Inferences
"""

import logging
import transformers
import warnings
import torch
from transformers import BitsAndBytesConfig
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def main(args):
    if args.llm_int8:
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        )
    else:
        quantization_config = None
    # Load Huggingface Model
    from utils.import_model import model_from_hf_path
    model = model_from_hf_path(args.model_path,
                args.use_cuda_graph,
                device_map='auto',
                quantization_config=quantization_config,
            ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    # Smooth Model
    if args.smoothquant:
        from lib.smoothquant.get_smooth_model import get_smoothquant_model
        get_smoothquant_model(model, tokenizer, args)

    if args.analyze_stats or args.get_layerwise_distance: # Dump reference weight
        fp_state_dict = dict()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                fp_state_dict[name] = module.weight.data.cpu()

    # Quantization
    if args.zeroquant:
        from lib.zeroquant.get_zeroquant import get_zeroquant_model
        model = get_zeroquant_model(model, tokenizer, device='cuda', args=args)
    else:
        # Weight Quantization
        if args.bits_w < 16:
            if args.lutgemm:
                from lib.lutgemm.quantize_bcq import quantize_lutgemm  # lutgemm-specific BCQ format
                # Case: lutgemm-only (BCQ format)
                quantize_lutgemm(model, args, dev='cuda')
                print("Applied LUT-GEMM with BCQ format.")
                if args.do_packing:
                    # [TODO] using kernel
                    raise NotImplementedError 
            else:
                from lib.quantization.weight_quant import quantize_gptq, quantize_nearest, quantize_spqr
                if args.gptq:
                    # Case: gptq-only
                    quantize_gptq(model, args, dev='cuda')
                    print("Applied GPTQ quantization.")
                elif args.spqr:
                    quantize_spqr(model, args, dev='cuda')
                else:
                    # Case: nearest-only
                    quantize_nearest(model, args, dev='cuda')
                    print("Applied nearest quantization.")
    
        # Activation Quantization
        if args.bits_a < 16 or args.analyze_stats: # Using custom Linear
            from lib.quantization.act_quant import add_act_quant
            add_act_quant(model, args)

    # KV Cache Quantization
    # KIVI
    if args.kivi:
        # delete Vanilla model
        del model
        torch.cuda.empty_cache()
        
        print("\n********** KV Cache Quantization: KIVI **********\n")

        from lib.kivi.models.llama_kivi_qllm import LlamaForCausalLM_KIVI

        # Support only INT4/INT2 Quantization of KV Cache
        assert args.kivi_k_bits in [4, 2] and args.kivi_v_bits in [4, 2]

        config = transformers.LlamaConfig.from_pretrained(args.model_path)
        config.k_bits = args.kivi_k_bits
        config.v_bits = args.kivi_v_bits
        config.group_size = args.kivi_group_size
        config.residual_length = args.kivi_residual_length
        config.prefill_with_quant = args.kivi_prefill_with_quant
        config.use_flash = True # for FlashAttention-2

        # load modified model
        model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            config=config,
        )

    # KVQuant
    if args.kvquant:
        # delete Vanilla model
        del model
        torch.cuda.empty_cache()
        
        print("\n********** KV Cache Quantization: KVQuant **********\n")
        
        from lib.kvquant.models.llama_kvquant_qllm import LlamaForCausalLM_KVQuant
        from lib.kvquant.quant.llama_simquant import get_modified_model_qllm

        # Support only 4/3/2-Bit Quantization of KV Cache
        assert args.kvquant_kv_bits in [4, 3, 2]

        model_name = args.model_path.split('/')[-1]
        quantizer_path = "lib/kvquant/quant/quantizers/"
        quantizer_path += "quantizers_{}_{}bits.pickle".format(model_name, args.kvquant_kv_bits)
        use_flash = True # for FlashAttention-2
        
        # load modified model
        model = get_modified_model_qllm(
            args.model_path, quantizer_path, use_flash,
            args.kvquant_prefill_with_quant, args.kvquant_kv_bits, 
            args.kvquant_nuq, args.kvquant_include_sparse, 
            args.kvquant_sparsity_threshold, args.kvquant_first_few_fp16,
            LlamaForCausalLM_KVQuant
        )

    # Analysis Tool
    if args.analyze_stats:
        from utils.statistics import summarize_stats
        stats = summarize_stats(model, tokenizer, fp_state_dict, args)
        return
    if args.get_layerwise_distance:
        from utils.statistics import get_layerwise_distance
        stats = get_layerwise_distance(model, tokenizer, fp_state_dict, args)
        return

    # Inference (Chatbot, NIAH, Perplexity, LM-Eval)
    ppls = dict()
    results = dict()
    if args.chat:
        from utils.chatbot import chatbot_play
        chatbot_play(model, tokenizer, max_new_tokens=128, device='cuda')
    if args.niah:
        from utils.needle_in_a_haystack.needle_in_a_haystack_example import niah_example
        niah_example(model, tokenizer)
    if args.eval_ppl:
        from utils.perplexity import eval_ppl
        ppls = eval_ppl(model if args.llm_int8 else model.cuda(), tokenizer, args)
    if len(args.tasks) > 0:
        import lm_eval
        lm = lm_eval.models.huggingface.HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                backend='causal',
                trust_remote_code=True,
            )
        results = lm_eval.evaluator.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )['results']
        logging.info(results)

    if args.logfile != 'none':
        import json
        with open(args.logfile, 'a') as file:
            file.write(json.dumps(vars(args), indent=4) + '\n')
            file.write(json.dumps(ppls, indent=4) + '\n')
            file.write(json.dumps(results, indent=4) + '\n')
            file.write('\n')
    return

if __name__ == '__main__':
    import argparse
    from utils.common import *
    parser = argparse.ArgumentParser() 
    # Model and Tasks
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--tasks', type=str2list, default=[])
    parser.add_argument('--num_fewshot', type=str2int, default='none')
    parser.add_argument('--limit',type=str2int, default='none')
    parser.add_argument('--eval_ppl', type=str2bool, default=False)
    parser.add_argument('--eval_ppl_seqlen', type=int, default=2048)
    parser.add_argument('--use_cuda_graph', type=str2bool, default=False)
    parser.add_argument('--seed',type=int, default=0)
    # Quantization Configs
    parser.add_argument('--bits_a', type=int, default=16)
    parser.add_argument('--sym_a', type=str2bool, default=False)
    parser.add_argument('--groupsize_a', type=int, default=-1)
    parser.add_argument('--bits_w', type=int, default=4)
    parser.add_argument('--sym_w', type=str2bool, default=False)
    parser.add_argument('--groupsize_w', type=int, default=-1)
    # SmoothQuant Configs
    parser.add_argument('--llm_int8', type=str2bool, default=False)
    parser.add_argument('--smoothquant', type=str2bool, default=False)
    parser.add_argument('--smoothquant_alpha', type=float, default=0.5)
    parser.add_argument('--smoothquant_dataset', type=str, default='pile')
    parser.add_argument('--smoothquant_nsamples', type=int, default=512)
    parser.add_argument('--smoothquant_seqlen', type=int, default=512)
    # ZeroQuant Configs
    parser.add_argument('--zeroquant', type=str2bool, default=False)
    parser.add_argument('--zeroquant_lkd', type=str2bool, default=False)
    # GPTQ Configs
    parser.add_argument('--gptq', type=str2bool, default=False)
    parser.add_argument('--gptq_dataset', type=str, default='c4')
    parser.add_argument('--gptq_nsamples', type=int, default=128)
    parser.add_argument('--gptq_seqlen', type=int, default=2048)
    parser.add_argument('--gptq_true_sequential', type=str2bool, default=False)
    parser.add_argument('--gptq_percdamp', type=float, default=.01)
    parser.add_argument('--gptq_act_order', type=str2bool, default=False)
    parser.add_argument('--gptq_static_groups', type=str2bool, default=False)
    # SpQR Configs
    parser.add_argument('--spqr', type=str2bool, default=False)
    parser.add_argument('--spqr_qq_scale_bits', type=int, default=4)
    parser.add_argument('--spqr_qq_zero_bits', type=int, default=4)
    parser.add_argument('--spqr_qq_zero_sym', type=str2bool, default=False)
    parser.add_argument('--spqr_qq_groupsize', type=int, default=16)
    parser.add_argument('--spqr_outlier_threshold', type=float, default=float("inf"))
    parser.add_argument('--spqr_simplified_outliers', type=str2bool, default=False)
    parser.add_argument('--spqr_offload_activations', type=str2bool, default=False)
    parser.add_argument('--spqr_load', type=str, default='./cache/spqr')
    parser.add_argument('--spqr_save', type=str, default='./cache/spqr')
    parser.add_argument('--spqr_skip_out_loss', type=str2bool, default=False)
    # KIVI Configs
    parser.add_argument('--kivi', type=str2bool, default=False)
    parser.add_argument('--kivi_k_bits', type=int, default=4)
    parser.add_argument('--kivi_v_bits', type=int, default=4)
    parser.add_argument('--kivi_group_size', type=int, default=32)
    parser.add_argument('--kivi_residual_length', type=int, default=128)
    parser.add_argument('--kivi_prefill_with_quant', type=str2bool, default=False)    
    # KVQuant Configs
    parser.add_argument('--kvquant', type=str2bool, default=False)
    parser.add_argument('--kvquant_kv_bits', type=int, default=4)
    parser.add_argument('--kvquant_nuq', type=str2bool, default=True)
    parser.add_argument('--kvquant_include_sparse', type=str2bool, default=True)
    parser.add_argument('--kvquant_sparsity_threshold', type=float, default=0.99)
    parser.add_argument('--kvquant_first_few_fp16', type=int, default=1)
    parser.add_argument('--kvquant_prefill_with_quant', type=str2bool, default=False)    
    # LUT-GEMM Configs
    parser.add_argument('--lutgemm', type=str2bool, default=False)
    parser.add_argument('--rtn', type=str2bool, default=False)
    parser.add_argument('--do_packing', type=str2bool, default=False)
    parser.add_argument('--round', type=int, default=1)
    # Others
    parser.add_argument('--chat', type=str2bool, default=False)
    parser.add_argument('--niah', type=str2bool, default=False)
    parser.add_argument('--logfile', type=str, default='./logs/dummy')
    # Analysis Tools
    parser.add_argument('--analyze_stats', type=str2bool, default=False)
    parser.add_argument('--stats_csv_path', type=str, default='./cache/stats.csv')
    parser.add_argument('--get_layerwise_distance', type=str2bool, default=False)

    args = parser.parse_args()
    set_seed(args.seed)
    logging.info(args)
    main(args)
