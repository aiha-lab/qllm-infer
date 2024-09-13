"""
HYU-AIHA-LAB
Quantization Framework for LLM Inferences
"""

import logging
import transformers
import warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def main(args):
    # Load Huggingface Model
    from utils.import_model import model_from_hf_path
    model = model_from_hf_path(args.model_path,
                args.use_cuda_graph,
                device_map='auto',
            )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)

    # Quantization
    if args.smoothquant:
        from lib.smoothquant.get_smooth_model import get_smoothquant_model
        get_smoothquant_model(model, tokenizer, args)
    if args.bits_w < 16:
        from lib.quantization.weight_quant import quantize_gptq, quantize_nearest
        if args.gptq:
            quantize_gptq(model, args, dev='cuda')
        else:
            quantize_nearest(model, args, dev='cuda')
    if args.bits_a < 16:
        from lib.quantization.act_quant import add_act_quant
        add_act_quant(model, args)

    # Inference (Chatbot, Perplexity, LM-Eval)
    ppls = dict()
    results = dict()
    if args.chat:
        from utils.chatbot import chatbot_play
        chatbot_play(model, tokenizer, max_new_tokens=128, device='cuda')
    if args.eval_ppl:
        from utils.perplexity import eval_ppl
        ppls = eval_ppl(model, tokenizer, args)
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
    parser.add_argument('--smoothquant', type=str2bool, default=False)
    parser.add_argument('--smoothquant_alpha', type=float, default=0.5)
    parser.add_argument('--smoothquant_dataset', type=str, default='pile')
    parser.add_argument('--smoothquant_nsamples', type=int, default=512)
    parser.add_argument('--smoothquant_seqlen', type=int, default=512)
    # GPTQ Configs
    parser.add_argument('--gptq', type=str2bool, default=False)
    parser.add_argument('--gptq_dataset', type=str, default='c4')
    parser.add_argument('--gptq_nsamples', type=int, default=128)
    parser.add_argument('--gptq_seqlen', type=int, default=2048)
    parser.add_argument('--gptq_true_sequential', type=str2bool, default=False)
    parser.add_argument('--gptq_percdamp', type=float, default=.01)
    parser.add_argument('--gptq_act_order', type=str2bool, default=False)
    parser.add_argument('--gptq_static_groups', type=str2bool, default=False)
    # Others
    parser.add_argument('--chat', type=str2bool, default=False)
    parser.add_argument('--logfile', type=str, default='./logs/dummy')

    args = parser.parse_args()
    set_seed(args.seed)
    logging.info(args)
    main(args)
