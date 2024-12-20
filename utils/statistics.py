import torch
import random
import logging
import gc
from datasets import load_dataset
from lib.quantization.act_quant import ActQuantLinear, ActQuantMatMul
from lib.quantization.metric import sqnr, mse

def summarize_stats(model, tokenizer, fp_state_dict, args):
    # For reference weight
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_buffer('fp_weight',fp_state_dict[name].data)
    # Get statistics from 10 wiki samples
    from utils.perplexity import eval_ppl
    ppls = eval_ppl(model.cuda(),
               tokenizer, args,
               nsamples=10, datasets=['wikitext2'],
           )
    # Summary: Average stats over 10 samples
    stats = dict()
    for k, v in model.state_dict().items():
        if 'stat' in k:
            stats[k] = v.mean().item()
    import pandas as pd
    from collections import OrderedDict
    parsed_stats = [
        {'layer': key.split('.stat_')[0],
         'metric_type': key.split('.stat_')[1], 'value': value}
        for key, value in stats.items()
    ]
    df = pd.DataFrame(parsed_stats)
    layer_order = list(OrderedDict.fromkeys(key.split('.stat_')[0] for key in stats.keys()))
    df['layer'] = pd.Categorical(df['layer'], categories=layer_order, ordered=True)
    result = df.pivot(index='layer', columns='metric_type', values='value')
    result.columns.name = None
    formatted_result = result.applymap(
                           lambda x: f"{x:.2e}" if (
                               isinstance(x, (int, float))
                               and abs(x)<1e-4
                               and abs(x)>0
                           )else x
                       )
    print(formatted_result)
    """Save your results if you want"""
    result.to_csv(args.stats_csv_path)
    return result

def block_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_output is None:
        module.raw_output = []
    module.raw_input=input[0].cpu().detach()
    module.raw_output=output.cpu().detach()

def forward_with_hooks(model, dataloader, device='cuda'):
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            setattr(module,'raw_input',None)
            setattr(module,'raw_output',None)
            hooks.append(module.register_forward_hook(block_forward_hook))

    # Forward (only first batch) import pdb; pdb.set_trace()
    with torch.no_grad():
        model(dataloader[0][0].to(model.device))

    for hook in hooks:
        hook.remove()

    # post-processing
    act_dict = {'input':dict(), 'output':dict()}
    for name, module in model.named_modules():
        if hasattr(module,'raw_input') and hasattr(module,'raw_output'):
            act_dict['input'][name] = module.raw_input
            act_dict['output'][name] = module.raw_output
            delattr(module,'raw_input')
            delattr(module,'raw_output')

    gc.collect()
    torch.cuda.empty_cache()

    return act_dict

def get_layerwise_distance(model, tokenizer, fp_state_dict, args):
    """Default: Wikitext trainset
    Sequence length: 2048
    """
    seqlen=2048
    seed=1234
    dataset = load_dataset('wikitext','wikitext-2-raw-v1',split='train[:1%]')
    logging.info('===> Load dataset done')
    trainenc = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(1): # Just a single sample
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    logging.info('===> Prepare dataloader done')

    # Quantized Forward
    q_act_dict = forward_with_hooks(model, trainloader)
    logging.info('===> Get quantized inference activations done')

    # Reset quantizer
    logging.info('===> Reset model')
    if args.zeroquant:
        # Delete original model
        del model
        from utils.import_model import model_from_hf_path
        model = model_from_hf_path(args.model_path,
                    args.use_cuda_graph,
                    device_map='auto',
                ).eval()
    else:
        for name, module in model.named_modules():
            if isinstance(module, ActQuantLinear):
                module.bit = 16
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.state_dict()['weight'].copy_(fp_state_dict[name].data)

    # FP16 Forward
    fp_act_dict = forward_with_hooks(model, trainloader)
    logging.info('===> Get full-precision inference activations done')

    # Get layerwise SQNR
    distance_dict = {'sqnr':dict(),'mse':dict()}
    for k, v in q_act_dict['input'].items():
        if args.zeroquant:
            k = k.split('module.')[1]
        distance_dict['sqnr'][k] = sqnr(fp_act_dict['input'][k],v)
        distance_dict['mse'][k] = mse(fp_act_dict['input'][k],v)

    logging.info('Layer-wise SQNR')
    for k, v in distance_dict['sqnr'].items():
        print(f'{k:50s}: {v:.4f}')
    logging.info('Layer-wise MSE')
    for k, v in distance_dict['mse'].items():
        print(f'{k:50s}: {v:.4f}')

    # Save sqnr and mse to csv
    # args.distance_csv_path = 'ld_l31_w48a16_b4_s600_lr5e6.csv'
    # import pandas as pd
    # df = pd.DataFrame(distance_dict)
    # df.to_csv(args.distance_csv_path)

    return distance_dict