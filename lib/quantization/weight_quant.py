# From https://github.com/IST-DASLab/gptq/blob/main/llama.py
# Disable cpu offloading because of conflicts with transformers version (FIXME)

import time

import torch
import torch.nn as nn

from lib.gptq.gptq import *
from lib.gptq.modelutils import *
from lib.quantization.quantizer import *

import logging

@torch.no_grad()
def llama_sequential(model, dataloader, dev, args=None):
    logging.info('Starting GPTQ ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    #model.model.embed_tokens = model.model.embed_tokens.to(dev)
    #model.model.norm = model.model.norm.to(dev)
    #layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.gptq_nsamples, args.gptq_seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    #layers[0] = layers[0].cpu()
    #model.model.embed_tokens = model.model.embed_tokens.cpu()
    #model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    for i in range(len(layers)):
        #layer = layers[i].to(dev)
        layer = layers[i]
        full = find_layers(layer)

        if args.gptq_true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.bits_w, perchannel=True, sym=args.sym_w, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.gptq_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                logging.info(f'Quantizing layer {i}: {name}')
                gptq[name].fasterquant(
                    percdamp=args.gptq_percdamp, groupsize=args.groupsize_w, actorder=args.gptq_act_order, static_groups=args.gptq_static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.gptq_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        #layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def quantize_gptq(model, args, dev):
    from utils.data_utils import get_loaders
    dataloader = get_loaders(
        args.gptq_dataset, nsamples=args.gptq_nsamples,
        seed=args.seed, model=args.model_path,
        seqlen=args.gptq_seqlen, cache_dir=args.cache_dir,
    )
    quantizers = llama_sequential(model, dataloader, dev, args)

def quantize_nearest(model, args, dev):
    layers = model.model.layers
    for i in range(len(layers)):
        logging.info(f'Quantizing layer {i}')
        #layer = layers[i].to(dev)
        layer = layers[i]
        
        subset = find_layers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(
                args.bits_w, perchannel=True, sym=args.sym_w, mse=False
            )
            W = subset[name].weight.data
            shape_ = W.shape
            if args.groupsize_w > 0:
                W = W.reshape(-1, args.groupsize_w)
            quantizer.find_params(W, weight=True)
            qW = quantize(
                W, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(next(iter(layer.parameters())).dtype)
            qW = qW.reshape(shape_)
            subset[name].weight.data = qW
