# From https://github.com/IST-DASLab/gptq/blob/main/llama.py
# Disable cpu offloading because of conflicts with transformers version (FIXME)
import os
import time
import sys
import torch
import torch.nn as nn

from lib.gptq.gptq import *
from lib.gptq.modelutils import *
from lib.quantization.quantizer import *

from lib.spqr.spqr_engine import SPQRUtil
from lib.spqr.modelutils import get_layers
from lib.spqr.modelutils import find_sublayers
from lib.spqr.modelutils import get_sequential_groups
import logging

@torch.no_grad()
def opt_sequential(model, dataloader, dev, args=None):
    logging.info('Starting GPTQ ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

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

    quantizers = {}
    for i in range(len(layers)):
        #layer = layers[i].to(dev)
        layer = layers[i]
        full = find_layers(layer)

        if args.gptq_true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.out_proj'],
                ['mlp.fc1'],
                ['mlp.fc2']
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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                logging.info(f'Quantizing layer {i}: {name}')
                gptq[name].fasterquant(
                    percdamp=args.gptq_percdamp, groupsize=args.groupsize_w, actorder=args.gptq_act_order, static_groups=args.gptq_static_groups
                )
                quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.gptq_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        #layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

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
    if 'llama' in args.model_path:
        quantizers = llama_sequential(model, dataloader, dev, args)
    elif 'opt' in args.model_path:
        quantizers = opt_sequential(model, dataloader, dev, args)
    else:
        raise NotImplementedError


def get_average_number_of_bits(
    wbits: int = 3,
    qq_scale_bits: int = 3,
    qq_zero_bits: int = 3,
    qqq_scale_bits: int = 16,
    qqq_zero_bits: int = 16,
    groupsize: int = 16,
    qq_groupsize: int = 16,
    round_zero: bool = False,
    global_ol_n_share: float = 0.00,
):
    # if not quantized stats are in full precision
    qq_scale_bits = qq_scale_bits or 16
    qq_zero_bits = qq_zero_bits or 16
    groupsize = groupsize or float("inf")
    qq_groupsize = qq_groupsize or float("inf")

    if groupsize is None:
        wbits_avg = wbits
    elif round_zero:
        wbits_avg = (
            wbits + (qq_scale_bits + wbits) / groupsize + (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
        )
    else:
        wbits_avg = (
            wbits
            + (qq_scale_bits + qq_zero_bits) / groupsize
            + 2 * (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
        )

    # correct accounting for outliers
    if global_ol_n_share > 0:
        wbits_avg += 32 * global_ol_n_share

    return round(wbits_avg, 2)

@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    logging.info("catching inputs from data")
    from lib.spqr.modelutils import get_layers
    layers = get_layers(model)

    nsamples = nsamples or args.gptq_nsamples

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.
    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, args.gptq_seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    forward_arg_names = [
        "attention_mask",
    ]
    from lib.spqr.modelutils import FALCON_TYPES, LLAMA_LIKE
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")
    elif model.config.model_type.lower() in LLAMA_LIKE:
        forward_arg_names.append("position_ids")
        
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)
    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args

@torch.no_grad()
def spqr_sequential(model, dataloader, args, dev):
    logging.info("\nStarting SPQR quantization ...")
    
    # get_inps in SpQR
    """mocks model launch to collect inputs to the first model layer"""
    logging.info("catching inputs from data")
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.gptq_nsamples, args.gptq_seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps, forward_args = get_inps(model, dataloader, args, dev="cpu" if args.spqr_offload_activations else dev)
    
    outs = torch.zeros_like(inps)
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "spqr_save", False)
    
    quantizers = {}

    normal_outlier_count_global, w_count_global = 0, 0

    layers = get_layers(model)
    for i in range(len(layers)):
        logging.info(f"\n---------------- Layer {i} of {len(layers)} ----------------")
        normal_outlier_count, w_count = 0, 0
        stats_payload = {}
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device  # quantized layer will return there
        logging.info(f"{layer_dev_original=}")
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(dev)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v

        if args.gptq_true_sequential:
            
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}

            spqr_handlers = {}
            for sublayer_name in subset:
                spqr_handlers[sublayer_name] = SPQRUtil(subset[sublayer_name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data, out.data)  # noqa: F821

                return tmp

            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))
            for j in range(args.gptq_nsamples):
                if 'llama' in args.model_path:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'], position_ids=forward_args['position_ids'])[0]
                elif 'opt' in args.model_path:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'])[0]
                #if args.spqr_offload_activations:
                #    outs[j] = outs[j].cpu()
            for h in handles:
                h.remove()

            torch.cuda.empty_cache()

            for sublayer_name in subset:
                logging.info(f"Quantizing module {sublayer_name} of layer {i}")
                quantized = spqr_handlers[sublayer_name].quantize(
                    percdamp=args.gptq_percdamp,
                    bits=args.bits_w,
                    groupsize=args.groupsize_w,
                    sym=args.sym_w,
                    perchannel=True,
                    qq_groupsize=args.spqr_qq_groupsize,
                    round_zero=False,
                    qq_scale_bits=args.spqr_qq_scale_bits,
                    qq_zero_bits=args.spqr_qq_zero_bits,
                    qq_zero_sym=args.spqr_qq_zero_sym,
                    outlier_relative_threshold=args.spqr_outlier_threshold,
                    permutation_order='act_order' if args.gptq_act_order is True else 'identity',
                    simplified_outliers=args.spqr_simplified_outliers,
                    save_quantization=save,
                )

                if save:
                    quantized.save_quant_dict["sublayer_name"] = sublayer_name
                    full_path = save + "/" + str(i) + "/"
                    os.makedirs(full_path, exist_ok=True)
                    torch.save(quantized.save_quant_dict, full_path + sublayer_name)

                spqr_handlers[sublayer_name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[sublayer_name].layer.weight.data.dtype
                )
                quantizers["model.layers.%d.%s" % (i, sublayer_name)] = ()  # to be updated

                # OUTLIER STATS per module:
                normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()
                stats_payload[f"n_{sublayer_name}_ol_share"] = (normal_outliers_count / quantized.weight.numel()).item()
                normal_outlier_count += normal_outliers_count.item()
                w_count += quantized.weight.numel()

        out_losses = []
        for j in range(args.gptq_nsamples):
            if 'llama' in args.model_path:
                outs_batch = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'], position_ids=forward_args['position_ids'])[0]
            elif 'opt' in args.model_path:
                outs_batch = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'])[0]
            if not args.spqr_skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j].to(layer_dev))
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                out_losses.append(outs_batch_loss.item())
            outs[j] = outs_batch
            if args.spqr_offload_activations:
                outs[j] = outs[j].cpu()
        del outs_batch

        layers[i] = layer.to(layer_dev_original)
        del layer
        del spqr_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["ol_share"] = normal_outlier_count / max(w_count, 1)
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
        stats_payload["Step"] = i

        normal_outlier_count_global += normal_outlier_count
        w_count_global += w_count

        logging.info(stats_payload)

    logging.info("=====================\nFinal stats:")
    logging.info(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")


    wbits_avg = get_average_number_of_bits(
        wbits=args.bits_w,
        qq_scale_bits=args.spqr_qq_scale_bits,
        qq_zero_bits=args.spqr_qq_zero_bits,
        qqq_scale_bits=16,
        qqq_zero_bits=16,
        groupsize=args.groupsize_w,
        qq_groupsize=args.spqr_qq_groupsize,
        round_zero=False,
        global_ol_n_share=normal_outlier_count_global / w_count_global,
    )
        
    if save:
        torch.save(vars(args), save + "/args.pt")
        already_saved_weights = set()
        for name, layer in nn.ModuleList(get_layers(model)).named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                already_saved_weights.add(layer.weight)
        not_quantized_weights = {
            name: param for name, param in model.named_parameters() if param not in already_saved_weights
        }
        torch.save(not_quantized_weights, save + "/not_quantized_weights.pt")

    model.config.use_cache = use_cache
    logging.info(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers, wbits_avg

def quantize_spqr(model, args, dev):
    from utils.data_utils import get_loaders
    logging.info("Loading data ...")
    dataloader = get_loaders(
        args.gptq_dataset, nsamples=args.gptq_nsamples,
        seed=args.seed, model=args.model_path,
        seqlen=args.gptq_seqlen, cache_dir=args.cache_dir,
    )
    quantizer, wbits_avg = spqr_sequential(model, dataloader, args, dev)
    logging.info(f'{wbits_avg=}')
    return quantizer


def quantize_nearest(model, args, dev):
    if 'llama' in args.model_path:
        layers = model.model.layers
    elif 'opt' in args.model_path:
        layers = model.model.decoder.layers
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
