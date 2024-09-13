from . import calibration
from . import smooth
import os
import logging
import torch

def get_smoothquant_model(model, tokenizer, args):
    possible_name = f'{args.model_path.split("/")[-1]}_{args.smoothquant_dataset}_{args.smoothquant_nsamples}_{args.smoothquant_seqlen}'
    possible_path = f'{args.cache_dir}/act_scales/{possible_name}.pt'

    if os.path.exists(possible_path):
        logging.info(f'Load act_scales form {possible_path}')
        act_scales = torch.load(possible_path)
    else:
        logging.info(f'Getting activation scales for SmoothQuant')
        act_scales = calibration.get_act_scales(
            model, tokenizer, args.smoothquant_dataset,
            args.smoothquant_nsamples, args.smoothquant_seqlen,
            args=args,
        )
    
        os.makedirs(os.path.dirname(possible_path), exist_ok=True)
        logging.info(f'Save act_scales to {possible_path}')
        torch.save(act_scales, possible_path)

    logging.info(f'Smoothing model')
    smooth.smooth_lm(model, act_scales, args.smoothquant_alpha)
