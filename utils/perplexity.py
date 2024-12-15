from utils import data_utils
import random
import torch
import logging
from tqdm import tqdm

def eval_ppl(model, tokenizer, args,
             nsamples=None, datasets=['wikitext2','c4', 'ptb'],
            ):
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    results = dict()

    for dataset in datasets:
        nsamples = None
        input_tok = data_utils.get_test_tokens(dataset,
                        seed=args.seed,
                        seqlen=args.eval_ppl_seqlen,
                        model=args.model_path,
                        cache_dir=args.cache_dir,
                    )
        #logging.info(f'{input_tok.numel()=}')
        #logging.info(f'{nsamples=}')
        if nsamples is None:
            nsamples = input_tok.numel() // args.eval_ppl_seqlen
            
        #logging.info(f'After nsample = {nsamples}')
        input_tok = input_tok[0, :(args.eval_ppl_seqlen * nsamples)].view(
            nsamples, args.eval_ppl_seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                           use_cache=False,
                           output_hidden_states=False,
                           output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        results[dataset] = ppl
        logging.info(f'{dataset} perplexity: {ppl}')
    return results
