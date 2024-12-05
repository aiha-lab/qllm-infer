import torch
import numpy as np
import gc
from tqdm import tqdm

from .utils import CompressionParameter, PACKER
from .rtn_parameter import RTNParameter

layers = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
def quantize_lutgemm(model, args, dev='cuda', parent_name=""):
    if args.lutgemm and args.rtn:
        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if len(list(module.children())) > 0:
                quantize_lutgemm(module, args, dev=dev, parent_name=full_name)
    
            if any(x in name for x in layers):
                print(full_name)
                original_weight = module.weight.clone().detach()
                # INT4 Quantization -> RTN
                w_rtn = RTNParameter(original_weight)
                scale, zero, w_quant, w_quant_shape = w_rtn.compress(
                    in_ch_wise=False, qbits=args.bits_w, group_size=args.groupsize_w,
                    perchannel=True, sym=False)
    
                w_rtn.decompress(scale, zero, w_quant, w_quant_shape, in_ch_wise=False)
#                import pdb; pdb.set_trace() 
                module.weight.data = w_rtn.data.to(module.weight.dtype)
                # Convert INT4 -> BCQ4
#                alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
#                    scale, zero, w_quant, qbits=args.qbits,
#                    do_packing=False, in_ch_wise=False)
#    
#                print("Parameter size before packing")
#                print("  alpha.size()  =", alpha.size())
#                print("  binary.size() =", binary.size())
#                print("="*30)
#    
#                # Packing BCQ4 -> Packed Weight (uint8)
#                alpha, binary, binary_shape, offset = w_rtn.convert_bcq_format(
#                    scale, zero, w_quant, qbits=args.qbits,
#                    do_packing=True, in_ch_wise=False)
#    
#                print("Parameter size after packing")
#                print("  alpha.size()  =", alpha.size())
#                print("  binary.size() =", binary.size())
#                print("="*30)
    else:
        for name, module in model.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
    
            if len(list(module.children())) > 0:
                quantize_lutgemm(module, args, dev=dev, parent_name=full_name)
    
            if any(x in name for x in layers):
                print(full_name)
                original_weight = module.weight.clone().detach()
                # INT4 Quantization -> BCQ
                w_bcq = BCQParameter(original_weight)
                ret, alpha, binary, binary_shape = w_bcq.compress(
                    do_packing=args.do_packing, 
                    in_ch_wise=False, qbits=args.bits_w,
                    rounds=args.round, group_size=args.groupsize_w)
    
    #            import pdb; pdb.set_trace() 
                print(f"Alpha shape : {alpha.size()}")
                print(f"Binary shape : {binary.size()}")
                print("="*30)
                module.weight.data = ret.to(module.weight.dtype)
    return model

@torch.inference_mode()
def quantize(w, qbits, rounds=15, group_size=-1, transpose=False, use_bst=True):
    '''
    Post-training Weighted Quantization (BCQ format)
    https://openreview.net/pdf?id=2Id6XxTjz7c

    rounds == 0: greedy algorithm
    rounds == 1: refined greedy algorithm
    rounds >= 2: alternating algorithm

    :param w: a weight tensor of layer
    :param qbits: number of quantization bits for the `w`
    :param rounds: number of iterations for refining both alpha and B
    :param group_size: number of weights in which a scaling factor can be shared
    :param transpose: if `transpose` is True, `w` is a transposed when using this method.
    :param use_bst: if `use_bst` is True(default), the binary matrix is calculated using BST algorithm.
                    if `use_bst` is False, the binary matrix is calculated with greedy algorithm.
    '''
    w_ = w.clone()
    w_ = w_.cuda()

    if transpose:
        assert len(w_.shape) == 2, f'Check your weight shape {w_.shape}'
        w_ = w_.transpose(1, 0).contiguous()
    
    orig_shape = w_.shape
    group_size = group_size if group_size > 0 else orig_shape[-1]
    w_ = w_.view([-1, group_size])
 
    # init weighted
    w_abs = w_.abs()
    ws, _ = w_abs.view(-1).sort()
    wf = torch.ones(w_.shape, dtype=torch.float32, device=w.device)
    wf = wf.to(w_.device)
    # greedy & alternating algo.
#    import pdb; pdb.set_trace() 
    ret, B, alpha = greedy_mean_torch(w_, n_bits=qbits, wf=wf)
    if rounds > 0 and qbits > 1:
        for _ in tqdm(range(rounds)):
            ret, B, alpha = refine_mean_torch(w_, ret, B, alpha, wf=wf, use_bst=use_bst)

#    if orig_shape[0] != orig_shape[1]:
#        import pdb; pdb.set_trace() 
    ret = ret.view(orig_shape) 
    if transpose:
        ret = ret.transpose(1, 0).contiguous()

    del w_
    
    B = B.reshape([orig_shape[0], orig_shape[1] // group_size, group_size, qbits])
    alpha = alpha.reshape([orig_shape[0], orig_shape[1] // group_size, qbits])

    B = B.to('cpu')
    alpha = alpha.to('cpu')
    torch.cuda.empty_cache()

    return ret, B, alpha, (wf != 0.0)

def greedy_mean_torch(w, n_bits=1, wf=None):
#    import pdb; pdb.set_trace() 
    B = torch.zeros(w.shape + (n_bits,), device=w.device)
    Alpha = torch.zeros(w.shape[0], n_bits, device=w.device)
  
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        b = r.sign()
        
        if wf is not None:
            a1sum = torch.sum(wf, dim=1)
            alpha = (r.abs()*wf).sum(dim=1) / torch.sum(wf, dim=1)
            alpha[torch.isnan(alpha)] = 0.
            alpha = alpha.view(alpha.shape[0], 1)
        else:
            alpha = r.abs().mean(dim=1, keepdim=True)
        
        r -= b * alpha
        w_hat += b * alpha
        B[:,:,i] = b
        Alpha[:,i] = alpha.view(-1)
    
    del r, b, alpha
    gc.collect()
    torch.cuda.empty_cache()

    return w_hat, B, Alpha

def refine_mean_torch(w, w_hat, B, Alpha, wf=None, use_bst=True):
    w = w.float()
    d1, d2 = w.shape
    with torch.no_grad():
        n_bits = B.shape[-1]
        Bt = B.transpose(1, 2)
        if wf is not None:
            Bt = Bt * wf.unsqueeze(1)
        B_cov = Bt.bmm(B)
        Btw = Bt.bmm(w.unsqueeze(-1)).view(d1, n_bits)

        Alpha_new = batch_cg_torch(B_cov, Btw, x=Alpha)
        Alpha_new, _ = Alpha_new.abs().sort(descending=True)

        if use_bst == False:
            r = w.clone()
            B_new = torch.zeros_like(B)
            for i in range(n_bits):
                B_new[:, :, i] = r.sign()
                r -= B_new[:, :, i] * Alpha_new[:, i].view([-1, 1])
            del r
        else:
            B_new = find_B_torch(w, Alpha_new)
            B_new = B_new * (wf != 0.0).unsqueeze(-1)
        w_hat_new = torch.einsum('ijl,il->ij', (B_new, Alpha_new))

    return w_hat_new, B_new, Alpha_new

def list_binary_vecs(n):
    ListBinaryVecs = {0 : [[]]}
    for m in range(1, n+1):
        ListBinaryVecs[m] = [[1.] + l for l in ListBinaryVecs[m-1]] + [[-1.] + l for l in ListBinaryVecs[m-1]]
    return ListBinaryVecs

def find_B_torch(w, Alpha):
    '''Find optimal quantization assignment via binary search (torch)'''
    n_bits = Alpha.shape[-1]

    ListBinaryVecs = list_binary_vecs(n_bits)
    bin_mat = torch.from_numpy(np.vstack(ListBinaryVecs[n_bits]).astype(np.float32)).to(w.device)

    d1, d2 = w.shape
    row_inds = torch.arange(d1, dtype=torch.long).view(d1, 1).repeat([1, d2]).view(-1)
    # w is d1xd2, Alpha is d1xk, v is d1x2^k
    v = Alpha.mm(bin_mat.t())
    v_sorted, inds = torch.sort(v)
    # Binary search to find nearest neighbor
    w_flat = w.view([-1])
    Left = torch.zeros(d1*d2, dtype=torch.long, device=w.device)
    Right = torch.ones(d1*d2, dtype=torch.long, device=w.device) * (2 ** n_bits - 1)
    for i in range(n_bits):
        Mid_Left = torch.div(Left + Right - 1, 2, rounding_mode='trunc')
        Mid_Right = Mid_Left + 1
        mid_vals = (v_sorted[row_inds, Mid_Left] + v_sorted[row_inds, Mid_Right]) / 2
        inds_left = (w_flat < mid_vals)
        Right[inds_left] = Mid_Left[inds_left]
        Left[~inds_left] = Mid_Right[~inds_left]
    assignment_inds = inds[row_inds, Left].view(d1, d2)
    return bin_mat[assignment_inds, :]

def batch_cg_torch(A, b, x=None):
    '''Batch conjugate gradient for solving Ax = b'''
    d1, k, _ = A.shape
    # Initialize
    x = x.clone().view(d1, k, 1)
    b = b.view(d1, k, 1)
    r = b - A.bmm(x)
    rtr_new = r.transpose(1, 2).bmm(r)
    p = r.clone()
    # Perform batch CG
    for i in range(k):
        rtr = rtr_new
        Ap = A.bmm(p)
        alpha = rtr / (p.transpose(1, 2).bmm(Ap) + 1e-6)
        x += alpha * p
        r -= alpha * Ap
        rtr_new = r.transpose(1, 2).bmm(r)
        beta = rtr_new / (rtr + 1e-6)
        p = r + beta * p
    return x.view(d1, k)

class BCQParameter(CompressionParameter):
    def compress(self, do_packing=False, in_ch_wise=False, **kwargs):
        global PACKER
        ret, binary, alpha, _ = quantize(self.data, transpose=in_ch_wise, **kwargs)

        binary_shape = binary.shape
        if do_packing == True:
            binary, binary_shape = PACKER.pack(binary)
            binary = binary.to(self.data.device)

        return ret, alpha, binary, binary_shape

    def decompress(self, alpha, binary, binary_shape, offset=None, do_packing=False, in_ch_wise=False):
        global PACKER

        if do_packing == True:
            binary = PACKER.unpack(binary, binary_shape, dtype=self.data.dtype)
            binary = binary.to(self.data.device)

        # w.shape = [out_ch, in_ch]
        # in_ch_wise == True
        #   -> binary.shape = [in_ch, out_ch//group_size, group_size, qbits]
        #   -> alpha.shape  = [in_ch, out_ch//group_size, qbits]
        #   -> offset.shape = [in_ch, out_ch//group_size, 1]
        # in_ch_wise == False
        #   -> binary.shape = [out_ch, in_ch//group_size, group_size, qbits]
        #   -> alpha.shape  = [out_ch, in_ch//group_size, qbits]
        #   -> offset.shape = [out_ch, in_ch//group_size, 1]

        if in_ch_wise == True:
            out_ch = binary_shape[1] * binary_shape[2]
            decomp_w = torch.einsum('iogb,iob->iog', (binary, alpha))
            if offset is not None:
                decomp_w = decomp_w + offset
            decomp_w = decomp_w.reshape([-1, out_ch]).T
        else:
            out_ch = binary_shape[0]
            decomp_w = torch.einsum('oigb,oib->oig', (binary, alpha))
            if offset is not None:
                decomp_w = decomp_w + offset
            decomp_w = decomp_w.reshape([out_ch, -1])
        self.data = decomp_w
