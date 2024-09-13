from types import MethodType
import logging
import torch.nn as nn
import torch.nn.functional as F

from . import attention
from lib.quantization.quantizer import *

class ActQuantLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias,args):
        super().__init__(in_features,out_features,bias)
        self.groupsize = args.groupsize_a
        self.bit = args.bits_a
        self.sym = args.sym_a
        self.quantizer = Quantizer()
        self.quantizer.configure(
            self.bit, perchannel=True, sym=self.sym, mse=False
        )

    def forward(self, x):
        shape_ = x.shape
        if self.groupsize > 0:
            x = x.reshape(-1, self.groupsize)
        else: # Token-wise
            x = x.reshape(-1, shape_[-1])
        self.quantizer.find_params(x, weight=True)
        qx = quantize(
            x, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        ).to(self.weight.dtype)
        qx = qx.reshape(shape_)
        return F.linear(qx,self.weight,self.bias)

class ActQuantMatMul(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.groupsize = args.groupsize_a
        assert self.groupsize < 0, 'Group-wise quantization on multi-head activation is not supported yet'
        self.bit = args.bits_a
        self.sym = args.sym_a
        self.quantizer_A = Quantizer()
        self.quantizer_A.configure(
            self.bit, perchannel=True, sym=self.sym, mse=False
        )
        self.quantizer_B = Quantizer()
        self.quantizer_B.configure(
            self.bit, perchannel=True, sym=self.sym, mse=False
        )

    def forward(self, A, B): # [b, h, s, d]
        # Quantizing A
        Ashape_ = A.shape
        A = A.reshape(-1, Ashape_[-1])
        self.quantizer_A.find_params(A, weight=True)
        qA = quantize(
            A, self.quantizer_A.scale, self.quantizer_A.zero, self.quantizer_A.maxq
        ).to(A.dtype)
        qA = qA.reshape(Ashape_)
        # Quantizing B
        B = B.transpose(-1,-2)
        Bshape_ = B.shape
        B = B.reshape(-1, Bshape_[-1])
        self.quantizer_B.find_params(B, weight=True)
        qB = quantize(
            B, self.quantizer_B.scale, self.quantizer_B.zero, self.quantizer_B.maxq
        ).to(B.dtype)
        qB = qB.reshape(Bshape_)
        qB = qB.transpose(-1,-2)
        return qA @ qB

def add_act_quant(model, args):
    from transformers.models.llama.modeling_llama import LlamaAttention
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            setattr(module, "matmul1", ActQuantMatMul(args))
            setattr(module, "matmul2", ActQuantMatMul(args))
            module.forward = MethodType(attention.llama_attn_forward, module)
            
    wrapped_modules={}
    module_dict={}
    it=[(name,m) for name,m in model.named_modules()]
    logging.info('Add quantized modules for activation')
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Linear) and 'head' not in name:
            idx = idx+1 if idx != 0 else idx
            new_m = ActQuantLinear(m.in_features,m.out_features,m.bias is not None,args=args)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)
