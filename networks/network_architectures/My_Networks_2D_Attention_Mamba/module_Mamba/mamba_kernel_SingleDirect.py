# from: https://github.com/alxndrTL/mamba.py


import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_kernel_pytorch_pscan import pscan

"""

This file closely follows the mamba_simple.py from the official Mamba implementation, and the mamba-minimal by @johnma2006.
The major differences are :
-the convolution is done with torch.nn.Conv1d
-the selective scan is done in PyTorch

A sequential version of the selective scan is also available for comparison. Also, it is possible to use the official Mamba implementation.

This is the structure of the torch modules :
- A Mamba model is composed of several layers, which are ResidualBlock.
- A ResidualBlock is composed of a MambaBlock, a normalization, and a residual connection : ResidualBlock(x) = mamba(norm(x)) + x
- This leaves us with the MambaBlock : its input x is (B, L, D) and its outputs y is also (B, L, D) (B=batch size, L=seq len, D=model dim).
First, we expand x into (B, L, 2*ED) (where E is usually 2) and split it into x and z, each (B, L, ED).
Then, we apply the short 1d conv to x, followed by an activation function (silu), then the SSM.
We then multiply it by silu(z).
See Figure 3 of the paper (page 8) for a visual representation of a MambaBlock.

"""


class Mamba_SingleDirect(nn.Module):
    def __init__(self, d_model, d_state = 16, d_conv = 4, expand_factor = 2):
        super().__init__()

        # -------------------------------------------------
        # configs

        self.d_model = d_model # D
        self.dt_rank = 'auto'
        self.d_state = d_state # N in paper/comments
        self.expand_factor = expand_factor # E in paper/comments
        self.d_conv = d_conv

        self.dt_min = 0.001
        self.dt_max = 0.1
        self.dt_init = "random" # "random" or "constant"
        self.dt_scale = 1.0
        self.dt_init_floor = 1e-4

        self.rms_norm_eps = 1e-5
        self.base_std = 0.02

        self.bias = False
        self.conv_bias = True
        self.inner_layernorms = False # apply layernorms to internal activations

        self.mup = False
        self.mup_base_width = 128 # width=d_model

        self.pscan = False # use parallel scan mode or sequential mode when training
        self.use_cuda = True # use official CUDA implementation when training (not compatible with (b)float16)


        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

        # -------------------------------------------------
        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=self.bias)

        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, 
                              kernel_size=self.d_conv, bias=self.conv_bias, 
                              groups=self.d_inner,
                              padding=self.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt initialization
        # dt weights
        # dt_scale: float = 1.0
        # dt_rank = math.ceil(self.d_model / 16)
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        # d_inner = expand_factor * d_model
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)  # if out < dt_init_floor, then return dt_init_floor
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        # (d_state): Values from 1 ~ d_state + 1 ---> (N, ED)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)  

        # log_e(A)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))   # (1, d_state)
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

        # used in jamba
        if self.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.dt_rank, self.rms_norm_eps, self.mup)
            self.B_layernorm = RMSNorm(self.d_state, self.rms_norm_eps, self.mup)
            self.C_layernorm = RMSNorm(self.d_state, self.rms_norm_eps, self.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.use_cuda:
            
            # try:
            #     from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            #     self.selective_scan_cuda = selective_scan_fn
            # except ImportError:
            #     print("Failed to import mamba_ssm. Falling back to mamba.py.")
            #     self.use_cuda = False

            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            self.selective_scan_cuda = selective_scan_fn

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)


        _, L, _ = x.shape

        xz = self.in_proj(x)     # (B, L, D) -> (B, L, 2*ED)

        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, L, ED) -> (B, ED, L)

        # (B, ED, L) -> (B, ED, L + d_conv - 1) -> (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter

        x = x.transpose(1, 2) # # (B, ED, L) -> (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)   # (B, L, ED), (B, L, ED) -> (B, L, ED)

        if self.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x, z):
        # x : (B, L, ED)
        # z : (B, L, ED) - only used in CUDA version
        # y : (B, L, ED)

        # e^x
        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()   # (ED)

        deltaBC = self.x_proj(x)  # (B, L, ED) -> (B, L, dt_rank+2*N)

        # (B, L, dt_rank+2N) -> delta: (B, L, dt_rank), B: (B, L, N), C: (B, L, N)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)

        # (ED, dt_rank) @ (B, dt_rank, L) -> (B, ED, L)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)   # (B, L, ED) -> (B, ED, L)
            B = B.transpose(1, 2)   # (B, L, N) -> (B, N, L)
            C = C.transpose(1, 2)   # (B, L, N) -> (B, N, L)
            z = z.transpose(1, 2)   # (B, L, ED) -> (B, ED, L)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else: 
            delta = delta.transpose(1, 2)   # (B, ED, L) -> (B, L, ED)

            # f(x) = log(1 + e^x)
            # As x → -∞, f(x) → 0
            # As x → +∞, f(x) → x (behaves like linear function)
            delta = F.softplus(delta + self.dt_proj.bias)   # (B, L, ED) -> (B, L, ED)

            # pscan: bool = True # use parallel scan mode or sequential mode when training
            if self.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)   # this one is slower than the parallel one

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        # e^x
        deltaA = torch.exp(delta.unsqueeze(-1) * A)     # (B, L, ED, 1) * (ED, N) -> (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)    # (B, L, ED, 1) * (B, L, 1, N) -> (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))    # (B, L, ED, N) * (B, L, ED, 1) -> (B, L, ED, N)
        
        hs = pscan(deltaA, BX)    # (B, L, ED, N) -> (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1) -> (B, L, ED)

        y = y + D * x   # (B, L, ED) + (ED) * (B, L, ED) -> (B, L, ED)

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        # (B, L, ED, 1) * (ED, N) -> (B, L, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)

        # (B, L, ED, 1) * (B, L, 1, N) -> (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
        
        # (B, L, ED, N) * (B, L, ED, 1) -> (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        # (B, ED, N)
        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device) # (B, ED, N)

        hs = []
        for t in range(0, L):
            # (B, ED, N) * (B, ED, N) + (B, ED, N) -> (B, ED, N)
            h = deltaA[:, t] * h + BX[:, t]

            hs.append(h)

        # List[(B, ED, N)] -> (B, L, ED, N)            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1) -> (B, L, ED)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # (B, L, ED) + (ED) * (B, L, ED) -> (B, L, ED)
        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        # (B, D) -> (B, 2*ED)
        xz = self.in_proj(x) # (B, 2*ED)

        # (B, 2*ED) -> x: (B, ED), z: (B, ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)  # (B, ED) -> (B, ED, 1)

        # [(B, ED, d_conv-1) + (B, ED, 1)] -> (B, ED, d_conv) -> (B, ED, 1) -> (B, ED)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)   # (B, ED), (B, ED, N) -> (B, ED), (B, ED, N)

        # z branch
        z = F.silu(z)

        # (B, ED) * (B, ED) -> (B, ED)
        output = y * z

        # (B, ED) -> (B, D)
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call

        # (B, ED, d_conv-1) -> (B, ED, d_conv-1)
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)

        cache = (h, inputs)   # (B, ED, N), (B, ED, d_conv-1)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        # (ED, N)
        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep

        # (ED)
        D = self.D.float()

        # (B, ED) -> (B, dt_rank+2*N)
        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        # (B, dt_rank+2N) -> delta: (B, dt_rank), B: (B, N), C: (B, N)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)

        # (B, dt_rank) -> (B, ED)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        # (B, ED, 1) * (ED, N) -> (B, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)

        # (B, ED, 1) * (B, 1, N) -> (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        # (B, ED, N) * (B, ED, 1) -> (B, ED, N)
        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device) # (B, ED, N)

        # (B, ED, N) * (B, ED, N) + (B, ED, N) -> (B, ED, N)
        h = deltaA * h + BX # (B, ED, N)

        # (B, ED, N) @ (B, N, 1) -> (B, ED, 1) -> (B, ED)
        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        # (B, ED) + (ED) * (B, ED) -> (B, ED)
        y = y + D * x

        # (B, ED), (B, ED, N)
        return y, h

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output
    