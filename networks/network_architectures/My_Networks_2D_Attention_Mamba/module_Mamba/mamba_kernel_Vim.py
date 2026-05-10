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


class Mamba_Vim(nn.Module):
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

        self.pscan = False # use parallel scan mode or sequential mode when training
        self.use_cuda = True # use official CUDA implementation when training (not compatible with (b)float16)

        self.bidirectional = True
        self.divide_output = True

        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

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
        dt_init_std = self.dt_rank**-0.5 * self.dt_scale
        if self.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        ).clamp(min=self.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Backward Parameters
        if self.bidirectional:
            A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log_b = nn.Parameter(torch.log(A_b))
            self.A_log_b._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                    kernel_size=self.d_conv, bias=self.conv_bias,
                                    groups=self.d_inner,
                                    padding=self.d_conv - 1)
            
            self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)

            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

        # used in jamba
        if self.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.dt_rank, self.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.d_state, self.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.d_state, self.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.use_cuda:

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
        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)
        x = F.silu(x)
        y = self.ssm(x=x, 
                     z=z,
                     A_log=self.A_log,
                     D=self.D,
                     x_proj=self.x_proj,
                     dt_proj=self.dt_proj)

        if self.bidirectional:
            xz_b = xz.flip([1]) # (B, L, 2*ED)
            x_b, z_b = xz_b.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)
            x_b = x_b.transpose(1, 2) # (B, ED, L)
            x_b = self.conv1d_b(x_b)[:, :, :L] # depthwise convolution over time, with a short filter
            x_b = x_b.transpose(1, 2) # (B, L, ED)
            x_b = F.silu(x_b)
            y_b = self.ssm(x=x_b,
                           z=z_b,
                           A_log=self.A_log_b,
                           D=self.D_b,
                           x_proj=self.x_proj_b,
                           dt_proj=self.dt_proj_b)

        if self.use_cuda:
            if not self.bidirectional:
                return self.out_proj(y)
            else:
                if self.divide_output:
                    return self.out_proj((y + y_b.flip([1])) / 2)
                else:
                    return self.out_proj(y + y_b.flip([1]))
        
        z = F.silu(z)
        y = y * z
        if not self.bidirectional:
            return self.out_proj(y)
        else:
            z_b = F.silu(z_b)
            y_b = y_b * z_b
            if self.divide_output:
                return self.out_proj((y + y_b.flip([1])) / 2)
            else:
                return self.out_proj(y + y_b.flip([1]))
    
    def ssm(self, x, z, A_log, D, x_proj, dt_proj):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(A_log.float()) # (ED, N)
        D = D.float()

        deltaBC = x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + dt_proj.bias)

            if self.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

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

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

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
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output