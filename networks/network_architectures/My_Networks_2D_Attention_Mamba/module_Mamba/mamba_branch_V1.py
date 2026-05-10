import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from .mamba_kernel_SS2D import SS2D
from .mamba_kernel_SingleDirect import Mamba_SingleDirect
from .mamba_kernel_Vim import Mamba_Vim


# mamba branch with several mamba layers
# add SS2D


# -------------------------------------------------
# Mamba block: SS2D

class SS2D_Block(nn.Module):
    def __init__(self, hidden_dim = 0, drop_path = 0, norm_layer = partial(nn.LayerNorm, eps=1e-6), attn_drop_rate = 0, d_state = 16):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):

        x = input + self.drop_path(self.self_attention(self.ln_1(input)))

        return x
    

# -------------------------------------------------
# mamba branch with several mamba layers 

class mamba_branch(nn.Module):
    def __init__(self, logger, dim, depth, attn_drop = 0., drop_path = 0., d_state = 16, norm_layer = nn.LayerNorm, mamba_type = 'SS2D_Standard'):
        super().__init__()

        self.logger = logger
        self.dim = dim

        # -------------------------------------------------
        self.blocks = nn.ModuleList()

        for i in range(depth):
            
            if mamba_type == 'SS2D_Standard':

                self.blocks.append(SS2D_Block(
                hidden_dim = dim,
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer = norm_layer,
                attn_drop_rate = attn_drop,
                d_state = d_state,
                ))

                self.logger.info('This type of Mamba is used:' + mamba_type)

            else:
                raise ValueError('this type of Mamba is not supported: ' + mamba_type)

        # -------------------------------------------------

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x










