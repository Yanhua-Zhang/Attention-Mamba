import math
import torch
from torch import nn
import torch.nn.functional as F

from ..module_Mamba.mamba_kernel_SS2D import SS2D
from ..module_Mamba.mamba_kernel_SingleDirect import Mamba_SingleDirect
from ..module_Mamba.mamba_kernel_Vim import Mamba_Vim


# add interleaved mamba_attention_module.
# add mamba_attention_module_V2: redesign channel attention

# add mamba_attention_module_V3: reverse the channel and spatial representations, and input into the 1-D Mamba

# ----------------------------------------------------------------
# use mamba for channel and spatial attention

class interleaved_mamba_attention_module(nn.Module):
    def __init__(self, classes, fea_dim = 256, d_state = 16, norm_layer = nn.LayerNorm, drop_path = 0, attn_drop_rate = 0):
        super(interleaved_mamba_attention_module, self).__init__()

        self.fea_dim = fea_dim

        self.classes = classes

        # -------------------
        # channel representation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # -------------------
        # cross channel relation
        self.channel_relation = Mamba_Vim(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)

        # channel attention
        self.fc1_channel = nn.Conv2d(self.fea_dim * 4, self.fea_dim // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2_channel = nn.Conv2d(self.fea_dim // 16, self.classes * 4, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # -------------------
        # cross spatial relation
        self.spatial_relation = Mamba_Vim(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)

        # spatial attention
        self.fc_spatial = nn.Linear(2, 1)

           
    def forward(self, fea_1, fea_2, fea_3, fea_4):

        b_fea, c_fea, h_fea, w_fea = fea_1.size()   # b, c, h, w

        n_fea = h_fea*w_fea

        assert fea_1.size() == fea_2.size() == fea_3.size() == fea_4.size(), 'fea size must be the same'

        # -------------------
        # channel representation

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_1)
        max_channel = self.max_pool(fea_1)
        rep_channel_fea_1 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_2)
        max_channel = self.max_pool(fea_2)
        rep_channel_fea_2 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_3)
        max_channel = self.max_pool(fea_3)
        rep_channel_fea_3 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_4)
        max_channel = self.max_pool(fea_4)
        rep_channel_fea_4 = avg_channel + max_channel  # b, c, 1, 1

        # -------------------
        # cross channel relation

        b_channel_fea, c_channel_fea, h_channel_fea, w_channel_fea = rep_channel_fea_1.size()
        assert h_channel_fea == w_channel_fea == 1, 'wrong!'

        # b, c, 1, 1 ---> b, c, 1 ---> b, c, 4, 1
        rep_channel_fea = torch.stack([rep_channel_fea_1.squeeze(3), rep_channel_fea_2.squeeze(3), rep_channel_fea_3.squeeze(3), rep_channel_fea_4.squeeze(3)], dim=2)
        
        # b, c, 4, 1 ---> b, c*4, 1
        interleaved_rep_channel_fea = rep_channel_fea.reshape(b_channel_fea, c_channel_fea * 4, 1)

        # b, 4*c, 1 ---> b, 4*c, 1 
        fea_channel_relation = self.channel_relation(interleaved_rep_channel_fea)

        # b, 4*c, 1 ---> b, c, 4, 1 ---> b, c, 1
        channel_fea1, channel_fea2, channel_fea3, channel_fea4 = fea_channel_relation.reshape(b_channel_fea, c_channel_fea, 4, 1).unbind(dim=2)

        # b, c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_restored = torch.cat([channel_fea1, channel_fea2, channel_fea3, channel_fea4], 1).unsqueeze(3)

        # b, 4*c, 1, 1 ---> b, c/4, 1, 1 ---> b, 4*classes, 1, 1
        fea_channel_attention = self.sigmoid(self.fc2_channel(self.relu(self.fc1_channel(fea_channel_restored))))

        # b, 4*classes, 1, 1 ---> [b, classes, 1, 1]*4
        fea_1_channel_attention = fea_channel_attention[:, :self.classes, :, :]
        fea_2_channel_attention = fea_channel_attention[:, self.classes:2*self.classes, :, :]
        fea_3_channel_attention = fea_channel_attention[:, 2*self.classes:3*self.classes, :, :]
        fea_4_channel_attention = fea_channel_attention[:, 3*self.classes:, :, :]
       
        # -------------------
        # spatial representation

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_1, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_1, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_1 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_2, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_2, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_2 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_3, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_3, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_3 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_4, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_4, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_4 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # -------------------
        # cross spatial relation

        # b, 2, h, w ---> b, 2, h*w ---> b, h*w, 2
        rep_spatial_fea_1 = rep_spatial_fea_1.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_2 = rep_spatial_fea_2.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_3 = rep_spatial_fea_3.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_4 = rep_spatial_fea_4.reshape(b_fea, 2, -1).permute(0, 2, 1)

        b_spatial_fea, l_spatial_fea, c_spatial_fea = rep_spatial_fea_1.size()
        assert c_spatial_fea == 2, 'wrong!'

        # b, h*w, 2 ---> # b, h*w, 4, 2
        rep_spatial_fea = torch.stack([rep_spatial_fea_1, rep_spatial_fea_2, rep_spatial_fea_3, rep_spatial_fea_4], dim=2)    

        interleaved_rep_spatial_fea = rep_spatial_fea.reshape(b_spatial_fea, l_spatial_fea * 4, c_spatial_fea)

        # b, 4*h*w, 2 
        fea_spatial_relation = self.spatial_relation(interleaved_rep_spatial_fea)
        
        # b, 4*h*w, 2 ---> b, h*w, 4, 2 ---> b, h*w, 2
        spatial_fea1, spatial_fea2, spatial_fea3, spatial_fea4 = fea_spatial_relation.reshape(b_spatial_fea, l_spatial_fea, 4, c_spatial_fea).unbind(dim=2)

        # b, h*w, 2 ---> b, 4*h*w, 2
        fea_spatial_relation_restored = torch.cat([spatial_fea1, spatial_fea2, spatial_fea3, spatial_fea4], 1)

        # b, 4*h*w, 2 ---> b, 4*h*w, 1
        fea_spatial_attention = self.sigmoid(self.fc_spatial(fea_spatial_relation_restored))

        # b, 4*h*w, 1 ---> [b, h*w, 1]*4 ---> [b, 1, h*w]*4 ---> [b, 1, h, w]*4
        fea_1_spatial_attention = fea_spatial_attention[:, :n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_2_spatial_attention = fea_spatial_attention[:, n_fea:2*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_3_spatial_attention = fea_spatial_attention[:, 2*n_fea:3*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_4_spatial_attention = fea_spatial_attention[:, 3*n_fea:, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)

        return [fea_1_channel_attention, fea_2_channel_attention, fea_3_channel_attention, fea_4_channel_attention], [fea_1_spatial_attention, fea_2_spatial_attention, fea_3_spatial_attention, fea_4_spatial_attention]


# ----------------------------------------------------------------
# use mamba for channel and spatial attention
# reverse the channel and spatial representations, and input into the 1-D Mamba
# no divide_output.

class mamba_attention_module_V4(nn.Module):
    def __init__(self, classes, fea_dim = 256, d_state = 16, norm_layer = nn.LayerNorm, drop_path = 0, attn_drop_rate = 0):
        super(mamba_attention_module_V4, self).__init__()

        self.fea_dim = fea_dim

        self.classes = classes

        # -------------------
        # channel representation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # -------------------
        # cross channel relation
        self.channel_relation = Mamba_SingleDirect(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)
        self.channel_relation_reverse = Mamba_SingleDirect(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)

        # channel attention
        self.fc1_channel = nn.Conv2d(self.fea_dim * 4, self.fea_dim // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2_channel = nn.Conv2d(self.fea_dim // 16, self.classes * 4, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # -------------------
        # cross spatial relation
        self.spatial_relation = Mamba_SingleDirect(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)
        self.spatial_relation_reverse = Mamba_SingleDirect(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)

        # spatial attention
        self.fc_spatial = nn.Linear(2, 1)

           
    def forward(self, fea_1, fea_2, fea_3, fea_4):

        b_fea, c_fea, h_fea, w_fea = fea_1.size()   # b, c, h, w

        n_fea = h_fea*w_fea

        assert fea_1.size() == fea_2.size() == fea_3.size() == fea_4.size(), 'fea size must be the same'

        # -------------------
        # channel representation

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_1)
        max_channel = self.max_pool(fea_1)
        rep_channel_fea_1 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_2)
        max_channel = self.max_pool(fea_2)
        rep_channel_fea_2 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_3)
        max_channel = self.max_pool(fea_3)
        rep_channel_fea_3 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_4)
        max_channel = self.max_pool(fea_4)
        rep_channel_fea_4 = avg_channel + max_channel  # b, c, 1, 1

        # -------------------
        # cross channel relation

        # b, c, 1, 1 ---> b, 4*c, 1, 1 ---> b, 4*c, 1
        rep_channel_fea = torch.cat([rep_channel_fea_1, rep_channel_fea_2, rep_channel_fea_3, rep_channel_fea_4], 1).squeeze(3)

        # b, 4*c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_relation = self.channel_relation(rep_channel_fea).unsqueeze(3)

        # b, 4*c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_relation_reverse = self.channel_relation_reverse(rep_channel_fea.flip([1])).flip([1]).unsqueeze(3)

        # b, 4*c, 1, 1
        fea_channel_fused = fea_channel_relation + fea_channel_relation_reverse

        # b, 4*c, 1, 1 ---> b, c/4, 1, 1 ---> b, 4*classes, 1, 1
        fea_channel_attention = self.sigmoid(self.fc2_channel(self.relu(self.fc1_channel(fea_channel_fused))))

        # b, 4*classes, 1, 1 ---> [b, classes, 1, 1]*4
        fea_1_channel_attention = fea_channel_attention[:, :self.classes, :, :]
        fea_2_channel_attention = fea_channel_attention[:, self.classes:2*self.classes, :, :]
        fea_3_channel_attention = fea_channel_attention[:, 2*self.classes:3*self.classes, :, :]
        fea_4_channel_attention = fea_channel_attention[:, 3*self.classes:, :, :]
       
        # -------------------
        # spatial representation

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_1, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_1, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_1 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_2, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_2, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_2 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_3, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_3, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_3 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_4, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_4, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_4 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # -------------------
        # cross spatial relation

        # b, 2, h, w ---> b, 2, h*w ---> b, h*w, 2
        rep_spatial_fea_1 = rep_spatial_fea_1.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_2 = rep_spatial_fea_2.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_3 = rep_spatial_fea_3.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_4 = rep_spatial_fea_4.reshape(b_fea, 2, -1).permute(0, 2, 1)

        # b, h*w, 2 ---> b, 4*h*w, 2
        rep_spatial_fea = torch.cat([rep_spatial_fea_1, rep_spatial_fea_2, rep_spatial_fea_3, rep_spatial_fea_4], 1)

        # b, 4*h*w, 2 
        fea_spatial_relation = self.spatial_relation(rep_spatial_fea)

        # b, 4*h*w, 2 
        fea_spatial_relation_reverse = self.spatial_relation_reverse(rep_spatial_fea.flip([1])).flip([1])

        fea_spatial_fused = fea_spatial_relation + fea_spatial_relation_reverse

        # b, 4*h*w, 2 ---> b, 4*h*w, 1
        fea_spatial_attention = self.sigmoid(self.fc_spatial(fea_spatial_fused))

        # b, 4*h*w, 1 ---> [b, h*w, 1]*4 ---> [b, 1, h*w]*4 ---> [b, 1, h, w]*4
        fea_1_spatial_attention = fea_spatial_attention[:, :n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_2_spatial_attention = fea_spatial_attention[:, n_fea:2*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_3_spatial_attention = fea_spatial_attention[:, 2*n_fea:3*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_4_spatial_attention = fea_spatial_attention[:, 3*n_fea:, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)

        return [fea_1_channel_attention, fea_2_channel_attention, fea_3_channel_attention, fea_4_channel_attention], [fea_1_spatial_attention, fea_2_spatial_attention, fea_3_spatial_attention, fea_4_spatial_attention]


# ----------------------------------------------------------------
# use mamba for channel and spatial attention
# reverse the channel and spatial representations, and input into the 1-D Mamba

class mamba_attention_module_V3(nn.Module):
    def __init__(self, classes, fea_dim = 256, d_state = 16, norm_layer = nn.LayerNorm, drop_path = 0, attn_drop_rate = 0):
        super(mamba_attention_module_V3, self).__init__()

        self.fea_dim = fea_dim

        self.classes = classes

        # -------------------
        # channel representation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # -------------------
        # cross channel relation
        self.channel_relation = Mamba_SingleDirect(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)
        self.channel_relation_reverse = Mamba_SingleDirect(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)

        # channel attention
        self.fc1_channel = nn.Conv2d(self.fea_dim * 4, self.fea_dim // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2_channel = nn.Conv2d(self.fea_dim // 16, self.classes * 4, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # -------------------
        # cross spatial relation
        self.spatial_relation = Mamba_SingleDirect(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)
        self.spatial_relation_reverse = Mamba_SingleDirect(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)

        # spatial attention
        self.fc_spatial = nn.Linear(2, 1)

           
    def forward(self, fea_1, fea_2, fea_3, fea_4):

        b_fea, c_fea, h_fea, w_fea = fea_1.size()   # b, c, h, w

        n_fea = h_fea*w_fea

        assert fea_1.size() == fea_2.size() == fea_3.size() == fea_4.size(), 'fea size must be the same'

        # -------------------
        # channel representation

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_1)
        max_channel = self.max_pool(fea_1)
        rep_channel_fea_1 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_2)
        max_channel = self.max_pool(fea_2)
        rep_channel_fea_2 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_3)
        max_channel = self.max_pool(fea_3)
        rep_channel_fea_3 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_4)
        max_channel = self.max_pool(fea_4)
        rep_channel_fea_4 = avg_channel + max_channel  # b, c, 1, 1

        # -------------------
        # cross channel relation

        # b, c, 1, 1 ---> b, 4*c, 1, 1 ---> b, 4*c, 1
        rep_channel_fea = torch.cat([rep_channel_fea_1, rep_channel_fea_2, rep_channel_fea_3, rep_channel_fea_4], 1).squeeze(3)

        # b, 4*c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_relation = self.channel_relation(rep_channel_fea).unsqueeze(3)

        # b, 4*c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_relation_reverse = self.channel_relation_reverse(rep_channel_fea.flip([1])).flip([1]).unsqueeze(3)

        # b, 4*c, 1, 1
        fea_channel_fused = (fea_channel_relation + fea_channel_relation_reverse) / 2

        # b, 4*c, 1, 1 ---> b, c/4, 1, 1 ---> b, 4*classes, 1, 1
        fea_channel_attention = self.sigmoid(self.fc2_channel(self.relu(self.fc1_channel(fea_channel_fused))))

        # b, 4*classes, 1, 1 ---> [b, classes, 1, 1]*4
        fea_1_channel_attention = fea_channel_attention[:, :self.classes, :, :]
        fea_2_channel_attention = fea_channel_attention[:, self.classes:2*self.classes, :, :]
        fea_3_channel_attention = fea_channel_attention[:, 2*self.classes:3*self.classes, :, :]
        fea_4_channel_attention = fea_channel_attention[:, 3*self.classes:, :, :]
       
        # -------------------
        # spatial representation

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_1, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_1, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_1 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_2, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_2, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_2 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_3, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_3, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_3 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_4, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_4, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_4 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # -------------------
        # cross spatial relation

        # b, 2, h, w ---> b, 2, h*w ---> b, h*w, 2
        rep_spatial_fea_1 = rep_spatial_fea_1.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_2 = rep_spatial_fea_2.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_3 = rep_spatial_fea_3.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_4 = rep_spatial_fea_4.reshape(b_fea, 2, -1).permute(0, 2, 1)

        # b, h*w, 2 ---> b, 4*h*w, 2
        rep_spatial_fea = torch.cat([rep_spatial_fea_1, rep_spatial_fea_2, rep_spatial_fea_3, rep_spatial_fea_4], 1)

        # b, 4*h*w, 2 
        fea_spatial_relation = self.spatial_relation(rep_spatial_fea)

        # b, 4*h*w, 2 
        fea_spatial_relation_reverse = self.spatial_relation_reverse(rep_spatial_fea.flip([1])).flip([1])

        fea_spatial_fused = (fea_spatial_relation + fea_spatial_relation_reverse) / 2

        # b, 4*h*w, 2 ---> b, 4*h*w, 1
        fea_spatial_attention = self.sigmoid(self.fc_spatial(fea_spatial_fused))

        # b, 4*h*w, 1 ---> [b, h*w, 1]*4 ---> [b, 1, h*w]*4 ---> [b, 1, h, w]*4
        fea_1_spatial_attention = fea_spatial_attention[:, :n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_2_spatial_attention = fea_spatial_attention[:, n_fea:2*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_3_spatial_attention = fea_spatial_attention[:, 2*n_fea:3*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_4_spatial_attention = fea_spatial_attention[:, 3*n_fea:, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)

        return [fea_1_channel_attention, fea_2_channel_attention, fea_3_channel_attention, fea_4_channel_attention], [fea_1_spatial_attention, fea_2_spatial_attention, fea_3_spatial_attention, fea_4_spatial_attention]


# ----------------------------------------------------------------
# use mamba for channel and spatial attention
# redesign channel attention

class mamba_attention_module_V2(nn.Module):
    def __init__(self, classes, fea_dim = 256, d_state = 16, norm_layer = nn.LayerNorm, drop_path = 0, attn_drop_rate = 0):
        super(mamba_attention_module_V2, self).__init__()

        self.fea_dim = fea_dim

        self.classes = classes

        # -------------------
        # channel representation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # -------------------
        # cross channel relation
        self.channel_relation = Mamba_Vim(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)

        # channel attention: fea1
        self.fea1_channel_fcn = nn.Sequential(
                                            nn.Conv2d(self.fea_dim, self.fea_dim // 16, 1, bias=False),
                                            nn.ReLU(),
                                            nn.Conv2d(self.fea_dim // 16, self.classes, 1, bias=False),
                                            )

        # channel attention: fea2
        self.fea2_channel_fcn = nn.Sequential(
                                            nn.Conv2d(self.fea_dim, self.fea_dim // 16, 1, bias=False),
                                            nn.ReLU(),
                                            nn.Conv2d(self.fea_dim // 16, self.classes, 1, bias=False),
                                            )

        # channel attention: fea3
        self.fea3_channel_fcn = nn.Sequential(
                                            nn.Conv2d(self.fea_dim, self.fea_dim // 16, 1, bias=False),
                                            nn.ReLU(),
                                            nn.Conv2d(self.fea_dim // 16, self.classes, 1, bias=False),
                                            )

        # channel attention: fea4
        self.fea4_channel_fcn = nn.Sequential(
                                            nn.Conv2d(self.fea_dim, self.fea_dim // 16, 1, bias=False),
                                            nn.ReLU(),
                                            nn.Conv2d(self.fea_dim // 16, self.classes, 1, bias=False),
                                            )

        self.sigmoid = nn.Sigmoid()

        # -------------------
        # cross spatial relation
        self.spatial_relation = Mamba_Vim(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)

        # spatial attention
        self.fc_spatial = nn.Linear(2, 1)

           
    def forward(self, fea_1, fea_2, fea_3, fea_4):

        b_fea, c_fea, h_fea, w_fea = fea_1.size()   # b, c, h, w

        n_fea = h_fea*w_fea

        assert fea_1.size() == fea_2.size() == fea_3.size() == fea_4.size(), 'fea size must be the same'

        # -------------------
        # channel representation

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_1)
        max_channel = self.max_pool(fea_1)
        rep_channel_fea_1 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_2)
        max_channel = self.max_pool(fea_2)
        rep_channel_fea_2 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_3)
        max_channel = self.max_pool(fea_3)
        rep_channel_fea_3 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_4)
        max_channel = self.max_pool(fea_4)
        rep_channel_fea_4 = avg_channel + max_channel  # b, c, 1, 1

        # -------------------
        # cross channel relation

        # b, c, 1, 1 ---> b, 4*c, 1, 1 ---> b, 4*c, 1
        rep_channel_fea = torch.cat([rep_channel_fea_1, rep_channel_fea_2, rep_channel_fea_3, rep_channel_fea_4], 1).squeeze(3)

        # b, 4*c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_relation = self.channel_relation(rep_channel_fea).unsqueeze(3)

        # b, 4*c, 1, 1 ---> b, c, 1, 1
        fea_1_channel_relation = fea_channel_relation[:, :c_fea, :, :]
        fea_2_channel_relation = fea_channel_relation[:, c_fea:2*c_fea, :, :]
        fea_3_channel_relation = fea_channel_relation[:, 2*c_fea:3*c_fea, :, :]
        fea_4_channel_relation = fea_channel_relation[:, 3*c_fea:, :, :]
        
        # b, c, 1, 1 ---> b, c/16, 1, 1 ---> b, classes, 1, 1
        fea_1_class_relation = self.fea1_channel_fcn(fea_1_channel_relation)
        fea_2_class_relation = self.fea2_channel_fcn(fea_2_channel_relation)
        fea_3_class_relation = self.fea3_channel_fcn(fea_3_channel_relation)
        fea_4_class_relation = self.fea4_channel_fcn(fea_4_channel_relation)

        # b, classes, 1, 1 ---> b, 4*classes, 1, 1
        fea_classes_relation = torch.cat([fea_1_class_relation, fea_2_class_relation, fea_3_class_relation, fea_4_class_relation], 1)

        # b, 4*classes, 1, 1
        fea_channel_attention = self.sigmoid(fea_classes_relation)

        # b, 4*classes, 1, 1 ---> [b, classes, 1, 1]*4
        fea_1_channel_attention = fea_channel_attention[:, :self.classes, :, :]
        fea_2_channel_attention = fea_channel_attention[:, self.classes:2*self.classes, :, :]
        fea_3_channel_attention = fea_channel_attention[:, 2*self.classes:3*self.classes, :, :]
        fea_4_channel_attention = fea_channel_attention[:, 3*self.classes:, :, :]
       
        # -------------------
        # spatial representation

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_1, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_1, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_1 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_2, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_2, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_2 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_3, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_3, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_3 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_4, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_4, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_4 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # -------------------
        # cross spatial relation

        # b, 2, h, w ---> b, 2, h*w ---> b, h*w, 2
        rep_spatial_fea_1 = rep_spatial_fea_1.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_2 = rep_spatial_fea_2.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_3 = rep_spatial_fea_3.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_4 = rep_spatial_fea_4.reshape(b_fea, 2, -1).permute(0, 2, 1)

        # b, h*w, 2 ---> b, 4*h*w, 2
        rep_spatial_fea = torch.cat([rep_spatial_fea_1, rep_spatial_fea_2, rep_spatial_fea_3, rep_spatial_fea_4], 1)

        # b, 4*h*w, 2 
        fea_spatial_relation = self.spatial_relation(rep_spatial_fea)

        # b, 4*h*w, 2 ---> b, 4*h*w, 1
        fea_spatial_attention = self.sigmoid(self.fc_spatial(fea_spatial_relation))

        # b, 4*h*w, 1 ---> [b, h*w, 1]*4 ---> [b, 1, h*w]*4 ---> [b, 1, h, w]*4
        fea_1_spatial_attention = fea_spatial_attention[:, :n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_2_spatial_attention = fea_spatial_attention[:, n_fea:2*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_3_spatial_attention = fea_spatial_attention[:, 2*n_fea:3*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_4_spatial_attention = fea_spatial_attention[:, 3*n_fea:, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)

        return [fea_1_channel_attention, fea_2_channel_attention, fea_3_channel_attention, fea_4_channel_attention], [fea_1_spatial_attention, fea_2_spatial_attention, fea_3_spatial_attention, fea_4_spatial_attention]


# ----------------------------------------------------------------
# use mamba for channel and spatial attention

class mamba_attention_module(nn.Module):
    def __init__(self, classes, fea_dim = 256, d_state = 16, norm_layer = nn.LayerNorm, drop_path = 0, attn_drop_rate = 0):
        super(mamba_attention_module, self).__init__()

        self.fea_dim = fea_dim

        self.classes = classes

        # -------------------
        # channel representation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # -------------------
        # cross channel relation
        self.channel_relation = Mamba_Vim(d_model = 1, d_state = d_state, d_conv = 4, expand_factor = 2)

        # channel attention
        self.fc1_channel = nn.Conv2d(self.fea_dim * 4, self.fea_dim // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2_channel = nn.Conv2d(self.fea_dim // 16, self.classes * 4, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        # -------------------
        # cross spatial relation
        self.spatial_relation = Mamba_Vim(d_model = 2, d_state = d_state, d_conv = 4, expand_factor = 2)

        # spatial attention
        self.fc_spatial = nn.Linear(2, 1)

           
    def forward(self, fea_1, fea_2, fea_3, fea_4):

        b_fea, c_fea, h_fea, w_fea = fea_1.size()   # b, c, h, w

        n_fea = h_fea*w_fea

        assert fea_1.size() == fea_2.size() == fea_3.size() == fea_4.size(), 'fea size must be the same'

        # -------------------
        # channel representation

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_1)
        max_channel = self.max_pool(fea_1)
        rep_channel_fea_1 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_2)
        max_channel = self.max_pool(fea_2)
        rep_channel_fea_2 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_3)
        max_channel = self.max_pool(fea_3)
        rep_channel_fea_3 = avg_channel + max_channel  # b, c, 1, 1

        # b, c, h, w ---> b, c, 1, 1
        avg_channel = self.avg_pool(fea_4)
        max_channel = self.max_pool(fea_4)
        rep_channel_fea_4 = avg_channel + max_channel  # b, c, 1, 1

        # -------------------
        # cross channel relation

        # b, c, 1, 1 ---> b, 4*c, 1, 1 ---> b, 4*c, 1
        rep_channel_fea = torch.cat([rep_channel_fea_1, rep_channel_fea_2, rep_channel_fea_3, rep_channel_fea_4], 1).squeeze(3)

        # b, 4*c, 1 ---> b, 4*c, 1 ---> b, 4*c, 1, 1
        fea_channel_relation = self.channel_relation(rep_channel_fea).unsqueeze(3)

        # b, 4*c, 1, 1 ---> b, c/4, 1, 1 ---> b, 4*classes, 1, 1
        fea_channel_attention = self.sigmoid(self.fc2_channel(self.relu(self.fc1_channel(fea_channel_relation))))

        # b, 4*classes, 1, 1 ---> [b, classes, 1, 1]*4
        fea_1_channel_attention = fea_channel_attention[:, :self.classes, :, :]
        fea_2_channel_attention = fea_channel_attention[:, self.classes:2*self.classes, :, :]
        fea_3_channel_attention = fea_channel_attention[:, 2*self.classes:3*self.classes, :, :]
        fea_4_channel_attention = fea_channel_attention[:, 3*self.classes:, :, :]
       
        # -------------------
        # spatial representation

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_1, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_1, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_1 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_2, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_2, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_2 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_3, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_3, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_3 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # b, c, h, w ---> b, 1, h, w
        avg_spatial = torch.mean(fea_4, dim=1, keepdim=True)
        max_spatial, _ = torch.max(fea_4, dim=1, keepdim=True)
        # b, 2, h, w
        rep_spatial_fea_4 = torch.cat([avg_spatial, max_spatial], dim=1)  

        # -------------------
        # cross spatial relation

        # b, 2, h, w ---> b, 2, h*w ---> b, h*w, 2
        rep_spatial_fea_1 = rep_spatial_fea_1.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_2 = rep_spatial_fea_2.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_3 = rep_spatial_fea_3.reshape(b_fea, 2, -1).permute(0, 2, 1)
        rep_spatial_fea_4 = rep_spatial_fea_4.reshape(b_fea, 2, -1).permute(0, 2, 1)

        # b, h*w, 2 ---> b, 4*h*w, 2
        rep_spatial_fea = torch.cat([rep_spatial_fea_1, rep_spatial_fea_2, rep_spatial_fea_3, rep_spatial_fea_4], 1)

        # b, 4*h*w, 2 
        fea_spatial_relation = self.spatial_relation(rep_spatial_fea)

        # b, 4*h*w, 2 ---> b, 4*h*w, 1
        fea_spatial_attention = self.sigmoid(self.fc_spatial(fea_spatial_relation))

        # b, 4*h*w, 1 ---> [b, h*w, 1]*4 ---> [b, 1, h*w]*4 ---> [b, 1, h, w]*4
        fea_1_spatial_attention = fea_spatial_attention[:, :n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_2_spatial_attention = fea_spatial_attention[:, n_fea:2*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_3_spatial_attention = fea_spatial_attention[:, 2*n_fea:3*n_fea, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)
        fea_4_spatial_attention = fea_spatial_attention[:, 3*n_fea:, :].permute(0, 2, 1).reshape(b_fea, 1, h_fea, w_fea)

        return [fea_1_channel_attention, fea_2_channel_attention, fea_3_channel_attention, fea_4_channel_attention], [fea_1_spatial_attention, fea_2_spatial_attention, fea_3_spatial_attention, fea_4_spatial_attention]

# ----------------------------------------------------------------
# from https://github.com/SLDGroup/CASCADE/blob/main/lib/decoders.py

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)   


# ----------------------------------------------------------------
# from https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    

# ----------------------------------------------------------------
# from: https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation.py

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
