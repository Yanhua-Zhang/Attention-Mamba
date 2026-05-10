import math
import torch
from torch import nn
import torch.nn.functional as F


# ----------------------------------------------------------------
# use Channel attention to reweight modality-specific features

class CA_reweight_module(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_reweight_module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # b, c, h, w ---> b, c, 1, 1
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)       
        self.relu = nn.ReLU() 
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.activ = nn.Sigmoid()  
           
    def forward(self, t2_fea, adc_fea, dwi_fea):

        # -------------------
        # concat

        b_fea, c_fea, _, _ = t2_fea.size()   # b, c, h, w

        concat_fea = torch.cat([t2_fea, adc_fea, dwi_fea], 1)  # b, c, h, w ---> b, 3c, h, w

        # -------------------
        # reweight

        # b, 3c, h, w ---> b, 3c, 1, 1 ---> b, 3c/r, 1, 1 ---> ReLU ---> b, 3c, 1, 1 ---> activation
        weights = self.activ(self.fc2(self.relu(self.fc1(self.avg_pool(concat_fea)))))

        # [b, 3c, 1, 1] * [b, 3c, h, w] ---> b, 3c, h, w
        concat_fea_reweight = weights * concat_fea

        # -------------------
        # split

        # b, 3c, h, w ---> b, c, h, w
        t2_fea_reweight = concat_fea_reweight[:, :c_fea, :, :]
        adc_fea_reweight = concat_fea_reweight[:, c_fea:2*c_fea, :, :]
        dwi_fea_reweight = concat_fea_reweight[:, 2*c_fea:, :, :]

        return t2_fea_reweight, adc_fea_reweight, dwi_fea_reweight

# ----------------------------------------------------------------


# ----------------------------------------------------------------
# use Spatial attention to reweight modality-specific features

class SA_reweight_module(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SA_reweight_module, self).__init__()

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)       
        self.relu = nn.ReLU() 
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.activ = nn.Sigmoid() 
           
    def forward(self, t2_fea, adc_fea, dwi_fea):

        # -------------------
        # avg

        b_fea, c_fea, h_fea, w_fea = t2_fea.size()   # b, c, h, w

        n_fea = h_fea*w_fea

        # b, c, h, w ---> b, 1, h, w
        avg_t2 = torch.mean(t2_fea, dim=1, keepdim=True)
        avg_adc = torch.mean(adc_fea, dim=1, keepdim=True)
        avg_dwi = torch.mean(dwi_fea, dim=1, keepdim=True)

        # -------------------
        # Flatten

        # b, 1, h, w ---> b, h, w ---> b, h*w, 1, 1
        flatten_t2 = avg_t2.squeeze(1).reshape(b_fea, n_fea, 1, 1)
        flatten_adc = avg_adc.squeeze(1).reshape(b_fea, n_fea, 1, 1)
        flatten_dwi = avg_dwi.squeeze(1).reshape(b_fea, n_fea, 1, 1)

        # -------------------
        # concat

        concat_fea = torch.cat([flatten_t2, flatten_adc, flatten_dwi], 1)  # b, h*w, 1, 1 ---> b, 3*h*w, 1, 1

        # -------------------
        # weights

        # b, 3*h*w, 1, 1 ---> b, 3*h*w/r, 1, 1 ---> ReLU ---> b, 3*h*w, 1, 1 ---> activation
        weights = self.activ(self.fc2(self.relu(self.fc1(concat_fea))))

        # -------------------
        # split and reshape

        # b, 3*h*w, 1, 1 ---> b, h*w, 1, 1 ---> b, h*w ---> b, 1, h*w ---> b, 1, h, w
        t2_weights = weights[:, :n_fea, :, :].reshape(b_fea, n_fea).unsqueeze(1).reshape(b_fea, 1, h_fea, w_fea)
        adc_weights = weights[:, n_fea:2*n_fea, :, :].reshape(b_fea, n_fea).unsqueeze(1).reshape(b_fea, 1, h_fea, w_fea)
        dwi_weights = weights[:, 2*n_fea:, :, :].reshape(b_fea, n_fea).unsqueeze(1).reshape(b_fea, 1, h_fea, w_fea)

        # -------------------
        # reweight

        # [b, 1, h, w] * [b, c, h, w] ---> b, c, h, w
        t2_fea_reweight = t2_weights * t2_fea
        adc_fea_reweight = adc_weights * adc_fea
        dwi_fea_reweight = dwi_weights * dwi_fea

        return t2_fea_reweight, adc_fea_reweight, dwi_fea_reweight

# ----------------------------------------------------------------