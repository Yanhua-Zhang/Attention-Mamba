import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential

# ----------------------------------------------------------------
# offset map 

class warp_grid(nn.Module):
    def __init__(self, in_channel, middle_channel):
        super(warp_grid, self).__init__()

        self.channel_change1 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=False)        
        self.channel_change2 = nn.Conv2d(in_channel, middle_channel, kernel_size=1, bias=False)    
        self.offset_map = nn.Conv2d(middle_channel*2, 2, kernel_size=3, padding=1, bias=False)    
           
    def forward(self, low_feature, h_feature):
        n, c, h, w = low_feature.size()           
        
        h_feature = self.channel_change1(h_feature)   
        h_feature_up = F.interpolate(h_feature, (h, w), mode='bilinear', align_corners=True)   
        low_feature = self.channel_change2(low_feature)   

        fuse_feature = torch.cat([low_feature, h_feature_up], 1)         
        flow_field = self.offset_map(fuse_feature)                       

        norm = torch.tensor([[[[w,h]]]]).type_as(low_feature).to(low_feature.device)   
        grid_h = torch.linspace(-1,1,h).view(-1,1).repeat(1,w)              
        grid_w = torch.linspace(-1,1,w).repeat(h,1)                         
        grid = torch.cat((grid_w.unsqueeze(2), grid_h.unsqueeze(2)), 2)                             
        grid = grid.repeat(n,1,1,1).type_as(low_feature).to(low_feature.device)    

        warp_grid = grid + flow_field.permute(0,2,3,1)/norm      

        # out_h_feature = F.grid_sample(h_feature_origin, warp_grid)  

        return warp_grid
    
# ----------------------------------------------------------------

