import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import copy
from timm.models.layers import trunc_normal_
from torchvision.models import resnet34

from .backbone.backbone_resnet_deep_stem import resnet18_deep_stem, resnet50_deep_stem
from .module.modules import warp_grid
from utils.util_init_weights_2D import init_weights_general_V2
from .module_Mamba.mamba_branch_V1 import mamba_branch
from .module.grid_attention_layer import GridAttentionBlock2D
from .module.attention_CA_SA_V5 import mamba_attention_module_V4


# -----------------------------------------------------------------------------

class Other_models(nn.Module):
    def __init__(self, logger, config, classes=2):
        super(Other_models, self).__init__()

        # -------------------------------------------------
        # about architecture

        self.classes = classes

        self.backbone_name = 'resnet18_deep_stem'   # resnet18_deep_stem    
        self.Dropout_Rate_Backbone = [0, 0, 0, 0, 0]    # [0, 0.3, 0.3, 0.3, 0.3]   [0, 0, 0, 0, 0]   [0, 0.1, 0.1, 0.1, 0.1]

        self.if_stage1_4_repeat_fuse = False
        self.HMSA_stage_choose = (1,2,3,4)
        self.fea_dim = 128     #  128    256
        
        self.mamba_type = 'SS2D_Standard'   # Mamba_G_L  SS2D_Standard
        self.depths_mamba = [2, 2, 2, 2]    # [2, 2, 2, 2]     [3, 3, 3, 3]
        self.attn_drop_rate_mamba = [0., 0., 0., 0.]
        self.dropout_rate_mamba = [0., 0., 0., 0.]   # 0.1
        self.norm_layer_mamba = nn.LayerNorm

        self.If_Local_GLobal_Fuison = False
        self.Local_Global_fusion_method = 'Sum_fusion'
        self.Dropout_Rate_Local_Global_Fusion = 0

        # -------------------------------------------------
        # general settings

        # self.config = config
        self.logger = logger
        self.If_weight_init = config.If_weight_init
        self.If_pretrained = config.If_pretrained

        self.if_check_fea_size = False
        self.If_in_deep_sup = True       # True
        
        # -------------------------------------------------
        # load backbone
        if self.backbone_name == 'resnet18_deep_stem':
            resnet = resnet18_deep_stem(pretrained = self.If_pretrained)
            stage_channels = [64, 128, 256, 512]  

        # -------------------------------------------------
        # add Dropout to Backbones

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, nn.Dropout2d(self.Dropout_Rate_Backbone[0]) if self.Dropout_Rate_Backbone[0] > 0. else nn.Identity())  

        self.layer1, self.layer2, self.layer3, self.layer4 = nn.Sequential(resnet.layer1, nn.Dropout2d(self.Dropout_Rate_Backbone[1]) if self.Dropout_Rate_Backbone[1] > 0. else nn.Identity()), nn.Sequential(resnet.layer2, nn.Dropout2d(self.Dropout_Rate_Backbone[2]) if self.Dropout_Rate_Backbone[2] > 0. else nn.Identity()), nn.Sequential(resnet.layer3, nn.Dropout2d(self.Dropout_Rate_Backbone[3]) if self.Dropout_Rate_Backbone[3] > 0. else nn.Identity()), nn.Sequential(resnet.layer4, nn.Dropout2d(self.Dropout_Rate_Backbone[4]) if self.Dropout_Rate_Backbone[4] > 0. else nn.Identity())

        del resnet 

        # -------------------------------------------------
        # channel change of all stages
        self.channel_changes = []   # orders: [stage1_feature, stage2_feature, stage3_feature, stage4_feature]
        for stage_channel in stage_channels:
            self.channel_changes.append(nn.Sequential(
                nn.Conv2d(stage_channel, self.fea_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.fea_dim),
                nn.ReLU(inplace=True),
                ))

        self.channel_changes = nn.ModuleList(self.channel_changes) 

        # -------------------------------------------------
        # up_branch:
        self.feature_fuses_up = []     # orders: [stage1_feature, stage2_feature, stage3_feature]
        for stage_channel in stage_channels[:-1]:
            self.feature_fuses_up.append(nn.Sequential(
                nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fea_dim),
                nn.ReLU(inplace=True),
            )) 
        self.feature_fuses_up = nn.ModuleList(self.feature_fuses_up)

        # -------------------------------------------------
        # down_branch:
        self.feature_fuses_down = []     # orders: [stage2_feature, stage3_feature, stage4_feature]
        for stage_channel in stage_channels[1:]:
            self.feature_fuses_down.append(nn.Sequential(
                nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fea_dim),
                nn.ReLU(inplace=True),
            )) 
        self.feature_fuses_down = nn.ModuleList(self.feature_fuses_down)

        # -------------------------------------------------
        # fuse the features of down annd up branch 
        if self.if_stage1_4_repeat_fuse:
            self.stage_fuses = []
            for i in range(len(stage_channels)):
                self.stage_fuses.append(nn.Sequential(
                    nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.fea_dim),
                    nn.ReLU(inplace=True),
                ))
            self.stage_fuses = nn.ModuleList(self.stage_fuses)
        else:
            # only fuse features of stage 2, 3  
            self.stage_fuses = []
            for i in range(len(stage_channels)-2):
                self.stage_fuses.append(nn.Sequential(
                    nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.fea_dim),
                    nn.ReLU(inplace=True),
                ))
            self.stage_fuses = nn.ModuleList(self.stage_fuses)  

        # -------------------------------------------------
        # add in_deep_sup here:

        if self.If_in_deep_sup:

            self.in_deep_sup_heads = nn.ModuleList()

            for i in range(len(stage_channels)):

                self.in_deep_sup_heads.append(nn.Sequential(
                    nn.Conv2d(self.fea_dim, self.classes, kernel_size=1, stride=1, padding=0, bias=True)
                    ))

        # -------------------------------------------------
        # mamba branch for global fea

        self.mamba_branches = nn.ModuleList()

        for i in range(len(stage_channels)):

            dpr_mamba = [x.item() for x in torch.linspace(0, self.dropout_rate_mamba[i], self.depths_mamba[i])]  # stochastic depth decay rule

            self.mamba_branches.append(mamba_branch(logger = self.logger, 
                                               dim = self.fea_dim, 
                                               depth = self.depths_mamba[i], 
                                               attn_drop = self.attn_drop_rate_mamba[i], 
                                               drop_path = dpr_mamba, 
                                               d_state = 16, 
                                               norm_layer = self.norm_layer_mamba, 
                                               mamba_type = self.mamba_type)
                                        )   
        
        # -------------------------------------------------
        # local global feature fusion:

        if self.If_Local_GLobal_Fuison:

            self.local_global_Fusions = nn.ModuleList()

            self.local_global_Fusions_drop = nn.ModuleList()

            if self.Local_Global_fusion_method == 'Sum_fusion':
                # local and global feature sum fusion + channel change (line fusion)
                for i in range(len(stage_channels)):
                    self.local_global_Fusions.append(nn.Sequential(
                        nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.fea_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(self.Dropout_Rate_Local_Global_Fusion) if self.Dropout_Rate_Local_Global_Fusion > 0. else nn.Identity()
                        )
                    )

            elif self.Local_Global_fusion_method == 'Attention_Gate':
                # use attention gate for local global feature fusion
                for i in range(len(stage_channels)):
                    self.local_global_Fusions.append(GridAttentionBlock2D(in_channels = self.fea_dim, 
                                                    gating_channels = self.fea_dim,
                                                    inter_channels = self.fea_dim // 2, 
                                                    final_out_channels = self.fea_dim,
                                                    mode='concatenation_MultiTrans',
                                                    sub_sample_factor = (1,1))
                                                    )
                                                
                    self.local_global_Fusions_drop.append(nn.Dropout2d(self.Dropout_Rate_Local_Global_Fusion) if self.Dropout_Rate_Local_Global_Fusion > 0. else nn.Identity())
                                                 
            elif self.Local_Global_fusion_method == 'Concat':
                # local and global feature Concat fusion
                for i in range(len(stage_channels)):
                    self.local_global_Fusions.append(nn.Sequential(
                        nn.Conv2d(self.fea_dim * 2, self.fea_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.fea_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(self.Dropout_Rate_Local_Global_Fusion) if self.Dropout_Rate_Local_Global_Fusion > 0. else nn.Identity()
                        )
                        )

            else:
                raise ValueError("If you choose to fuse local and global feautures, you need to choose the way to fuse them.")

        # -------------------------------------------------
        # if self.use_Multi_loss:
        self.multi_loss_head = []    # [stage1_fuse_feature, stage2_fuse_feature, stage3_fuse_feature, stage4_fuse_feature]

        for i in range(len(stage_channels)):
            self.multi_loss_head.append(
                    nn.Sequential(
                        # nn.Conv2d(self.fea_dim, self.fea_dim, kernel_size=3, stride=1, padding=1),
                        # nn.BatchNorm2d(self.fea_dim),
                        # nn.ReLU(),
                        # nn.Dropout2d(0.1),
                        nn.Conv2d(self.fea_dim, self.classes, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.multi_loss_head = nn.ModuleList(self.multi_loss_head)

        # -------------------------------------------------
        # offset map
        self.stages_offset = []
        # offsets between stage 1 and stage 2, stage 2 and stage 3, stage 3 and stage 4 
        for i in range(len(stage_channels)-1):   
                self.stages_offset.append(warp_grid(self.fea_dim, self.fea_dim//2))
        self.stages_offset = nn.ModuleList(self.stages_offset)

        # -------------------------------------------------
        # spatial-attention

        self.multi_scale_attentions = mamba_attention_module_V4(classes = self.classes, fea_dim = self.fea_dim, d_state = 16)

        # -------------------------------------------------
        # initialization

        if self.If_weight_init:
            self.logger.info('Start init weights.')
            self._init_weights()
        else:
            self.logger.info('Do not init weights.')

        # -------------------------------------------------


    def _init_weights(self):

        for m in self.modules():
            m.apply(init_weights_general_V2)   


    def forward(self, x):

        # for grey images
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # if single channel ---> 3 channel

        # --------------------------------------------------------------
        # backbone
        x_size = x.size()
        
        out_in = x
        x = self.layer0(x)
        out0 = x
        x = self.layer1(x)
        out1 = x
        stage1_feature = x
        x = self.layer2(x)
        out2 = x
        stage2_feature = x
        x = self.layer3(x)  
        out3 = x
        stage3_feature = x
        x = self.layer4(x)
        out4 = x
        stage4_feature = x

        # --------------------------------------------------------------
        # channel change
        stage_features = [stage1_feature, stage2_feature, stage3_feature, stage4_feature]
        compress_stage_features = []
        for i in range(len(stage_features)):
            compress_stage_features.append(self.channel_changes[i](stage_features[i]))

        # --------------------------------------------------------------
        # Top-down branch 
        stage_features_up = [compress_stage_features[0], compress_stage_features[1], compress_stage_features[2]]
        f = compress_stage_features[3]                            
        FPN_features_up = [f]    
        for i in reversed(range(len(stage_features_up))):
            stage_feature = stage_features_up[i]
            f = F.interpolate(f, (stage_feature.size())[2:], mode='bilinear', align_corners=True)   
            f = self.feature_fuses_up[i](f + stage_feature)        
            FPN_features_up.append(f)                             # [stage4, stage3, stage2, stage1]

        # --------------------------------------------------------------
        # Bottom-up branch 
        stage_features_down = [compress_stage_features[1], compress_stage_features[2], compress_stage_features[3]]
        f = compress_stage_features[0]                              
        FPN_features_down = [f]    
        for i in range(len(stage_features_down)):
            stage_feature = stage_features_down[i]  
            f = F.interpolate(f, (stage_feature.size())[2:], mode='bilinear', align_corners=True)  
            f = self.feature_fuses_down[i](f + stage_feature)       
            FPN_features_down.append(f)                             # [stage1, stage2, stage3, stage4]

        # ---------------------------------------------------------------
        # Bottom-up and Top-down branch feature fuse
        FPN_features_up.reverse()       # [stage1, stage2, stage3, stage4]
        if self.if_stage1_4_repeat_fuse:
            fuse_features = []
            for i in range(len(FPN_features_up)):
                fuse_features.append(self.stage_fuses[i](FPN_features_up[i]+FPN_features_down[i]))
        else:
            # only feature fuse of stage 2, stage3
            fuse_features = [FPN_features_up[0]]           # up stage1 
            j = 0
            for i in range(1, len(FPN_features_up)-1):
                fuse_features.append(self.stage_fuses[j](FPN_features_up[i]+FPN_features_down[i]))
                j += 1
            fuse_features.append(FPN_features_down[3])     # down stage4

        # ----------------------------------------------------------------
        # offsets between stage 1 and stage 2, stage 2 and stage 3, stage 3 and stage 4
        stages_warp_grid = []
        for i in range(len(fuse_features)-1):
            stages_warp_grid.append(self.stages_offset[i](fuse_features[i], fuse_features[i+1]))

        # recursive alignment
        for i in self.HMSA_stage_choose:
            if i != 1:  # up to the size of stage1 
                for k in reversed(range(i-1)):
                    fuse_features[i-1] = F.grid_sample(fuse_features[i-1], stages_warp_grid[k], align_corners=True)   

        # ---------------------------------------------------------------
        in_deep_sup_outs = []  # [stage1, stage2, stage3, stage4]

        if self.If_in_deep_sup:

            for i in range(len(fuse_features)):

                in_deep_sup_score = self.in_deep_sup_heads[i](fuse_features[i])
                in_deep_sup_score_up = F.interpolate(in_deep_sup_score, x_size[2:], mode='bilinear', align_corners=True)

                in_deep_sup_outs.append(in_deep_sup_score_up)

        # ---------------------------------------------------------------
        # add mamba branch on each level:

        branch_out_fea = []

        for i in range(len(fuse_features)):
            
            fea_locals = fuse_features[i]
            # B, C*4, H, W ---> B, H, W, C*4
            fea_locals_permuted = fea_locals.permute(0, 2, 3, 1)  
            fea_global = self.mamba_branches[i](fea_locals_permuted)   # input request: B, H, W, C
            # B, H, W, C*4 ---> B, C*4, H, W
            fea_global = fea_global.permute(0, 3, 1, 2)

            # --------------------------------
            if self.If_Local_GLobal_Fuison:

                if self.Local_Global_fusion_method == 'Sum_fusion':

                    fuse = fea_global + fea_locals
                    out_fuse = self.local_global_Fusions[i](fuse) 

                elif self.Local_Global_fusion_method == 'Attention_Gate':
                    
                    out_fuse = self.local_global_Fusions[i](fea_locals, fea_global)
                    out_fuse = self.local_global_Fusions_drop[i](out_fuse)
                                            
                elif self.Local_Global_fusion_method == 'Concat':
                    
                    fuse = torch.cat([fea_locals, fea_global], 1)
                    out_fuse = self.local_global_Fusions[i](fuse)

                else:
                    raise ValueError("If you choose to fuse local and global feautures, you need to choose the way to fuse them.")
            
            else:

                out_fuse = fea_global

            branch_out_fea.append(out_fuse)

        # -----------------------------------------------------------------
        # multi_loss_scores

        multihead_supervison_outs = []
        multi_loss_scores = []

        for i in range(len(branch_out_fea)):

            stage_score = self.multi_loss_head[i](branch_out_fea[i])

            if (i+1) in self.HMSA_stage_choose:
                stage_score_up = F.interpolate(stage_score, x_size[2:], mode='bilinear', align_corners=True)
                multihead_supervison_outs.append(stage_score_up)

            multi_loss_scores.append(stage_score)

        # -----------------------------------------------------------------
        # calculate final_score

        channel_attentions, spatial_attentions = self.multi_scale_attentions(branch_out_fea[0], branch_out_fea[1], branch_out_fea[2], branch_out_fea[3])

        HMSA_stage_scores = []

        for i in self.HMSA_stage_choose:

            stage_score_map = multi_loss_scores[i-1]

            stage_attention_map = channel_attentions[i-1] * spatial_attentions[i-1]

            HMSA_stage_score = stage_score_map*stage_attention_map

            HMSA_stage_scores.append(HMSA_stage_score)

        final_score  = sum(HMSA_stage_scores)

        logits = F.interpolate(final_score, x_size[2:], mode='bilinear', align_corners=True)

        # -------------------------------------------------

        if self.training:

            return [logits] + multihead_supervison_outs + in_deep_sup_outs

        else:
            return logits


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    