import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time

from utils.util import str2bool, set_seed_and_deterministic

from data_loaders.datasets_configs import fun_choose_dataset

# ----------------------------------------------------------------
parser = argparse.ArgumentParser()

# -------------------
# General settings

# choose network and it's version
parser.add_argument('--model_name', type=str, default='My_Attention_Mamba_2D', help='experiment_name')  # choose model name: My_Model, My_Networks_2D
parser.add_argument('--model_type', type=str, default='2D', help='model type')  # choose model type: 2D or 3D
parser.add_argument('--model_version_name', type=str, default='My_Attention_Mamba_2D', help='model_version_name')  # choose model version

# about datset
parser.add_argument('--dataset', type=str, default='Datase1_Synapse_8classes', help='experiment_name')   
parser.add_argument('--fold_name', type=int, default=0, help='5-fold cross validation: 0, 1, 2, 3, 4')

# training
parser.add_argument('--if_training', type=str2bool, default='True', help='if_training')
parser.add_argument('--val_interval', type=int, default=100, help='val_interval')

# inference
parser.add_argument('--if_inferece_valset', type=str2bool, default='True', help='if_inferece_valset')
parser.add_argument('--if_inferece_testset', type=str2bool, default='True', help='if_inferece_testset')
parser.add_argument('--if_inferece_Second_testset', type=str2bool, default='false', help='if_inferece_Second_testset')  # add the Secondary testset
parser.add_argument('--if_calc_metrics_val', type=str2bool, default='True', help='if_calc_metrics_val')
parser.add_argument('--if_calc_metrics_test', type=str2bool, default='True', help='if_calc_metrics_test')
parser.add_argument('--if_calc_metrics_Second_test', type=str2bool, default='false', help='if_calc_metrics_Second_test')  # add the Secondary testset
parser.add_argument('--if_do_ensemble_5_folds', type=str2bool, default='false', help='if_do_ensemble_5_folds')
parser.add_argument('--if_do_ensemble_5_folds_Second_test', type=str2bool, default='false', help='if_do_ensemble_5_folds_Second_test') # add the Secondary testset

# save acc
parser.add_argument('--if_avg_5_folds_val', type=str2bool, default='false', help='if_avg_5_folds_val')
parser.add_argument('--if_avg_5_folds_test', type=str2bool, default='false', help='if_avg_5_folds_test')
parser.add_argument('--if_avg_5_folds_second_test', type=str2bool, default='false', help='if_avg_5_folds_second_test')  # add the Secondary testset
parser.add_argument('--fold_ids_select', nargs='+', type=int, help='selected fold ids for avg and Execel') # [0, 1, 2, 3, 4, 5]: [fold_0, fold_1, fold_2, fold_3, fold_4, ensemble]
parser.add_argument('--metrics_thresh', type=float, default=0.5, help='threshold for calculating metrics')

# Visualization
parser.add_argument('--if_save_Transformed_img', type=str2bool, default='false', help='if_save_Transformed_img')
parser.add_argument('--if_inferece_valtset_SaveFeaMaps', type=str2bool, default='false', help='if_inferece_valtset_SaveFeaMaps')
parser.add_argument('--if_inferece_testset_SaveFeaMaps', type=str2bool, default='false', help='if_inferece_testset_SaveFeaMaps')

# -------------------
# about dataset loader: reduce neg, extend pos
parser.add_argument('--if_reduce_neg_slices_2D', type=str2bool, default='false', help='if_reduce_neg_slices_2D') # for 2D model
parser.add_argument('--if_extend_pos_cases', type=str2bool, default='false', help='if_extend_pos_cases')
parser.add_argument('--select_interval', type=int, default=1, help='select_interval')
parser.add_argument('--max_neg', type=int, default=100000000000000000, help='max_neg')
parser.add_argument('--if_mask_to_long_type', type=str2bool, default='true', help='if_mask_to_long_type')    #  for the BCE_and_wIoU_loss of Polyp dataset

# -------------------
# augmentation type
parser.add_argument('--augmentation_type', type=str, default='ACDC_2D_Median_V1', help='augmentation_type')  

# -------------------
# inference type:
parser.add_argument('--inference_type', type=str, default='MS_inference', help='inference_type')  # MS_inference or SS_inference
parser.add_argument('--inference_scales', nargs='+', type=float, help='inference_scales for MS inference')  # 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4
parser.add_argument('--if_inf_flip', type=str2bool, default='True', help='if flip when inference')
parser.add_argument('--load_model_type', type=str, default='load_best_val_acc_model', help='load_best_loss_model')  # load_best_val_acc_model or load_best_val_loss_model or load_best_train_loss_model or load_best_train_main_loss_model or load_best_the_last_model

# -------------------
# about optimizer
parser.add_argument('--optimizer_name', type=str, default='AdamW', help='optimizer_name')  # AdamW    SGD
parser.add_argument('--betas', nargs='+', type=float, help='betas for AdamW')  # [0.9, 0.999]
# parser.add_argument('--betas', type=list, default=[0.9, 0.999], help='betas for AdamW')  # [0.9, 0.999]
parser.add_argument('--base_lr', type=float, default=0.05, help='segmentation network learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--eps', type=float, default=1e-8, help='eps')   # AdamW

# -------------------
# about scheduler
parser.add_argument('--scheduler_name', type=str, default='cosine_annealing', help='scheduler_name')  # cosine_annealing_wr   poly_lr

# poly_lr
parser.add_argument('--power', type=float, default=0.9, help='power for poly_lr')

# cosine_annealing_wr
parser.add_argument('--t_0_epochs', type=int, default=20, help='t_0_epochs for cosine_annealing_wr')  
parser.add_argument('--t_mult', type=int, default=1, help='t_mult for cosine_annealing_wr')
parser.add_argument('--min_lr', type=float, default=1e-6, help='min_lr for cosine_annealing_wr')

# cosine_annealing
parser.add_argument('--T_max', type=int, default=100, help='T_max for cosine_annealing')  
parser.add_argument('--eta_min', type=float, default=0.00001, help='eta_min for cosine_annealing')
parser.add_argument('--last_epoch', type=int, default=-1, help='last_epoch for cosine_annealing')

# warm up
parser.add_argument('--if_use_warmup', type=str2bool, default='False', help='if_use_warmup')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup_epochs')  

# -------------------
# about loss functions

# or calc_MultiOutput_Align_Loss or calc_MultiOutput_Inconsistenc_Loss or calc_MultipleOutput_Loss
parser.add_argument('--loss_calculator_name', type=str, default='calc_MultipleOutput_CE_and_Dice_Loss', help='loss calculator name') 

parser.add_argument('--loss_weights', nargs='+', type=float, help='loss_weights: 1 + ...')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha')   # FL_and_CE_loss

# for calc_MultiOutput_Inconsistenc_Loss
parser.add_argument('--if_add_inconsist_loss', type=str2bool, default='False', help='if_add_inconsist_loss')

parser.add_argument('--weight_inconsist', type=float, default=0.5, help='weight_inconsist')
parser.add_argument('--nonlin_name', type=str, default='Sigmoid', help='nonlin name') 
parser.add_argument('--inconsist_loss_type', type=str, default='L2', help='inconsist loss type') 

parser.add_argument('--class_weights', nargs='+', type=float, help='class_weights: [0.5, 1, 1, 1]') # weitght of each class in the CE_and_Dice_loss_V2

# -------------------
# Training
parser.add_argument('--max_epochs', type=int, default=200, help='maximum epoch number to train')  
parser.add_argument('--batch_size', type=int, default=48, help='batch_size per gpu')

# -------------------
# for calculating metrics from softmax
parser.add_argument('--num_thresholds', type=int, default=100, help='num_thresholds')
parser.add_argument('--dynamic_threshold_factor', type=int, default=2.5, help='dynamic_threshold_factor')
parser.add_argument('--min_voxels_detection', type=int, default=10, help='min_voxels_detection')
parser.add_argument('--num_parallel_calls', type=int, default=3, help='num_parallel_calls')
parser.add_argument("--candidate_prob_method", type=str, default='max')  # 'average'  'median'

# -------------------
# others
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--marker', type=str, default='full_architecture', help='marker: to distinguish different ablation studies')

# -------------------
# about My_Networks_2D and My_Networks_3D

parser.add_argument('--img_size', nargs='+', type=int, help='img_size')  # y, x (height, width) or z, y, x (depth, height, width)

# about initialization
parser.add_argument('--If_weight_init', type=str2bool, default='false', help='If_weight_init') 

# multi-scales/predictions
parser.add_argument('--HMSA_stage_choose', nargs='+', type=int, help='HMSA_stage_choose')  # [1, 2, 3, 4]

# about backbone
parser.add_argument('--If_pretrained', type=str2bool, default='True', help='If_pretrained')
parser.add_argument('--backbone_name', type=str, default='resnet18_deep', help='backbone_name')
parser.add_argument('--Share_backbone', type=str2bool, default=False, help="Share_backbone")
parser.add_argument('--Dropout_Rate_CNN', nargs='+', type=float, help='Dropout_Rate_CNN for backbone')

# about U-shape decoder
parser.add_argument('--Dropout_Rate_UNet', nargs='+', type=float, help='Dropout_Rate_UNet for U-shape decoder part')

# about Transformer
parser.add_argument('--If_use_position_embedding', type=str2bool, default=True, help="If_use_position_embedding")
parser.add_argument('--If_out_side', type=str2bool, default=True, help="If_out_side")
parser.add_argument('--Self_Attention_Name',type=str, default='ESA_MultiTrans', help='Self_Attention_Name')   # 'ESA_MultiTrans' or 'SSA' or 'ESA_xxx'
parser.add_argument('--Drop_path_rate_Trans', type=float, default=0.1, help='Drop_path_rate_Trans')

# about Mamba
parser.add_argument('--depths_mamba', nargs='+', type=int, help='depths_mamba')  # [2, 2, 2, 2]

# modality features fusion
parser.add_argument('--if_align_features', type=str2bool, default=True, help="if_align_features")
parser.add_argument('--Modality_fea_fusion_name', type=str, default='Sum', help='Modality_fea_fusion_name') # 'Sum', 'Concat' 'Identity'
parser.add_argument('--Dropout_Rate_modality_fea_fusion', type=float, default=0, help='Dropout_Rate_modality_fea_fusion')

parser.add_argument('--if_use_CA_reweight', type=str2bool, default=False, help="if use channel attention for feature reweight")
parser.add_argument('--if_use_SA_reweight', type=str2bool, default=False, help="if use spatial attention for feature reweight")

# multi-scale features fusion
parser.add_argument('--If_use_Multi_scale_fusion', type=str2bool, default=True, help="If_use_Multi_scale_fusion")
parser.add_argument('--Dropout_multi_scale_fea_fusion', type=float, default=0, help='Dropout_multi_scale_fea_fusion')
parser.add_argument('--fea_compress_name', type=str, default='Avg', help='fea_compress_name')     # 'Avg'  'Avg+Max_Sum'  'Avg+Max_Concat' 'Deformable_Atten_Adapt_filter_size' or 'Deformable_Atten_Fix_filter_size'
parser.add_argument('--num_fuse_attention_heads', type=int, default=4, help='num_fuse_attention_heads')
parser.add_argument('--if_cross_attention', type=str2bool, default=False, help="if_cross_attention")

# local-global feature fusion
parser.add_argument('--If_Local_GLobal_Fuison', type=str2bool, default=True, help="If_Local_GLobal_Fuison")
parser.add_argument('--Local_Global_fusion_method', type=str, default='Sum_fusion', help='Local_Global_fusion_method')     # 'Sum_fusion'   
parser.add_argument('--Dropout_Rate_Local_Global_Fusion', type=float, default=0, help='Dropout_Rate_Local_Global_Fusion')

# aux supervison
parser.add_argument('--aux_heads_choose', nargs='+', type=int, help='aux_heads_choose')
parser.add_argument('--If_Deep_Fusion_Supervision', type=str2bool, default=False, help="If_Deep_Fusion_Supervision")
parser.add_argument('--Dropout_Rate_Aux_SegHead', type=float, default=0.1, help='Dropout_Rate_Aux_SegHead')

parser.add_argument('--If_Multi_scale_Fusion_Supervision', type=str2bool, default=False, help="If_Multi_scale_Fusion_Supervision")

# seg head
parser.add_argument('--Dropout_SegHead', type=float, default=0.0, help='Dropout_SegHead')

# -------------------
args = parser.parse_args()

# -------------------


# ----------------------------------------------------------------
if __name__ == "__main__":

    # ----------------------------------------------------------------
    # set seed:
    set_seed_and_deterministic(args.seed)

    # ----------------------------------------------------------------
    # choose dataset

    args.path_dataset, args.path_split, args.num_classes, args.input_channel_dim = fun_choose_dataset(dataset_name=args.dataset)

    if args.loss_calculator_name == 'calc_MultipleOutput_BCE_and_wIoU_loss':
        args.num_classes = 1
    elif args.loss_calculator_name == 'calc_MultipleOutput_BCE_and_Dice_loss':
        args.num_classes = 1

    # ----------------------------------------------------------------
    # creat folder for saving training, testing inf
    
    from utils.util_creat_folders import fun_creat_folders

    fun_creat_folders(args)

    # ----------------------------------------------------------------
    # build log, writer

    from utils.util import build_logger
    from tensorboardX import SummaryWriter
    
    # -------------------
    logger = build_logger('Train, Val, and Test Inf', args.Log_path + "/train_val_test.log")

    logger.info("-----------------------------------------------------------------------------")
    logger.info('All args: ' + str(args))

    # -------------------
    writer = SummaryWriter(args.TensorboardX_path)  

    # ----------------------------------------------------------------
    # load my model
    if args.model_name == 'My_Attention_Mamba_2D':

        from network_architectures.model_loader_2D_My_Attention_Mamba import fun_load_model_2D

        net = fun_load_model_2D(args, logger)

    # ----------------------------------------------------------------
    # loss function
    from loss_functions.calc_loss_loader import fun_calc_loss_loader

    args.calc_loss_fun = fun_calc_loss_loader(args, logger)

    # -------------------
    # optimizer
    from optimizers.optimizer_loader import get_optimizer

    optimizer = get_optimizer(net, args, logger)

    # -------------------
    # scheduler
    from optimizers.scheduler_loader import get_scheduler

    lr_scheduler_train = get_scheduler(optimizer, args, logger)

    if args.if_use_warmup:
        args.scheduler_name = "LambdaLR_warm_up"
        lr_scheduler_warmup = get_scheduler(optimizer, args, logger)
    else:
        lr_scheduler_warmup = None
        logger.info("Do not use warm up")

    # ----------------------------------------------------------------
    from utils.util import FPS_counter_2D_3D, summary_architecture, GFLOPs_params_counter_torchinfo_thop

    if not (args.if_inferece_testset_SaveFeaMaps or args.if_inferece_valtset_SaveFeaMaps):
        summary_architecture(summary_path=args.Summary_path, model_name=args.model_name, model=net)
        GFLOPs_params_counter_torchinfo_thop(model=net, summary_path=args.Summary_path, model_name=args.model_name, input_size=(1, args.input_channel_dim) + tuple(args.img_size))
        FPS_counter_2D_3D(model=net, summary_path=args.Summary_path, model_name=args.model_name, input_size=(1, args.input_channel_dim) + tuple(args.img_size), iteration=100, if_GPU=True)

    # ----------------------------------------------------------------
    # train and val

    from engines.main_2D_net_ACDC_V4 import main_ACDC_2D

    from engines.main_2D_net_ISIC_Binarized import main_ISIC_2D

    trainer = {'2D': {'Datase1_Synapse_8classes': main_ACDC_2D, 'Dataset2_ACDC': main_ACDC_2D, 'Dataset3_ISIC2018_PH2': main_ISIC_2D}}

    trainer[args.model_type][args.dataset](args, net.cuda(), logger, writer, optimizer, lr_scheduler_train, lr_scheduler_warmup) # put Net back on GPU

    # ----------------------------------------------------------------
    # postprocessing

    from postprocessing.postprocessing_Seg_3D import fun_postprocessing_Seg_3D
    from postprocessing.postprocessing_ISIC_2D import fun_postprocessing_ISIC_2D

    if args.dataset == 'Datase1_Synapse_8classes':
        fun_postprocessing_Seg_3D(logger, args)

    elif args.dataset == 'Dataset2_ACDC':
        fun_postprocessing_Seg_3D(logger, args)

    elif args.dataset == 'Dataset3_ISIC2018_PH2':
        fun_postprocessing_ISIC_2D(logger, args)

    else:
        raise ValueError('This dataset is not supported for postprocessing: ' + args.dataset)