from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric, MeanIoU, compute_hausdorff_distance, compute_surface_dice
from monai.transforms import AsDiscrete, Activations
import numpy as np
from sklearn.metrics import confusion_matrix


# add calculate_all_metrics_BinarySeg


# --------------------------------------------------------------------------------------------------
# Dice Similarity Coefficient (DSC), Normalized Surface Dice (NSD), Mean Intersection over Union (mIoU), Hausdorff Distance 95% (HD95)

class MONAIMultiClassMetrics:
    def __init__(self, num_classes, reduction='mean_batch', include_background=False, voxelspacing=(1.0, 1.0, 1.0)):
        """
        Initialize MONAI metrics calculator for multi-class segmentation
        
        Args:
            num_classes: Number of classes (including background)
            include_background: Whether to include background class in metrics
        """
        self.num_classes = num_classes
        self.include_background = include_background
        self.reduction = reduction
        self.voxelspacing = voxelspacing
        
        # Initialize metrics
        # It supports single-channel label maps or multi-channel images with class segmentations per channel. This allows the computation for both multi-class and multi-label tasks.
        # If either prediction y_pred or ground truth y have shape BCHW[D], it is expected that these represent one- hot segmentations for C number of classes. If either shape is B1HW[D], it is expected that these are label maps and the number of classes must be specified by the num_classes parameter.
        # Typically this implies that raw predictions from a network must first be activated and possibly made into label maps, eg. for a multi-class prediction tensor softmax and then argmax should be applied over the channel dimensions to produce a label map.
        # The include_background parameter can be set to False to exclude the first category (channel index 0) which is by convention assumed to be background.
        self.dice_metric = DiceMetric(
            include_background=include_background, 
            reduction=self.reduction,
        )

        # y_pred is expected to have binarized predictions and y should be in one-hot format
        # The include_background parameter can be set to False to exclude the first category (channel index 0) which is by convention assumed to be background.
        # y_pred and y can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
        self.mIoU_metric = MeanIoU(
            include_background=include_background,
            reduction=self.reduction,
        )


    def calculate_all_metrics(self, y_pred_oh, y_true_oh):
        """
        Calculate all metrics: DSC, HD95, NSD, and mIoU
        
        Args:
            y_pred: Model predictions [B, C, H, W, D] (logits or probabilities)
            y_true: Ground truth labels [B, 1, H, W, D]
        
        Returns:
            Dictionary containing all metrics
        """
        
        # Calculate metrics
        dice_scores = self.dice_metric(y_pred_oh, y_true_oh)
        iou_scores = self.mIoU_metric(y_pred_oh, y_true_oh)

        hd95_scores = [] 
        for i in range(y_pred_oh.size()[0]):

            # y_pred – It must be one-hot format and first dim is batch
            # y – ground truth to compute mean the distance. It must be one-hot format and first dim is batch. The values should be binarized.
            # include_background: whether to include distance computation on the first channel of the predicted output
            # y_pred, y: C, 1, z, y, x ---> 1, z, y, x ---> 1, 1, z, y, x
            hd95_scores.append(compute_hausdorff_distance(y_pred=y_pred_oh[i].unsqueeze(0), 
                                                          y=y_true_oh[i].unsqueeze(0),
                                                          include_background=self.include_background,  
                                                          percentile=95, 
                                                          distance_metric="euclidean",
                                                          spacing=self.voxelspacing,
                                                          ).cpu().numpy())
        hd95_scores = np.array(hd95_scores)

        nsd_scores = [] 
        for i in range(y_pred_oh.size()[0]):

            # This function computes the (Normalized) Surface Dice
            # y_pred – Predicted segmentation, typically segmentation model output. It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            # y – Reference segmentation. It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            nsd_scores.append(compute_surface_dice(y_pred=y_pred_oh[i].unsqueeze(0), 
                                                   y=y_true_oh[i].unsqueeze(0),
                                                   include_background=self.include_background,
                                                    # class_thresholds=[1.0] * (num_classes if include_background else num_classes - 1),
                                                   class_thresholds=[1.0],
                                                   distance_metric="euclidean",
                                                   spacing=self.voxelspacing,
                                                   ).cpu().numpy())
        nsd_scores = np.array(nsd_scores)
        
        
        # # Calculate mIoU using Dice metric (IoU = DSC / (2 - DSC))
        # iou_scores = self._dice_to_iou(dice_scores)
        
        # Reset metrics for next calculation
        self._reset_metrics()
        
        return {
            'DSC': dice_scores.cpu().numpy(),
            'IoU': iou_scores.cpu().numpy(),
            'HD95': hd95_scores,
            'NSD': nsd_scores,
            'mean_DSC': np.mean(dice_scores.cpu().numpy()),
            'mean_IoU': np.mean(iou_scores.cpu().numpy()),
            'mean_HD95': np.nanmean(hd95_scores),     # np.nanmean important!!!
            'mean_NSD': np.mean(nsd_scores)
        }


    def _reset_metrics(self):
        """Reset all metric states"""
        self.dice_metric.reset()
        self.mIoU_metric.reset()


# --------------------------------------------------------------------------------------------------
# calculate Surface Dice of multiple classes
# modified from: https://github.com/yhygao/CBIM-Medical-Image-Segmentation/blob/main/metric/utils.py

from .utils_metrics_AvgSurfaceDice import compute_surface_distances, compute_average_surface_distance, compute_robust_hausdorff

def calculate_distance(label_pred, label_true, spacing, C, percentage=95):
    # the input args are torch tensors
    if label_pred.is_cuda:
        label_pred = label_pred.cpu()
        label_true = label_true.cpu()

    label_pred = label_pred.numpy()
    label_true = label_true.numpy()
    spacing = spacing.numpy()

    ASD_list = np.zeros(C-1)
    HD_list = np.zeros(C-1)

    for i in range(C-1):
        tmp_surface = compute_surface_distances(label_true==(i+1), label_pred==(i+1), spacing)
        dis_gt_to_pred, dis_pred_to_gt = compute_average_surface_distance(tmp_surface)
        ASD_list[i] = (dis_gt_to_pred + dis_pred_to_gt) / 2 

        HD = compute_robust_hausdorff(tmp_surface, percentage)
        HD_list[i] = HD

    return ASD_list, HD_list


# --------------------------------------------------------------------------------------------------
# 2D/3D case-level DSC, mIoU: y_det and y_true's value must be 0/1

from medpy import metric
import math
import torch

# -------------------
# CPU

# DSC: only consider the forground, because the Background is so large
def calculate_dsc_binary(y_det, y_true):
    """Calculate Dice similarity coefficient (DSC) for N-D Arrays"""

    y_det_np = y_det.cpu().numpy().squeeze() 
    y_true_np = y_true.cpu().numpy().squeeze()  

    assert y_det_np.shape == y_true_np.shape, "Should have the same shape"

    epsilon = 1e-8
    dsc_num = np.sum(y_det_np[y_true_np == 1]) * 2.0   # Intersection * 2
    dsc_denom = np.sum(y_det_np) + np.sum(y_true_np)   # Union + Intersection

    return float((dsc_num + epsilon) / (dsc_denom + epsilon))


# mIoU: only consider the forground, because the Background is so large
def calculate_iou_binary(y_det, y_true):
    """Calculate Intersection over Union (IoU) for N-D Arrays"""

    epsilon = 1e-8
    iou_num = np.sum(y_det[y_true == 1])  # Intersection
    iou_denom = np.sum(y_det) + np.sum(y_true) - iou_num   # Union

    return float((iou_num + epsilon) / (iou_denom + epsilon))



def calculate_hd95_binary(pred, gt, voxelspacing = (3.0, 0.5, 0.5)):

    if pred.sum() > 0 and gt.sum()>0:
        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt, voxelspacing = voxelspacing)  
        return float(hd95)
    
    elif pred.sum() > 0 and gt.sum()==0:   
        # return float(0)    
        return math.nan
     
    else:
        # return float(0)
        return math.nan


# -------------------
# GPU

def calculate_dsc_binary_GPU(y_det, y_true):
    """Calculate Dice similarity coefficient (DSC) for N-D Arrays using PyTorch"""

    y_det_gpu = torch.tensor(y_det, device='cuda')
    y_true_gpu = torch.tensor(y_true, device='cuda')
    
    # Ensure tensors are on the same device
    assert y_det_gpu.device == y_true_gpu.device, "Input tensors must be on the same device"
    
    epsilon = 1e-8
    # Calculate numerator: intersection * 2
    dsc_num = torch.sum(y_det_gpu[y_true_gpu == 1]) * 2.0
    # Calculate denominator: sum of predictions + sum of ground truth
    dsc_denom = torch.sum(y_det_gpu) + torch.sum(y_true_gpu)
    
    # Move result to CPU and convert to Python float if it's on GPU
    dsc = (dsc_num + epsilon) / (dsc_denom + epsilon)

    return dsc.item() if dsc.is_cuda else float(dsc)


def calculate_iou_binary_GPU(y_pred, y_true):

    epsilon = 1e-8

    y_pred_gpu = torch.tensor(y_pred, device='cuda')
    y_true_gpu = torch.tensor(y_true, device='cuda')

    # Ensure tensors are on the same device (GPU/CPU)
    assert y_pred_gpu.device == y_true_gpu.device, "Tensors must be on the same device!"

    # Calculate intersection (foreground only)
    intersection = torch.sum(y_pred_gpu[y_true_gpu == 1])

    # Calculate union: (pred_sum + true_sum - intersection)
    union = torch.sum(y_pred_gpu) + torch.sum(y_true_gpu) - intersection

    # Compute IoU with epsilon smoothing
    iou = (intersection + epsilon) / (union + epsilon)

    # Convert to Python float (handles GPU tensors automatically)
    return iou.item()


def calculate_hd95_binary_GPU(pred, gt, voxelspacing=(3.0, 0.5, 0.5)):

    pred_gpu = torch.tensor(pred, device='cuda').unsqueeze(0).unsqueeze(0)
    gt_gpu = torch.tensor(gt, device='cuda').unsqueeze(0).unsqueeze(0)

    if pred_gpu.sum() > 0 and gt_gpu.sum()>0:

        # Compute the HD95 value
        hd95_value = compute_hausdorff_distance(pred_gpu, gt_gpu, percentile=95, distance_metric="euclidean", spacing=voxelspacing)

        return hd95_value.item() if hd95_value.is_cuda else float(hd95_value)
    
    elif pred_gpu.sum() > 0 and gt_gpu.sum()==0:   
        # return float(0)    
        return math.nan
     
    else:
        # return float(0)
        return math.nan

# --------------------------------------------------------------------------------------------------
# calculate the IoU, DSC, hd95, and Surface Dice of multiple classes


def calculate_all_metrics_MultiClass(y_pred_oh, y_true_oh, if_include_background = False, voxelspacing=(1.0, 1.0, 1.0)):

    # onehot_pred: C, 1, z, y, x
    # onehot_GT: C, 1, z, y, x
    
    # -------------------
    # dice
    dice_scores = []

    for i in range(y_pred_oh.size()[0]):
        
        # C, 1, z, y, x ---> 1, z, y, x ---> z, y, x
        dice_scores.append(calculate_dsc_binary_GPU(y_det = y_pred_oh[i].squeeze(0), y_true = y_true_oh[i].squeeze(0)))

    if not if_include_background:
        dice_scores = dice_scores[1:]

    dice_scores = np.array(dice_scores)

    # -------------------
    # iou
    iou_scores = []

    for i in range(y_pred_oh.size()[0]):
        
        # C, 1, z, y, x ---> 1, z, y, x ---> z, y, x
        iou_scores.append(calculate_iou_binary_GPU(y_pred = y_pred_oh[i].squeeze(0), y_true = y_true_oh[i].squeeze(0)))

    if not if_include_background:
        iou_scores = iou_scores[1:]

    iou_scores = np.array(iou_scores)

    # -------------------
    # hd95
    hd95_scores = [] 

    for i in range(y_pred_oh.size()[0]):
        
        # C, 1, z, y, x ---> 1, z, y, x ---> z, y, x
        hd95_scores.append(calculate_hd95_binary_GPU(pred = y_pred_oh[i].squeeze(0), gt = y_true_oh[i].squeeze(0), voxelspacing = voxelspacing))

    if not if_include_background:
        hd95_scores = hd95_scores[1:]

    nsd_scores = np.array(hd95_scores)

    # -------------------
    nsd_scores = [] 
    for i in range(y_pred_oh.size()[0]):

        # This function computes the (Normalized) Surface Dice
        # y_pred – Predicted segmentation, typically segmentation model output. It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
        # y – Reference segmentation. It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
        nsd_scores.append(compute_surface_dice(y_pred = y_pred_oh[i].unsqueeze(0), 
                                                y = y_true_oh[i].unsqueeze(0),
                                                include_background = if_include_background,
                                                # class_thresholds=[1.0] * (num_classes if include_background else num_classes - 1),
                                                class_thresholds = [1.0],
                                                distance_metric = "euclidean",
                                                spacing = voxelspacing,
                                                ).cpu().numpy())
        
    if not if_include_background:
        nsd_scores = nsd_scores[1:]

    nsd_scores = np.array(nsd_scores).squeeze(-1).squeeze(-1) # N, 1, 1 ---> N, 1 ---> N
    
    # -------------------
    
    return {
        'DSC': dice_scores,
        'IoU': iou_scores,
        'HD95': hd95_scores,
        'NSD': nsd_scores,
        'mean_DSC': np.mean(dice_scores),
        'mean_IoU': np.mean(iou_scores),
        'mean_HD95': np.nanmean(hd95_scores),     # np.nanmean important!!!
        'mean_NSD': np.mean(nsd_scores)
    }




# --------------------------------------------------------------------------------------------------
# calculate the accuracy, sensitivity, specificity, f1_or_dsc, and miou of Binary Seg
# modified from: https://github.com/JCruan519/MALUNet


def calculate_all_metrics_BinarySeg(y_pred_tensor, y_true_tensor):

    # y_pred_tensor: 1, 1, y, x
    # y_true_tensor: 1, 1, y, x
    
    # -------------------
    y_pre = np.array(y_pred_tensor.cpu()).reshape(-1)
    y_true = np.array(y_true_tensor.cpu()).reshape(-1)

    # y_pre = np.where(preds>=config.threshold, 1, 0)
    # y_true = np.where(gts>=0.5, 1, 0)

    # -------------------
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0      # Sensitivity(Recall)
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    precision = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0      
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    
    # -------------------
    nd_accuracy = np.array([accuracy])
    nd_sensitivity = np.array([sensitivity])
    nd_specificity = np.array([specificity])
    nd_precision = np.array([precision])
    nd_f1_or_dsc = np.array([f1_or_dsc])
    nd_miou = np.array([miou])

    # -------------------
    
    return {
        'Acc': nd_accuracy,
        'Sen(Recall)': nd_sensitivity,
        'Spe': nd_specificity,
        'Prec': nd_precision,
        'f1(dsc)': nd_f1_or_dsc,
        'mIoU': nd_miou,
        'mean_Acc': np.mean(nd_accuracy),
        'mean_Sen(Recall)': np.mean(nd_sensitivity),
        'mean_Spe': np.mean(nd_specificity),
        'mean_Prec': np.mean(nd_precision),
        'mean_f1(dsc)': np.mean(nd_f1_or_dsc),
        'mean_mIoU': np.mean(nd_miou)
    }