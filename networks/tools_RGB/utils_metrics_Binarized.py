import numpy as np
import torch


# --------------------------------------------------------------------------------------------------
# modified from: E:\Code_Python\3_PYTHON_code\0_Semantic_segmentation\102_3_Polyp-PVT\Polyp-PVT-main\Train.py

def cal_Dice(logger, pre_logits, gt_tensor, if_check_inf = False):
    # pre_logits: 1, 1, y, x
    # gt: 1, 1, y, x

    # -------------------
    if if_check_inf:
        logger.info("--------------------------------------")
        logger.info("size of pre_logits: " + str(pre_logits.size()))
        logger.info('Unique of pre_logits: ' + str(torch.unique(pre_logits)))

    # .sigmoid(): Maps values to the range (0, 1).
    # .data: Accesses the underlying tensor without tracking autograd. avoid using .data Better alternative: with torch.no_grad():
    # .squeeze(): Removes all dimensions of size 1.
    pre_sig_np = pre_logits.sigmoid().cpu().numpy().squeeze()
    if if_check_inf:
        logger.info("size of pre_sig_np, after squeeze(): " + str(pre_sig_np.shape))
        logger.info('Unique of pre_sig_np, after sigmoid(): ' + str(np.unique(pre_sig_np)))

    pre_normalized = (pre_sig_np - pre_sig_np.min()) / (pre_sig_np.max() - pre_sig_np.min() + 1e-8)
    if if_check_inf:
        logger.info("size of pre_normalized, after normalize: " + str(pre_normalized.shape))
        logger.info('Unique of pre_normalized, after normalize: ' + str(np.unique(pre_normalized)))
    input = pre_normalized

    # -------------------
    if if_check_inf:
        logger.info("size of gt_tensor: " + str(gt_tensor.size()))
        logger.info('Unique of gt_tensor: ' + str(torch.unique(gt_tensor)))

    gt_np = np.asarray(gt_tensor.cpu(), np.float32)
    gt_np_0_1 = gt_np / (gt_np.max() + 1e-8)
    if if_check_inf:
        logger.info("size of gt_np_0_1, after normalize: " + str(gt_np_0_1.shape))
        logger.info('Unique of gt_np_0_1, after normalize: ' + str(np.unique(gt_np_0_1)))

    target = np.array(gt_np_0_1)
    if if_check_inf:
        logger.info("size of target, after normalize, array: " + str(target.shape))
        logger.info('Unique of target, after normalize, array: ' + str(np.unique(target)))

    # -------------------
    smooth = 1
    input_flat = np.reshape(input, (-1))
    target_flat = np.reshape(target, (-1))
    intersection = (input_flat * target_flat)
    dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
    dice = '{:.4f}'.format(dice)
    dice = float(dice)

    return dice


# --------------------------------------------------------------------------------------------------
def cal_Dice_V2(logger, pre_logits, gt_tensor, if_check_inf = False):
    # pre_logits: 1, 1, y, x
    # gt: 1, 1, y, x

    # -------------------
    if if_check_inf:
        logger.info("--------------------------------------")
        logger.info("size of pre_logits: " + str(pre_logits.size()))
        logger.info('Unique of pre_logits: ' + str(torch.unique(pre_logits)))

    # .sigmoid(): Maps values to the range (0, 1).
    # .data: Accesses the underlying tensor without tracking autograd. avoid using .data Better alternative: with torch.no_grad():
    # .squeeze(): Removes all dimensions of size 1.
    # logits ---> 0~1 ---> np 
    # 1, 1, y, x ---> y, x
    pre_sig_np = pre_logits.sigmoid().cpu().numpy().squeeze()
    if if_check_inf:
        logger.info("size of pre_sig_np: " + str(pre_sig_np.shape))
        logger.info('Unique of pre_sig_np: ' + str(np.unique(pre_sig_np)))

    # pre_normalized = (pre_sig_np - pre_sig_np.min()) / (pre_sig_np.max() - pre_sig_np.min() + 1e-8)
    pre_normalized = pre_sig_np
    if if_check_inf:
        logger.info("size of pre_normalized: " + str(pre_normalized.shape))
        logger.info('Unique of pre_normalized: ' + str(np.unique(pre_normalized)))

    detection_mask = pre_normalized.copy()
    detection_mask[pre_normalized >= 0.5] = 1
    detection_mask[pre_normalized < 0.5] = 0
    if if_check_inf:
        logger.info("size of detection_mask: " + str(detection_mask.shape))
        logger.info('Unique of detection_mask: ' + str(np.unique(detection_mask)))

    # -------------------
    if if_check_inf:
        logger.info("size of gt_tensor: " + str(gt_tensor.size()))
        logger.info('Unique of gt_tensor: ' + str(torch.unique(gt_tensor)))

    # 1, 1, y, x ---> np ---> y, x
    gt_np = gt_tensor.cpu().numpy().squeeze()
    if if_check_inf:
        logger.info("size of gt_np: " + str(gt_np.shape))
        logger.info('Unique of gt_np: ' + str(np.unique(gt_np)))

    label_mask = gt_np.copy()
    label_mask[gt_np > 0] = 1
    label_mask[gt_np <= 0] = 0
    if if_check_inf:
        logger.info("size of label_mask: " + str(label_mask.shape))
        logger.info('Unique of label_mask: ' + str(np.unique(label_mask)))

    # -------------------
    epsilon = 1e-8
    dsc_num = np.sum(detection_mask[label_mask == 1]) * 2.0   # Intersection * 2
    dsc_denom = np.sum(detection_mask) + np.sum(label_mask)   # Union + Intersection

    dice = float((dsc_num + epsilon) / (dsc_denom + epsilon))

    return dice



# --------------------------------------------------------------------------------------------------
def cal_Dice_V3(logger, pre_sigmoid, gt_tensor, if_check_inf = False):
    # pre_sigmoid: 1, 1, y, x
    # gt: 1, 1, y, x

    # -------------------
    if if_check_inf:
        logger.info("--------------------------------------")
        logger.info("size of pre_sigmoid: " + str(pre_sigmoid.size()))
        logger.info('Unique of pre_sigmoid: ' + str(torch.unique(pre_sigmoid)))

    # .data: Accesses the underlying tensor without tracking autograd. avoid using .data Better alternative: with torch.no_grad():
    # .squeeze(): Removes all dimensions of size 1.
    # ---> np ---> 1, 1, y, x ---> y, x
    pre_sig_np = pre_sigmoid.cpu().numpy().squeeze()
    if if_check_inf:
        logger.info("size of pre_sig_np: " + str(pre_sig_np.shape))
        logger.info('Unique of pre_sig_np: ' + str(np.unique(pre_sig_np)))

    # pre_normalized = (pre_sig_np - pre_sig_np.min()) / (pre_sig_np.max() - pre_sig_np.min() + 1e-8)
    pre_normalized = pre_sig_np
    if if_check_inf:
        logger.info("size of pre_normalized: " + str(pre_normalized.shape))
        logger.info('Unique of pre_normalized: ' + str(np.unique(pre_normalized)))

    detection_mask = pre_normalized.copy()
    detection_mask[pre_normalized >= 0.5] = 1
    detection_mask[pre_normalized < 0.5] = 0
    if if_check_inf:
        logger.info("size of detection_mask: " + str(detection_mask.shape))
        logger.info('Unique of detection_mask: ' + str(np.unique(detection_mask)))

    # -------------------
    if if_check_inf:
        logger.info("size of gt_tensor: " + str(gt_tensor.size()))
        logger.info('Unique of gt_tensor: ' + str(torch.unique(gt_tensor)))

    # 1, 1, y, x ---> np ---> y, x
    gt_np = gt_tensor.cpu().numpy().squeeze()
    if if_check_inf:
        logger.info("size of gt_np: " + str(gt_np.shape))
        logger.info('Unique of gt_np: ' + str(np.unique(gt_np)))

    label_mask = gt_np.copy()
    label_mask[gt_np > 0] = 1
    label_mask[gt_np <= 0] = 0
    if if_check_inf:
        logger.info("size of label_mask: " + str(label_mask.shape))
        logger.info('Unique of label_mask: ' + str(np.unique(label_mask)))

    # -------------------
    epsilon = 1e-8
    dsc_num = np.sum(detection_mask[label_mask == 1]) * 2.0   # Intersection * 2
    dsc_denom = np.sum(detection_mask) + np.sum(label_mask)   # Union + Intersection

    dice = float((dsc_num + epsilon) / (dsc_denom + epsilon))

    return dice