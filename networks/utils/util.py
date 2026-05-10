import os
import numpy as np
from PIL import Image
import argparse
import logging

import torch
from torch import nn
import random


# -----------------------------------------------------------------------------
def set_seed_and_deterministic(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
def read_config(file_path):
    with open(file_path, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    cfg={}
    for key in cfg_from_file:
        # print(key)
        # items()
        if type(cfg_from_file[key]) == dict:
            for k, v in cfg_from_file[key].items():
                # print(k)
                # print(v)
                cfg[k] = v
        else:
            cfg[key] = cfg_from_file[key]
    return cfg


# -----------------------------------------------------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset() # reset val, sum when initialization 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# -----------------------------------------------------------------------------
# learning rate
def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


# -----------------------------------------------------------------------------
# calculate mIoU on CPU
def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

# calculate mIoU on GPU
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)   
    target = target.view(-1)
    output[target == ignore_index] = ignore_index    
    intersection = output[output == target]   

    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)    
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection       
    return area_intersection, area_union, area_target


# -----------------------------------------------------------------------------
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# -----------------------------------------------------------------------------
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


# -----------------------------------------------------------------------------
def get_args(yaml_path):
    # read yaml file
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')  # Creat ArgumentParser 

    parser.add_argument('--config', type=str,
                        default=yaml_path, help='config file')

    parser.add_argument('opts', help='see config/***/***.yaml for all options',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args() 

    cfg = read_yaml.read_config(args.config)   
    return cfg

def build_logger(logger_name, file_name_log):
    # logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)   # the level of output

    # Creat StreamHandler: output Inf in the Terminal
    handler = logging.StreamHandler()               
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # creat result.log and write inf
    file_handler = logging.FileHandler(file_name_log)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    return logger


# -----------------------------------------------------------------------------
def summary_architecture(summary_path, model_name, model):
             
    file_name_log = summary_path + '/' + 'architecture_summary.log'  # the name of the log file
    logger = build_logger('summary_architecture', file_name_log)

    logger.info('model name: ' + model_name + ': ')
    logger.info(model)
    logger.info('-----------------------------------------------------------------------')
    logger.info('        ')


from torchinfo import summary
def GFLOPs_params_counter_torchinfo(model, summary_path, model_name, input_size):

    file_name_log = summary_path + '/' + 'model_GFLOPs_params.log'  # logger 
    logger = build_logger('summary_GFLOPs_params', file_name_log)

    logger.info('model name: ' + model_name + ': ')
    logger.info('input size: ' + str(input_size))

    num_threads = torch.get_num_threads()
    logger.info('num of threads: ' + str(num_threads))

    logger.info('use torchinfo:')
    logger.info(summary(model, input_size=input_size))
    logger.info('End')
    logger.info('-----------------------------------------------------------------------')
    logger.info('        ')


from thop import profile
def GFLOPs_params_counter_torchinfo_thop(model, summary_path, model_name, input_size):

    file_name_log = summary_path + '/' + 'model_GFLOPs_params.log'  # logger 
    logger = build_logger('summary_GFLOPs_params', file_name_log)

    logger.info('model name: ' + model_name + ': ')
    logger.info('input size: ' + str(input_size))

    num_threads = torch.get_num_threads()
    logger.info('num of threads: ' + str(num_threads))

    logger.info('-----------------------------------------------------------------------')
    logger.info('use torchinfo:')
    logger.info(summary(model, input_size=input_size))
    logger.info('End')
    logger.info('-----------------------------------------------------------------------')

    logger.info('use thop profile:')
    input_size_tensor = torch.randn(input_size).cuda()
    flops, params = profile(model, inputs=(input_size_tensor,))
    logger.info('flops (G): ' + str(flops/1e9))			
    logger.info('params (M): ' + str(params/1e6))			

    total = sum(p.numel() for p in model.parameters())
    logger.info('Total params (M): ' + str(total/1e6))

    logger.info('End')
    logger.info('-----------------------------------------------------------------------')
    logger.info('        ')


import time
def FPS_counter_2D_3D(model, summary_path, model_name, input_size, iteration=100, if_GPU=True): 

    file_name_log = summary_path + '/' + 'model_FPS.log'  # logger
    logger = build_logger('summary_FPS', file_name_log)

    logger.info('model name: ' + model_name + ': ')
    logger.info('Use the method provided by SFNet to count FPS: ')
    logger.info('input size: ' + str(input_size))

    num_threads = torch.get_num_threads()
    logger.info('num of threads: ' + str(num_threads))

    if if_GPU:
        input_t = torch.rand(input_size).cuda()
        model.eval()
        model.cuda()
    else:
        input_t = torch.rand(input_size).cpu()
        model.eval()
        model.cpu()

    logger.info("start warm up")
    for i in range(10):
        model(input_t)
    logger.info("warm up done")

    start_ts = time.time()
    for i in range(iteration):
        model(input_t)

    if if_GPU:
        torch.cuda.synchronize() # Waits for all kernels in all streams on a CUDA device to complete.
    end_ts = time.time()

    t_cnt = end_ts - start_ts
  
    logger.info("FPS: %f" % (100 / t_cnt))
    logger.info(f"Inference time {t_cnt/100*1000} ms")
    logger.info('End')
    logger.info('-----------------------------------------------------------------------')
    logger.info('        ')


# -----------------------------------------------------------------------------

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt)  
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:   
        return 1, 0     
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 1, N, x, y ---> N, x, y
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)  # N, x, y
        for ind in range(image.shape[0]):  
            slice = image[ind, :, :]    # N, x, y ---> x, y
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # x, y ---> 1, 1, x, y ---> float ---> GPU
            
            net.eval()
            with torch.no_grad():
                outputs = net(input)

                # 1, classes, x, y ---> 1, 1, x, y ---> 1, x, y
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  
                else:
                    pred = out
                prediction[ind] = pred  # 1, x, y ---> N, x, y
    else:
        # np ---> Tensor ---> x, y ---> 1, 1, x, y
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))  

    # save
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))  
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))  
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

# -----------------------------------------------------------------------------
# the mDice, mIoU of RGB

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

def calculate_metric_percase_appended(pred, gt):
    # pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)

        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt)  

        mIoU = mean_iou_np(pred, gt)
        dice2 = mean_dice_np(pred, gt)

        return dice, hd95, mIoU, dice2
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 1
    else:
        return 0, 0, 0, 0

# -----------------------------------------------------------
def test_RGB_image(args, image, label, net, classes, test_save_path=None, case=None):    
    # # 1, 3, x, y ---> 3, x, y ---> cpu ---> np
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # x, y = image.shape[2], image.shape[3]

    # image = image.transpose(1, 2, 0)  # 3, x, y ---> x, y, 3 
    # image_resized = cv2.resize(image, (patch_size[0], patch_size[1]), interpolation = cv2.INTER_LINEAR)
    # image_resized = image_resized.transpose(2, 0, 1)  # x, y, 3 ---> 3, x, y 
    
    # input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # 3, x, y ---> 1, 3, x, y ---> float ---> GPU

    label = label.squeeze(0).cpu().detach().numpy()  # 
    w, h = label.shape

    input = image.float().cuda()

    # inference
    net.eval()
    with torch.no_grad():
        outputs = net(input)

        # 1, classes, x, y ---> 1, x, y ---> x, y
        if args.num_classes == 1:
            out = outputs.sigmoid().data.cpu().numpy().squeeze()
            out = 1*(out > 0.5)
        else:
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
            out = out.cpu().detach().numpy()

        x, y = out.shape[0], out.shape[1]

        # print(out.shape)

        if x !=w or y != h:
            # pred = zoom(out, (x / w, y / h), order=0)  
            pred = cv2.resize(out, (h, w), interpolation=cv2.INTER_NEAREST)
        else:
            pred = out

        prediction = pred  # 

    metric_list = []
    # for i in range(1, classes):
    metric_list.append(calculate_metric_percase_appended(prediction, label))  

    # save
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))  
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))

        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.png")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.png")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.png")
        
    return metric_list

# -----------------------------------------------------------------------------
# for the bool input of argparse
def str2bool(str):
    return True if str.lower() == 'true' else False

# -----------------------------------------------------------------------------
# for logging inf
import logging
def build_logger(logger_name, file_name_log):
    # logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)   # inf level

    # creat StreamHandler, output logger in the terminal
    handler = logging.StreamHandler()               # output Inf in the Terminal
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # creat result.log, and write in logger
    file_handler = logging.FileHandler(file_name_log)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    return logger

# -----------------------------------------------------------------------------
# calculate Dice-Similarity Coefficient (DSC)

def cal_dice(pred, target, C): 
    N = pred.shape[0]
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.) 

    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred, 1.) 

    intersection= pred_mask * target_mask
    summ = pred_mask + target_mask

    intersection = intersection.sum(0).type(torch.float32)
    summ = summ.sum(0).type(torch.float32)
    
    eps = torch.rand(C, dtype=torch.float32)
    eps = eps.fill_(1e-7)

    summ += eps.cuda()
    dice = 2 * intersection / summ

    return dice, intersection, summ

# -----------------------------------------------------------------------------
import SimpleITK 
import pickle

def save_transformed_img_3D(args, logger, i_batch, dict_data):

    # MetaTensor (monai.transforms) ---> tensor ---> np
    img_transformed = dict_data['Img'].as_tensor().numpy().astype(np.float32)      # b, 3, z, x, y 
    label_lesion_transformed = dict_data['GT_lesion'].as_tensor().numpy().astype(np.int16)    # b, 1, z, x, y 
    label_prostate_transformed = dict_data['GT_prostate'].as_tensor().numpy().astype(np.int16)   # b, 1, z, x, y 
    file_name = dict_data['file_name']

    for j in range(img_transformed.shape[0]):

        logger.info("label_prostate_transformed unique: {}".format(str(np.unique(label_prostate_transformed[j]))))
        logger.info("label_prostate_transformed size: {}".format(str(label_prostate_transformed[j].shape)))
        logger.info("label_prostate_transformed[0] unique: {}".format(str(np.unique(label_prostate_transformed[j][0]))))
        logger.info("label_prostate_transformed[0] size: {}".format(str(label_prostate_transformed[j][0].shape)))
        logger.info("label_prostate_transformed[0] astype: {}".format(str(label_prostate_transformed[j][0].astype)))

        # -------------------
        pkl_path = args.path_dataset + '/' + 'Train_Val' + '/' + file_name[j] + '.pkl'
        pkl_list = pickle.load(open(pkl_path, 'rb'))

        Inf_Spacing = pkl_list[0]['Spacing']
        Inf_Origin = pkl_list[0]['Origin']
        Inf_Direction = pkl_list[0]['Direction']

        # -------------------
        # recover and save transformed img

        t2w_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch) + '_transformed_t2w.nii.gz'
        adc_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch)  + '_transformed_adc.nii.gz'
        dwi_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch)  + '_transformed_dwi.nii.gz'
        lesion_mask_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch)  + '_transformed_lesion_mask.nii.gz'
        Prostate_mask_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch)  + '_transformed_Prostate_mask.nii.gz'

        logger.info("transformed t2w_path_save: {}".format(t2w_path_save))
        logger.info("transformed adc_path_save: {}".format(adc_path_save))
        logger.info("transformed dwi_path_save: {}".format(dwi_path_save))
        logger.info("transformed lesion_mask_path_save: {}".format(lesion_mask_path_save))
        logger.info("transformed Prostate_mask_path_save: {}".format(Prostate_mask_path_save))

        t2w_np = img_transformed[j][0]
        t2w_Img = SimpleITK.GetImageFromArray(t2w_np)
        t2w_Img.SetSpacing(Inf_Spacing)
        t2w_Img.SetOrigin(Inf_Origin)
        t2w_Img.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(t2w_Img, t2w_path_save)

        adc_np = img_transformed[j][1]
        adc_Img = SimpleITK.GetImageFromArray(adc_np)
        adc_Img.SetSpacing(Inf_Spacing)
        adc_Img.SetOrigin(Inf_Origin)
        adc_Img.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(adc_Img, adc_path_save)

        dwi_np = img_transformed[j][2]
        dwi_Img = SimpleITK.GetImageFromArray(dwi_np)
        dwi_Img.SetSpacing(Inf_Spacing)
        dwi_Img.SetOrigin(Inf_Origin)
        dwi_Img.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(dwi_Img, dwi_path_save)

        lesion_mask_np = label_lesion_transformed[j][0]
        lesion_mask_Img = SimpleITK.GetImageFromArray(lesion_mask_np)
        lesion_mask_Img.SetSpacing(Inf_Spacing)
        lesion_mask_Img.SetOrigin(Inf_Origin)
        lesion_mask_Img.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(lesion_mask_Img, lesion_mask_path_save)

        Prostate_mask_np = label_prostate_transformed[j][0]
        Prostate_mask_Img = SimpleITK.GetImageFromArray(Prostate_mask_np)
        Prostate_mask_Img.SetSpacing(Inf_Spacing)
        Prostate_mask_Img.SetOrigin(Inf_Origin)
        Prostate_mask_Img.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(Prostate_mask_Img, Prostate_mask_path_save)


def save_img_3D(args, logger, img, file_name, suffix):

    # -------------------
    pkl_path = args.path_dataset + '/' + 'Test' + '/' + file_name + '.pkl'
    if not os.path.exists(pkl_path):
        pkl_path = args.path_dataset + '/' + 'Train_Val' + '/' + file_name + '.pkl'

    pkl_list = pickle.load(open(pkl_path, 'rb'))

    Inf_Spacing = pkl_list[0]['Spacing']
    Inf_Origin = pkl_list[0]['Origin']
    Inf_Direction = pkl_list[0]['Direction']

    # -------------------
    # recover and save transformed img

    img = img.cpu().numpy().astype(np.float32)

    t2w_path_save = args.Visual_path + '/' + file_name + '_' + suffix + '_t2w.nii.gz'
    adc_path_save = args.Visual_path + '/' + file_name + '_' + suffix + '_adc.nii.gz'
    dwi_path_save = args.Visual_path + '/' + file_name + '_' + suffix + '_dwi.nii.gz'

    logger.info("t2w_path_save for checking: {}".format(t2w_path_save))
    logger.info("adc_path_save for checking: {}".format(adc_path_save))
    logger.info("dwi_path_save for checking: {}".format(dwi_path_save))

    t2w_np = img[0]
    t2w_Img = SimpleITK.GetImageFromArray(t2w_np)
    t2w_Img.SetSpacing(Inf_Spacing)
    t2w_Img.SetOrigin(Inf_Origin)
    t2w_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(t2w_Img, t2w_path_save)

    adc_np = img[1]
    adc_Img = SimpleITK.GetImageFromArray(adc_np)
    adc_Img.SetSpacing(Inf_Spacing)
    adc_Img.SetOrigin(Inf_Origin)
    adc_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(adc_Img, adc_path_save)

    dwi_np = img[2]
    dwi_Img = SimpleITK.GetImageFromArray(dwi_np)
    dwi_Img.SetSpacing(Inf_Spacing)
    dwi_Img.SetOrigin(Inf_Origin)
    dwi_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(dwi_Img, dwi_path_save)


def save_transformed_img_2D(args, logger, i_batch, dict_data):

    # MetaTensor (monai.transforms) ---> tensor ---> np
    img_transformed = dict_data['Img'].as_tensor().permute(1, 0, 2, 3).numpy().astype(np.float32)      # b, 3, y, x  ---> 3, b, y, x
    label_lesion_transformed = dict_data['GT_lesion'].as_tensor().permute(1, 0, 2, 3).numpy().astype(np.int16)    # b, 1, y, x ---> 1, b, y, x
    label_prostate_transformed = dict_data['GT_prostate'].as_tensor().permute(1, 0, 2, 3).numpy().astype(np.int16)   # b, 1, y, x ---> 1, b, y, x
    file_name = dict_data['file_name']

    logger.info("label_prostate_transformed unique: {}".format(str(np.unique(label_prostate_transformed))))
    logger.info("label_prostate_transformed size: {}".format(str(label_prostate_transformed.shape)))
    logger.info("label_prostate_transformed[0] unique: {}".format(str(np.unique(label_prostate_transformed[0]))))
    logger.info("label_prostate_transformed[0] size: {}".format(str(label_prostate_transformed[0].shape)))
    logger.info("label_prostate_transformed[0] astype: {}".format(str(label_prostate_transformed[0].astype)))

    # -------------------
    pkl_path = args.path_dataset + '/' + 'Train_Val' + '/' + file_name[0] + '.pkl'
    pkl_list = pickle.load(open(pkl_path, 'rb'))

    Inf_Spacing = pkl_list[0]['Spacing']
    Inf_Origin = pkl_list[0]['Origin']
    Inf_Direction = pkl_list[0]['Direction']

    # -------------------
    # recover and save transformed img

    t2w_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch) + '_transformed_t2w.nii.gz'
    adc_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch)  + '_transformed_adc.nii.gz'
    dwi_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch)  + '_transformed_dwi.nii.gz'
    lesion_mask_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch)  + '_transformed_lesion_mask.nii.gz'
    Prostate_mask_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch)  + '_transformed_Prostate_mask.nii.gz'

    logger.info("transformed t2w_path_save: {}".format(t2w_path_save))
    logger.info("transformed adc_path_save: {}".format(adc_path_save))
    logger.info("transformed dwi_path_save: {}".format(dwi_path_save))
    logger.info("transformed lesion_mask_path_save: {}".format(lesion_mask_path_save))
    logger.info("transformed Prostate_mask_path_save: {}".format(Prostate_mask_path_save))

    t2w_np = img_transformed[0]   # 3, b, y, x ---> b, y, x (depth, height, width)
    t2w_Img = SimpleITK.GetImageFromArray(t2w_np)   # b, y, x (depth, height, width) ---> x, y, b
    t2w_Img.SetSpacing(Inf_Spacing)
    t2w_Img.SetOrigin(Inf_Origin)
    t2w_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(t2w_Img, t2w_path_save)

    adc_np = img_transformed[1]
    adc_Img = SimpleITK.GetImageFromArray(adc_np)
    adc_Img.SetSpacing(Inf_Spacing)
    adc_Img.SetOrigin(Inf_Origin)
    adc_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(adc_Img, adc_path_save)

    dwi_np = img_transformed[2]
    dwi_Img = SimpleITK.GetImageFromArray(dwi_np)
    dwi_Img.SetSpacing(Inf_Spacing)
    dwi_Img.SetOrigin(Inf_Origin)
    dwi_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(dwi_Img, dwi_path_save)

    lesion_mask_np = label_lesion_transformed[0]
    lesion_mask_Img = SimpleITK.GetImageFromArray(lesion_mask_np)
    lesion_mask_Img.SetSpacing(Inf_Spacing)
    lesion_mask_Img.SetOrigin(Inf_Origin)
    lesion_mask_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(lesion_mask_Img, lesion_mask_path_save)

    Prostate_mask_np = label_prostate_transformed[0]
    Prostate_mask_Img = SimpleITK.GetImageFromArray(Prostate_mask_np)
    Prostate_mask_Img.SetSpacing(Inf_Spacing)
    Prostate_mask_Img.SetOrigin(Inf_Origin)
    Prostate_mask_Img.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(Prostate_mask_Img, Prostate_mask_path_save)


# -----------------------------------------------------------------------------

def save_transformed_img_2D_One_modality(args, logger, i_batch, dict_data):

    # MetaTensor (monai.transforms) ---> tensor ---> np
    img_transformed = dict_data['Img'].as_tensor().permute(1, 0, 2, 3).numpy().astype(np.float32)      # b, 1, y, x  ---> 1, b, y, x
    label_transformed = dict_data['GT'].as_tensor().permute(1, 0, 2, 3).numpy().astype(np.int16)    # b, 1, y, x ---> 1, b, y, x
    file_name = dict_data['file_name']

    logger.info("label_transformed unique: {}".format(str(np.unique(label_transformed))))
    logger.info("label_transformed size: {}".format(str(label_transformed.shape)))

    # -------------------
    pkl_path = args.path_dataset + '/' + 'Train_Val' + '/' + file_name[0] + '.pkl'
    pkl_list = pickle.load(open(pkl_path, 'rb'))

    Inf_Spacing = pkl_list[0]['Spacing']
    Inf_Origin = pkl_list[0]['Origin']
    Inf_Direction = pkl_list[0]['Direction']

    # -------------------
    # recover and save transformed img

    img_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch) + '_transformed_Img.nii.gz'
    GT_path_save = args.Visual_path + '/' + file_name[0] + '_' + str(i_batch)  + '_transformed_GT.nii.gz'

    logger.info("transformed Img_path_save: {}".format(img_path_save))
    logger.info("transformed GT_path_save: {}".format(GT_path_save))

    img_np = img_transformed[0]   # 1, b, y, x ---> b, y, x (depth, height, width)
    img_ITK = SimpleITK.GetImageFromArray(img_np)   # b, y, x (depth, height, width) ---> x, y, b
    img_ITK.SetSpacing(Inf_Spacing)
    img_ITK.SetOrigin(Inf_Origin)
    img_ITK.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(img_ITK, img_path_save)

    lesion_np = label_transformed[0]
    lesion_ITK = SimpleITK.GetImageFromArray(lesion_np)
    lesion_ITK.SetSpacing(Inf_Spacing)
    lesion_ITK.SetOrigin(Inf_Origin)
    lesion_ITK.SetDirection(Inf_Direction)
    SimpleITK.WriteImage(lesion_ITK, GT_path_save)


def save_transformed_img_3D_One_modality(args, logger, i_batch, dict_data):

    # MetaTensor (monai.transforms) ---> tensor ---> np
    img_transformed = dict_data['Img'].as_tensor().numpy().astype(np.float32)      # b, 1, z, x, y 
    label_transformed = dict_data['GT'].as_tensor().numpy().astype(np.int16)    # b, 1, z, x, y 
    file_name = dict_data['file_name']

    for j in range(img_transformed.shape[0]):
        
        # b, 1, z, x, y ---> 1, z, x, y
        logger.info("label_transformed unique: {}".format(str(np.unique(label_transformed[j]))))
        logger.info("label_transformed size: {}".format(str(label_transformed[j].shape)))

        # 1, z, x, y ---> z, x, y
        logger.info("label_transformed[0] unique: {}".format(str(np.unique(label_transformed[j][0]))))
        logger.info("label_transformed[0] size: {}".format(str(label_transformed[j][0].shape)))
        logger.info("label_transformed[0] astype: {}".format(str(label_transformed[j][0].astype)))

        # -------------------
        pkl_path = args.path_dataset + '/' + 'Train_Val' + '/' + file_name[j] + '.pkl'
        pkl_list = pickle.load(open(pkl_path, 'rb'))

        Inf_Spacing = pkl_list[0]['Spacing']
        Inf_Origin = pkl_list[0]['Origin']
        Inf_Direction = pkl_list[0]['Direction']

        # -------------------
        # recover and save transformed img

        img_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch) + '_transformed_img.nii.gz'
        GT_path_save = args.Visual_path + '/' + file_name[j] + '_' + str(i_batch)  + '_transformed_GT.nii.gz'

        logger.info("transformed img_path_save: {}".format(img_path_save))
        logger.info("transformed GT_path_save: {}".format(GT_path_save))

        # b, 1, z, x, y ---> 1, z, x, y ---> z, x, y
        img_np = img_transformed[j][0]
        img_ITK = SimpleITK.GetImageFromArray(img_np)
        img_ITK.SetSpacing(Inf_Spacing)
        img_ITK.SetOrigin(Inf_Origin)
        img_ITK.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(img_ITK, img_path_save)

        # b, 1, z, x, y ---> 1, z, x, y ---> z, x, y
        lable_np = label_transformed[j][0]
        lable_ITK = SimpleITK.GetImageFromArray(lable_np)
        lable_ITK.SetSpacing(Inf_Spacing)
        lable_ITK.SetOrigin(Inf_Origin)
        lable_ITK.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(lable_ITK, GT_path_save)


# -----------------------------------------------------------------------------

def save_transformed_img_RGB(args, logger, i_batch, dict_data):

    # MetaTensor (monai.transforms) ---> tensor ---> np
    img_transformed = dict_data['Img'].as_tensor().permute(0, 2, 3, 1).numpy().astype(np.float32)      # b, 3, y, x  ---> b, y, x, 3
    label_transformed = dict_data['GT'].as_tensor().permute(0, 2, 3, 1).squeeze(-1).numpy().astype(np.int16)    # b, 1, y, x ---> b, y, x, 1 ---> b, y, x
    file_name = dict_data['file_name']

    min, max = img_transformed.min(), img_transformed.max()
    logger.info("img_transformed min, max: {}".format(str([min, max])))
    logger.info("img_transformed size: {}".format(str(img_transformed.shape)))

    logger.info("label_transformed unique: {}".format(str(np.unique(label_transformed))))
    logger.info("label_transformed size: {}".format(str(label_transformed.shape)))

    # -------------------
    # recover and save transformed img

    batch_size = len(file_name)

    assert batch_size == img_transformed.shape[0], "the batch size should be the same"

    for i in range(batch_size):

        # pkl_path = args.path_dataset + '/' + 'Train_Val' + '/' + file_name[i] + '.pkl'
        # pkl_list = pickle.load(open(pkl_path, 'rb'))

        img_path_save = args.Visual_path + '/' + file_name[i] + '_transformed_Img.png'
        GT_path_save = args.Visual_path + '/' + file_name[i] + '_transformed_GT.png'

        arr_uint8 = (img_transformed[i] * 255).clip(0, 255).astype(np.uint8)   # the clip is very necessary, or the visualizations look weird.
        img_save = Image.fromarray(arr_uint8, mode='RGB')
        img_save.save(img_path_save)

        arr_uint8 = (label_transformed[i] * 255).astype(np.uint8)
        img_save = Image.fromarray(arr_uint8, mode='L')   # 'L' = grayscale
        img_save.save(GT_path_save)

        logger.info("transformed Img_path_save: {}".format(img_path_save))
        logger.info("transformed GT_path_save: {}".format(GT_path_save))
