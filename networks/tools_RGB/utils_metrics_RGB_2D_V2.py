import numpy as np
import pickle
import torch
# from nnunet.utilities.io import (checksum, path_exists, read_json, refresh_file_list, write_json)
import SimpleITK 
import os
from PIL import Image
from monai.transforms import AsDiscrete, Activations

from .utils_metrics_all import calculate_all_metrics_MultiClass
from utils.util_io import read_json, write_json


# use compute_hausdorff_distance to calculate HD95: so that we can set Spacing
# use compute_surface_dice to calculate NSD: so that we can set Spacing
# fix the bug, when calculating mIoU and mDSC: FP of Pre, No class 2 of GT, Why the IoU, DSC and HD95 are NaN?
# add Binarized processing to calculate_from_folder

# remove the dependence on nnunet.utilities.io


# --------------------------------------------------------------------------------------------------
# save Detection map

def fun_save_Detection_map_to_png(logger, path_save_folders, img_np_0_1, file_name):

    path = path_save_folders.replace(path_save_folders.split('/')[-1], '')

    path_save_folder = path + 'Seg_map_pre'

    if not os.path.exists(path_save_folder):
        os.makedirs(path_save_folder)  

    path_save = path_save_folder + '/' + file_name + '.png'

    arr_uint8 = (img_np_0_1 * 255).astype(np.uint8)
    img_save = Image.fromarray(arr_uint8, mode='L')   # 'L' = grayscale
    img_save.save(path_save)

    logger.info('successfully save this case: ' + str(file_name) + 'to: ' + str(path_save))


# --------------------------------------------------------------------------------------------------
# main function

def calculate_from_folder(logger, softmax_folder, GT_folder, path_save_folder, num_thresholds, name_list, num_classes):

    # -------------------
    # get the spacing from one of the patient, so that we can set spacing when calculating HausdorffDistanceMetric 

    voxelspacing = (1.0, 1.0)       # x, y, z ---> z, y, x
    logger.info('spacing for HausdorffDistanceMetric: ' + str(voxelspacing))

    # -------------------
    # decide the num_classes  

    path_softmax = softmax_folder  + '/' + name_list[0] + '.npy'
    softmax_np = np.load(path_softmax)    # C, y, x
    softmax_tensor = torch.from_numpy(softmax_np)    # C, y, x ---> tensor

    if softmax_tensor.size()[0] != num_classes:
        raise ValueError('Something wrong of the softmax output!')

    if softmax_tensor.size()[0] > 1:
        logger.info("-----------------------------------------------------------------------------")
        logger.info('softmax has multiply classes')

    elif softmax_tensor.size()[0] == 1:
        logger.info("-----------------------------------------------------------------------------")
        logger.info('softmax has been binarized')
        assert num_classes == 1, "the num_classes for binarized softmax must be 1"
        num_classes = num_classes + 1   # add background

    # -------------------
    # for one-hot 
    # Transformation for converting to one-hot format
    # MONAI's AsDiscrete can handle both logits and probabilities (seems only probabilities)
    # dim=0, keepdim=True, dtype=torch.float are the default settings of argmax
    # self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes, dim=0, keepdim=True, dtype=torch.float)
    fun_onehot_pred = AsDiscrete(argmax=False, to_onehot=num_classes)
    fun_onehot_label = AsDiscrete(to_onehot=num_classes)

    # -------------------
    # for the calculation of avg DSC, mIoU, HD95, NSD of all cases of all classes

    all_DSC_array = np.zeros((len(name_list), num_classes-1)) 
    all_IoU_array = np.zeros((len(name_list), num_classes-1))
    all_HD95_array = np.zeros((len(name_list), num_classes-1))
    all_NSD_array = np.zeros((len(name_list), num_classes-1))

    # -------------------
    # save to dict
    all_cases_dict = {}

    num_row = 0

    index = 0

    for name in name_list:

        logger.info("-----------------------------------------------------------------------------")
        logger.info('start to process this patient: ' + str(name))

        # -------------------
        # load softmax prediction
        path_softmax = softmax_folder  + '/' + name + '.npy'
        softmax_np = np.load(path_softmax)    # C, y, x
        softmax_tensor = torch.from_numpy(softmax_np).to(torch.float32).cuda()    # C, y, x ---> tensor ---> GPU

        logger.info('shape of softmax: ' + str(softmax_tensor.size()))
        logger.info('min and max of softmax: ' + str([torch.min(softmax_tensor), torch.max(softmax_tensor)]))

        # -------------------
        # load Ground truth
        path_GT = GT_folder  + '/' + name + '.npy'
        npy_np = np.load(path_GT)
        GT_np = npy_np[3]             # 3, y, x ---> y, x 
        GT_tensor = torch.from_numpy(GT_np).long().cuda()    # y, x ---> long ---> GPU

        logger.info('shape of GT: ' + str(GT_tensor.size()))
        logger.info('Unique of GT: ' + str(torch.unique(GT_tensor)))

        # -------------------
        # use monai, and also save the detection map

        # detection map
        if softmax_tensor.size()[0] > 1:
            # detection map of multiply classes
            label_pred = torch.argmax(softmax_tensor, 0, keepdim=False)       # C, y, x ---> y, x
            logger.info('shape of label_pred, after argmax: ' + str(label_pred.size()))
            logger.info('Unique of label_pred, after argmax: ' + str(torch.unique(label_pred)))

        elif softmax_tensor.size()[0] == 1:
            # detection map of binarized softmax
            label_pred = softmax_tensor.clone()              # 1, y, x 
            label_pred[softmax_tensor >= 0.5] = 1            # 1, y, x
            label_pred[softmax_tensor < 0.5] = 0             # 1, y, x
            label_pred = label_pred.squeeze(0)               # 1, y, x ---> y, x

            if index % 50 == 0:
                logger.info('forgroud num of label_pred, after Binarization: ' + str(torch.sum(label_pred)))
                logger.info('one value num of label_pred, after Binarization: ' + str((label_pred == 1).sum().item()))
                logger.info('zero value num of label_pred, after Binarization: ' + str((label_pred == 0).sum().item()))
                logger.info('all pixel num of label_pred, after Binarization: ' + str(label_pred.numel()))

            logger.info('shape of label_pred, after Binarization: ' + str(label_pred.size()))
            logger.info('Unique of label_pred, after Binarization: ' + str(torch.unique(label_pred)))

        # save the detection map
        label_pred_np = label_pred.cpu().numpy().astype(np.int32)
        fun_save_Detection_map_to_png(logger, path_save_folder, label_pred_np, name)

        # required by the monai
        label_pred = label_pred.unsqueeze(0).unsqueeze(0)  # y, x ---> 1, 1, y, x
        GT_tensor = GT_tensor.unsqueeze(0).unsqueeze(0)  # y, x ---> 1, 1, y, x

        # Convert to one-hot format
        onehot_pred = fun_onehot_pred(label_pred)   # 1, 1, y, x ---> C, 1, y, x
        logger.info('shape of onehot_pred: ' + str(onehot_pred.size()))
        logger.info('Unique of onehot_pred: ' + str(torch.unique(onehot_pred)))

        if index % 50 == 0:
            onehot_pred_class1 = onehot_pred[1]
            logger.info('forgroud num of onehot_pred_class1, after Binarization: ' + str(torch.sum(onehot_pred_class1)))
            logger.info('one value num of onehot_pred_class1, after Binarization: ' + str((onehot_pred_class1 == 1).sum().item()))
            logger.info('zero value num of onehot_pred_class1, after Binarization: ' + str((onehot_pred_class1 == 0).sum().item()))
            logger.info('all pixel num of onehot_pred_class1, after Binarization: ' + str(onehot_pred_class1.numel()))
        index += 1

        onehot_GT = fun_onehot_label(GT_tensor)    # 1, 1, y, x ---> C, 1, y, x
        logger.info('shape of onehot_GT: ' + str(onehot_GT.size()))
        logger.info('Unique of onehot_GT: ' + str(torch.unique(onehot_GT)))

        results = calculate_all_metrics_MultiClass(onehot_pred, onehot_GT, if_include_background = False, voxelspacing = voxelspacing)  # exclude the background
        logger.info('Results: ' + str(results))

        # -------------------
        # record the DSC, mIoU, HD95, NSD of all cases of all classes

        # print(results['DSC'].shape)
        # print(results['NSD'].shape)
        # print(all_DSC_array.shape)
        # print(all_DSC_array[num_row,:].shape)

        all_DSC_array[num_row,0:] = results['DSC'] 
        all_IoU_array[num_row,0:] = results['IoU'] 
        all_HD95_array[num_row,0:] = results['HD95']
        all_NSD_array[num_row,0:] = results['NSD']

        # logger.info('the DSC of all cases of all classes: ' + str(all_DSC_array))
        # logger.info('the mIoU of all cases of all classes: ' + str(all_IoU_array))
        # logger.info('the HD95 of all cases of all classes: ' + str(all_HD95_array))
        # logger.info('the NSD of all cases of all classes: ' + str(all_NSD_array))

        num_row += 1    

        # -------------------
        case_dict={}

        for i in range(num_classes-1):
            case_dict['DSC_class_'+str(i+1)] = np.round(np.float64(results['DSC'][i]), 6)
        for i in range(num_classes-1):
            case_dict['IoU_class_'+str(i+1)] = np.round(np.float64(results['IoU'][i]), 6)
        for i in range(num_classes-1):
            case_dict['HD95_class_'+str(i+1)] = np.round(np.float64(results['HD95'][i]), 6)
        for i in range(num_classes-1):
            case_dict['NSD_class_'+str(i+1)] = np.round(np.float64(results['NSD'][i]), 6)

        case_dict['mean_DSC'] = np.round(np.float64(results['mean_DSC']), 6)
        case_dict['mean_IoU'] = np.round(np.float64(results['mean_IoU']), 6)
        case_dict['mean_HD95'] = np.round(np.float64(results['mean_HD95']), 6)
        case_dict['mean_NSD'] = np.round(np.float64(results['mean_NSD']), 6)

        all_cases_dict[name] = case_dict
        # -------------------

    # -------------------
    # calculation of avg DSC, mIoU, HD95, NSD of all cases of all classes

    logger.info('the DSC of all cases of all classes: ' + str(all_DSC_array))
    logger.info('the mIoU of all cases of all classes: ' + str(all_IoU_array))
    logger.info('the HD95 of all cases of all classes: ' + str(all_HD95_array))
    logger.info('the NSD of all cases of all classes: ' + str(all_NSD_array))
    
    avg_each_class_all_DSC_array = np.nanmean(all_DSC_array, axis=0)
    avg_each_class_all_IoU_array = np.nanmean(all_IoU_array, axis=0)
    avg_each_class_all_HD95_array = np.nanmean(all_HD95_array, axis=0)
    avg_each_class_all_NSD_array = np.nanmean(all_NSD_array, axis=0)

    logger.info('the avg DSC of each cases of all classes: ' + str(avg_each_class_all_DSC_array))
    logger.info('the avg mIoU of each cases of all classes: ' + str(avg_each_class_all_IoU_array))
    logger.info('the avg HD95 of each cases of all classes: ' + str(avg_each_class_all_HD95_array))
    logger.info('the avg NSD of each cases of all classes: ' + str(avg_each_class_all_NSD_array))

    avg_all_DSC_array = np.mean(avg_each_class_all_DSC_array)
    avg_all_IoU_array = np.mean(avg_each_class_all_IoU_array)
    avg_all_HD95_array = np.mean(avg_each_class_all_HD95_array)
    avg_all_NSD_array = np.mean(avg_each_class_all_NSD_array)

    logger.info('the avg DSC of all cases of all classes: ' + str(avg_all_DSC_array))
    logger.info('the avg mIoU of all cases of all classes: ' + str(avg_all_IoU_array))
    logger.info('the avg HD95 of all cases of all classes: ' + str(avg_all_HD95_array))
    logger.info('the avg NSD of all cases of all classes: ' + str(avg_all_NSD_array))

    all_case_dict={}

    for i in range(num_classes-1):
        all_case_dict['DSC_class_'+str(i+1)] = np.round(np.float64(avg_each_class_all_DSC_array[i]), 6)
    for i in range(num_classes-1):
        all_case_dict['IoU_class_'+str(i+1)] = np.round(np.float64(avg_each_class_all_IoU_array[i]), 6)
    for i in range(num_classes-1):
        all_case_dict['HD95_class_'+str(i+1)] = np.round(np.float64(avg_each_class_all_HD95_array[i]), 6)
    for i in range(num_classes-1):
        all_case_dict['NSD_class_'+str(i+1)] = np.round(np.float64(avg_each_class_all_NSD_array[i]), 6)

    all_case_dict['mean_DSC'] = np.round(np.float64(avg_all_DSC_array), 6)
    all_case_dict['mean_IoU'] = np.round(np.float64(avg_all_IoU_array), 6)
    all_case_dict['mean_HD95'] = np.round(np.float64(avg_all_HD95_array), 6)
    all_case_dict['mean_NSD'] = np.round(np.float64(avg_all_NSD_array), 6)

    all_cases_dict['all'] = all_case_dict

    # -------------------
    save_suffix = 'each_cases'
    write_json(path_save_folder + '/metrics_' + save_suffix + '.json', all_cases_dict)

        

