import numpy as np
import SimpleITK 
import pickle
import shutil
import os

# from tools_Seg_3D.utils_metrics_Seg_3D import calculate_from_folder
# from tools_Seg_3D.utils_metrics_Seg_3D_V1 import calculate_from_folder
from tools_Seg_3D.utils_metrics_Seg_3D_V2 import calculate_from_folder


# -----------------------------------------------------------------------------
# from npy, and pkl to lesion GT (nii.gz)

def from_npy_pkl_2_nii(logger, path_split, path_from, path_to, fold_name, mode='val'):

    if not os.path.exists(path_to):
        os.makedirs(path_to)  

    file_split = pickle.load(open(path_split, 'rb'))
    name_list = file_split[fold_name][mode]  

    logger.info('fold_name: ' + str(fold_name))
    logger.info('mode: ' + mode)
    logger.info('length of name_list: ' + str(len(name_list)))
    logger.info('name_list: ' + str(name_list))

    for name in name_list:
        
        # -------------------
        path_npy = path_from + '/' + name + '.npy'
        if not os.path.exists(path_npy):
            raise ValueError('this file is lost: ' + str(path_npy))

        path_pkl = path_from + '/' + name + '.pkl'
        if not os.path.exists(path_pkl):
            raise ValueError('this file is lost: ' + str(path_pkl))

        shutil.copy(path_pkl, path_to) # move pkl to aim folder

        # -------------------
        save_path_ITK = path_to + '/' + name + '.nii.gz'

        npy_np = np.load(path_npy)   # 2, z, y, x
        pkl_list = pickle.load(open(path_pkl, 'rb'))

        Inf_Spacing = pkl_list[0]['Spacing']
        Inf_Origin = pkl_list[0]['Origin']
        Inf_Direction = pkl_list[0]['Direction']

        GT_np = npy_np[1].astype(np.int32)  # 2, z, y, x ---> z, y, x ---> int
        gt_ITK = SimpleITK.GetImageFromArray(GT_np)   # z, y, x ---> x, y, z
        gt_ITK.SetSpacing(Inf_Spacing)
        gt_ITK.SetOrigin(Inf_Origin)
        gt_ITK.SetDirection(Inf_Direction)
        SimpleITK.WriteImage(gt_ITK, save_path_ITK)

        # -------------------
        # save GT as npz
        npz_path = path_to + '/' + name + '.npz'
        np.savez(npz_path, label = GT_np)

# -----------------------------------------------------------------------------

def move_softmax_for_analysis_V2(logger, path_split, path_from, path_to, fold_name, mode='val'):

    if not os.path.exists(path_to):
        os.makedirs(path_to)  

    file_split = pickle.load(open(path_split, 'rb'))
    name_list = file_split[fold_name][mode]   

    logger.info('fold_name: ' + str(fold_name))
    logger.info('mode: ' + mode)
    logger.info('length of name_list: ' + str(len(name_list)))
    logger.info('name_list: ' + str(name_list))

    for j in range(len(name_list)):
        
        path_file = path_from + '/' + name_list[j] + '.npz'

        shutil.copy(path_file, path_to)


# -----------------------------------------------------------------------------
def extract_Metrics_from_Softmax(logger, path_split, fold_name, softmax_folder, GT_folder, save_path, num_classes, num_thresholds=0.5, mode='val'):

    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    file_split = pickle.load(open(path_split, 'rb'))
    name_list = file_split[fold_name][mode]   

    logger.info('fold_name: ' + str(fold_name))
    logger.info('mode: ' + mode)
    logger.info('length of name_list: ' + str(len(name_list)))
    logger.info('name_list: ' + str(name_list))

    calculate_from_folder(logger, softmax_folder=softmax_folder, GT_folder=GT_folder, path_save_folder=save_path, num_thresholds=num_thresholds, name_list=name_list, num_classes=num_classes)


# -----------------------------------------------------------------------------
import statistics

def fun_avg(input_list):

    if len(input_list) != 0:
        output = statistics.mean(input_list)
    else:
        output = 0

    return output

def fun_std(input_list):
    
    if len(input_list) != 0:
        output = statistics.pstdev(input_list)
    else:
        output = 0

    return output