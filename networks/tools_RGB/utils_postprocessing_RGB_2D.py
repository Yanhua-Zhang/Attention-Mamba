import numpy as np
import SimpleITK 
import pickle
import shutil
import os


# from tools_RGB.utils_metrics_RGB_2D import calculate_from_folder
from tools_RGB.utils_metrics_RGB_2D_V2 import calculate_from_folder
from tools_RGB.utils_metrics_RGB_2D_ISIC import calculate_from_folder_ISIC


# add calculate_from_folder_ISIC


# -----------------------------------------------------------------------------
def move_npy_for_analysis(logger, path_split, path_from, path_to, mode='val'):

    if not os.path.exists(path_to):
        os.makedirs(path_to)  

    file_split = pickle.load(open(path_split, 'rb'))
    name_list = file_split[mode]   

    logger.info('mode: ' + mode)
    logger.info('length of name_list: ' + str(len(name_list)))
    logger.info('name_list: ' + str(name_list))

    for j in range(len(name_list)):
        
        path_file = path_from + '/' + name_list[j] + '.npy'

        shutil.copy(path_file, path_to)


# -----------------------------------------------------------------------------
def extract_Metrics_from_Softmax_RGB(logger, path_split, softmax_folder, GT_folder, save_path, num_classes, num_thresholds=0.5, mode='val'):

    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    file_split = pickle.load(open(path_split, 'rb'))
    name_list = file_split[mode]   

    logger.info('mode: ' + mode)
    logger.info('length of name_list: ' + str(len(name_list)))
    logger.info('name_list: ' + str(name_list))

    calculate_from_folder(logger, softmax_folder=softmax_folder, GT_folder=GT_folder, path_save_folder=save_path, num_thresholds=num_thresholds, name_list=name_list, num_classes=num_classes)


# -----------------------------------------------------------------------------
def extract_Metrics_from_Softmax_2D_ISIC(logger, path_split, softmax_folder, GT_folder, save_path, num_classes, num_thresholds=0.5, mode='val'):

    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    file_split = pickle.load(open(path_split, 'rb'))
    name_list = file_split[mode]   

    logger.info('mode: ' + mode)
    logger.info('length of name_list: ' + str(len(name_list)))
    logger.info('name_list: ' + str(name_list))

    calculate_from_folder_ISIC(logger, softmax_folder=softmax_folder, GT_folder=GT_folder, path_save_folder=save_path, num_thresholds=num_thresholds, name_list=name_list, num_classes=num_classes)


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