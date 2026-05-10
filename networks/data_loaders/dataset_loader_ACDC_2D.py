import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import pickle

import monai.transforms as transforms


# add ACDC_2D_Median_V3
# add self.if_use_gamma

# ---------------------------------------------------------------
class Dataset_ACDC_2D(Dataset):
    def __init__(self, logger, args, path_dataset, path_split, mode='Train', fold=0, transform=None):

        self.path_dataset = path_dataset
        self.path_split = path_split
        self.mode = mode
        self.fold = fold
        self.transform = transform

        self.logger = logger

        # -------------------
        # some argumentions are added here:
        if self.mode == 'Train':
            if args.augmentation_type == 'PCa_2D_V1':
                self.if_use_gamma = True
                self.RandAdjustContrast = transforms.RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5))  # gamma transform
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_2D_Median_V1':
                self.if_use_gamma = True
                self.RandAdjustContrast = transforms.RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5))  # gamma transform
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_2D_Median_V2':
                self.if_use_gamma = True
                self.per_channel = False
                self.std = 0.7
                self.mean = 0
                self.RandAdjustContrast = transforms.RandAdjustContrast(prob=1, gamma=(0.5, 1.6), retain_stats=True)  # gamma transform
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_2D_Median_V3':
                self.if_use_gamma = False
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            else:
                raise ValueError('This type of augmentation is not supported: ' + args.augmentation_type)

        # -------------------
        # load split file
        if not os.path.exists(path_split):
            raise ValueError('Split file is lost.')
        else:
            name_list_all = pickle.load(open(path_split, 'rb'))

        # -------------------
        # return img folder path, lists of train, val, test
        if self.mode == 'Train':
            path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all[self.fold]['train']  # train list of corresponding fold
        elif self.mode == 'Val':
            path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all[self.fold]['val']   # val list of corresponding fold
        elif self.mode == 'Test':
            path_img = self.path_dataset + '/Test'
            # path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all[5]['test']   # test list
        else:
            raise ValueError('This mode is not supported: ' + self.mode)
        
        # -------------------
        # save to list
        self.list_img = []
        self.list_GT = []
        self.list_file_name = []

        for name in name_list:

            # -------------------
            logger.info("Process this case: {}".format(name))

            # -------------------
            # file path
            path_npy = path_img + '/' + name + '.npy'
            if not os.path.exists(path_npy):
                raise ValueError('This file is lost:' + path_npy)
            
            # -------------------
            # load t2w, adc, dwi, lesion mask, prostate mask
            np_all = np.load(path_npy)  # 2, z, y, x 

            np_img = np_all[0]   # z, y, x
            np_GT = np_all[1]  # z, y, x

            # -------------------
            if self.mode == 'Train':
                if self.if_use_gamma:
                    # apply gamma transform, only for training set
                    np_img = self.random_gamma(np_img)
                    logger.info("Add random_gamma.")
                else:
                    logger.info("Do Not Add random_gamma.")

            # -------------------
            # np ---> tensor.float/long
            tensor_img = torch.from_numpy(np_img).to(torch.float32)  # z, y, x
            tensor_GT = torch.from_numpy(np_GT).long()   # z, y, x

            # -------------------
            if self.mode == 'Train':
                if args.augmentation_type == 'ACDC_2D_Median_V2':
                    # apply brightness_additive, only for training set
                    tensor_img = self.brightness_additive(tensor_img)
                    logger.info("Add brightness_additive.")

            # -------------------
            logger.info("Shape of this case, after processing, img: {}".format(tensor_img.shape))
            logger.info('min and max of img: ' + str([torch.min(tensor_img), torch.max(tensor_img)]))
            logger.info("Shape of this case, after processing, GT: {}".format(tensor_GT.shape))
            logger.info('Unique of GT: ' + str(torch.unique(tensor_GT)))

            # -------------------
            # save to list
            if self.mode == 'Train':

                # -------------------
                # for training, 3D ---> 2D slices
                z, y, x = tensor_img.shape  # z, y, x

                logger.info("Shape of this case, before extending: {}".format(tensor_img.shape))
                
                # 3D ---> 2D
                for j in range(z):
                    self.list_img.append(tensor_img[j])  # z, y, x ---> y, x
                    self.list_GT.append(tensor_GT[j])   # z, y, x ---> y, x

                    self.list_file_name.append(name) # return file name

            elif self.mode == 'Val' or 'Test':
                # still 3D
                self.list_img.append(tensor_img)          # z, y, x 
                self.list_GT.append(tensor_GT)            # z, y, x

                self.list_file_name.append(name) # return file name

            else:
                raise ValueError('This mode is not supported: ' + self.mode)

        # return a subset of the dataset
        # self.list_img = self.list_img[:50]          
        # self.list_GT = self.list_GT[:50]     
        # self. = self.[:50]    
        # self.list_file_name = self.list_file_name[:50]  # return file name

        logger.info("load done, length of dataset: {}".format(len(self.list_img)))

        
    def __len__(self):
        return len(self.list_img)


    def __getitem__(self, idx):
        
        tensor_image = self.list_img[idx]   # y, x or z, y, x 
        tensor_GT = self.list_GT[idx]       # y, x or z, y, x 

        file_name = self.list_file_name[idx]
       
        if self.mode == 'Train':
            # for using the Transform fun of monai, need to: y, x ---> 1, y, x  
            tensor_image = tensor_image.unsqueeze(0)   # y, x ---> 1, y, x
            tensor_GT = tensor_GT.unsqueeze(0)      # y, x ---> 1, y, x

        elif self.mode == 'Val' or 'Test':

            tensor_image = tensor_image               # z, y, x 
            tensor_GT = tensor_GT                     # z, y, x

        else:
            raise ValueError('This mode is not supported: ' + self.mode)
        
        data_dict = {'Img': tensor_image, 'GT': tensor_GT, 'file_name': file_name}

        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict
        

    def random_gamma(self, img):

        img_gamma = self.RandAdjustContrast(img)

        img_gamma = img_gamma.numpy()  # MetaTensor ---> np.ndarray

        assert img_gamma.shape == img.shape, "must keep the same shape, after gamma"

        return img_gamma
    

    def brightness_additive(self, tensor_img):

        tensor_img = tensor_img.unsqueeze(0)  # z, y, x ---> 1, z, y, x 

        if self.per_channel:
            C = tensor_img.shape[0]
        else:
            C = 1

        if len(tensor_img.shape) == 4:
            rand_brightness = torch.normal(self.mean, self.std, size=(C, 1, 1, 1)).to(tensor_img.device)

        elif len(tensor_img.shape) == 3:
            rand_brightness = torch.normal(self.mean, self.std, size=(C, 1, 1)).to(tensor_img.device)

        else:
            raise ValueError('Invalid input tensor dimension, should be 4d for volume image or 3d for 2d image')
        
        tensor_img = tensor_img + rand_brightness

        tensor_img = tensor_img.squeeze(0)  # 1, z, y, x ---> z, y, x

        assert tensor_img.shape == tensor_img.shape, "must keep the same shape, after brightness_additive"

        return tensor_img


    