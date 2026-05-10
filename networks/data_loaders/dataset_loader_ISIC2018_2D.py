import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import pickle

import monai.transforms as transforms


# add self.if_use_gamma


# ---------------------------------------------------------------
class Dataset_ISIC2018_2D(Dataset):
    def __init__(self, logger, args, path_dataset, path_split, mode='train', transform=None):

        self.path_dataset = path_dataset
        self.path_split = path_split
        self.mode = mode
        self.transform = transform

        self.logger = logger

        # -------------------
        # some argumentions are added here:
        if self.mode == 'train':

            if args.augmentation_type == 'ISIC_Aug_V1':
                self.if_use_gamma = True
                self.RandAdjustContrast = transforms.RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5))  # gamma transform
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ISIC_Aug_V2':
                self.if_use_gamma = True
                self.per_channel = False
                self.std = 0.7
                self.mean = 0
                self.RandAdjustContrast = transforms.RandAdjustContrast(prob=1, gamma=(0.5, 1.6), retain_stats=True)  # gamma transform
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ISIC_Aug_V3':
                self.if_use_gamma = True
                self.RandAdjustContrast = transforms.RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5))  # gamma transform
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ISIC_Aug_V4':
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
        if self.mode == 'train':
            path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all['train']  # train list 
        elif self.mode == 'test_all':
            path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all['test_all']   # val list
        elif self.mode == 'test_ISIC2018':
            path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all['test_ISIC2018']   # test list
        elif self.mode == 'test_PH2':
            path_img = self.path_dataset + '/Train_Val'
            name_list = name_list_all['test_PH2']   # test list
        else:
            raise ValueError('This mode is not supported: ' + self.mode)
        
        # -------------------
        # save to list
        self.list_img = []
        self.list_GT = []
        self.list_file_name = []

        index = 0

        for name in name_list:

            # -------------------
            logger.info("Process this case: {}".format(name))

            # -------------------
            # file path
            path_npy = path_img + '/' + name + '.npy'
            if not os.path.exists(path_npy):
                raise ValueError('This file is lost:' + path_npy)
            
            # -------------------
            # load RGB + masks
            np_all = np.load(path_npy)  # 4, y, x 

            np_img = np_all[0:3]   # 3, y, x: 1-th, 2-th, 3-th channels are R, G, B
            np_GT = np_all[3]      # 1, y, x

            # -------------------
            if self.mode == 'train':

                if self.if_use_gamma:
                    # apply gamma transform, only for training set
                    np_img = self.random_gamma(np_img)
                    logger.info("Add random_gamma.")
                else:
                    logger.info("Do Not Add random_gamma.")

            # -------------------
            # np ---> tensor.float/long
            tensor_img = torch.from_numpy(np_img).to(torch.float32)   # 3, y, x

            if args.if_mask_to_long_type:
                tensor_GT = torch.from_numpy(np_GT).unsqueeze(0).long()   # y, x ---> 1, y, x

            else:
                # for the BCE_and_wIoU_loss of Polyp dataset
                # the values of mask is 0~1, not 0 or 1
                tensor_GT = torch.from_numpy(np_GT).unsqueeze(0)   # y, x ---> 1, y, x

            # -------------------
            if self.mode == 'train':
                if args.augmentation_type == 'ISIC_Aug_V2':
                    # apply brightness_additive, only for training set
                    tensor_img = self.brightness_additive(tensor_img)
                    logger.info("Add brightness_additive.")

            # -------------------
            if index % 30 == 0:
                logger.info("Shape of this case, after processing, img: {}".format(tensor_img.shape))
                logger.info('min and max of img: ' + str([torch.min(tensor_img), torch.max(tensor_img)]))
                logger.info("Shape of this case, after processing, GT: {}".format(tensor_GT.shape))
                logger.info('min and max of GT: ' + str([torch.min(tensor_GT), torch.max(tensor_GT)]))
                logger.info('Unique of GT: ' + str(torch.unique(tensor_GT)))
            
            index += 1

            # -------------------
            # save to list

            self.list_img.append(tensor_img)  # 3, y, x
            self.list_GT.append(tensor_GT)    # 1, y, x
            self.list_file_name.append(name)  # return file name

        logger.info("load done, length of dataset: {}".format(len(self.list_img)))

        
    def __len__(self):
        return len(self.list_img)


    def __getitem__(self, idx):
        
        tensor_image = self.list_img[idx]   # 3, y, x 
        tensor_GT = self.list_GT[idx]       # 1, y, x 
        file_name = self.list_file_name[idx]
       
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

        if self.per_channel:
            C = tensor_img.shape[0]
        else:
            C = 1

        if len(tensor_img.shape) == 3:
            rand_brightness = torch.normal(self.mean, self.std, size=(C, 1, 1)).to(tensor_img.device)

        else:
            raise ValueError('Invalid input tensor dimension, should be 3d for 2d image')
        
        tensor_img_brightness = tensor_img + rand_brightness

        assert tensor_img.shape == tensor_img_brightness.shape, "must keep the same shape, after brightness_additive"

        return tensor_img_brightness


    