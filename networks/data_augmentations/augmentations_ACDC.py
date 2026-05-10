import monai.transforms as transforms

# ---------------------------------------------------------------
def get_augmentations(args, logger, mode):

    if mode=='train':

        if args.model_type=='2D':

            if args.augmentation_type == 'PCa_2D_V1':
                # this is used by PCa task, and nnUNet

                transform = [

                    transforms.SpatialPadd(keys=["Img", "GT_lesion", "GT_prostate"], spatial_size=args.img_size),

                    transforms.CenterSpatialCropd(keys=["Img", "GT_lesion", "GT_prostate"], roi_size=args.img_size),

                    # -------------------
                    transforms.RandRotated(keys=["Img", "GT_lesion", "GT_prostate"], mode=["bilinear", "nearest", "nearest"], prob=0.2, range_x=[-3.14159265, 3.14159265], range_y=0.0, range_z=0.0),

                    # -------------------
                    transforms.RandFlipd(keys=["Img", "GT_lesion", "GT_prostate"], prob=0.5, spatial_axis=1), # horizontally

                    transforms.RandFlipd(keys=["Img", "GT_lesion", "GT_prostate"], prob=0.5, spatial_axis=0), # vertically
                    
                    # -------------------
                    transforms.RandZoomd(keys=["Img", "GT_lesion", "GT_prostate"], mode=["area", "nearest", "nearest"], prob=0.2, min_zoom=0.7, max_zoom=1.4),

                    # -------------------
                    # transforms.RandCoarseDropoutd(keys=["Img", "GT_lesion", "GT_prostate"], holes=20, spatial_size=(14, 14), fill_value=0, prob=0.5),
                    
                    # transforms.RandGibbsNoised(keys=["Img"], prob=0.5, alpha=0.5),
                
                ]

                logger.info("Successfully loaded augmentations: gamma + SpatialPadd + CenterSpatialCropd + RandRotated + RandFlipd + RandZoomd for 2D models.")
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_2D_Median_V1':

                transform = [

                    transforms.SpatialPadd(keys=["Img", "GT"], spatial_size=args.img_size),

                    # transforms.CenterSpatialCropd(keys=["Img", "GT"], roi_size=args.img_size),
                    transforms.RandSpatialCropd(keys=["Img", "GT"], roi_size=args.img_size, random_size=False),

                    # -------------------
                    transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-3.14159265, 3.14159265], range_y=0.0, range_z=0.0),

                    # transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-0.5236, 0.5236], range_y=0.0, range_z=0.0),

                    # -------------------
                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=1), # horizontally

                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=0), # vertically
                    
                    # -------------------
                    transforms.RandZoomd(keys=["Img", "GT"], mode=["area", "nearest"], prob=0.2, min_zoom=0.7, max_zoom=1.4),

                    # -------------------
                    # transforms.RandCoarseDropoutd(keys=["Img", "GT"], holes=20, spatial_size=(14, 14), fill_value=0, prob=0.5),
                    
                    # transforms.RandGibbsNoised(keys=["Img"], prob=0.5, alpha=0.5),
                
                ]

                # logger.info("Successfully loaded augmentations: gamma + SpatialPadd + RandSpatialCropd + RandRotated (small x-y rotation: 180 ---> 30) + RandFlipd + RandZoomd for 2D models.")
                logger.info("Successfully loaded augmentations: gamma + SpatialPadd + RandSpatialCropd + Large RandRotated + RandFlipd + RandZoomd for 2D models.")
                # logger.info("Successfully loaded augmentations: gamma + SpatialPadd + CenterSpatialCropd + RandRotated + RandFlipd + RandZoomd for 2D models.")
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_2D_Median_V2':

                transform = [

                    # -------------------
                    transforms.RandGaussianNoised(keys=["Img"], prob=1, mean=0.0, std=0.02),

                    # -------------------
                    transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=1, range_x=[-3.14159265, 3.14159265], range_y=0.0, range_z=0.0),

                    # -------------------
                    transforms.RandZoomd(keys=["Img", "GT"], mode=["area", "nearest"], prob=1, min_zoom=0.7, max_zoom=1.3),

                    # -------------------
                    transforms.SpatialPadd(keys=["Img", "GT"], spatial_size=args.img_size),
                    transforms.RandSpatialCropd(keys=["Img", "GT"], roi_size=args.img_size, random_size=False),
                
                ]

                logger.info("Successfully loaded augmentations: RandGaussianNoised, RandRotated, RandZoomd, RandSpatialCropd for 2D models.")
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_2D_Median_V3':

                transform = [

                    transforms.SpatialPadd(keys=["Img", "GT"], spatial_size=args.img_size),
                    transforms.RandSpatialCropd(keys=["Img", "GT"], roi_size=args.img_size, random_size=False),

                    # -------------------
                    transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-3.14159265, 3.14159265], range_y=0.0, range_z=0.0),

                    # -------------------
                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=1), # horizontally

                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=0), # vertically
                
                ]

                logger.info("Successfully loaded augmentations: SpatialPadd + RandSpatialCropd + Large RandRotated + RandFlipd for 2D models.")
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            else:
                raise ValueError('This type of augmentation is not supported: ' + args.augmentation_type)

        elif args.model_type=='3D':

            if args.augmentation_type == 'PCa_3D_V1':
                # this is used by PCa task, and nnUNet

                transform = [

                    transforms.SpatialPadd(keys=["Img", "GT_lesion", "GT_prostate"], spatial_size=args.img_size),

                    transforms.CenterSpatialCropd(keys=["Img", "GT_lesion", "GT_prostate"], roi_size=args.img_size),

                    # -------------------
                    # transforms.RandRotated(keys=["Img", "GT_lesion", "GT_prostate"], mode=["bilinear", "nearest", "nearest"], prob=0.2, range_x=[-0.261799, 0.261799], range_y=0.0, range_z=0.0),  # only rotated in-plane x-y

                    # transforms.RandRotated(keys=["Img", "GT_lesion", "GT_prostate"], mode=["bilinear", "nearest", "nearest"], prob=0.2, range_x=[-0.261799, 0.261799], range_y=[-0.174533, 0.174533], range_z=[-0.174533, 0.174533]),  # rotated in-plane x-y, x-z, y-z at the same time

                    transforms.RandRotated(keys=["Img", "GT_lesion", "GT_prostate"], mode=["bilinear", "nearest", "nearest"], prob=0.2, range_x=[-3.14159265, 3.14159265], range_y=0.0, range_z=0.0),   # only rotated in-plane x-y (-180, 180)

                    transforms.RandRotated(keys=["Img", "GT_lesion", "GT_prostate"], mode=["bilinear", "nearest", "nearest"], prob=0.2, range_x=0.0, range_y=[-0.5236, 0.5236], range_z=0.0),   # only rotated in-plane x-z (-30, 30)     

                    transforms.RandRotated(keys=["Img", "GT_lesion", "GT_prostate"], mode=["bilinear", "nearest", "nearest"], prob=0.2, range_x=0.0, range_y=0.0, range_z=[-0.5236, 0.5236]),   # only rotated in-plane y-z (-30, 30)            
                    
                    # -------------------
                    transforms.RandFlipd(keys=["Img", "GT_lesion", "GT_prostate"], prob=0.5, spatial_axis=2), # horizontally

                    transforms.RandFlipd(keys=["Img", "GT_lesion", "GT_prostate"], prob=0.5, spatial_axis=1), # vertically

                    transforms.RandFlipd(keys=["Img", "GT_lesion", "GT_prostate"], prob=0.5, spatial_axis=0), # z-axis

                    # -------------------
                    transforms.RandZoomd(keys=["Img", "GT_lesion", "GT_prostate"], mode=["area", "nearest", "nearest"], prob=0.2, min_zoom=0.7, max_zoom=1.4),

                    # -------------------
                    # transforms.RandCoarseDropoutd(keys=["Img", "GT_lesion", "GT_prostate"], holes=20, spatial_size=(-1, 14, 14), fill_value=0, prob=0.5),
                    
                    # transforms.RandGibbsNoised(keys=["Img"], prob=0.5, alpha=0.5),

                ]

                logger.info("Successfully loaded augmentations: gamma + SpatialPadd + CenterSpatialCropd + RandRotated + RandFlipd + RandZoomd for 3D models.")
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            elif args.augmentation_type == 'ACDC_3D_Median_V1':

                transform = [

                    transforms.SpatialPadd(keys=["Img", "GT"], spatial_size=args.img_size),

                    # transforms.CenterSpatialCropd(keys=["Img", "GT"], roi_size=args.img_size),
                    transforms.RandSpatialCropd(keys=["Img", "GT"], roi_size=args.img_size, random_size=False),

                    # -------------------
                    # transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-0.261799, 0.261799], range_y=0.0, range_z=0.0),  # only rotated in-plane x-y

                    # transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-0.261799, 0.261799], range_y=[-0.174533, 0.174533], range_z=[-0.174533, 0.174533]),  # rotated in-plane x-y, x-z, y-z at the same time

                    # transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-3.14159265, 3.14159265], range_y=0.0, range_z=0.0),   # only rotated in-plane x-y (-180, 180)

                    transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=[-0.5236, 0.5236], range_y=0.0, range_z=0.0),   # only rotated in-plane x-y (-30, 30)

                    transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=0.0, range_y=[-0.5236, 0.5236], range_z=0.0),   # only rotated in-plane x-z (-30, 30)     

                    transforms.RandRotated(keys=["Img", "GT"], mode=["bilinear", "nearest"], prob=0.2, range_x=0.0, range_y=0.0, range_z=[-0.5236, 0.5236]),   # only rotated in-plane y-z (-30, 30)            
                    
                    # -------------------
                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=2), # horizontally

                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=1), # vertically

                    transforms.RandFlipd(keys=["Img", "GT"], prob=0.5, spatial_axis=0), # z-axis

                    # -------------------
                    transforms.RandZoomd(keys=["Img", "GT"], mode=["area", "nearest"], prob=0.2, min_zoom=0.7, max_zoom=1.4),

                    # -------------------
                    # transforms.RandCoarseDropoutd(keys=["Img", "GT"], holes=20, spatial_size=(-1, 14, 14), fill_value=0, prob=0.5),
                    
                    # transforms.RandGibbsNoised(keys=["Img"], prob=0.5, alpha=0.5),

                ]

                logger.info("Successfully loaded augmentations: gamma + SpatialPadd + RandSpatialCropd + RandRotated (small x-y rotation: 180 ---> 30) + RandFlipd + RandZoomd for 3D models.")
                logger.info("This type of augmentation is loaded: " + args.augmentation_type)

            else:
                raise ValueError('This type of augmentation is not supported: ' + args.augmentation_type)

        else:
            raise ValueError('This model_type is not supported: ' + args.model_type)
        
        return transforms.Compose(transform)

    else:
        raise ValueError('This mode is not supported: ' + mode)

