

def fun_choose_dataset(dataset_name):

    dataset_config = {
    'Datase1_Synapse_8classes': {
        # do spacing for x, y, Not z
        # median spacing of x, y: 0.7617189288139343, 0.7617189288139343
        # no Crop
        # 13 classes ---> 8 classes
        # use the foreground (8 classes), use the whole training set to decide the upbnd, lowbnd, mn, std, stacked:
        # mn_masked: 71.19013633601502
        # std_masked: 127.95561355400633
        # upper_bnd_masked: 277.0
        # lower_bnd_masked: -956.0
        'path_dataset': '../datasets_preprocessed/Datase1_Synapse_8classes/npy_npz_pkl',    
        'path_split': '../datasets_split/Datase1_Synapse_8classes/splits_Synapse.pkl', 
        'num_classes': 9,   # always including the background
        'input_channel_dim': 1,
    },
    'Dataset2_ACDC': {
        # only do spacing for x-y plain, no z
        # the Spacing decided only by the Median of Train (Train + val, 100 patients) ---> Spacing = [0, 1.5625, 1.5625] 
        # normalization: no clipping, Z-score normalization
        'path_dataset': '../datasets_preprocessed/Dataset2_ACDC/npy_npz_pkl',    
        'path_split': '../datasets_split/Dataset2_ACDC/splits_ACDC.pkl',
        'num_classes': 4,   # always including the background
        'input_channel_dim': 1,
    },
    'Dataset3_ISIC2018_PH2': {
        # resize to the same size: 256 × 256. 
        # use PIL Image to read img and gt
        # record the numb of pixels of background, and different classes (after preprocessed). 
        # calculate mean and std w/o masks
        # mean and std: average over all patients
        # mean and std: average over 'Stacked' patients

        # in this way to Resize methods: PIL.Img ---> resize [256 × 256] ---> to np ---> normalize by using: [mn, st] ---> normalize to [0, 1] ---> *255 to [0, 255] ---> save
        # average over Stacked patients, use Only the Training set: mn = 155.25308, st = 46.459187
        # Test set is Also resized: the GT is also resized by BILINEAR, while the GT of Training is also resized by BILINEAR
        'path_dataset': '../datasets_preprocessed/Dataset3_ISIC2018_PH2/npy_npz_pkl',    
        'path_split': '../datasets_split/Dataset3_ISIC2018_PH2/splits_ISIC2018.pkl', 
        'num_classes': 2,   # always including the background
        'input_channel_dim': 3,
    },
    }

    path_dataset = dataset_config[dataset_name]['path_dataset']
    path_split = dataset_config[dataset_name]['path_split']
    num_classes = dataset_config[dataset_name]['num_classes']
    input_channel_dim = dataset_config[dataset_name]['input_channel_dim']

    return path_dataset, path_split, num_classes, input_channel_dim