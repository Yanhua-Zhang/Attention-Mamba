from tools_Seg_3D.utils_postprocessing_Seg_3D import *


#---------------------------------------------------------------
# calculate metrics from softmax pre of Val and Test set
# works for both 2D and 3D models

def fun_analysis_Val_Test(args, logger, mode):

    if mode == 'val':

        fold_name = args.fold_name
        mode = 'val'

        path_GT_from = args.path_dataset + '/Train_Val'
        path_GT_to = args.analysis_val_path + '/GT'

        path_softmax_from = args.Softmax_val_path
        path_softmax_to = args.analysis_val_path + '/softmax_pre'

        save_metrics_path = args.analysis_val_path + '/metrics_from_softmax'

    elif mode == 'test':

        fold_name = 5
        mode = 'test'
        path_GT_from = args.path_dataset + '/Test'
        path_GT_to = args.analysis_test_path + '/GT'

        path_softmax_from = args.Softmax_test_path
        path_softmax_to = args.analysis_test_path + '/softmax_pre'

        save_metrics_path = args.analysis_test_path + '/metrics_from_softmax'

    else:
        raise ValueError('this mode is not supported: ' + str(mode))

    # -------------------
    # move GT

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start moving GT from: ' + path_GT_from + ' to: ' + path_GT_to)

    # process GT and save GT as nii.gz
    from_npy_pkl_2_nii(logger, args.path_split, path_GT_from, path_GT_to, fold_name=fold_name, mode=mode)

    logger.info('finish moving GT')

    # -------------------

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start moving softmax from: ' + path_softmax_from + ' to: ' + path_softmax_to)

    # move softmax 
    move_softmax_for_analysis_V2(logger, args.path_split, path_softmax_from, path_softmax_to, fold_name=fold_name, mode=mode)

    logger.info('finish moving softmax')

    # -------------------

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start extracting metrics, save to: ' + save_metrics_path)

    # from softmax, calculate metrics
    extract_Metrics_from_Softmax(logger, args.path_split, fold_name=fold_name, softmax_folder=path_softmax_to, GT_folder=path_GT_to, save_path=save_metrics_path, num_classes=args.num_classes, num_thresholds=0.5, mode=mode)

    logger.info('finish extracting metrics')
    



