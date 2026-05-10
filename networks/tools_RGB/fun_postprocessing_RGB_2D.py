from tools_RGB.utils_postprocessing_RGB_2D import *


# add fun_analysis_Val_Test_ISIC:


#---------------------------------------------------------------
# calculate metrics from softmax pre of Val and Test set
# works for both 2D and 3D models

def fun_analysis_Val_Test(args, logger, mode, path_GT_from, path_GT_to, path_softmax_from, path_softmax_to, save_metrics_path):

    # -------------------
    # move GT

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start moving GT from: ' + path_GT_from + ' to: ' + path_GT_to)

    # process GT and save GT as nii.gz
    move_npy_for_analysis(logger, args.path_split, path_GT_from, path_GT_to, mode=mode)

    logger.info('finish moving GT')

    # -------------------

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start moving softmax from: ' + path_softmax_from + ' to: ' + path_softmax_to)

    # move softmax 
    move_npy_for_analysis(logger, args.path_split, path_softmax_from, path_softmax_to, mode=mode)

    logger.info('finish moving softmax')

    # -------------------

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start extracting metrics, save to: ' + save_metrics_path)

    # from softmax, calculate metrics
    extract_Metrics_from_Softmax_RGB(logger, args.path_split, softmax_folder=path_softmax_to, GT_folder=path_GT_to, save_path=save_metrics_path, num_classes=args.num_classes, num_thresholds=0.5, mode=mode)

    logger.info('finish extracting metrics')
    



#---------------------------------------------------------------
# calculate metrics from softmax pre of Val and Test set
# works for both 2D and 3D models

def fun_analysis_Val_Test_ISIC(args, logger, mode, path_GT_from, path_GT_to, path_softmax_from, path_softmax_to, save_metrics_path):

    # -------------------
    # move GT

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start moving GT from: ' + path_GT_from + ' to: ' + path_GT_to)

    # process GT and save GT as nii.gz
    move_npy_for_analysis(logger, args.path_split, path_GT_from, path_GT_to, mode=mode)

    logger.info('finish moving GT')

    # -------------------

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start moving softmax from: ' + path_softmax_from + ' to: ' + path_softmax_to)

    # move softmax 
    move_npy_for_analysis(logger, args.path_split, path_softmax_from, path_softmax_to, mode=mode)

    logger.info('finish moving softmax')

    # -------------------

    logger.info("-----------------------------------------------------------------------------")

    logger.info('start extracting metrics, save to: ' + save_metrics_path)

    # from softmax, calculate metrics
    extract_Metrics_from_Softmax_2D_ISIC(logger, args.path_split, softmax_folder=path_softmax_to, GT_folder=path_GT_to, save_path=save_metrics_path, num_classes=args.num_classes, num_thresholds=0.5, mode=mode)

    logger.info('finish extracting metrics')