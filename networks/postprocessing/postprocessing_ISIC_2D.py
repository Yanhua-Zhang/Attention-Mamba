import torch
import os
import time

from tools_RGB.fun_postprocessing_RGB_2D import fun_analysis_Val_Test_ISIC
from tools_RGB.fun_avg_2_Execel_Certain_folds_for_Single_file_OneClasses_V3 import run_analysis_avg_To_Execel_val_test_for_Single_file_OneClasses
from tools_RGB.fun_Execel_2_Execel_ISIC import run_Execel_2_Execel


# --------------------------------------------------------------------------
# for three classes:

def fun_postprocessing_ISIC_2D(logger, args):

    # --------------------------------------------------------------------------
    # set CPU environment
    max_cpu_threads = os.cpu_count()
    logger.info('num of threads on the system: ' + str(max_cpu_threads))

    num_threads = torch.get_num_threads()
    logger.info('num of threads, before setting: ' + str(num_threads))
    cpu_num = max_cpu_threads - 10
    torch.set_num_threads(cpu_num)
    num_threads = torch.get_num_threads()
    logger.info('num of threads, after setting: ' + str(num_threads))

    # --------------------------------------------------------------------------
    
    if args.if_calc_metrics_test:

        all_path_softmax_from = [args.Softmax_test_ISIC2018_path, args.Softmax_test_PH2_path, args.Softmax_test_all_path]

        all_analysis_test_path = [args.analysis_test_ISIC2018_path, args.analysis_test_PH2_path, args.analysis_test_all_path]

        index = 0

        for mode in ['test_ISIC2018', 'test_PH2', 'test_all']:

            logger.info("start calc_metrics_test, this set: " + mode)
            start_ts = time.time()

            fun_analysis_Val_Test_ISIC(args, logger, mode=mode, path_GT_from=args.path_dataset + '/Train_Val', path_GT_to=all_analysis_test_path[index] + '/GT', path_softmax_from=all_path_softmax_from[index], path_softmax_to=all_analysis_test_path[index] + '/softmax_pre', save_metrics_path=all_analysis_test_path[index] + '/metrics_from_softmax')

            end_ts = time.time()
            t_cnt = end_ts - start_ts
            logger.info(f"Time use: {t_cnt/100*1000} ms")
            logger.info("end calc_metrics_test, this set: " + mode)

            index += 1

    # --------------------------------------------------------------------------
    # calculate the avg of 5 folds, and save to Execel

    if args.if_avg_5_folds_test:

        all_path_softmax_from = [args.Softmax_test_ISIC2018_path, args.Softmax_test_PH2_path, args.Softmax_test_all_path]

        all_analysis_test_path = [args.analysis_test_ISIC2018_path, args.analysis_test_PH2_path, args.analysis_test_all_path]
        
        index = 0

        for mode in ['test_ISIC2018', 'test_PH2', 'test_all']:

            path_folders = all_analysis_test_path[index].replace('/fold_' + str(args.fold_name), '')

            run_analysis_avg_To_Execel_val_test_for_Single_file_OneClasses(args, path_folders=path_folders, mode=mode, file_name='metrics_each_cases.json', suffix='Seg')

            index += 1

        all_analysis_test_paths = [args.analysis_test_ISIC2018_path, args.analysis_test_PH2_path, args.analysis_test_all_path]
        suffix_names = ['test_ISIC2018', 'test_PH2', 'test_all']

        # save the results of all test sets to one execel
        run_Execel_2_Execel(args, all_analysis_test_paths, suffix_names)







