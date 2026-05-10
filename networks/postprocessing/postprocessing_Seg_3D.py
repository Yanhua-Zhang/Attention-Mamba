import torch
import os
import time

from tools_Seg_3D.fun_postprocessing_Seg_3D_V1 import fun_analysis_Val_Test
from tools_Seg_3D.fun_avg_2_Execel_Certain_folds_for_Single_file import run_analysis_avg_To_Execel_val_test_for_Single_file
from tools_Seg_3D.fun_avg_2_Execel_Certain_folds_for_Single_file_TwoClasses import run_analysis_avg_To_Execel_val_test_for_Single_file_TwoClasses

from tools_Seg_3D.fun_avg_2_Execel_Certain_folds_for_Single_file_MultiClasses import run_analysis_avg_To_Execel_val_test_for_Single_file_MulitClasses

# --------------------------------------------------------------------------
# for three classes:

def fun_postprocessing_Seg_3D(logger, args):

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
    if args.if_calc_metrics_val:
        logger.info("start calc_metrics_val:")
        start_ts = time.time()

        fun_analysis_Val_Test(args, logger, mode='val')

        end_ts = time.time()
        t_cnt = end_ts - start_ts
        logger.info(f"Time use: {t_cnt/100*1000} ms")
        logger.info("end calc_metrics_val.")

    if args.if_calc_metrics_test:
        logger.info("start calc_metrics_test:")
        start_ts = time.time()

        fun_analysis_Val_Test(args, logger, mode='test')

        end_ts = time.time()
        t_cnt = end_ts - start_ts
        logger.info(f"Time use: {t_cnt/100*1000} ms")
        logger.info("end calc_metrics_test.")

    # --------------------------------------------------------------------------
    # calculate the avg of 5 folds, and save to Execel
    
    # has three forground classes
    if (args.num_classes - 1) == 3:

        if args.if_avg_5_folds_val:

            run_analysis_avg_To_Execel_val_test_for_Single_file(args, mode='val', file_name='metrics_each_cases.json', suffix='Seg')


        if args.if_avg_5_folds_test:

            run_analysis_avg_To_Execel_val_test_for_Single_file(args, mode='test', file_name='metrics_each_cases.json', suffix='Seg')
    
    # has two forground classes
    elif (args.num_classes - 1) == 2:

        if args.if_avg_5_folds_val:

            run_analysis_avg_To_Execel_val_test_for_Single_file_TwoClasses(args, mode='val', file_name='metrics_each_cases.json', suffix='Seg')


        if args.if_avg_5_folds_test:

            run_analysis_avg_To_Execel_val_test_for_Single_file_TwoClasses(args, mode='test', file_name='metrics_each_cases.json', suffix='Seg')

    # has 8 forground classes
    elif (args.num_classes - 1) == 8:

        if args.if_avg_5_folds_val:

            run_analysis_avg_To_Execel_val_test_for_Single_file_MulitClasses(args, classes = (args.num_classes-1), mode='val', file_name='metrics_each_cases.json', suffix='Seg')


        if args.if_avg_5_folds_test:

            run_analysis_avg_To_Execel_val_test_for_Single_file_MulitClasses(args, classes = (args.num_classes-1), mode='test', file_name='metrics_each_cases.json', suffix='Seg')



