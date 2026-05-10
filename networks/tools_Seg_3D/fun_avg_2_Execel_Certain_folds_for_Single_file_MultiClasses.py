import numpy as np
import os
import pandas
# from nnunet.utilities.io import (checksum, path_exists, read_json, refresh_file_list, write_json)

from tools_Seg_3D.utils_postprocessing_Seg_3D import *
from utils.util_io import read_json, write_json


# remove the dependence on nnunet.utilities.io


#---------------------------------------------------------------
# calculate avg metrics across the 5 folds on the val, test

def run_analysis_avg_To_Execel_val_test_for_Single_file_MulitClasses(args, classes, mode='val', file_name=None, suffix=None):

    # --------------------------------------------------------------------------
    # only save certain folds

    sub_folder_names = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']

    sub_folder_names_selected = []

    for id in args.fold_ids_select:
        if id != 5:
            sub_folder_names_selected.append(sub_folder_names[id])

    # lenth for calc avg
    if 5 in args.fold_ids_select:
        lenth = len(args.fold_ids_select)-1  # ensemble results not involved in calc avg
        fold_ids_for_avg = args.fold_ids_select[:-1]

    else:
        lenth = len(args.fold_ids_select)
        fold_ids_for_avg = args.fold_ids_select.copy()

    # --------------------------------------------------------------------------
    # path name

    if mode == 'val':

        path_folders = "../results/{}/{}/{}/analysis_val".format(args.save_folder, args.inference_type, args.load_model_type)

        execel_path_save = path_folders + '/Execel_' + mode + '_' + suffix + '.xlsx' 

    elif mode == 'test':

        path_folders = "../results/{}/{}/{}/analysis_test".format(args.save_folder, args.inference_type, args.load_model_type)

        execel_path_save = path_folders + '/Execel_' + mode + '_' + suffix + '.xlsx'

    else:
        raise ValueError("something wrong.")

    if not os.path.exists(path_folders):
        raise ValueError("path_folders not exists.")

    # --------------------------------------------------------------------------
    name_list = []

    mean_DSC_list = []
    mean_DSC_list_2 = []

    mean_IoU_list = []
    mean_IoU_list_2 = []

    mean_HD95_list = []
    mean_HD95_list_2 = []

    mean_NSD_list = []
    mean_NSD_list_2 = []

    for folder_name in sub_folder_names:

        if folder_name in sub_folder_names_selected:

            # --------------------------------------------------------------------------
            metrics_analysis_path = path_folders + '/' + folder_name + '/metrics_from_softmax/' + file_name

            if not os.path.exists(metrics_analysis_path):

                raise ValueError('This path does not exist: ' + str(metrics_analysis_path))

            # --------------------------------------------------------------------------
            # all metrics:
            if os.path.exists(metrics_analysis_path):

                metrics_analysis = read_json(metrics_analysis_path)

                name_list.append(folder_name + '_' + suffix)

                # --------------------------------------------------------------------------
                # DSC:

                mean_DSC_list.append(metrics_analysis["all"]["mean_DSC"]*100)
                mean_DSC_list_2.append(round(metrics_analysis["all"]["mean_DSC"]*100, 2))

                # --------------------------------------------------------------------------
                # IoU:

                mean_IoU_list.append(metrics_analysis["all"]["mean_IoU"]*100)
                mean_IoU_list_2.append(round(metrics_analysis["all"]["mean_IoU"]*100, 2))

                # ---------------------------------------------------------------------------
                # HD95
                mean_HD95_list.append(metrics_analysis["all"]["mean_HD95"])
                mean_HD95_list_2.append(round(metrics_analysis["all"]["mean_HD95"], 2))

                # --------------------------------------------------------------------------
                # NSD:

                mean_NSD_list.append(metrics_analysis["all"]["mean_NSD"]*100)
                mean_NSD_list_2.append(round(metrics_analysis["all"]["mean_NSD"]*100, 2))

            # the file not exist:
            else:
                
                # --------------------------------------------------------------------------
                # if file not exist, remove the corresponding iD when calculating avg and std.

                fold_ids_for_avg.remove(int(folder_name.split('_')[-1])) # this remove will change the value of args.fold_ids_select!!! use: .copy()

                name_list.append(folder_name + '_' + suffix)

                # --------------------------------------------------------------------------
                # DSC:

                mean_DSC_list.append(str('Nan'))
                mean_DSC_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # IoU:

                mean_IoU_list.append(str('Nan'))
                mean_IoU_list_2.append(str('Nan'))

                # ---------------------------------------------------------------------------
                # HD95
                mean_HD95_list.append(str('Nan'))
                mean_HD95_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # NSD:

                mean_NSD_list.append(str('Nan'))
                mean_NSD_list_2.append(str('Nan'))
            
        else:

            name_list.append(folder_name + '_' + suffix)

            # --------------------------------------------------------------------------
            # DSC:

            mean_DSC_list.append(0)
            mean_DSC_list_2.append(0)

            # --------------------------------------------------------------------------
            # IoU:

            mean_IoU_list.append(0)
            mean_IoU_list_2.append(0)

            # ---------------------------------------------------------------------------
            # HD95
            mean_HD95_list.append(0)
            mean_HD95_list_2.append(0)

            # --------------------------------------------------------------------------
            # NSD:

            mean_NSD_list.append(0)
            mean_NSD_list_2.append(0)

    # --------------------------------------------------------------------------
    # average:

    # ------------------------------------------
    name_list.append('average_' + suffix)

    mean_DSC_list_2.append(str(round(fun_avg([mean_DSC_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([mean_DSC_list[i] for i in fold_ids_for_avg]), 2)))

    mean_IoU_list_2.append(str(round(fun_avg([mean_IoU_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([mean_IoU_list[i] for i in fold_ids_for_avg]), 2)))

    mean_HD95_list_2.append(str(round(fun_avg([mean_HD95_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([mean_HD95_list[i] for i in fold_ids_for_avg]), 2)))

    mean_NSD_list_2.append(str(round(fun_avg([mean_NSD_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([mean_NSD_list[i] for i in fold_ids_for_avg]), 2)))

    # ------------------------------------------

    all_dict = {'name': name_list, 'mean_DSC': mean_DSC_list_2, 'mean_IoU': mean_IoU_list_2, 'mean_HD95': mean_HD95_list_2, 'mean_NSD': mean_NSD_list_2}

    # --------------------------------------------------------------------------
    # record each classes:

    for index_class in range(classes): 

        index_class += 1

        DSC_class_list = []
        DSC_class_list_2 = []

        IoU_class_list = []
        IoU_class_list_2 = []

        HD95_class_list = []
        HD95_class_list_2 = []

        NSD_class_list = []
        NSD_class_list_2 = []

        for folder_name in sub_folder_names:

            if folder_name in sub_folder_names_selected:

                # --------------------------------------------------------------------------
                metrics_analysis_path = path_folders + '/' + folder_name + '/metrics_from_softmax/' + file_name

                if not os.path.exists(metrics_analysis_path):

                    raise ValueError('This path does not exist: ' + str(metrics_analysis_path))

                # --------------------------------------------------------------------------
                # all metrics:
                if os.path.exists(metrics_analysis_path):

                    metrics_analysis = read_json(metrics_analysis_path)

                    # --------------------------------------------------------------------------
                    # DSC:

                    DSC_class_list.append(metrics_analysis["all"]["DSC_class_" + str(index_class)]*100)
                    DSC_class_list_2.append(round(metrics_analysis["all"]["DSC_class_" + str(index_class)]*100, 2))

                    # --------------------------------------------------------------------------
                    # IoU:

                    IoU_class_list.append(metrics_analysis["all"]["IoU_class_" + str(index_class)]*100)
                    IoU_class_list_2.append(round(metrics_analysis["all"]["IoU_class_" + str(index_class)]*100, 2))

                    # ---------------------------------------------------------------------------
                    # HD95

                    HD95_class_list.append(metrics_analysis["all"]["HD95_class_" + str(index_class)])
                    HD95_class_list_2.append(round(metrics_analysis["all"]["HD95_class_" + str(index_class)], 2))

                    # --------------------------------------------------------------------------
                    # NSD:

                    NSD_class_list.append(metrics_analysis["all"]["NSD_class_" + str(index_class)]*100)
                    NSD_class_list_2.append(round(metrics_analysis["all"]["NSD_class_" + str(index_class)]*100, 2))

                # the file not exist:
                else:
                    
                    # --------------------------------------------------------------------------
                    # DSC:

                    DSC_class_list.append(str('Nan'))
                    DSC_class_list_2.append(str('Nan'))

                    # --------------------------------------------------------------------------
                    # IoU:

                    IoU_class_list.append(str('Nan'))
                    IoU_class_list_2.append(str('Nan'))

                    # ---------------------------------------------------------------------------
                    # HD95

                    HD95_class_list.append(str('Nan'))
                    HD95_class_list_2.append(str('Nan'))

                    # --------------------------------------------------------------------------
                    # NSD:

                    NSD_class_list.append(str('Nan'))
                    NSD_class_list_2.append(str('Nan'))
                
            else:

                # --------------------------------------------------------------------------
                # DSC:

                DSC_class_list.append(0)
                DSC_class_list_2.append(0)

                # --------------------------------------------------------------------------
                # IoU:

                IoU_class_list.append(0)
                IoU_class_list_2.append(0)

                # ---------------------------------------------------------------------------
                # HD95

                HD95_class_list.append(0)
                HD95_class_list_2.append(0)

                # --------------------------------------------------------------------------
                # NSD:

                NSD_class_list.append(0)
                NSD_class_list_2.append(0)

        # --------------------------------------------------------------------------
        # average:

        # ------------------------------------------

        DSC_class_list_2.append(str(round(fun_avg([DSC_class_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([DSC_class_list[i] for i in fold_ids_for_avg]), 2)))

        IoU_class_list_2.append(str(round(fun_avg([IoU_class_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([IoU_class_list[i] for i in fold_ids_for_avg]), 2)))

        HD95_class_list_2.append(str(round(fun_avg([HD95_class_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([HD95_class_list[i] for i in fold_ids_for_avg]), 2)))

        NSD_class_list_2.append(str(round(fun_avg([NSD_class_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([NSD_class_list[i] for i in fold_ids_for_avg]), 2)))

        # ------------------------------------------

        sub_dict = {'DSC_class_' + str(index_class): DSC_class_list_2, 'IoU_class_' + str(index_class): IoU_class_list_2, 'HD95_class_' + str(index_class): HD95_class_list_2, 'NSD_class_' + str(index_class): NSD_class_list_2}

        all_dict.update(sub_dict)

    # --------------------------------------------------------------------------
    writer = pandas.ExcelWriter(execel_path_save)

    sheetNames = all_dict.keys()  

    all_execel = pandas.DataFrame(all_dict)

    all_execel.to_excel(writer, sheet_name='ACDC_3D_Median')

    # writer.save()
    writer.close()


