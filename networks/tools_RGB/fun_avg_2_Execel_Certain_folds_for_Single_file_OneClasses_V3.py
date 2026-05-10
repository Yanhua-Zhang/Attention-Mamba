import numpy as np
import os
import pandas
# from nnunet.utilities.io import (checksum, path_exists, read_json, refresh_file_list, write_json)

from tools_Seg_3D.utils_postprocessing_Seg_3D import *
from utils.util_io import read_json, write_json


# remove the dependence on nnunet.utilities.io
# fix the bug of Postprocessing for one class type: no need for avg of multiple classess

# add calculation of f1, Acc, Recall, Spe, Prec 

#---------------------------------------------------------------
# calculate avg metrics across the 5 folds on the val, test

def run_analysis_avg_To_Execel_val_test_for_Single_file_OneClasses(args, path_folders, mode='val', file_name=None, suffix=None):

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

    execel_path_save = path_folders + '/Execel_' + mode + '_' + suffix + '.xlsx'

    if not os.path.exists(path_folders):
        raise ValueError("path_folders not exists.")

    # --------------------------------------------------------------------------
    name_list = []

    DSC_class_1_list = []
    DSC_class_1_list_2 = []

    IoU_class_1_list = []
    IoU_class_1_list_2 = []

    HD95_class_1_list = []
    HD95_class_1_list_2 = []

    NSD_class_1_list = []
    NSD_class_1_list_2 = []

    Acc_class_1_list = []
    Acc_class_1_list_2 = []

    Recall_class_1_list = []
    Recall_class_1_list_2 = []

    Spe_class_1_list = []
    Spe_class_1_list_2 = []

    Prec_class_1_list = []
    Prec_class_1_list_2 = []

    f1_class_1_list = []
    f1_class_1_list_2 = []


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

                DSC_class_1_list.append(metrics_analysis["all"]["DSC_class_1"]*100)
                DSC_class_1_list_2.append(round(metrics_analysis["all"]["DSC_class_1"]*100, 2))

                # --------------------------------------------------------------------------
                # IoU:

                IoU_class_1_list.append(metrics_analysis["all"]["IoU_class_1"]*100)
                IoU_class_1_list_2.append(round(metrics_analysis["all"]["IoU_class_1"]*100, 2))

                # ---------------------------------------------------------------------------
                # HD95

                HD95_class_1_list.append(metrics_analysis["all"]["HD95_class_1"])
                HD95_class_1_list_2.append(round(metrics_analysis["all"]["HD95_class_1"], 2))

                # --------------------------------------------------------------------------
                # NSD:

                NSD_class_1_list.append(metrics_analysis["all"]["NSD_class_1"]*100)
                NSD_class_1_list_2.append(round(metrics_analysis["all"]["NSD_class_1"]*100, 2))

                # --------------------------------------------------------------------------
                # Acc:

                Acc_class_1_list.append(metrics_analysis["all"]["Acc_class_1"]*100)
                Acc_class_1_list_2.append(round(metrics_analysis["all"]["Acc_class_1"]*100, 2))

                # --------------------------------------------------------------------------
                # Recall:

                Recall_class_1_list.append(metrics_analysis["all"]["Recall_class_1"]*100)
                Recall_class_1_list_2.append(round(metrics_analysis["all"]["Recall_class_1"]*100, 2))

                # --------------------------------------------------------------------------
                # Spe:

                Spe_class_1_list.append(metrics_analysis["all"]["Spe_class_1"]*100)
                Spe_class_1_list_2.append(round(metrics_analysis["all"]["Spe_class_1"]*100, 2))

                # --------------------------------------------------------------------------
                # Prec:

                Prec_class_1_list.append(metrics_analysis["all"]["Prec_class_1"]*100)
                Prec_class_1_list_2.append(round(metrics_analysis["all"]["Prec_class_1"]*100, 2))

                # --------------------------------------------------------------------------
                # f1:

                f1_class_1_list.append(metrics_analysis["all"]["f1_class_1"]*100)
                f1_class_1_list_2.append(round(metrics_analysis["all"]["f1_class_1"]*100, 2))

            # the file not exist:
            else:
                
                # --------------------------------------------------------------------------
                # if file not exist, remove the corresponding iD when calculating avg and std.

                fold_ids_for_avg.remove(int(folder_name.split('_')[-1])) # this remove will change the value of args.fold_ids_select!!! use: .copy()

                name_list.append(folder_name + '_' + suffix)

                # --------------------------------------------------------------------------
                # DSC:

                DSC_class_1_list.append(str('Nan'))
                DSC_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # IoU:

                IoU_class_1_list.append(str('Nan'))
                IoU_class_1_list_2.append(str('Nan'))

                # ---------------------------------------------------------------------------
                # HD95

                HD95_class_1_list.append(str('Nan'))
                HD95_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # NSD:

                NSD_class_1_list.append(str('Nan'))
                NSD_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # Acc:

                Acc_class_1_list.append(str('Nan'))
                Acc_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # Recall:

                Recall_class_1_list.append(str('Nan'))
                Recall_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # Spe:

                Spe_class_1_list.append(str('Nan'))
                Spe_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # Prec:

                Prec_class_1_list.append(str('Nan'))
                Prec_class_1_list_2.append(str('Nan'))

                # --------------------------------------------------------------------------
                # f1:

                f1_class_1_list.append(str('Nan'))
                f1_class_1_list_2.append(str('Nan'))

        else:

            name_list.append(folder_name + '_' + suffix)

            # --------------------------------------------------------------------------
            # DSC:

            DSC_class_1_list.append(0)
            DSC_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # IoU:

            IoU_class_1_list.append(0)
            IoU_class_1_list_2.append(0)

            # ---------------------------------------------------------------------------
            # HD95

            HD95_class_1_list.append(0)
            HD95_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # NSD:

            NSD_class_1_list.append(0)
            NSD_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # Acc:

            Acc_class_1_list.append(0)
            Acc_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # Recall:

            Recall_class_1_list.append(0)
            Recall_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # Spe:

            Spe_class_1_list.append(0)
            Spe_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # Prec:

            Prec_class_1_list.append(0)
            Prec_class_1_list_2.append(0)

            # --------------------------------------------------------------------------
            # f1:

            f1_class_1_list.append(0)
            f1_class_1_list_2.append(0)

    # --------------------------------------------------------------------------
    # average:

    # ------------------------------------------
    name_list.append('average_' + suffix)

    DSC_class_1_list_2.append(str(round(fun_avg([DSC_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([DSC_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    IoU_class_1_list_2.append(str(round(fun_avg([IoU_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([IoU_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    HD95_class_1_list_2.append(str(round(fun_avg([HD95_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([HD95_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    NSD_class_1_list_2.append(str(round(fun_avg([NSD_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([NSD_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    Acc_class_1_list_2.append(str(round(fun_avg([Acc_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([Acc_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    Recall_class_1_list_2.append(str(round(fun_avg([Recall_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([Recall_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    Spe_class_1_list_2.append(str(round(fun_avg([Spe_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([Spe_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    Prec_class_1_list_2.append(str(round(fun_avg([Prec_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([Prec_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    f1_class_1_list_2.append(str(round(fun_avg([f1_class_1_list[i] for i in fold_ids_for_avg]), 2)) + ' std: ' + str(round(fun_std([f1_class_1_list[i] for i in fold_ids_for_avg]), 2)))

    # ------------------------------------------

    all_dict = {'name': name_list, 'DSC_class_1': DSC_class_1_list_2, 'IoU_class_1': IoU_class_1_list_2, 'HD95_class_1': HD95_class_1_list_2, 'NSD_class_1': NSD_class_1_list_2, 'Acc_class_1': Acc_class_1_list_2, 'Recall_class_1': Recall_class_1_list_2, 'Spe_class_1': Spe_class_1_list_2, 'Prec_class_1': Prec_class_1_list_2, 'f1_class_1': f1_class_1_list_2}

    # --------------------------------------------------------------------------
    writer = pandas.ExcelWriter(execel_path_save)

    sheetNames = all_dict.keys()  

    all_execel = pandas.DataFrame(all_dict)

    all_execel.to_excel(writer, sheet_name='Results')

    # writer.save()
    writer.close()


