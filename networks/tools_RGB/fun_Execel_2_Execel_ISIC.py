import numpy as np
from statistics import mean, pstdev
from sklearn.metrics import auc
import pandas
import os


# summarize the results from Execels to one Execel

# add calculation of f1, Acc, Recall, Spe, Prec 


#---------------------------------------------------------------
# calculate avg metrics across the 5 folds on the val, test, or ensemble sets

def run_Execel_2_Execel(args, all_analysis_test_paths, suffix_names):

    # --------------------------------------------------------------------------
    # save path

    save_path = "../results/{}/{}/{}".format(args.save_folder, args.inference_type, args.load_model_type)

    execel_path_save = save_path + '/Execel_all_Testset.xlsx' 

    # --------------------------------------------------------------------------
    all_dict = {'name': ['fold_0_Seg', 'fold_1_Seg', 'fold_2_Seg', 'fold_3_Seg', 'fold_4_Seg', 'average_Seg']}
    # all_dict = {}

    index = 0

    for all_analysis_test_path in all_analysis_test_paths:
        
        # -------------------------------------------------
        path_folder = all_analysis_test_path.replace('/fold_' + str(args.fold_name), '')

        execel_path = path_folder + '/Execel_' + suffix_names[index] + '_Seg.xlsx'

        if not os.path.exists(execel_path):
            raise ValueError("path not exists: " + execel_path)
        
        # -------------------------------------------------
        execel_inf = pandas.read_excel(execel_path)

        value_1_all = []
        value_2_all = []
        value_3_all = []
        value_4_all = []
        value_5_all = []
        value_6_all = []
        value_7_all = []
        value_8_all = []
        value_9_all = []

        for i in range(6):
            value_1 = execel_inf.loc[i,"DSC_class_1"]
            value_2 = execel_inf.loc[i,"IoU_class_1"]
            value_3 = execel_inf.loc[i,"HD95_class_1"]
            value_4 = execel_inf.loc[i,"NSD_class_1"]
            value_5 = execel_inf.loc[i,"Acc_class_1"]
            value_6 = execel_inf.loc[i,"Recall_class_1"]
            value_7 = execel_inf.loc[i,"Spe_class_1"]
            value_8 = execel_inf.loc[i,"Prec_class_1"]
            value_9 = execel_inf.loc[i,"f1_class_1"]

            value_1_all.append(value_1)
            value_2_all.append(value_2)
            value_3_all.append(value_3)
            value_4_all.append(value_4)
            value_5_all.append(value_5)
            value_6_all.append(value_6)
            value_7_all.append(value_7)
            value_8_all.append(value_8)
            value_9_all.append(value_9)

        sub_dict = {suffix_names[index] + '_DSC': value_1_all, suffix_names[index] + '_IoU': value_2_all, suffix_names[index] + '_HD95': value_3_all, suffix_names[index] + '_NSD': value_4_all, suffix_names[index] + '_Acc': value_5_all, suffix_names[index] + '_Recall': value_6_all, suffix_names[index] + '_Spe': value_7_all, suffix_names[index] + '_Prec': value_8_all, suffix_names[index] + '_f1': value_9_all}

        all_dict.update(sub_dict)

        # -------------------------------------------------
        index += 1

    # --------------------------------------------------------------------------
    writer = pandas.ExcelWriter(execel_path_save)

    sheetNames = all_dict.keys()  

    all_execel = pandas.DataFrame(all_dict)

    # for sheetName in sheetNames:
    #     all_execel.to_excel(writer, sheet_name=sheetName)
    all_execel.to_excel(writer, sheet_name='ISIC')

    # writer.save()
    writer.close()


