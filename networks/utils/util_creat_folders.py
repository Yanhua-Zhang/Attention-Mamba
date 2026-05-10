import os


# add weight_decay in save_folder
# add model_version_name in save_folder


def fun_creat_folders(args):

    # args.save_folder = args.dataset + '_' + args.model_name + '_epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr) + '_' + args.loss_calculator_name + '_' + args.marker + '_seed' + str(args.seed)

    args.save_folder = args.dataset + '_' + args.model_name + '_' + args.model_version_name + '_epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr) + '_wd' + str(args.weight_decay) + '_' + args.loss_calculator_name + '_' + args.marker + '_seed' + str(args.seed)

    args.TensorboardX_path = "../results/{}/TensorboardX/Train/{}".format(args.save_folder, 'fold_' + str(args.fold_name))
    args.Log_path = "../results/{}/{}/Log".format(args.save_folder, 'fold_' + str(args.fold_name))
    args.Model_path = "../results/{}/{}/Train/Model".format(args.save_folder, 'fold_' + str(args.fold_name))
    args.Model_best_path = "../results/{}/{}/Train/Model_best".format(args.save_folder, 'fold_' + str(args.fold_name))
    args.Summary_path = "../results/{}/{}/Train/Summary".format(args.save_folder, 'fold_' + str(args.fold_name))
    args.Visual_path = "../results/{}/{}/Train/Visualization".format(args.save_folder, 'fold_' + str(args.fold_name))

    if not os.path.exists(args.Log_path):
        os.makedirs(args.Log_path)

    if not os.path.exists(args.TensorboardX_path):
        os.makedirs(args.TensorboardX_path)

    if not os.path.exists(args.Model_path):
        os.makedirs(args.Model_path)

    if not os.path.exists(args.Model_best_path):
        os.makedirs(args.Model_best_path)

    if not os.path.exists(args.Summary_path):
        os.makedirs(args.Summary_path)

    if not os.path.exists(args.Visual_path):
        os.makedirs(args.Visual_path)


    if args.dataset.split('_')[1] == 'ISIC2018':

        args.Softmax_test_all_path = "../results/{}/{}/test_all/softmax_pre".format(args.save_folder, 'fold_' + str(args.fold_name))
        args.analysis_test_all_path = "../results/{}/{}/{}/analysis_test_all/{}".format(args.save_folder, args.inference_type, args.load_model_type, 'fold_' + str(args.fold_name))
        if not os.path.exists(args.Softmax_test_all_path):
            os.makedirs(args.Softmax_test_all_path)
        if not os.path.exists(args.analysis_test_all_path):
            os.makedirs(args.analysis_test_all_path)

        args.Softmax_test_ISIC2018_path = "../results/{}/{}/test_ISIC2018/softmax_pre".format(args.save_folder, 'fold_' + str(args.fold_name))
        args.analysis_test_ISIC2018_path = "../results/{}/{}/{}/analysis_test_ISIC2018/{}".format(args.save_folder, args.inference_type, args.load_model_type, 'fold_' + str(args.fold_name))
        if not os.path.exists(args.Softmax_test_ISIC2018_path):
            os.makedirs(args.Softmax_test_ISIC2018_path)
        if not os.path.exists(args.analysis_test_ISIC2018_path):
            os.makedirs(args.analysis_test_ISIC2018_path)

        args.Softmax_test_PH2_path = "../results/{}/{}/test_PH2/softmax_pre".format(args.save_folder, 'fold_' + str(args.fold_name))
        args.analysis_test_PH2_path = "../results/{}/{}/{}/analysis_test_PH2/{}".format(args.save_folder, args.inference_type, args.load_model_type, 'fold_' + str(args.fold_name))
        if not os.path.exists(args.Softmax_test_PH2_path):
            os.makedirs(args.Softmax_test_PH2_path)
        if not os.path.exists(args.analysis_test_PH2_path):
            os.makedirs(args.analysis_test_PH2_path)

    else:

        args.Softmax_val_path = "../results/{}/{}/Val/softmax_pre".format(args.save_folder, 'fold_' + str(args.fold_name))
        args.analysis_val_path = "../results/{}/{}/{}/analysis_val/{}".format(args.save_folder, args.inference_type, args.load_model_type, 'fold_' + str(args.fold_name))
        if not os.path.exists(args.Softmax_val_path):
            os.makedirs(args.Softmax_val_path)
        if not os.path.exists(args.analysis_val_path):
            os.makedirs(args.analysis_val_path)

        args.Softmax_test_path = "../results/{}/{}/Test/softmax_pre".format(args.save_folder, 'fold_' + str(args.fold_name))
        args.analysis_test_path = "../results/{}/{}/{}/analysis_test/{}".format(args.save_folder, args.inference_type, args.load_model_type, 'fold_' + str(args.fold_name))
        args.analysis_test_ensemble_path = "../results/{}/{}/{}/analysis_test/ensemble".format(args.save_folder, args.inference_type, args.load_model_type)
        if not os.path.exists(args.Softmax_test_path):
            os.makedirs(args.Softmax_test_path)
        if not os.path.exists(args.analysis_test_path):
            os.makedirs(args.analysis_test_path)
        if not os.path.exists(args.analysis_test_ensemble_path):
            os.makedirs(args.analysis_test_ensemble_path)

        args.Softmax_Second_test_path = "../results/{}/{}/Test_Second/softmax_pre".format(args.save_folder, 'fold_' + str(args.fold_name)) # add Secondary Testset
        args.analysis_Second_test_path = "../results/{}/{}/{}/analysis_Second_test/{}".format(args.save_folder, args.inference_type, args.load_model_type, 'fold_' + str(args.fold_name))
        args.analysis_Second_test_ensemble_path = "../results/{}/{}/{}/analysis_Second_test/ensemble".format(args.save_folder, args.inference_type, args.load_model_type)
        if not os.path.exists(args.Softmax_Second_test_path):
            os.makedirs(args.Softmax_Second_test_path)
        if not os.path.exists(args.analysis_Second_test_path):
            os.makedirs(args.analysis_Second_test_path)
        if not os.path.exists(args.analysis_Second_test_ensemble_path):
            os.makedirs(args.analysis_Second_test_ensemble_path)