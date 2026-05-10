

def fun_load_model_2D(args, logger):

    if args.loss_calculator_name == 'calc_MultipleOutput_CE_and_Dice_Loss':

        if args.model_version_name == 'My_Attention_Mamba_2D':
            logger.info("Load the model used for this loss calculator: " + args.loss_calculator_name + '. The model is loaded from: ' + args.model_version_name)

            from network_architectures.My_Networks_2D_Attention_Mamba.configs_My_Attention_Mamba_2D import get_configs, fun_renew_configs
            from network_architectures.My_Networks_2D_Attention_Mamba.model_My_Attention_Mamba_2D import Other_models

        else:
            raise ValueError('this model version is not supported: ' + args.model_version_name)
        

    elif args.loss_calculator_name == 'calc_MultipleOutput_BCE_and_wIoU_loss':

        if args.model_version_name == 'My_Attention_Mamba_2D':
            logger.info("Load the model used for this loss calculator: " + args.loss_calculator_name + '. The model is loaded from: ' + args.model_version_name)

            from network_architectures.My_Networks_2D_Attention_Mamba.configs_My_Attention_Mamba_2D import get_configs, fun_renew_configs
            from network_architectures.My_Networks_2D_Attention_Mamba.model_My_Attention_Mamba_2D import Other_models

        else:
            raise ValueError('this model version is not supported: ' + args.model_version_name)


    elif args.loss_calculator_name == 'calc_MultipleOutput_BCE_and_Dice_loss':

        if args.model_version_name == 'My_Attention_Mamba_2D':
            logger.info("Load the model used for this loss calculator: " + args.loss_calculator_name + '. The model is loaded from: ' + args.model_version_name)

            from network_architectures.My_Networks_2D_Attention_Mamba.configs_My_Attention_Mamba_2D import get_configs, fun_renew_configs
            from network_architectures.My_Networks_2D_Attention_Mamba.model_My_Attention_Mamba_2D import Other_models

        else:
            raise ValueError('this model version is not supported: ' + args.model_version_name)

    else:
        raise ValueError('this loss calculator is not supported: ' + args.loss_calculator_name)


    config = get_configs()
    config = fun_renew_configs(config, args)

    logger.info("load this network: " + args.model_name + '. And its version is: ' + args.model_version_name)
    
    net = Other_models(logger, config, classes=args.num_classes).cuda()

    logger.info('All configs: ' + str(config))

    return net