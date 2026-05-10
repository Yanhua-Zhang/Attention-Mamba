

def fun_calc_loss_loader(args, logger):

    from loss_functions.loss_loader import get_loss
    from loss_functions.calc_loss import calc_MultipleOutput_Loss, calc_MultiOutput_Align_Loss, calc_MultiOutput_Inconsistenc_Loss, calc_MultiOutput_Align_Fea_Inconsist_Loss, calc_MultiOutput_Align_Multi_Scale_Fus_Loss, calc_MultipleOutput_CE_Loss, calc_MultipleOutput_CE_Loss_V2

    if args.loss_calculator_name == 'calc_MultipleOutput_Loss':
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        args.loss_name = 'FL_and_CE_loss'
        loss_lesion = get_loss(args, logger)

        calc_loss_fun = calc_MultipleOutput_Loss(loss=loss_lesion, loss_weights=args.loss_weights)


    elif args.loss_calculator_name == 'calc_MultipleOutput_BCE_and_wIoU_loss':
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        args.loss_name = 'BCE_and_wIoU_loss'
        loss_seg = get_loss(args, logger)

        calc_loss_fun = calc_MultipleOutput_Loss(loss=loss_seg, loss_weights=args.loss_weights)

    elif args.loss_calculator_name == 'calc_MultipleOutput_BCE_and_Dice_loss':
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        args.loss_name = 'BCE_and_Dice_loss'
        loss_seg = get_loss(args, logger)

        calc_loss_fun = calc_MultipleOutput_Loss(loss=loss_seg, loss_weights=args.loss_weights)

    elif args.loss_calculator_name == 'calc_MultipleOutput_CE_and_Dice_Loss':
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        args.loss_name = 'CE_and_Dice_loss'
        loss_seg = get_loss(args, logger)

        calc_loss_fun = calc_MultipleOutput_CE_Loss(loss=loss_seg, loss_weights=args.loss_weights)

    elif args.loss_calculator_name == 'calc_MultipleOutput_CE_and_Dice_Loss_V2':
        # used for the balanced CE loss: CE_and_Dice_loss_V2
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        args.loss_name = 'CE_and_Dice_loss_V2'
        loss_seg = get_loss(args, logger)

        calc_loss_fun = calc_MultipleOutput_CE_Loss_V2(loss=loss_seg, loss_weights=args.loss_weights)

    elif args.loss_calculator_name == 'calc_MultiOutput_Align_Loss':
        
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        if not args.If_Deep_Fusion_Supervision:
            # raise ValueError('must use Deep_Fusion_Supervision for this loss calculator: ' + args.loss_calculator_name)
            logger.info('the Deep_Fusion_Supervision for this loss calculator is disabled: ' + args.loss_calculator_name)

        args.loss_name = 'FL_and_CE_loss'
        loss_lesion = get_loss(args, logger)

        args.loss_name = 'CE_and_Dice_loss'
        loss_prostate = get_loss(args, logger)

        calc_loss_fun = calc_MultiOutput_Align_Loss(args, loss_lesion, loss_prostate)

    elif args.loss_calculator_name == 'calc_MultiOutput_Align_Multi_Scale_Fus_Loss':
        
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        if not args.If_Multi_scale_Fusion_Supervision:
            raise ValueError('must use Multi_scale_Fusion_Supervision for this loss calculator: ' + args.loss_calculator_name)

        args.loss_name = 'FL_and_CE_loss'
        loss_lesion = get_loss(args, logger)

        args.loss_name = 'CE_and_Dice_loss'
        loss_prostate = get_loss(args, logger)

        calc_loss_fun = calc_MultiOutput_Align_Multi_Scale_Fus_Loss(args, loss_lesion, loss_prostate)

    elif args.loss_calculator_name == 'calc_MultiOutput_Inconsistenc_Loss':
        
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        if not args.If_Deep_Fusion_Supervision:
            raise ValueError('must use Deep_Fusion_Supervision for this loss calculator: ' + args.loss_calculator_name)

        args.loss_name = 'FL_and_CE_loss'
        loss_lesion = get_loss(args, logger)

        args.loss_name = 'CE_and_Dice_loss'
        loss_prostate = get_loss(args, logger)

        calc_loss_fun = calc_MultiOutput_Inconsistenc_Loss(args, loss_lesion, loss_prostate)

    elif args.loss_calculator_name == 'calc_MultiOutput_Align_Fea_Inconsist_Loss':
        
        logger.info("Use this loss calculator: " + args.loss_calculator_name)

        if not args.If_Deep_Fusion_Supervision:
            raise ValueError('must use Deep_Fusion_Supervision for this loss calculator: ' + args.loss_calculator_name)

        args.loss_name = 'FL_and_CE_loss'
        loss_lesion = get_loss(args, logger)

        args.loss_name = 'CE_and_Dice_loss'
        loss_prostate = get_loss(args, logger)

        calc_loss_fun = calc_MultiOutput_Align_Fea_Inconsist_Loss(args, loss_lesion, loss_prostate)

    else:
        raise ValueError('this loss calculator is not supported: ' + args.loss_calculator_name)

    return calc_loss_fun