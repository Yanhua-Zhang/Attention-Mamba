from loss_functions.loss import CE_and_Dice_loss, FL_and_CE_loss, BCE_and_wIoU_loss, BCE_and_Dice_loss
from loss_functions.loss_V2 import CE_and_Dice_loss_V2

# -----------------------------------------------------------------------------
def get_loss(args, logger):

    if args.loss_name == 'CE_and_Dice_loss':
        loss = CE_and_Dice_loss(num_classes = args.num_classes)
        logger.info("Successfully loaded: " + args.loss_name)

    elif args.loss_name == 'CE_and_Dice_loss_V2':
        # balanced CE loss
        loss = CE_and_Dice_loss_V2(class_weights = args.class_weights)
        logger.info("Successfully loaded: " + args.loss_name)

    elif args.loss_name == 'FL_and_CE_loss':
        loss = FL_and_CE_loss(alpha = args.alpha)
        logger.info("Successfully loaded: " + args.loss_name)

    elif args.loss_name == 'BCE_and_wIoU_loss':
        loss = BCE_and_wIoU_loss()
        logger.info("Successfully loaded: " + args.loss_name)

    elif args.loss_name == 'BCE_and_Dice_loss':
        loss = BCE_and_Dice_loss()
        logger.info("Successfully loaded: " + args.loss_name)

    else:
        raise ValueError('this loss is not supported: ' + args.loss_name)

    return loss

# -----------------------------------------------------------------------------