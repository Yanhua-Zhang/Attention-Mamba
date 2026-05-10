from typing import Dict
import torch.optim as optim


# ----------------------------------------------------------------
def optim_adam(model, args, logger):
    # betas: default: (0.9, 0.999)
    # eps: (default: 1e-8)
    # weight_decay: (default: 0)
    adam = optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    logger.info("base_lr: " + str(args.base_lr))
    logger.info("weight_decay: " + str(args.weight_decay))
    return adam


# ----------------------------------------------------------------
def optim_sgd(model, args, logger):
    adam = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    logger.info("base_lr: " + str(args.base_lr))
    logger.info("weight_decay: " + str(args.weight_decay))
    logger.info("momentum: " + str(args.momentum))
    return adam


# ----------------------------------------------------------------
def optim_adamw(model, args, logger):
    adam = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        betas=args.betas,   # default: (0.9, 0.999)
        weight_decay=args.weight_decay,   # default: 1e-2
        # amsgrad=True,    # default: False
        eps=args.eps,          # 1e-5 default: 1e-8  
    )
    logger.info("base_lr: " + str(args.base_lr))
    logger.info("weight_decay: " + str(args.weight_decay))
    logger.info("betas: " + str(args.betas))
    logger.info("eps: " + str(args.eps))
    return adam


# ----------------------------------------------------------------
def get_optimizer(model, args, logger):

    if args.optimizer_name == "Adam":
        logger.info("Use this optimizer: " + args.optimizer_name)
        return optim_adam(model, args, logger)
    
    elif args.optimizer_name == "AdamW":
        logger.info("Use this optimizer: " + args.optimizer_name)
        return optim_adamw(model, args, logger)
    
    elif args.optimizer_name == "SGD":
        logger.info("Use this optimizer: " + args.optimizer_name)
        return optim_sgd(model, args, logger)
    
    else:
        raise ValueError("optimizer not supported, right now")
