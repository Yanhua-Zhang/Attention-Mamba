from typing import Dict
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ----------------------------------------------------------------
def scheduler_for_warmup(args, optimizer, logger):
    """
    Linearly ramps up the learning rate within X
    number of epochs to the working epoch.
    Args:
        optimizer (_type_): _description_
        warmup_epochs (_type_): _description_
        warmup_lr (_type_): warmup lr should be the starting lr we want.
    """
    
    lambda1 = lambda epoch: (
        (epoch + 1) * 1.0 / args.warmup_epochs
    )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, verbose=False)
    
    return scheduler

# ----------------------------------------------------------------
def scheduler_reducelronplateau(args, optimizer, logger):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        mode=args.mode,
        patience=args.patience,
        verbose=False,
        min_lr=args.min_lr,
    )

    return scheduler


# ----------------------------------------------------------------
def scheduler_cosine_annealing_wr(args, optimizer, logger):
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t_0_epochs,
        T_mult=args.t_mult,
        eta_min=args.min_lr,
        last_epoch=-1,
        verbose=False,
    )
    logger.info("t_0_epochs: " + str(args.t_0_epochs))
    logger.info("t_mult: " + str(args.t_mult))
    logger.info("min_lr: " + str(args.min_lr))
    logger.info("last_epoch: " + str(-1))
    return scheduler


# ----------------------------------------------------------------
def scheduler_cosine_annealing(args, optimizer, logger):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = args.T_max,
        eta_min = args.eta_min,
        last_epoch = args.last_epoch
    )
    logger.info("T_max: " + str(args.T_max))
    logger.info("eta_min: " + str(args.eta_min))
    logger.info("last_epoch: " + str(args.last_epoch))
    return scheduler


# ----------------------------------------------------------------
def scheduler_poly_lr(args, optimizer, logger):
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer=optimizer,
        total_iters=args.max_epochs,
        power=args.power,
    )
    logger.info("max_epochs: " + str(args.max_epochs))
    logger.info("power: " + str(args.power))
    return scheduler


# ----------------------------------------------------------------
def get_scheduler(optimizer, args, logger):

    if args.scheduler_name == "reducelronplateau":

        logger.info("Use this scheduler: " + args.scheduler_name)

        return scheduler_reducelronplateau(args, optimizer, logger)
    
    elif args.scheduler_name == "cosine_annealing_wr":

        logger.info("Use this scheduler: " + args.scheduler_name)

        return scheduler_cosine_annealing_wr(args, optimizer, logger)
        
    elif args.scheduler_name == "poly_lr":

        logger.info("Use this scheduler: " + args.scheduler_name)

        return scheduler_poly_lr(args, optimizer, logger)
    
    elif args.scheduler_name == "LambdaLR_warm_up":

        logger.info("Use this scheduler for warm up: " + args.scheduler_name)

        return scheduler_for_warmup(args, optimizer, logger)
    
    elif args.scheduler_name == "cosine_annealing":

        logger.info("Use this scheduler: " + args.scheduler_name)

        return scheduler_cosine_annealing(args, optimizer, logger)
    
    else:
        raise NotImplementedError("This Scheduler is not implemented")

