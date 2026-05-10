import ml_collections

def get_configs():
  
    config = ml_collections.ConfigDict()

    # -------------------------------------------------
    # backbone


    # -------------------------------------------------
    # general settings

    config.If_weight_init = False

    config.If_pretrained = True

    # -------------------------------------------------

    return config


def fun_renew_configs(config, args):

    config.img_size  = args.img_size

    config.If_weight_init  = args.If_weight_init

    config.If_pretrained  = args.If_pretrained

    return config
