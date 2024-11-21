import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'InvDN':
        from .InvDN_model import InvDN_Model as M
    elif model == 'InvDN_C':
        from .InvDN_model_C import InvDN_Model_C as M
    elif model == 'Unet':
        from .Unet_model import Unet_model as M
    elif model == 'Unet_aug':
        from .Unet_model_aug import Unet_model_aug as M
    elif model == 'DnCNN':
        from .DnCNN_model import DnCNN_model as M
    elif model == 'DnCNN_aug':
        from .DnCNN_model_aug import DnCNN_model_aug as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
