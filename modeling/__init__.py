from modeling.localization import Localization
from modeling.localization_HiSA import Localization_HiSA

import torch


def build(cfg):
    if cfg.MODEL_NAME == 'HiSA':
        return Localization_HiSA(cfg)

