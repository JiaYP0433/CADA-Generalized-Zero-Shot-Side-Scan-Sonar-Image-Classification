import argparse
from .photowct import PhotoWCT
from .liwct import SonarNoiseWCT
from .stytr import StyTrans
from .asepa import AsePaTrans
from .ccpl import CplTrans

transfer_list = {
    "photowct": PhotoWCT,
    "liwct": SonarNoiseWCT,
    "stytr": StyTrans,
    "asepa": AsePaTrans,
    "ccpl": CplTrans,
}


def build_transfer(style_mode, mode='none'):
    return transfer_list[style_mode](mode=mode)    # cfg, style_mode=style_mode
