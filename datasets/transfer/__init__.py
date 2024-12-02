from .photowct import PhotoWCT
from .liwct import SonarNoiseWCT
from .stytr import StyTrans
from .asepa import AsePaTrans
from .gcea import GCEATrans
from .ccpl import CplTrans

transfer_list = {
    "photowct": PhotoWCT,
    "liwct": SonarNoiseWCT,
    "stytr": StyTrans,
    "asepa": AsePaTrans,
    "gcea": GCEATrans,
    "ccpl": CplTrans,
}


def build_transfer(style_mode, mode='none'):
    return transfer_list[style_mode](mode=mode) 
