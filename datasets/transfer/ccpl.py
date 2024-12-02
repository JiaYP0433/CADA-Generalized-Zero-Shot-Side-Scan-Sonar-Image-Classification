import torch
import torch.nn as nn
from .utils import VGGEncoder, VGGDecoder, nor_mean_std, nor_mean, calc_cov, calc_mean_std, WCT


class SCT(nn.Module):
    def __init__(self):
        super(SCT, self).__init__()
        self.cnet = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 128, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.snet = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 128, 3, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.uncompress = nn.Conv2d(32, 512, 1, 1, 0)

    def forward(self, content, style, mode='none'):
        cF_nor = nor_mean_std(content)

        if mode in ['custom']:
            mask_c = torch.mean(content, dim=1, keepdim=True)
            mask_c = (mask_c - mask_c.min()) / (mask_c.max() - mask_c.min() + 0.00001)
            cF_nor = cF_nor * mask_c

        sF_nor, smean = nor_mean(style)
        cF = self.cnet(cF_nor)
        sF = self.snet(sF_nor)
        b, c, w, h = cF.size()
        s_cov = calc_cov(sF)
        gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b,c,w,h)
        gF = self.uncompress(gF)
        gF = gF + smean.expand(cF_nor.size())

        # if mode in ['custom']:
        #     gF = gF * mask_c

        return gF


class CplTrans(nn.Module):
    """ This is the versatile style transformer module """
    def __init__(self, mode='none'):
        super().__init__()
        style_mode = 'ccpl'
        enc_path = 'datasets/transfer/params/{}/vgg_normalised.pth'.format(style_mode)
        if mode in ['custom']:
            trans_path = 'datasets/transfer/params/{}/mybackoff_sct_iter_40000.pth'.format(style_mode)
            dec_path = 'datasets/transfer/params/{}/mybackoff_decoder_iter_40000.pth'.format(style_mode)
        else:
            trans_path = 'datasets/transfer/params/{}/sct_iter_160000.pth'.format(style_mode)
            dec_path = 'datasets/transfer/params/{}/decoder_iter_160000.pth'.format(style_mode)

        self.encoder = VGGEncoder(vgg=torch.load(enc_path), mode=style_mode)
        self.transformer = SCT()
        self.transformer.load_state_dict(torch.load(trans_path))
        self.decoder = VGGDecoder(torch.load(dec_path), style_mode=style_mode, transfer_mode=mode)
        self.encoder.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.transfer_mode = mode

    def forward(self, content, style, alpha=1.0):
        assert (0.0 <= alpha <= 1.0)
        content_f, pool_idx = self.encoder(content)
        style_f, _ = self.encoder(style)

        feat = self.transformer(content_f, style_f, mode=self.transfer_mode)
        feat = self.decoder(feat, pool_dix=pool_idx) if self.transfer_mode == 'li' else self.decoder(feat)

        return feat
