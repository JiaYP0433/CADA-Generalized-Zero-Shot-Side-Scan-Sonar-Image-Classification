import torch
import torch.nn as nn
from .utils import VGGEncoder, VGGDecoder, nor_mean_std


class GCEA_Block(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio=0.25,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(GCEA_Block, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        self.mk = nn.Linear(self.inplanes, 64, bias=False)
        self.mv = nn.Linear(64, self.inplanes, bias=False)
        self.softmax = nn.Softmax(dim=1)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, Fs, F_s):
        batch, channel, height, width = Fs.size()
        if self.pooling_type == 'att':
            input_Fs = Fs
            input_Fs = input_Fs.view(batch, channel, height * width)   # [N, C, H * W]
            input_Fs = input_Fs.unsqueeze(1)    # [N, 1, C, H * W]
            context_mask = self.conv_mask(F_s)  # [N, 1, H, W]
            context_mask = context_mask.view(batch, 1, height * width)  # [N, 1, H * W]
            context_mask = self.softmax(context_mask)   # [N, 1, H * W]
            context_mask = context_mask.unsqueeze(-1)   # [N, 1, H * W, 1]
            context = torch.matmul(input_Fs, context_mask)  # [N, 1, C, H * W]# [N, 1, H * W, 1]
            context = context.view(batch, channel, 1, 1)    # [N, C, 1, 1]
        else:
            context = self.avg_pool(F_s)    # [N, C, 1, 1]

        return context

    def forward(self, content, style):
        Fs = style
        F_s = nor_mean_std(style)
        context = self.spatial_pool(Fs, F_s)    # [N, C, 1, 1]
        batch, channel, height, width = content.size()
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))    # [N, C, 1, 1]
            context = content * channel_mul_term    # [N, C, 1, 1]
            context = context.view(batch, height * width, channel)  # [N, H*W,C]
            attn = self.mk(context)  # bs,n,c
            attn = self.softmax(attn)  # bs,n,c
            attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,c
            out = self.mv(attn)  # bs,n,c
            out = out.view(batch, channel, height, width)

        if self.channel_add_conv is not None:
            context = content + context  # [N, C, 1, 1]
            context = context.view(batch, height * width, channel)  # [N, H*W,C]
            attn = self.mk(context)  # bs,n,c
            attn = self.softmax(attn)  # bs,n,c
            attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,c
            out = self.mv(attn)  # bs,n,c
            out = out.view(batch, channel, height, width)

        return out


class GCEA(nn.Module):
    def __init__(self, in_planes):
        super(GCEA, self).__init__()
        self.sanet4_1 = GCEA_Block(inplanes=in_planes)
        self.sanet5_1 = GCEA_Block(inplanes=in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        self.upsample5_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]), mode='nearest')
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) +
                                                   self.upsample5_1(self.sanet5_1(content5_1, style5_1))))


class GCEATrans(nn.Module):
    """ This is the style transform transformer module """
    def __init__(self, mode='none'):
        super().__init__()
        style_mode = 'gcea'
        enc_path = 'datasets/transfer/params/{}/vgg_normalised.pth'.format(style_mode)
        trans_path = 'datasets/transfer/params/{}/transformer_iter_160000.pth'.format(style_mode)
        dec_path = 'datasets/transfer/params/{}/decoder_iter_160000.pth'.format(style_mode)

        self.encoder = VGGEncoder(vgg=torch.load(enc_path), mode=style_mode)
        self.transformer = GCEA(in_planes=512)
        self.transformer.load_state_dict(torch.load(trans_path))
        self.decoder = VGGDecoder(torch.load(dec_path), style_mode=style_mode, transfer_mode=mode)
        self.encoder.eval()
        self.transformer.eval()
        self.decoder.eval()

    def forward(self, content, style):
        content4_1 = self.encoder.get_features(content, 4)
        content5_1 = self.encoder.get_features(content, 5)
        style4_1 = self.encoder.get_features(style, 4)
        style5_1 = self.encoder.get_features(style, 5)

        hs = self.transformer(content4_1, style4_1, content5_1, style5_1)
        feat = self.decoder(hs)

        return feat
