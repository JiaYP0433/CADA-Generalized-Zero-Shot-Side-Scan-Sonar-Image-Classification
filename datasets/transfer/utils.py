import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def nor_mean_std(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    nor_feat = (feat - mean.expand(size)) / std.expand(size)
    return nor_feat


def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean


def nor_mean(feat):
    size = feat.size()
    mean = calc_mean(feat)
    nor_feat = feat - mean.expand(size)
    return nor_feat, mean


def calc_cov(feat):
    feat = feat.flatten(2, 3)
    f_cov = torch.bmm(feat, feat.permute(0,2,1)).div(feat.size(2))
    return f_cov


def WCT(s_feat, t_feat, alpha=1.):
    sf_c, sf_h, sf_w = s_feat.size(0), s_feat.size(1), s_feat.size(2)
    tf_c, tf_h, tf_w = t_feat.size(0), t_feat.size(1), t_feat.size(2)
    s_feat = s_feat.view(sf_c, -1).clone()
    t_feat = t_feat.view(tf_c, -1).clone()

    sF_Size = s_feat.size()
    s_mean = torch.mean(s_feat, 1)  # c x (h x w)
    s_mean = s_mean.unsqueeze(1).expand_as(s_feat)
    s_feat = s_feat - s_mean

    iden = torch.eye(sF_Size[0]).cuda()  # .double()
    s_Conv = torch.mm(s_feat, s_feat.t()).div(sF_Size[1] - 1) + iden
    s_u, s_e, s_v = torch.svd(s_Conv, some=False)
    k_s = sF_Size[0]
    for i in range(sF_Size[0] - 1, -1, -1):
        if s_e[i] >= 0.00001:
            k_s = i + 1
            break

    tF_Size = t_feat.size()
    t_mean = torch.mean(t_feat, 1)
    t_feat = t_feat - t_mean.unsqueeze(1).expand_as(t_feat)
    t_Conv = torch.mm(t_feat, t_feat.t()).div(tF_Size[1] - 1)
    t_u, t_e, t_v = torch.svd(t_Conv, some=False)
    k_t = tF_Size[0]
    for i in range(tF_Size[0] - 1, -1, -1):
        if t_e[i] >= 0.00001:
            k_t = i + 1
            break

    s_d = (s_e[0:k_s]).pow(-0.5)
    step1 = torch.mm(s_v[:, 0:k_s], torch.diag(s_d))
    step2 = torch.mm(step1, (s_v[:, 0:k_s].t()))
    whiten_sF = torch.mm(step2, s_feat)

    t_d = (t_e[0:k_t]).pow(0.5)
    t_Feature = torch.mm(torch.mm(torch.mm(t_v[:, 0:k_t], torch.diag(t_d)), (t_v[:, 0:k_t].t())), whiten_sF)
    t_Feature = t_Feature + t_mean.unsqueeze(1).expand_as(t_Feature)

    t_Feature = t_Feature.view_as(s_feat)
    ccsF = alpha * t_Feature + (1.0 - alpha) * s_feat
    ccsF = ccsF.float().reshape(sf_c, sf_h, sf_w)

    return ccsF


class VGGEncoder(nn.Module):
    def __init__(self, vgg=None, mode='ccpl', level=5):
        super(VGGEncoder, self).__init__()
        self.level = level
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2)
        self.mode = mode

        # ----- Level0 ----- #
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        # ----- Level1 ----- #
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        if vgg is not None:
            self.conv0.weight = nn.Parameter(torch.FloatTensor(vgg['0.weight']))
            self.conv0.bias = nn.Parameter(torch.FloatTensor(vgg['0.bias']))
            self.conv1_1.weight = nn.Parameter(torch.FloatTensor(vgg['2.weight']))
            self.conv1_1.bias = nn.Parameter(torch.FloatTensor(vgg['2.bias']))
        if level < 2:
            return

        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # ----- Level2 ----- #
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        if vgg is not None:
            self.conv1_2.weight = nn.Parameter(torch.FloatTensor(vgg['5.weight']))
            self.conv1_2.bias = nn.Parameter(torch.FloatTensor(vgg['5.bias']))
            self.conv2_1.weight = nn.Parameter(torch.FloatTensor(vgg['9.weight']))
            self.conv2_1.bias = nn.Parameter(torch.FloatTensor(vgg['9.bias']))
        if level < 3:
            return

        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # ----- Level3 ----- #
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        if vgg is not None:
            self.conv2_2.weight = nn.Parameter(torch.FloatTensor(vgg['12.weight']))
            self.conv2_2.bias = nn.Parameter(torch.FloatTensor(vgg['12.bias']))
            self.conv3_1.weight = nn.Parameter(torch.FloatTensor(vgg['16.weight']))
            self.conv3_1.bias = nn.Parameter(torch.FloatTensor(vgg['16.bias']))
        if level < 4:
            return

        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # ----- Level4 ----- #
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        if vgg is not None:
            self.conv3_2.weight = nn.Parameter(torch.FloatTensor(vgg['19.weight']))
            self.conv3_2.bias = nn.Parameter(torch.FloatTensor(vgg['19.bias']))
            self.conv3_3.weight = nn.Parameter(torch.FloatTensor(vgg['22.weight']))
            self.conv3_3.bias = nn.Parameter(torch.FloatTensor(vgg['22.bias']))
            self.conv3_4.weight = nn.Parameter(torch.FloatTensor(vgg['25.weight']))
            self.conv3_4.bias = nn.Parameter(torch.FloatTensor(vgg['25.bias']))
            self.conv4_1.weight = nn.Parameter(torch.FloatTensor(vgg['29.weight']))
            self.conv4_1.bias = nn.Parameter(torch.FloatTensor(vgg['29.bias']))
        if level < 5:
            return

        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # ----- Level5 ----- #
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)
        if vgg is not None and level >= 5:
            self.conv4_2.weight = nn.Parameter(torch.FloatTensor(vgg['32.weight']))
            self.conv4_2.bias = nn.Parameter(torch.FloatTensor(vgg['32.bias']))
            self.conv4_3.weight = nn.Parameter(torch.FloatTensor(vgg['35.weight']))
            self.conv4_3.bias = nn.Parameter(torch.FloatTensor(vgg['35.bias']))
            self.conv4_4.weight = nn.Parameter(torch.FloatTensor(vgg['38.weight']))
            self.conv4_4.bias = nn.Parameter(torch.FloatTensor(vgg['38.bias']))
            self.conv5_1.weight = nn.Parameter(torch.FloatTensor(vgg['42.weight']))
            self.conv5_1.bias = nn.Parameter(torch.FloatTensor(vgg['42.bias']))

    def forward(self, x, s=None):
        pool_idx = []
        if self.mode in ['asepa']:
            skips, s_skips = {}, {}
            for level in [1, 2, 3, 4]:
                x = self.encode(x, skips, s=s, s_skips=s_skips, mode=self.mode)
            return x
        else:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            if self.mode not in ['asepa', 'ccpl'] and self.level < 2:
                return out
            out = self.relu(self.conv1_2(self.pad(out)))
            out, pool_idx1 = self.maxpool1(out)
            pool_idx.append(pool_idx1)
            out = self.relu(self.conv2_1(self.pad(out)))
            if self.mode not in ['asepa', 'ccpl'] and self.level < 3:
                return out, pool_idx
            out = self.relu(self.conv2_2(self.pad(out)))
            out, pool_idx2 = self.maxpool2(out)
            pool_idx.append(pool_idx2)
            out = self.relu(self.conv3_1(self.pad(out)))
            if self.mode not in ['asepa', 'ccpl'] and self.level < 4:
                return out, pool_idx
            out = self.relu(self.conv3_2(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))
            out, pool_idx3 = self.maxpool3(out)
            pool_idx.append(pool_idx3)
            out = self.relu(self.conv4_1(self.pad(out)))

            return out, pool_idx

    def forward_multiple(self, x):
        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
        if self.level < 2:
            return out
        out1 = out
        out = self.relu(self.conv1_2(self.pad(out)))
        out, pool_idx1 = self.maxpool1(out)
        out = self.relu(self.conv2_1(self.pad(out)))
        if self.level < 3:
            return out, out1
        out2 = out
        out = self.relu(self.conv2_2(self.pad(out)))
        out, pool_idx2 = self.maxpool2(out)
        out = self.relu(self.conv3_1(self.pad(out)))
        if self.level < 4:
            return out, out2, out1
        out3 = out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        out, pool_idx3 = self.maxpool3(out)
        out = self.relu(self.conv4_1(self.pad(out)))

        return out, out3, out2, out1

    def encode(self, x, skips):
        is_maxpool = False

        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
        skips['conv1_1'] = out

        out = self.relu(self.conv1_2(self.pad(out)))
        skips['conv1_2'] = out
        resize_w, resize_h = out.size(2), out.size(3)
        pooled_feature = self.pool(out)
        HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
        skips['pool1'] = HH
        if is_maxpool:
            pooled_feature, _ = self.maxpool1(out)

        out = self.relu(self.conv2_1(self.pad(pooled_feature)))
        skips['conv2_1'] = out
        out = self.relu(self.conv2_2(self.pad(out)))
        skips['conv2_2'] = out
        resize_w, resize_h = out.size(2), out.size(3)
        pooled_feature = self.pool(out)
        HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
        skips['pool2'] = HH
        if is_maxpool:
            pooled_feature, _ = self.maxpool2(out)

        out = self.relu(self.conv3_1(self.pad(pooled_feature)))
        skips['conv3_1'] = out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        skips['conv3_4'] = out
        resize_w, resize_h = out.size(2), out.size(3)
        pooled_feature = self.pool(out)
        HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
        skips['pool3'] = HH
        if is_maxpool:
            pooled_feature, _ = self.maxpool3(out)

        out = self.relu(self.conv4_1(self.pad(pooled_feature)))
        skips['conv4_1'] = out
        out = self.relu(self.conv4_2(self.pad(out)))
        out = self.relu(self.conv4_3(self.pad(out)))
        out = self.relu(self.conv4_4(self.pad(out)))
        skips['conv4_4'] = out
        resize_w, resize_h = out.size(2), out.size(3)
        pooled_feature = self.pool(out)
        HH = out - F.interpolate(pooled_feature, size=[resize_w, resize_h], mode='nearest')
        skips['pool4'] = HH
        if is_maxpool:
            pooled_feature, _ = self.maxpool4(out)

        out = self.relu(self.conv5_1(self.pad(pooled_feature)))
        skips['conv5_1'] = out

        return out

    def get_features(self, x, level):
        is_maxpool = False

        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
        if level == 1:
            return out

        out = self.relu(self.conv1_2(self.pad(out)))
        pooled_feature = self.pool(out)
        ##################################
        if is_maxpool:
            pooled_feature = self.maxpool1(out)
        out = self.relu(self.conv2_1(self.pad(pooled_feature)))
        if level == 2:
            return out

        out = self.relu(self.conv2_2(self.pad(out)))
        pooled_feature = self.pool(out)
        ##################################
        if is_maxpool:
            pooled_feature = self.maxpool2(out)
        out = self.relu(self.conv3_1(self.pad(pooled_feature)))
        if level == 3:
            return out

        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        pooled_feature = self.pool(out)
        ##################################
        if is_maxpool:
            pooled_feature = self.maxpool3(out)
        out = self.relu(self.conv4_1(self.pad(pooled_feature)))
        if level == 4:
            return out

        out = self.relu(self.conv4_2(self.pad(out)))
        out = self.relu(self.conv4_3(self.pad(out)))
        out = self.relu(self.conv4_4(self.pad(out)))
        pooled_feature = self.pool(out)
        ####################################
        if is_maxpool:
            pooled_feature = self.maxpool4(out)
        out = self.relu(self.conv5_1(self.pad(pooled_feature)))
        if level == 5:
            return out


class VGGDecoder(nn.Module):
    def __init__(self, de_vgg=None, style_mode='ccpl', transfer_mode='custom', level=4):
        super(VGGDecoder, self).__init__()
        self.level = level
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=False)
        self.nearest_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.max_upsample = nn.MaxUnpool2d(kernel_size=2, stride=2)

        if level > 3:
            self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
            self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
            self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        if level > 2:
            self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
            self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        if level > 1:
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        if level > 0:
            self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

        if style_mode in ['stytr', 'ccpl']:
            self.conv4_1.weight = nn.Parameter(torch.FloatTensor(de_vgg['1.weight']))
            self.conv4_1.bias = nn.Parameter(torch.FloatTensor(de_vgg['1.bias']))
            self.conv3_4.weight = nn.Parameter(torch.FloatTensor(de_vgg['5.weight']))
            self.conv3_4.bias = nn.Parameter(torch.FloatTensor(de_vgg['5.bias']))
            self.conv3_3.weight = nn.Parameter(torch.FloatTensor(de_vgg['8.weight']))
            self.conv3_3.bias = nn.Parameter(torch.FloatTensor(de_vgg['8.bias']))
            self.conv3_2.weight = nn.Parameter(torch.FloatTensor(de_vgg['11.weight']))
            self.conv3_2.bias = nn.Parameter(torch.FloatTensor(de_vgg['11.bias']))
            self.conv3_1.weight = nn.Parameter(torch.FloatTensor(de_vgg['14.weight']))
            self.conv3_1.bias = nn.Parameter(torch.FloatTensor(de_vgg['14.bias']))
            self.conv2_2.weight = nn.Parameter(torch.FloatTensor(de_vgg['18.weight']))
            self.conv2_2.bias = nn.Parameter(torch.FloatTensor(de_vgg['18.bias']))
            self.conv2_1.weight = nn.Parameter(torch.FloatTensor(de_vgg['21.weight']))
            self.conv2_1.bias = nn.Parameter(torch.FloatTensor(de_vgg['21.bias']))
            self.conv1_2.weight = nn.Parameter(torch.FloatTensor(de_vgg['25.weight']))
            self.conv1_2.bias = nn.Parameter(torch.FloatTensor(de_vgg['25.bias']))
            self.conv1_1.weight = nn.Parameter(torch.FloatTensor(de_vgg['28.weight']))
            self.conv1_1.bias = nn.Parameter(torch.FloatTensor(de_vgg['28.bias']))

        self.style_mode = style_mode
        self.transfer_mode = transfer_mode

    def forward(self, x, pool_dix=None, c_skips=None, s_skips=None, noise_scale=0.7):
        if self.style_mode in ['asepa']:
            x = self.decode(x, c_skips, s_skips)
            return x
        else:
            out = x
            if self.level > 3:
                o = self.relu(self.conv4_1(self.pad(out)))
                out = self.max_upsample(o, pool_dix[2]) if self.style_mode in ['photowct'] else \
                    self.max_nearest_upsample(o, pool_dix[2], ns=noise_scale) \
                    if self.transfer_mode in ['li'] or self.style_mode in ['liwct'] and \
                    self.style_mode not in ['stytr'] else self.nearest_upsample(o)    # +
                out = self.relu(self.conv3_4(self.pad(out)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_2(self.pad(out)))
            if self.level > 2:
                o = self.relu(self.conv3_1(self.pad(out)))
                out = self.max_upsample(o, pool_dix[1]) if self.style_mode in ['photowct'] else \
                    self.max_nearest_upsample(o, pool_dix[1], ns=noise_scale) \
                    if self.transfer_mode in ['li'] or self.style_mode in ['liwct'] and \
                    self.style_mode not in ['stytr'] else self.nearest_upsample(o)    # +
                out = self.relu(self.conv2_2(self.pad(out)))
            if self.level > 1:
                o = self.relu(self.conv2_1(self.pad(out)))
                out = self.max_upsample(o, pool_dix[0]) if self.style_mode in ['photowct'] else \
                    self.max_nearest_upsample(o, pool_dix[0], ns=noise_scale) \
                    if self.transfer_mode in ['li'] or self.style_mode in ['liwct'] and \
                    self.style_mode not in ['stytr'] else self.nearest_upsample(o)    # +
                out = self.relu(self.conv1_2(self.pad(out)))
            if self.level > 0:
                out = self.conv1_1(self.pad(out))

            return out

    def max_nearest_upsample(self, o, pool, ns=0.7):
        x = self.max_upsample(o, pool)
        n = self.nearest_upsample(o)
        n = n - x
        n = (n * torch.rand(n.shape).cuda()) * ns

        return x + n

    def decode(self, stylized_feat, content_skips, style_skips):
        out = self.relu(self.conv4_1(self.pad(stylized_feat)))
        resize_w, resize_h = content_skips['conv3_4'].size(2), content_skips['conv3_4'].size(3)
        unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
        out = unpooled_feat

        out = self.relu(self.conv3_4(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_2(self.pad(out)))
        out = WCT(out[0], style_skips['conv3_1'][0]).unsqueeze(0)

        out = self.relu(self.conv3_1(self.pad(out)))
        resize_w, resize_h = content_skips['conv2_2'].size(2), content_skips['conv2_2'].size(3)
        unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
        out = unpooled_feat
        out = self.relu(self.conv2_2(self.pad(out)))
        out = WCT(out[0], style_skips['conv2_1'][0]).unsqueeze(0)

        out = self.relu(self.conv2_1(self.pad(out)))
        resize_w, resize_h = content_skips['conv1_2'].size(2), content_skips['conv1_2'].size(3)
        unpooled_feat = F.interpolate(out, size=[resize_w, resize_h], mode='nearest')
        out = unpooled_feat
        out = self.relu(self.conv1_2(self.pad(out)))
        out = WCT(out[0], style_skips['conv1_1'][0]).unsqueeze(0)
        out = self.conv1_1(self.pad(out))

        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # d_model embedding dim
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, pos)
        v = memory
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)
