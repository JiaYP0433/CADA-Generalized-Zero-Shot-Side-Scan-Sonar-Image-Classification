import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .utils import VGGEncoder, VGGDecoder, nor_mean_std, gram_matrix


def size_arrange(x):
    x_w, x_h = x.size(2), x.size(3)

    if (x_w % 2) != 0:
        x_w = (x_w // 2) * 2
    if (x_h % 2) != 0:
        x_h = (x_h // 2) * 2
    if (x_h > 1024) or (x_w > 1024):
        old_x_w = x_w
        x_w = x_w // 2
        x_h = int(x_h * x_w / old_x_w)

    return F.interpolate(x, size=(x_w, x_h))


def SwitchWhiten2d(x):
    N, C, H, W = x.size()
    in_data = x.view(N, C, -1)
    eye = in_data.data.new().resize_(C, C)
    eye = torch.nn.init.eye_(eye).view(1, C, C).expand(N, C, C)
    # calculate other statistics
    mean_in = in_data.mean(-1, keepdim=True)
    x_in = in_data - mean_in
    # (N x g) x C x C
    cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)
    mean = mean_in
    cov = cov_in + 1e-5 * eye
    # perform whitening using Newton's iteration
    Ng, c, _ = cov.size()
    P = torch.eye(c).to(cov).expand(Ng, c, c)
    rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
    cov_N = cov * rTr
    for k in range(5):
        P = torch.baddbmm(P, torch.matrix_power(P, 3), cov_N, beta=1.5, alpha=-0.5)
    wm = P.mul_(rTr.sqrt())
    x_hat = torch.bmm(wm, in_data - mean)

    return x_hat, wm, mean


def Bw_wct_core(content_feat, style_feat, weight=1, registers=None, device='cpu'):
    N, C, H, W = content_feat.size()
    cont_min = content_feat.min().item()
    cont_max = content_feat.max().item()
    whiten_cF, _, _ = SwitchWhiten2d(content_feat)
    _, wm_s, s_mean = SwitchWhiten2d(style_feat)

    targetFeature = torch.bmm(torch.inverse(wm_s), whiten_cF)
    targetFeature = targetFeature.view(N, C, H, W)
    targetFeature = targetFeature + s_mean.unsqueeze(2).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature


def feature_wct_simple(content_feat, style_feat, alpha=1):
    target_feature = Bw_wct_core(content_feat, style_feat)
    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat

    return target_feature


class AdaptiveMultiAdaAttN_v2(nn.Module):
    def __init__(self, in_planes, out_planes, max_sample=256 * 256, query_planes=None, key_planes=None):
        super(AdaptiveMultiAdaAttN_v2, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(query_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)

        b, style_c, style_h, style_w = H.size()
        H = torch.nn.functional.interpolate(H, (h_g, w_g), mode='bicubic')
        if h_g * w_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(h_g * w_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, h_g * w_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, h_g * w_g).transpose(1, 2).contiguous()
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        _, _, ch, cw = content.size()
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return std * nor_mean_std(content) + mean, S


class AdaptiveMultiAttn_Transformer_v2(nn.Module):
    def __init__(self, in_planes, out_planes, query_planes=None, key_planes=None, shallow_layer=False):
        super(AdaptiveMultiAttn_Transformer_v2, self).__init__()
        self.attn_adain_4_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes,
                                                      query_planes=query_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaptiveMultiAdaAttN_v2(in_planes=in_planes, out_planes=out_planes,
                                                      query_planes=query_planes, key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(out_planes, out_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, content4_1_key, style4_1_key, content5_1_key,
                style5_1_key, seed=None):
        feature_4_1, attn_4_1 = self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed)
        feature_5_1, attn_5_1 = self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed)
        stylized_results = self.merge_conv(self.merge_conv_pad(
            feature_4_1 + nn.functional.interpolate(feature_5_1, size=(feature_4_1.size(2), feature_4_1.size(3)))))

        return stylized_results, feature_4_1, feature_5_1, attn_4_1, attn_5_1


class AsePaTrans(nn.Module):
    """ This is the aesthetic pattern-aware style transfer module """
    def __init__(self, mode='none'):
        super().__init__()
        style_mode = 'asepa'
        enc_path = 'datasets/transfer/params/{}/vgg_normalised.pth'.format(style_mode)
        trans_path = 'datasets/transfer/params/{}/transformer_model.pth'.format(style_mode)
        dec_path = 'datasets/transfer/params/{}/dec_model.pth'.format(style_mode)
        query_channels = 512
        key_channels = 512 + 256 + 128 + 64

        self.encoder = VGGEncoder(vgg=torch.load(enc_path), mode=style_mode)
        self.transformer = AdaptiveMultiAttn_Transformer_v2(in_planes=512, out_planes=512, query_planes=query_channels,
                                                            key_planes=key_channels)
        self.decoder = VGGDecoder(style_mode=style_mode, transfer_mode=mode)
        self.transformer.load_state_dict(torch.load(trans_path)['state_dict'])
        self.decoder.load_state_dict(torch.load(dec_path)['state_dict'])
        self.encoder.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.transfer_mode = mode

    def encode(self, x, skips):
        o = self.encoder.encode(x, skips)
        return o

    def decode(self, stylized_feat, content_skips, style_skips):
        return self.decoder.decode(stylized_feat, content_skips, style_skips)

    def adaptive_get_keys(self, feat_skips, start_layer_idx, last_layer_idx, target_feat, mode='none'):
        B, C, th, tw = target_feat.shape
        results = []
        target_conv_layer = 'conv' + str(last_layer_idx) + '_1'
        _, _, h, w = feat_skips[target_conv_layer].shape
        for i in range(start_layer_idx, last_layer_idx + 1):
            target_conv_layer = 'conv' + str(i) + '_1'
            feat = feat_skips[target_conv_layer]
            if mode in ['custom']:
                mask_c = torch.mean(feat, dim=1, keepdim=True)
                mask_c = (mask_c - mask_c.min()) / (mask_c.max() - mask_c.min() + 0.00001)
                feat = feat * mask_c
            if i == last_layer_idx:
                results.append(nor_mean_std(feat))
            else:
                results.append(nor_mean_std(nn.functional.interpolate(feat, (h, w))))

        return nn.functional.interpolate(torch.cat(results, dim=1), (th, tw))

    def forward(self, content, style, mode=''):
        gray_content = torchvision.transforms.functional.rgb_to_grayscale(content).repeat(1, 3, 1, 1)
        gray_style = torchvision.transforms.functional.rgb_to_grayscale(style).repeat(1, 3, 1, 1)
        adaptive_alpha = ((self.adaptive_gram_weight(gray_style, 1, 8) + self.adaptive_gram_weight(gray_style, 2, 8) +
                           self.adaptive_gram_weight(gray_style, 3, 8)) / 3).unsqueeze(1).cuda()
        content_ = size_arrange(gray_content)
        style_ = size_arrange(style)

        content_feat, content_skips = content_, {}
        style_feat, style_skips = style_, {}
        content_feat = self.encode(content_feat, content_skips)
        style_feat = self.encode(style_feat, style_skips)

        local_transformed_feature, attn_style_4_1, attn_style_5_1, attn_map_4_1, attn_map_5_1 = self.transformer(
            content_skips['conv4_1'], style_skips['conv4_1'], content_skips['conv5_1'], style_skips['conv5_1'],
            self.adaptive_get_keys(content_skips, 4, 4, target_feat=content_skips['conv4_1'], mode=self.transfer_mode),
            self.adaptive_get_keys(style_skips, 1, 4, target_feat=style_skips['conv4_1']),
            self.adaptive_get_keys(content_skips, 5, 5, target_feat=content_skips['conv5_1'], mode=self.transfer_mode),
            self.adaptive_get_keys(style_skips, 1, 5, target_feat=style_skips['conv5_1']))
        content = content_skips['conv4_1']
        if self.transfer_mode in ['custom']:
            mask_c = torch.mean(content, dim=1, keepdim=True)
            mask_c = (mask_c - mask_c.min()) / (mask_c.max() - mask_c.min() + 0.00001)
            content = content * mask_c
        global_transformed_feat = feature_wct_simple(content, style_skips['conv4_1'])
        transformed_feature = global_transformed_feat * (
                1 - adaptive_alpha.unsqueeze(-1).unsqueeze(-1)) + adaptive_alpha.unsqueeze(-1).unsqueeze(
            -1) * local_transformed_feature
        if self.transfer_mode in ['custom']:
            mask_c = torch.mean(transformed_feature, dim=1, keepdim=True)
            mask_c = (mask_c - mask_c.min()) / (mask_c.max() - mask_c.min() + 0.00001)
            transformed_feature = transformed_feature * mask_c
        stylization = self.decode(transformed_feature, content_skips, style_skips)

        return stylization

    def extract_image_patches(self, x, kernel, stride=1):
        b, c, h, w = x.shape
        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.contiguous().view(b, c, -1, kernel, kernel)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()

        return patches.view(b, -1, c, kernel, kernel)

    def adaptive_gram_weight(self, image, level, ratio):
        if level == 0:
            encoded_features = image
        else:
            encoded_features = self.encoder.get_features(image, level)  # B x C x W x H
        global_gram = gram_matrix(encoded_features)

        B, C, w, h = encoded_features.size()
        target_w, target_h = w // ratio, h // ratio
        # assert target_w==target_h
        patches = self.extract_image_patches(encoded_features, target_w, target_h)
        _, patches_num, _, _, _ = patches.size()
        cos = torch.nn.CosineSimilarity(eps=1e-6)

        intra_gram_statistic = []
        inter_gram_statistic = []
        comb = torch.combinations(torch.arange(patches_num), r=2)
        if patches_num >= 10:
            sampling_num = int(comb.size(0) * 0.05)
        else:
            sampling_num = comb.size(0)
        for idx in range(B):
            if patches_num < 2:
                continue
            cos_gram = []
            for patch in range(0, patches_num):
                cos_gram.append(cos(global_gram, gram_matrix(patches[idx][patch].unsqueeze(0))).mean().item())

            intra_gram_statistic.append(torch.tensor(cos_gram))
            cos_gram = []
            for idxes in random.choices(list(comb), k=sampling_num):
                cos_gram.append(cos(gram_matrix(patches[idx][idxes[0]].unsqueeze(0)),
                                    gram_matrix(patches[idx][idxes[1]].unsqueeze(0))).mean().item())

            inter_gram_statistic.append(torch.tensor(cos_gram))

        intra_gram_statistic = torch.stack(intra_gram_statistic).mean(dim=1)
        inter_gram_statistic = torch.stack(inter_gram_statistic).mean(dim=1)
        results = (intra_gram_statistic + inter_gram_statistic) / 2
        # For boosting value
        results = (1 / (1 + torch.exp(-10 * (results - 0.6))))

        return results


