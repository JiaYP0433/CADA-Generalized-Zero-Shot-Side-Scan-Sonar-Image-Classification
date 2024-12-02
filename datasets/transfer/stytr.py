import torch
import torchvision
import torch.nn as nn
import collections.abc as container_abcs
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from torch import Tensor
from .utils import \
    VGGDecoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder
from collections import OrderedDict
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.new_ps = nn.Conv2d(512, 512, (1, 1))
        self.averagepooling = nn.AdaptiveAvgPool2d(18)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, style, mask, content, pos_embed_s):
        # Content-Aware Positional Embedding (CAPE)
        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_embed_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])
        # flatten NxCxHxW to HWxNxC
        style = style.flatten(2).permute(2, 0, 1)
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1)
        content = content.flatten(2).permute(2, 0, 1)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)
        style = self.encoder_s(style, src_key_padding_mask=mask, pos=pos_embed_s)
        content = self.encoder_c(content, src_key_padding_mask=mask, pos=pos_embed_c)
        hs = self.decoder(content, style, memory_key_padding_mask=mask,
                          pos=pos_embed_s, query_pos=pos_embed_c)[0]
        # HWxNxC to NxCxHxW
        N, B, C = hs.shape
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1, H)
        return hs


class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    def __init__(self, mode='none'):
        super().__init__()
        style_mode = 'stytr'
        enc_path = 'datasets/transfer/params/{}/embedding_iter_160000.pth'.format(style_mode)
        trans_path = 'datasets/transfer/params/{}/transformer_iter_160000.pth'.format(style_mode)
        dec_path = 'datasets/transfer/params/{}/decoder_iter_160000.pth'.format(style_mode)

        self.embedding = PatchEmbed()
        self.transformer = Transformer()
        self.decoder = VGGDecoder(torch.load(dec_path), style_mode=style_mode, transfer_mode=mode)
        new_state_dict = OrderedDict()
        state_dict = torch.load(enc_path)
        for k, v in state_dict.items():
            namekey = k
            new_state_dict[namekey] = v
        self.embedding.load_state_dict(new_state_dict)
        new_state_dict = OrderedDict()
        state_dict = torch.load(trans_path)
        for k, v in state_dict.items():
            namekey = k
            new_state_dict[namekey] = v
        self.transformer.load_state_dict(new_state_dict)

        self.embedding.eval()
        self.transformer.eval()
        self.decoder.eval()
        self.transfer_mode = mode

    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        hs = self.transformer(style, None, content, None)
        feat = self.decoder(hs)

        return feat
