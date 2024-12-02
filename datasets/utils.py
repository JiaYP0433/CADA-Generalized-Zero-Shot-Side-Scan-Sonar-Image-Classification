import os
import random
# import gdown
import tarfile
import zipfile
import torch

import os.path as osp
import torch.nn as nn
import numpy as np

from PIL import Image
from collections import defaultdict, OrderedDict
from torchvision import transforms


def listdir_nohidden(path, sort=False):
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


class Datum:
    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    dataset_dir = ''
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None, style=None):
        self._train_x = train_x
        self._train_u = train_u
        self._val = val
        self._test = test
        self._style = style
        tx = []
        for t in train_x:
            tx = tx + t
        self._num_classes = self.get_num_classes(tx)
        self._lab2cname, self._classnames = self.get_lab2cname(tx)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def style(self):
        return self._style

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if len(data_sources) == 1:
            data_sources = data_sources[0]
        num_w = [t.__len__() for t in data_sources]
        if num_shots < 1:
            return data_sources, num_w
        else:
            print(f'Creating a {num_shots}-shot dataset')
            output = []
            for i, data_source in enumerate(data_sources):
                if i > 0:
                    n_shot = min(num_w[i], num_shots)
                    output.append(data_source[:n_shot])
                    num_w[i] = n_shot
                else:
                    output.append(data_source)

            return output, num_w

    def split_dataset_by_label(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.domain].append(item)

        return output


class ResizeAPad(nn.Module):
    def __init__(self, target_size, scale=1., scaling=False, interpolation=None):
        super().__init__()
        self.target_size = target_size
        self.scale = scale
        self.scaling = scaling
        self.interpolation = interpolation

    def forward(self, img):
        width, height = img.size
        if self.scaling is not False:
            ratio = self.scale
            scaling_width = int(width * np.random.uniform(ratio, 1.)) if np.random.uniform() < 0.5 else \
                int(width / np.random.uniform(ratio, 1.))
            scaling_height = int(height * np.random.uniform(ratio, 1.)) if np.random.uniform() < 0.5 else \
                int(height / np.random.uniform(ratio, 1.))
            tran_resize = transforms.Resize((scaling_height, scaling_width), interpolation=self.interpolation)
            img = tran_resize(img)
            if scaling_width < width or scaling_height < height:
                left = (max(scaling_width, width) - scaling_width) // 2
                up = (max(scaling_height, height) - scaling_height) // 2
                right = max(scaling_width, width) - left - scaling_width
                down = max(scaling_height, height) - up - scaling_height
                pad0 = transforms.Pad((left, up, right, down), fill=0, padding_mode='edge')
                img = pad0(img)
                left2 = max(scaling_width - width, 0) // 2
                up2 = max(scaling_height - height, 0) // 2
                img = Image.fromarray(np.array(img)[up2:up2+height, left2:left2+width])

        width, height = img.size
        scale = min(self.target_size / width, self.target_size / height)
        new_width, new_height = int(width * scale), int(height * scale)
        tran_resize = transforms.Resize((new_height, new_width), interpolation=self.interpolation)
        resized_img = tran_resize(img)

        left, up = (self.target_size - new_width) // 2, (self.target_size - new_height) // 2
        right, down = self.target_size - left - new_width, self.target_size - up - new_height
        pad = transforms.Pad((left, up, right, down), fill=0, padding_mode='edge')
        padded_img = pad(resized_img)

        return padded_img


def read_image(path):
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def _convert_image_to_gray(image):
    return image.convert("L")


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred[:, 0]
        correct = pred.eq(target).contiguous()
        # global, mean
        correct_k = correct.float().sum(0, keepdim=True)
        correct_m = torch.zeros(1)
        t_list = target.unique().tolist()
        for t in t_list:
            c = target == t
            correct_m += correct[c].sum(0, keepdim=True) / c.sum(0, keepdim=True)
        correct_m /= t_list.__len__()

        return correct_k.mul_(100.0 / batch_size), correct_m.mul_(100.0)
