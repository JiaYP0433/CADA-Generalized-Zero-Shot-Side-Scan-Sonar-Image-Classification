import torch
import cv2
import math
import itertools
from .utils import DatasetBase, OrderedDict, listdir_nohidden, Datum, transforms, nn, os, np, random, Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from torchvision.transforms import functional


def _convert_channel(image):
    return image.repeat(3, 1, 1)


def _convert_torch_to_pil(image):
    return transforms.ToPILImage()(image)


# class Laplacian(nn.Module):
#     def __init__(self):
#         super(Laplacian, self).__init__()
#         self.kernel = nn.Parameter(torch.tensor([[-0.707, -1.0, -0.707],
#                                                  [-1.0, 6.828, -1.0],
#                                                  [-0.707, -1.0, -0.707]], dtype=torch.float))
#
#     def forward(self, x):
#         lap = F.conv2d(x.unsqueeze(0), self.kernel.unsqueeze(0).unsqueeze(0).to(x.device), padding=1)[0]
#         lap[:, 0, :] = 0.
#         lap[:, -1, :] = 0.
#         lap[:, :, 0] = 0.
#         lap[:, :, -1] = 0.
#         return lap


# class RandomRotation(nn.Module):
#     def __init__(self, degrees, p=0.2, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
#         super().__init__()
#         self.rr = transforms.RandomRotation(
#             degrees, interpolation=interpolation, expand=expand, center=center, fill=fill)
#         self.p = p
#
#     def forward(self, img):
#         return self.rr(img) if torch.rand(1) < self.p else img


class ResizeAPad(nn.Module):
    def __init__(self, target_size, interpolation=None):
        super().__init__()
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img, w_rand=1., h_rand=1.):
        width, height = img.size
        scaling_width = int(width * w_rand)
        scaling_height = int(height * h_rand)
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


class SCL(nn.Module):
    def __init__(self, target_size, interpolation=None):
        super().__init__()
        self.resize_pad = ResizeAPad(target_size, interpolation=interpolation)
        self.tensor = transforms.ToTensor()

    def __call__(self, img, w_rand=1., h_rand=1.):
        img = self.resize_pad(img)
        img = self.tensor(img)

        return img


class SSSg(DatasetBase):
    def __init__(self, folder='datasets/gzsss_data', num_shots=0):
        self.train_dir = os.path.join(folder, 'train')
        self.test_dir = os.path.join(folder, 'test')
        text_file = os.path.join(folder, "classnames.txt")
        classnames = self.read_classnames(text_file)
        train, test = self.read_data(classnames)
        train, num_w = self.generate_fewshot_dataset(train, num_shots=num_shots)
        self.num_w = np.array(num_w)
        self.seenclasses = np.array([0, 2])
        self.unseenclasses = np.array([1, 3])
        super().__init__(train_x=train, test=test)

    @staticmethod
    def read_classnames(text_file):
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, novelty=False):
        folders = sorted(f.name for f in os.scandir(self.train_dir) if f.is_dir())
        train_items = [[] for _ in range(classnames.__len__())]
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(self.train_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(self.train_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                train_items[label].append(item)

        folders = sorted(f.name for f in os.scandir(self.test_dir) if f.is_dir())
        test_items = []
        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(self.test_dir, folder))
            classname = classnames[folder]
            if novelty is False and classname == 'others':
                continue
            for imname in imnames:
                impath = os.path.join(self.test_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                test_items.append(item)

        return train_items, test_items


class RS2SSS(nn.Module):
    def __init__(self, target_size, scale_mul=(0.05, 0.05, 0.1), scale_gauss=20):
        super().__init__()
        self.target_size = target_size
        self.scale_mul = scale_mul
        self.scale_gauss = scale_gauss
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, img, refer=None):
        if refer is None:
            img_up = img + np.random.normal(loc=0., scale=self.scale_gauss, size=img.shape)
            return img_up
        # 降采样 - d2 -> d4
        img_d2 = self.pool_max2(img, self.target_size)
        img_d4 = self.pool_max2(img_d2, self.target_size // 2)
        img_d8 = self.pool_max2(img_d4, self.target_size // 4)
        refer_d2 = self.pool_max2(refer, self.target_size)
        refer_d4 = self.pool_max2(refer_d2, self.target_size // 2)
        refer_d8 = self.pool_max2(refer_d4, self.target_size // 4)
        # 根据refer，寻找mid和highlight信息，对img进行灰度校正
        img, img_d2, img_d4, img_d8 = \
            img.astype(float), img_d2.astype(float), img_d4.astype(float), img_d8.astype(float)
        refer, refer_d2, refer_d4, refer_d8 = \
            refer.astype(float), refer_d2.astype(float), refer_d4.astype(float), refer_d8.astype(float)
        img = self.gray_correction(img, refer)
        img_d2 = self.gray_correction(img_d2, refer_d2)
        img_d4 = self.gray_correction(img_d4, refer_d4)
        img_d8 = self.gray_correction(img_d8, refer_d8)
        # 生成，融合
        img_up4 = self.upsample(img_d4, img_d8, std_gauss=0., std_multi=self.scale_mul[0])
        img_up2 = self.upsample(img_d2, img_up4, std_gauss=0., std_multi=self.scale_mul[1])
        img_up = self.upsample(
            img, img_up2, refer=refer, std_gauss=self.scale_gauss, std_multi=self.scale_mul[2], fus=True)

        return img_up

    def pool_max2(self, img, w):
        return np.max(img.reshape(w // 2, 2, w // 2, 2).transpose((0, 2, 1, 3)).reshape(w // 2, w // 2, -1), axis=-1)

    def gray_correction(self, img, refer):
        img = np.clip(img + (np.random.random(img.shape) * 0.5 - 1.), 0., img.max())
        min_diff = max(img.min() - 10., 0.)
        img = img - min_diff
        lg, mg, hg = self.find_mid_hl(img)
        lg_r, mg_r, hg_r = self.find_mid_hl(refer)
        r_gamma = np.log((mg_r - lg_r) / (hg_r - lg_r)) / np.log(((mg - lg) / (hg - lg)))
        img_output = (img <= lg) * img + \
                     (img > hg) * ((img - lg) / (hg - lg) * (hg_r - lg_r) + lg) + \
                     ((img > lg) * (img <= hg)) * ((np.maximum(img - lg, 0.) / (hg - lg)) ** r_gamma *
                                                   (hg_r - lg_r) + lg)

        return img_output

    def find_mid_hl(self, img):
        sort = np.sort(img.reshape(-1))
        return np.mean(sort[:int(sort.size * 0.01)]), sort[int(sort.size * 0.5)], np.mean(sort[int(sort.size * 0.99):])

    def variance_diff(self, img):
        img_fNlMD = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        diff = (img - img_fNlMD.astype(np.float64))
        mean_value = np.mean(diff)
        diff_squared = (diff - mean_value) ** 2.

        return diff_squared.mean() ** 0.5

    def upsample(self, img_d, img_d2, refer=None, std_gauss=20., std_multi=0.02, fus=False):
        sort = np.sort(img_d.reshape(-1))
        mid_d = sort[int(sort.size * 0.5)]
        img_de = np.zeros(img_d.shape)
        img_up = img_d2[:, np.newaxis, :, np.newaxis].repeat(2, axis=1).repeat(2, axis=3).reshape(
            (img_d.shape[1], img_d.shape[0]))
        img_de = \
            img_de + (np.random.rayleigh(scale=std_multi, size=img_d.shape) + 1.) * img_up + 1. * \
            np.random.normal(loc=0., scale=std_gauss, size=img_d.shape)

        ratio = (img_d < mid_d) * (img_d - img_d.min()) / (mid_d - img_d.min()) * 1. + \
                (img_d >= mid_d) * (img_d.max() - img_d) / (img_d.max() - mid_d) * 1.
        # 融合
        if fus:
            img_de = self.fusion(img_de, refer, ratio=ratio)

        return (img_de + img_d) / 2.

    def clip_random_normal(self, img, loc=0., std=10.):
        return img + img / 255 * np.clip(np.random.normal(loc=loc, scale=std, size=img.shape), -std-20, std+20)

    def clip_random_reduction(self, img, min=0.7):
        return np.random.rand(img.shape[0], img.shape[1]) * (1. - min) + min

    def fusion(self, img, refer, ratio=0.5):
        w_, h_ = refer.shape[0], refer.shape[1]
        refer_m = np.mean(refer.reshape(w_ // 16, 16, h_ // 16, 16).transpose((0, 2, 1, 3)).
                          reshape(w_ // 16, h_ // 16, -1), axis=-1)
        refer_m = cv2.resize(refer_m.astype(np.uint8), (w_, h_), interpolation=cv2.INTER_LINEAR).astype(float)
        output = img + (refer - refer_m) * ratio

        return np.clip(output, 0., output.max())


class IWrapper0Loader(Dataset):         # no style transform
    def __init__(self, root, classnames, trans_s1=None, trans_s2=None, in_seen=None, in_unseen=None,
                 device=torch.device("cuda")):
        self.root_path = list(itertools.chain.from_iterable(root))
        self.transform_s1 = trans_s1
        self.transform_s2 = trans_s2
        self.class_names = classnames
        self.num_class = classnames.__len__()
        self.in_seen = in_seen
        self.in_unseen = in_unseen
        self.idx_floor = classnames.index('seafloor')
        self.device = device
        self.n_sample = self.root_path.__len__()
        self.hor_flip = [False, True, False, True]
        self.ver_flip = [False, False, True, True]

        assert self.transform_s1 is not None
        assert self.transform_s2 is not None

    def __len__(self):
        return int(self.n_sample * 4)

    def __getitem__(self, index):
        ind = index % self.n_sample
        hvf = index // self.n_sample
        item = self.root_path[ind]
        path = item.impath
        label = item.label

        sample = Image.open(path).convert("L")
        sample = self.transform_s1(sample).repeat(3, 1, 1)
        sample = functional.hflip(sample) if self.hor_flip[hvf] else sample
        sample = functional.vflip(sample) if self.ver_flip[hvf] else sample
        sample = self.transform_s2(sample)

        return sample, label


class IWrapper1Loader(Dataset):     # style transform but not contrastion
    def __init__(self, model, root, classnames, trans_s1=None, trans_s2=None, in_seen=None, in_unseen=None, mode_t=None,
                 device=torch.device("cuda")):
        self.root_path = root
        self.sample_path = list(itertools.chain.from_iterable(root))
        self.style_model = model
        self.style_mode = mode_t
        self.transform_s1 = trans_s1
        self.transform_s2 = trans_s2
        self.class_names = classnames
        self.num_class = classnames.__len__()
        self.in_seen = in_seen
        self.in_unseen = in_unseen
        self.idx_floor = classnames.index('seafloor')
        self.device = device
        self.n_sample = self.sample_path.__len__()
        self.hor_flip = [False, True, False, True]
        self.ver_flip = [False, False, True, True]

        assert self.transform_s1 is not None
        assert self.transform_s2 is not None

    def __len__(self):
        return int(self.n_sample * 4)

    def __getitem__(self, index):
        ind = index % self.n_sample
        hvf = index // self.n_sample
        item = self.sample_path[ind]
        path = item.impath
        label = item.label

        sample = Image.open(path).convert("L")
        sample = self.transform_s1(sample).repeat(3, 1, 1)
        sample = functional.hflip(sample) if self.hor_flip[hvf] else sample
        sample = functional.vflip(sample) if self.ver_flip[hvf] else sample

        if label in self.in_unseen:  # for refer
            style = self.transform_s1(self.get_()).repeat(3, 1, 1)
            with torch.no_grad():
                sample = self.style_model(
                    sample.unsqueeze(0).to(self.device), style.unsqueeze(0).to(self.device))[0].cpu()

        sample = self.transform_s2(sample)

        return sample, label

    def get_(self):
        ncs = np.random.choice(self.in_seen)        # np.random.choice(self.in_seen)  self.idx_floor
        path_s = random.choice(self.root_path[ncs]).impath
        return Image.open(path_s).convert("L")


class IWrapper2Loader(Dataset):
    def __init__(self, model, root, classnames, hist, trans_s1=None, trans_s2=None,
                 in_seen=None, in_unseen=None, flag=('ccpl', 'none'), std_n=(0.01, 0.1), device=torch.device("cuda")):
        self.root_path = root
        self.sample_path = []
        self.style_model = model
        self.transform_s1 = trans_s1
        self.transform_s2 = trans_s2
        self.class_names = classnames
        self.num_class = classnames.__len__()
        self.hist_batch = hist
        self.n_batch = int(np.sum(hist))
        self.in_seen = in_seen
        self.in_unseen = in_unseen
        self.idx_floor = classnames.index('seafloor')
        self.flag_transfer = flag[0]
        self.flag_mode = flag[1]
        self.std_multi = std_n[0]
        self.std_plus = std_n[1]
        self.device = device

        assert self.transform_s1 is not None
        assert self.transform_s2 is not None

    def __len__(self):
        return int(self.root_path[self.idx_floor].__len__() // self.hist_batch[self.idx_floor] * self.n_batch * 4)

    def __getitem__(self, index):
        ind = index % self.n_batch
        if ind == 0:
            self.re_arrange()

        item = self.sample_path[ind]
        path = item.impath
        label = item.label

        sample = Image.open(path).convert("L")

        sample1 = self.transform_s1[0](sample).repeat(3, 1, 1)
        sample2 = self.transform_s1[0](sample).repeat(3, 1, 1)
        sample3 = self.transform_s1[1](sample).repeat(3, 1, 1)
        if torch.rand(1) < 0.5:
            sample1 = functional.hflip(sample1)
            sample2 = functional.hflip(sample2)
        if torch.rand(1) < 0.5:
            sample1 = functional.vflip(sample1)
            sample2 = functional.vflip(sample2)

        if label != self.idx_floor:  # for refer
            style2 = self.transform_s1[0](self.get_()).repeat(3, 1, 1)
            style3 = self.transform_s1[0](self.get_()).repeat(3, 1, 1)
            refer2 = self.transform_s1[0](self.get_(flag='refer')).repeat(3, 1, 1)
            refer3 = self.transform_s1[0](self.get_(flag='refer')).repeat(3, 1, 1)
            with torch.no_grad():
                sample2 = self.style_model(
                    sample2.unsqueeze(0).to(self.device), style2.unsqueeze(0).to(self.device))[0].cpu()
                sample3 = self.style_model(
                    sample3.unsqueeze(0).to(self.device), style3.unsqueeze(0).to(self.device))[0].cpu()
                if self.flag_mode in ['custom']:
                    sample2 = self.process(sample2, refer2, label=label)
                    sample3 = self.process(sample3, refer3, label=label)
        else:
            if self.flag_mode in ['custom']:
                sample2 = self.process(sample2, label=label)
                sample3 = self.process(sample3, label=label)

        sample2 = self.after_process(sample2)
        sample3 = self.after_process(sample3)

        sample1 = self.transform_s2[0](sample1)
        sample2 = self.transform_s2[0](sample2)
        sample3 = self.transform_s2[1](sample3)

        return sample1, sample2, sample3, label

    def get_(self, flag='style'):
        ncs = self.idx_floor if flag == 'refer' else np.random.choice(self.in_seen)
        path_s = random.choice(self.root_path[ncs]).impath
        return Image.open(path_s).convert("L")

    def process(self, sample, refer=None, label=0):
        output = torch.mean(sample, dim=0, keepdim=True)
        if self.flag_transfer in ['ccpl']:
            if refer is not None:
                refer = torch.mean(refer, dim=0, keepdim=True)
                med_f = output.median()
                med_r = refer.median()
                refer = torch.clip(refer, 0., 1.) ** (med_r / med_f)
                mask_refer = (- (output - med_f).abs() * 20).exp()
                output = output * (1. - mask_refer) + refer * mask_refer
            noise = (np.random.rayleigh(scale=self.std_multi, size=output.shape) + 1.) * output.cpu().numpy() + \
                np.random.normal(loc=0., scale=self.std_plus, size=output.shape)\
                # \
                # if label in self.in_unseen else \
                # np.random.normal(loc=0., scale=self.std_plus/2, size=output.shape)
            output = output + torch.from_numpy(noise).to(output.device)
            output = output.repeat(3, 1, 1) / output.max()

        return output

    def after_process(self, sample):
        output = (sample - sample.min()) / (sample.max() - sample.min()) \
            if self.flag_transfer in ['asepa'] else torch.clip(sample / sample.max(), 0., 1.)
        return output

    def re_arrange(self):
        self.sample_path = []
        for n in range(self.num_class):
            path = self.root_path[n]
            bz = self.hist_batch[n]
            item_list = np.random.choice(path, size=bz, replace=False if path.__len__() >= bz else True).tolist()
            self.sample_path += item_list


class MyWrapperLoader(Dataset):
    def __init__(self, model, root, classnames, hist, trans_s1=None, trans_s2=None, in_seen=None, in_unseen=None,
                 flag=('ccpl', 'none'), std_n=(0.01, 0.1), device=torch.device("cuda")):
        self.root_path = root
        self.sample_path = []
        self.style_model = model
        self.transform_s1 = trans_s1
        self.transform_s2 = trans_s2
        self.class_names = classnames
        self.num_class = classnames.__len__()
        self.hist_batch = hist
        self.n_batch = int(np.sum(hist))
        self.in_seen = in_seen
        self.in_unseen = in_unseen
        self.idx_floor = classnames.index('seafloor')
        self.flag_transfer = flag[0]
        self.flag_mode = flag[1]
        self.std_multi = std_n[0]
        self.std_plus = std_n[1]
        self.device = device

        assert self.transform_s1 is not None
        assert self.transform_s2 is not None

    def __len__(self):
        return int(self.root_path[self.idx_floor].__len__() // self.hist_batch[self.idx_floor] * self.n_batch * 4)

    def __getitem__(self, index):
        ind = index % self.n_batch
        if ind == 0:
            self.re_arrange()

        item = self.sample_path[ind]
        path = item.impath
        label = item.label

        sample = Image.open(path).convert("L")
        sample = self.transform_s1[0](sample).repeat(3, 1, 1)
        hor_flip = True if torch.rand(1).item() > 0.5 else False
        ver_flip = True if torch.rand(1).item() > 0.5 else False
        sample = functional.hflip(sample) if hor_flip else sample
        sample = functional.vflip(sample) if ver_flip else sample

        if label in self.in_unseen:
            style = self.transform_s1[1](self.get_()).repeat(3, 1, 1)
            refer = self.transform_s1[0](self.get_(flag='refer')).repeat(3, 1, 1)
            with torch.no_grad():
                sample2 = self.style_model(
                    sample.unsqueeze(0).to(self.device), style.unsqueeze(0).to(self.device))[0].detach().cpu()
                sample2 = self.process(sample2, refer, label=label) \
                    if self.flag_mode in ['custom'] else \
                    self.process(sample2, label=label)    # self.flag_transfer in ['ccpl'] and # sample2
        else:
            sample2 = self.process(sample, label=label)

        sample2 = self.after_process(sample2)

        sample = self.transform_s2(sample)
        sample2 = self.transform_s2(sample2)

        return sample, sample2, label

    def get_(self, flag='style'):
        ncs = self.idx_floor if flag == 'refer' else np.random.choice(self.in_seen)
        path_s = random.choice(self.root_path[ncs]).impath
        return Image.open(path_s).convert("L")

    def process(self, sample, refer=None, label=0):
        output = torch.mean(sample, dim=0, keepdim=True)
        if refer is not None:       # self.flag_transfer in ['ccpl'] and
            refer = torch.mean(refer, dim=0, keepdim=True)
            med_f = output.median()
            med_r = refer.median()
            rd = torch.randn(1) * 0.1 + 1.
            refer = torch.clip(refer * rd.to(med_r.device), 0., 1.)     # ** (med_r / med_f)
            mask_refer = (- (output - med_f).abs() * 20).exp()
            output = output * (1. - mask_refer) + refer * mask_refer
        noise = (np.random.rayleigh(scale=self.std_multi, size=output.shape) + 1.) * output.cpu().numpy() + \
            np.random.normal(loc=0., scale=self.std_plus, size=output.shape)
            # if label in self.in_unseen else \
            # np.random.normal(loc=0., scale=self.std_plus/2, size=output.shape)
        output = output + torch.from_numpy(noise).to(output.device)
        output = output.repeat(3, 1, 1) / output.max()

        return output

    def after_process(self, sample):
        output = (sample - sample.min()) / (sample.max() - sample.min()) \
            if self.flag_transfer in ['asepa'] else torch.clip(sample / sample.max(), 0., 1.)
        return output

    def re_arrange(self):
        self.sample_path = []
        for n in range(self.num_class):
            path = self.root_path[n]
            bz = self.hist_batch[n]
            item_list = np.random.choice(path, size=bz, replace=False if path.__len__() >= bz else True).tolist()
            self.sample_path += item_list


class My2WrapperLoader(Dataset):
    def __init__(self, model, root, classnames, hist, trans_s1=None, trans_s2=None, in_seen=None, in_unseen=None,
                 flag=('ccpl', 'none'), std_n=(0.01, 0.1), device=torch.device("cuda")):
        self.root_path = root
        self.sample_path = []
        self.style_model = model
        self.transform_s1 = trans_s1
        self.transform_s2 = trans_s2
        self.class_names = classnames
        self.num_class = classnames.__len__()
        self.hist_batch = hist
        self.n_batch = int(np.sum(hist))
        self.in_seen = in_seen
        self.in_unseen = in_unseen
        self.idx_floor = classnames.index('seafloor')
        self.flag_transfer = flag[0]
        self.flag_mode = flag[1]
        self.std_multi = std_n[0]
        self.std_plus = std_n[1]
        self.device = device

        assert self.transform_s1 is not None
        assert self.transform_s2 is not None

    def __len__(self):
        return int(self.root_path[self.idx_floor].__len__() // self.hist_batch[self.idx_floor] * self.n_batch * 4)

    def __getitem__(self, index):
        ind = index % self.n_batch
        if ind == 0:
            self.re_arrange()

        item = self.sample_path[ind]
        path = item.impath
        label = item.label

        sample = Image.open(path).convert("L")
        sample = self.transform_s1[0](sample).repeat(3, 1, 1)
        hor_flip = True if torch.rand(1).item() > 0.5 else False
        ver_flip = True if torch.rand(1).item() > 0.5 else False
        sample = functional.hflip(sample) if hor_flip else sample
        sample = functional.vflip(sample) if ver_flip else sample

        if label in self.in_unseen:
            style = self.transform_s1[1](self.get_()).repeat(3, 1, 1)
            refer = self.transform_s1[0](self.get_(flag='refer')).repeat(3, 1, 1)
            neg_idx = np.random.choice(np.delete(self.in_unseen, np.where(self.in_unseen == label)[0][0]))
            neg_ = Image.open(random.choice(self.root_path[neg_idx]).impath).convert("L")
            neg_ = self.transform_s1[0](neg_).repeat(3, 1, 1)
            neg_ = functional.hflip(neg_) if hor_flip else neg_
            neg_ = functional.vflip(neg_) if ver_flip else neg_
            with torch.no_grad():
                sample2 = self.style_model(
                    sample.unsqueeze(0).to(self.device), style.unsqueeze(0).to(self.device))[0].detach().cpu()
                sample2 = self.process(sample2, refer, label=label) \
                    if self.flag_transfer in ['ccpl'] and self.flag_mode in ['custom'] else \
                    self.process(sample, label=label)
                neg = self.style_model(
                    neg_.unsqueeze(0).to(self.device), style.unsqueeze(0).to(self.device))[0].detach().cpu()
                neg = self.process(neg, refer, label=label) \
                    if self.flag_transfer in ['ccpl'] and self.flag_mode in ['custom'] else \
                    self.process(neg_, label=label)
        else:
            sample2 = self.process(sample, label=label)
            neg_idx = np.random.choice(self.in_unseen)
            neg_ = Image.open(random.choice(self.root_path[neg_idx]).impath).convert("L")
            neg_ = self.transform_s1[0](neg_).repeat(3, 1, 1)
            neg_ = functional.hflip(neg_) if hor_flip else neg_
            neg_ = functional.vflip(neg_) if ver_flip else neg_
            style = self.transform_s1[1](Image.open(path).convert("L")).repeat(3, 1, 1)
            refer = Image.open(path).convert("L") if label == self.idx_floor else \
                Image.open(random.choice(self.root_path[self.idx_floor]).impath)
            refer = self.transform_s1[0](refer).repeat(3, 1, 1)
            with torch.no_grad():
                neg = self.style_model(
                    neg_.unsqueeze(0).to(self.device), style.unsqueeze(0).to(self.device))[0].detach().cpu()
                neg = self.process(neg, refer, label=label) \
                    if self.flag_transfer in ['ccpl'] and self.flag_mode in ['custom'] else \
                    self.process(neg_, label=label)

        sample2 = self.after_process(sample2)
        neg = self.after_process(neg)

        sample = self.transform_s2(sample)
        sample2 = self.transform_s2(sample2)
        neg = self.transform_s2(neg)

        return sample, sample2, neg, label

    def get_(self, flag='style'):
        ncs = self.idx_floor if flag == 'refer' else np.random.choice(self.in_seen)
        path_s = random.choice(self.root_path[ncs]).impath
        return Image.open(path_s).convert("L")

    def process(self, sample, refer=None, label=0):
        output = torch.mean(sample, dim=0, keepdim=True)
        if self.flag_transfer in ['ccpl'] and refer is not None:
            refer = torch.mean(refer, dim=0, keepdim=True)
            med_f = output.median()
            med_r = refer.median()
            rd = torch.randn(1) * 0.1 + 1.
            refer = torch.clip(refer * rd.to(med_r.device), 0., 1.)     # ** (med_r / med_f)
            mask_refer = (- (output - med_f).abs() * 20).exp()
            output = output * (1. - mask_refer) + refer * mask_refer
        noise = (np.random.rayleigh(scale=self.std_multi, size=output.shape) + 1.) * output.cpu().numpy() + \
            np.random.normal(loc=0., scale=self.std_plus, size=output.shape)\
            # if label in self.in_unseen else \
            # np.random.normal(loc=0., scale=self.std_plus/2, size=output.shape)
        output = output + torch.from_numpy(noise).to(output.device)
        output = output.repeat(3, 1, 1) / output.max()

        return output

    def after_process(self, sample):
        output = (sample - sample.min()) / (sample.max() - sample.min()) \
            if self.flag_transfer in ['asepa'] else torch.clip(sample / sample.max(), 0., 1.)
        return output

    def re_arrange(self):
        self.sample_path = []
        for n in range(self.num_class):
            path = self.root_path[n]
            bz = self.hist_batch[n]
            item_list = np.random.choice(path, size=bz, replace=False if path.__len__() >= bz else True).tolist()
            self.sample_path += item_list


class INaturalist(Dataset):
    def __init__(self, root, trans=None):
        self.root_path = root
        self.transform = trans
        self.n_sample = self.root_path.__len__()
        self.hor_flip = [False, True, False, True]
        self.ver_flip = [False, False, True, True]

        assert self.transform is not None

    def __len__(self):
        return int(self.n_sample * 4)

    def __getitem__(self, index):
        ind = index % self.n_sample
        hvf = index // self.n_sample
        item = self.root_path[ind]
        path = item.impath
        label = item.label
        with open(path, 'rb') as f:
            sample = Image.open(f).convert("L")

        sample = self.transform(sample)
        sample = functional.hflip(sample) if self.hor_flip[hvf] else sample
        sample = functional.vflip(sample) if self.ver_flip[hvf] else sample

        return sample, label


def prior_label(w, seen, unseen, ratio, device):
    nc = w.shape[0]
    log_p_y = torch.zeros(nc)
    p_y_seen = torch.zeros(seen.shape[0])
    p_y_unseen = torch.zeros(unseen.shape[0])
    for i, n in enumerate(seen):
        p_y_seen[i] = w[n]
    p_y_seen /= p_y_seen.sum()
    for i, n in enumerate(unseen):
        p_y_unseen[i] = w[n]
    p_y_seen /= p_y_seen.sum()
    log_p_y[seen] = p_y_seen.log()
    log_p_y[unseen] = p_y_unseen.log()

    log_p0_y = torch.zeros(nc)
    p0_s = 1 / (1 + 1 / ratio)
    p0_u = (1 - p0_s)
    log_p0_y[seen] = math.log(p0_s)
    log_p0_y[unseen] = math.log(p0_u)

    return log_p_y.to(device), log_p0_y.to(device)


def sample_assignment(w, nc, b):
    left = 0
    n_left = 0
    output = np.zeros(nc).astype(int)
    mask = np.ones(nc).astype(bool)
    for n in range(nc):
        if w[n] < b:
            output[n] = w[n]
            left += b - w[n]
            n_left += 1
            mask[n] = False
        else:
            output[n] = b
    f = False
    for n in range(nc):
        if mask[n] == True:
            if f == False:
                output[n] = left - (left // n_left) * (n_left - 1) + b if n_left > 1 else left + b
            else:
                if n_left > 1:
                    output[n] = left // n_left + b
            f = True

    return output
