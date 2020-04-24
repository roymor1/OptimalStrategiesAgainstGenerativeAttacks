# imports
import os
import sys
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from data_handling.utils import list_files, list_dir, list_files_rec


########################################################################################################################
# GIM Datasets
########################################################################################################################
class ImgGIMDataSet(Dataset):
    """"""
    def __init__(self, root, split,
                 img_channels, img_size,
                 m, n, si, example_cnt_per_class,
                 img_suffix='.jpg', hierarchical=False, mirror=True):
        """"""
        self.root = root
        self.split = split
        self.img_channels = img_channels
        self.img_size = img_size
        self.m = m
        self.n = n
        self.si = si
        self.min_imgs_per_cls = m + n + si
        self.example_cnt_per_class = example_cnt_per_class
        self.img_suffix = img_suffix
        self.data_dir = os.path.join(root, split)

        augment_transforms = []
        if mirror:
            augment_transforms.append(transforms.RandomHorizontalFlip())
        self.augment_transforms = transforms.Compose(augment_transforms)

        if hierarchical:
            print("Loading ds class dirs")
            self._class_dir_names = []
            parent_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            for pdir in tqdm(parent_dirs):
                pdir_path = os.path.join(self.data_dir, pdir)
                subdirs = [os.path.join(pdir, d) for d in os.listdir(pdir_path) if os.path.isdir(os.path.join(pdir_path, d))]
                self._class_dir_names.extend(subdirs)
        else:
            self._class_dir_names = list_dir(self.data_dir)

        print()
        print("Filtering classes with less then n+m+k images")
        self._class_dir_names = [d for d in self._class_dir_names if
                                 self.at_least_min_num_imgs(min_num_imgs=m+n+si, dir_rel_path=d, suffix=img_suffix)]
        self.n_classes = len(self._class_dir_names)

    def __len__(self):
        return len(self._class_dir_names) * self.example_cnt_per_class

    def __getitem__(self, index):
        """"""
        cls_idx = index // self.example_cnt_per_class
        label_str = self._class_dir_names[cls_idx]
        cls_dir_path = os.path.join(self.data_dir, self._class_dir_names[cls_idx])

        # get img_paths
        cls_img_paths = [
            os.path.join(cls_dir_path, img_name)
            for img_name in [f for f in os.listdir(cls_dir_path) if f.endswith(self.img_suffix)]
        ]
        n_cls_imgs = len(cls_img_paths)

        # sample indices
        sampled_indices = random.sample(list(range(n_cls_imgs)), self.m + self.n + self.si)
        leaked_indices = sampled_indices[:self.m]
        real_indices = sampled_indices[self.m:self.m + self.n]
        si_indices = sampled_indices[self.m + self.n:]
        assert len(leaked_indices) == self.m
        assert len(real_indices) == self.n
        assert len(si_indices) == self.si

        # load images
        real_sample = self.load_images(cls_img_paths, real_indices)
        leaked_sample = self.load_images(cls_img_paths, leaked_indices)
        si_sample = self.load_images(cls_img_paths, si_indices)

        # set up example
        example = {
            "real_sample": real_sample,
            "leaked_sample": leaked_sample,
            "si_sample": si_sample,
            "class": cls_idx,
            "class_name": label_str
        }
        return example

    def load_images(self, img_paths, indices):
        images = []
        for idx in indices:
            images.append(load_image(img_paths[idx], self.img_size, augment_transforms=self.augment_transforms))
        images = torch.stack(images, dim=0)
        return images

    def at_least_min_num_imgs(self, min_num_imgs, dir_rel_path, suffix='.jpg'):
        dir_path = os.path.join(self.data_dir, dir_rel_path)
        num_imgs = len([f for f in os.listdir(dir_path) if f.endswith(suffix)])
        return num_imgs >= min_num_imgs


class OmniglotGIMDataSet(Dataset):
    """
    Each example is sampled from T images of some class c
    public_sample: m images of class c
    real_sample: n images of class c
    src_info: T-n images of class c. where T is the total number of images of class c.
    """
    NUM_EXAMPLES_PER_CLASS = 20
    def __init__(self, root, split, img_channels, img_size, m, n, si, example_cnt_per_class):
        """"""
        if m + n + si > self.NUM_EXAMPLES_PER_CLASS:
            raise ValueError("Max allowed value for m+n+si is {}".format(self.NUM_EXAMPLES_PER_CLASS))
        self.example_cnt_per_class = example_cnt_per_class
        self.root = root
        self.split = split
        self.img_channels = img_channels
        self.img_size = img_size
        self.m = m
        self.n = n
        self.si = si
        self.example_cnt_per_class = example_cnt_per_class
        self.data_path = os.path.join(root, split)

        self._alphabets = list_dir(self.data_path)
        self._characters = sum([[os.path.join(a, c) for c in list_dir(os.path.join(self.data_path, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in list_files(os.path.join(self.data_path, character), ('.png', '.jpg', 'jpeg', '.JPG', 'JPEG'))]
                                  for idx, character in enumerate(self._characters)]
        self.augment_transforms = None
        self._load_data()
        self.n_classes = len(self._characters)

    def __len__(self):
        return len(self._characters) * self.example_cnt_per_class

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        # get character
        char_class = index // self.example_cnt_per_class

        # process images
        n_examples = self.data[char_class].size(0)
        images = self.data[char_class]

        # draw samples
        sampled_indices = random.sample(list(range(n_examples)), self.m + self.n + self.si)
        leaked_indices = sampled_indices[:self.m]
        real_indices = sampled_indices[self.m:self.m + self.n]
        si_indices = sampled_indices[self.m + self.n:]
        assert len(leaked_indices) == self.m
        assert len(real_indices) == self.n
        assert len(si_indices) == self.si
        real_sample = images[real_indices]
        leaked_sample = images[leaked_indices]
        si_sample = images[si_indices]

        example = {
            "real_sample": real_sample,
            "leaked_sample": leaked_sample,
            "si_sample": si_sample,
            "class": char_class,
            "class_name": self._characters[char_class]
        }
        return example

    def _load_data(self):
        """"""
        num_classes = len(self._characters)
        self.data = [None for _ in range(num_classes)]
        for char_class, char_img_names in enumerate(self._character_images):
            n_examples = len(char_img_names)
            # assert n_examples <= self.NUM_EXAMPLES_PER_CLASS
            # process images
            images = []
            for i in range(n_examples):
                img_name, img_char_class = self._character_images[char_class][i]
                assert (img_char_class == char_class), "Wrong image class"
                img_path = os.path.join(self.data_path, self._characters[char_class], img_name)
                images.append(
                    load_image(
                        img_path=img_path,
                        img_size=self.img_size,
                        augment_transforms=self.augment_transforms,
                        img_mode='L'
                    )
                )
            images = torch.stack(images, dim=0)
            self.data[char_class] = images


########################################################################################################################
# Arcface Datasets
########################################################################################################################
class ArcfaceDataSet(Dataset):
    """"""
    def __init__(self, root, split,
                 img_channels, img_size,
                 example_cnt_per_class,
                 img_suffix='.jpg', mirror=True):
        """"""
        self.root = root
        self.split = split
        self.img_channels = img_channels
        self.img_size = img_size
        self.example_cnt_per_class = example_cnt_per_class
        self.img_suffix = img_suffix
        self.data_dir = os.path.join(root, split)

        augment_transforms = []
        if mirror:
            augment_transforms.append(transforms.RandomHorizontalFlip())
        self.augment_transforms = transforms.Compose(augment_transforms)

        self._class_dir_names = list_dir(self.data_dir)
        self.n_classes = len(self._class_dir_names)
        self.class_img_paths = {}

    def __len__(self):
        return len(self._class_dir_names) * self.example_cnt_per_class

    def __getitem__(self, index):
        """"""
        cls_idx = index // self.example_cnt_per_class
        cls_dir_path = os.path.join(self.data_dir, self._class_dir_names[cls_idx])

        # get img_paths
        if not(cls_idx in self.class_img_paths):
            self.class_img_paths[cls_idx] = [
                os.path.join(cls_dir_path, img_name)
                for img_name in [f for f in list_files_rec(cls_dir_path, self.img_suffix)]
            ]
        n_cls_imgs = len(self.class_img_paths[cls_idx])

        # sample image
        img_idx = random.choice(list(range(n_cls_imgs)))

        # load image
        img = self.load_images(self.class_img_paths[cls_idx], [img_idx]).squeeze(0)

        return img, cls_idx

    def load_images(self, img_paths, indices):
        images = []
        for idx in indices:
            images.append(load_image(img_paths[idx], self.img_size, augment_transforms=self.augment_transforms))
        images = torch.stack(images, dim=0)
        return images


########################################################################################################################
# Utility functions
########################################################################################################################
def adjust_dynamic_range(data, drange_in, drange_out=(-1,1)):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def load_image(img_path, img_size, drange_net=(-1, 1), augment_transforms=None, img_mode='RGB'):
    """"""
    img = Image.open(img_path, mode='r').convert(img_mode)
    return process_pil_image(img, img_size=img_size, drange_net=drange_net, augment_transforms=augment_transforms)


def load_glow_image(img_path, img_size, drange_net=(-1, 1), augment_transforms=None, img_mode='RGB'):
    """"""
    img = Image.open(img_path, mode='r').convert(img_mode)
    return process_pil_image_glow(img, img_size=img_size, drange_net=drange_net, augment_transforms=augment_transforms)


def process_pil_image(pil_img, img_size, drange_net=(-1, 1), augment_transforms=None):
    """"""
    img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
    if augment_transforms is not None:
        img = augment_transforms(img)
    img = transforms.ToTensor()(img) # scales from [0,255] to [0,1]
    img = adjust_dynamic_range(img, (0., 1.), drange_net)
    return img


def process_pil_image_bw(pil_img, img_size, drange_net=(-1, 1), augment_transforms=None):
    """"""
    img = pil_img.convert('L')
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    if augment_transforms is not None:
        img = augment_transforms(img)
    img = transforms.ToTensor()(img) # scales from [0,255] to [0,1]
    img = adjust_dynamic_range(img, (0., 1.), drange_net)
    return img


def process_pil_image_glow(pil_img, img_size, drange_net=(-1, 1), augment_transforms=None): # TODO
    """"""
    img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
    if augment_transforms is not None:
        img = augment_transforms(img)
    img = np.array(img)
    if len(img.shape) == 2:
        img = torch.from_numpy(img).unsqueeze(0)
    elif len(img.shape) == 3:
        img = torch.from_numpy(img.transpose((2, 0, 1)))
    else:
        raise TypeError("Only supports imgs with 1 or channels")

    if isinstance(img, torch.ByteTensor):
        img =  img.float().div(256)
    img = adjust_dynamic_range(img, (0., 1.), drange_net)
    return img
