import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as torchvision_transforms
import numpy as np
from PIL import Image
import glob
import os
from modules.custom_transforms import *


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.extend(glob.glob(os.path.join(data_dir, "*.gif")))
    data.sort()
    filenames.sort()
    return data, filenames


class FusionDataset(Dataset):
    def __init__(self, args):
        super(FusionDataset, self).__init__()
        self.args = args
        data_dir_mr = args.mr_dir
        data_dir_ct = args.ct_dir
        self.filepath_mr, self.filenames_mr = prepare_data_path(data_dir_mr)
        self.filepath_ct, self.filenames_ct = prepare_data_path(data_dir_ct)
        self.length = min(len(self.filenames_mr), len(self.filenames_ct))

    def __getitem__(self, index):
        mr_path = self.filepath_mr[index]
        ct_path = self.filepath_ct[index]
        # image_mr = np.array(Image.open(mr_path))
        # image_ct = np.array(Image.open(ct_path))
        image_mr = np.array(Image.open(mr_path).convert('RGB'))
        image_ct = np.array(Image.open(ct_path).convert('RGB'))
        # image_mr = (np.asarray(Image.fromarray(image_mr), dtype=np.float32).transpose((2, 0, 1))/ 255.0)
        image_mr = np.asarray(image_mr, dtype=np.float32) / 255.0
        image_ct = np.asarray(image_ct, dtype=np.float32) / 255.0
        # image_ct = np.expand_dims(image_ct, axis=0)
        name = self.filenames_mr[index]
        if not self.args.one_in_rgb:
            return (
                torch.tensor(np.expand_dims(image_mr[:, :, 0], axis=0)),
                torch.tensor(np.expand_dims(image_ct[:, :, 0], axis=0)),
                name,
                index,
            )
        else:
            image_mr = np.array(Image.open(mr_path))
            image_mr = (
                    np.asarray(Image.fromarray(image_mr), dtype=np.float32).transpose(
                        (2, 0, 1)
                    )
                    / 255.0
            )
            return (
                torch.tensor(image_mr),
                torch.tensor(np.expand_dims(image_ct[:, :, 0], axis=0)),
                name,
                index,
            )

    def __len__(self):
        return self.length


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, args,mode='train',inv_list=None, eqv_list=None, scale=(0.5, 1)):
        self.source = args.fused_dir
        self.res1 = [args.seg_res1_h, args.seg_res1_w]
        self.res2 = [args.seg_res2_h, args.seg_res2_w]
        self.labeldir = args.label_dir
        self.mode = mode
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.scale = scale
        self.view = -1
        self.imdb = self.load_imdb()
        self.N = len(self.imdb)
        self.reshuffle()

    def load_imdb(self):
        imdb = []
        for fname in os.listdir(self.source):
            image_path = os.path.join(self.source, fname)
            imdb.append(image_path)
        return imdb

    def __getitem__(self, index):
        index = self.shuffled_indices[index]
        ipath = self.imdb[index]
        image = Image.open(ipath).convert('RGB')
        image = self.transform_image(index, image)
        label = self.transform_label(index)
        return (index,) + image + label

    def reshuffle(self):
        self.shuffled_indices = np.arange(len(self.imdb))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()

    def transform_image(self, index, image):
        image = self.transform_base(index, image)
        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                image = self.transform_inv(index, image, 1)
                image = torchvision_transforms.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            return (image,)
        elif self.mode == 'train':
            # Invariance transform.
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image1)
            image2 = self.transform_inv(index, image, 1)
            image2 = torchvision_transforms.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)
            return (image1, image2)

    def transform_inv(self, index, image, ver):
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        return image

    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)
        return image

    def init_transforms(self):
        N = self.N
        # Base transform.
        self.transform_base = BaseTransform(self.res2)
        # Transforms for invariance.
        # Color jitter (4), gray scale, blur.
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in
                                        range(2)]  # Control this later (NOTE)]
        self.random_color_contrast = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in
                                      range(2)]  # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in
                                        range(2)]  # Control this later (NOTE)
        self.random_color_hue = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]  # Control this later (NOTE)
        self.random_gray_scale = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]
        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip = RandomVerticalFlip(N=N)
        self.random_resized_crop = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)
        # Tensor transform.
        self.transform_tensor = TensorTransform()

    def transform_label(self, index):
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)
            # X1 = int(np.sqrt(label1.shape[0]))
            # X2 = int(np.sqrt(label2.shape[0]))
            # label1 = label1.view(X1, X1)
            # label2 = label2.view(X2, X2)
            label1 = label1[0,:,:]
            label2 = label2[0,:,:]
            return label1, label2
        return (None,)

    def __len__(self):
        return len(self.imdb)


class CUTSDataset(Dataset):
    def __init__(self, args):
        super(CUTSDataset, self).__init__()
        self.args = args
        data_dir_fu = args.fused_dir
        self.filepath_fu, self.filenames_fu = prepare_data_path(data_dir_fu)
        self.length = len(self.filenames_fu)

    def __getitem__(self, index):
        fu_path = self.filepath_fu[index]
        image_fu = np.array(Image.open(fu_path).convert('RGB'))
        image_fu = np.asarray(image_fu, dtype=np.float32) / 255.0
        name = self.filenames_fu[index]
        return (
            torch.tensor(image_fu),
            name
        )

    def __len__(self):
        return self.length


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--ct_dir', type=str, default='E:/Datasets/AANLIB_multimodal/CT')
    args.add_argument('--mr_dir', type=str, default='E:/Datasets/AANLIB_multimodal/MR')
    args = args.parse_args()
    train_dataset = FusionDataset(args)
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    for it, (image_mr, image_ct, name) in enumerate(train_loader):
        if it == 3:
            image_mr.numpy()
            print(image_mr.shape)
            image_ct.numpy()
            print(image_ct.shape)
            break
