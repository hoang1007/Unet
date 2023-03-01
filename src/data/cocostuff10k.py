import os.path as osp

from PIL import Image
import scipy.io as sio

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from pytorch_lightning import LightningDataModule


class CocoStuff10kDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        self.train_data = CocoStuff10k(self.data_dir, split="train")
        self.val_data = CocoStuff10k(self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class CocoStuff10k(Dataset):
    def __init__(self, data_dir: str, split: str = "train", warp_image: bool = True):
        assert split in ["train", "test"]

        self.split = split
        self.root = data_dir
        self.warp_image = warp_image
        self.images_dir = osp.join(data_dir, "images")
        self.labels_dir = osp.join(data_dir, "labels")

        self.NUM_CLASSES = 182
        self.MEAN = [0.43931922, 0.41310471, 0.37480941]
        self.STD = [0.24272706, 0.23649098, 0.23429529]

        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=self.MEAN, std=self.STD)]
        )

        self.setup()

    def setup(self):
        file_list = osp.join(self.root, "imageLists", self.split + ".txt")
        self.files = [name.rstrip() for name in tuple(open(file_list, "r"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_id = self.files[index]
        image_path = osp.join(self.root, "images", image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", image_id + ".mat")

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        label = sio.loadmat(label_path)["S"]
        label -= 1  # unlabeled (0 -> -1)
        label[label == -1] = 255
        if self.warp_image:
            image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)
            label = np.asarray(
                Image.fromarray(label).resize((513, 513), resample=Image.NEAREST)
            )

        image = self.transforms(image)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return {'image': image, 'label': label}
