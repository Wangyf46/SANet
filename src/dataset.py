from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import h5py
import cv2
import os


class listDataset(Dataset):
    def __init__(self, root, shape = None, shuffle = True, transform = None, train = False, batch_size = 1, num_workers = 4):
        if train:
            root = root * 4
        random.shuffle(root)
        self.nSamples = len(root)
        self.root = root
        self.shape = shape
        self.transform = transform
        self.train = True
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        img_path = self.root[index]
        gt_density_map_path = img_path.replace("images", "ground_truth")
        gt_density_map_path = os.path.splitext(gt_density_map_path)[0] + ".h5"
        img = Image.open(img_path).convert("RGB")
        gt_density_map_file = h5py.File(gt_density_map_path)
        gt_density_map = np.asarray(gt_density_map_file["density"])
        if self.transform is not None:
            img = self.transform(img)
        return img, gt_density_map
