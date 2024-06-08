from pathlib import Path
import pickle
import numpy as np
import torch
from os.path import join, isfile, basename
from torch.utils.data import Dataset, DataLoader
import cv2
import random


class NpyDatasetCache:
    def __init__(self, img_data_root, gt_data_root, dataset_cache_path="./dataset_cache_npy.pkl"):
        if not Path(dataset_cache_path).exists():
            print("Npy Dataset cache not found. Creating...")
            img_file_paths = sorted([str(path) for path in Path(img_data_root).rglob('*.npy') if path.is_file()])
            gt_file_paths = [str(Path(gt_data_root) / Path(path).relative_to(img_data_root)) for path in img_file_paths]
            self.slice_dataset_cache = {"img_path": img_file_paths, "gt_path": gt_file_paths}
            # 使用with语句打开文件以保证即使发生错误也能正确关闭文件
            with open(Path(dataset_cache_path), 'wb') as file:
                # 使用pickle.dump()函数将字典写入文件
                pickle.dump(self.slice_dataset_cache, file)
                print(f"Saving dataset cache to {dataset_cache_path}...")
        else:
            # 使用with语句打开文件以保证即使发生错误也能正确关闭文件
            with open(Path(dataset_cache_path), 'rb') as file:
                # 使用pickle.load()函数从文件中加载对象
                self.slice_dataset_cache = pickle.load(file)
                print(f"Using Npy dataset cache from {dataset_cache_path}...")

    def divide(self, train_cache_path="./train_dataset_cache_npy.pkl", val_cache_path="./val_dataset_cache_npy.pkl"):
        if not Path(train_cache_path).exists() and not Path(val_cache_path).exists():
            print("Train dataset cache or val dataset cache not found. Creating...")
            train_img_path = []
            train_gt_path = []
            val_img_path = []
            val_gt_path = []
            flag = 0
            for i in range(len(self.slice_dataset_cache["img_path"])):
                if flag < 20:
                    train_img_path.append(self.slice_dataset_cache["img_path"][i])
                    train_gt_path.append(self.slice_dataset_cache["gt_path"][i])
                    flag += 1
                else:
                    val_img_path.append(self.slice_dataset_cache["img_path"][i])
                    val_gt_path.append(self.slice_dataset_cache["gt_path"][i])
                    flag = 0
            train_cache = {"img_path": train_img_path, "gt_path": train_gt_path}
            val_cache = {"img_path": val_img_path, "gt_path": val_gt_path}
            with open(Path(train_cache_path), 'wb') as file:
                # 使用pickle.dump()函数将字典写入文件
                pickle.dump(train_cache, file)
                print(f"Saving train dataset cache to {train_cache_path}...")
            with open(Path(val_cache_path), 'wb') as file:
                # 使用pickle.dump()函数将字典写入文件
                pickle.dump(val_cache, file)
                print(f"Saving val dataset cache to {val_cache_path}...")
        else:
            with open(Path(train_cache_path), 'rb') as file:
                # 使用pickle.load()函数从文件中加载对象
                train_cache = pickle.load(file)
                print(f"Using train dataset cache from {train_cache_path}...")
            with open(Path(val_cache_path), 'rb') as file:
                # 使用pickle.load()函数从文件中加载对象
                val_cache = pickle.load(file)
                print(f"Using train dataset cache from {val_cache_path}...")
        return train_cache, val_cache


class NpyDataset(Dataset):
    def __init__(self, dataset_cache, image_size=256, bbox_shift=10, data_aug=True):
        # self.data_root = data_root
        self.gt_path = dataset_cache["gt_path"]
        self.img_path = dataset_cache["img_path"]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def __len__(self):
        return len(self.gt_path)

    def __getitem__(self, index):
        img_name = basename(self.gt_path[index])
        assert img_name == basename(self.img_path[index]), 'img gt name error' + self.gt_path[index] + self.img_path[
            index]
        img_3c = np.load(self.img_path[index], 'r', allow_pickle=True)  # (H, W, 3)

        # Resizing and normalization
        img_resize = self.resize_longest_side(img_3c)
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8,
                                                               a_max=None)  # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize)  # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (3, 256, 256)
        assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path[index], 'r', allow_pickle=True)  # multiple labels [0, 1,4,5...], (256,256)
        assert gt.max() >= 1, 'gt should have at least one label'
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = self.pad_image(gt)  # (256, 256)
        label_ids = np.unique(gt)[1:]
        try:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))  # only one label, (256, 256)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt))  # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :, :]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(),  # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3:  ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:  ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded