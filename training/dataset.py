import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch


class InferenceDataset(Dataset):
    """Basic Pytorch datatset"""
    def __init__(self, image_dir, masks_dir, n_classes, augmentation=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.n_classes = n_classes
        self.augmentation = augmentation

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.image_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_name = self.ids[idx]

        # get file path regardless of extension
        image_path = glob(os.path.join(self.image_dir, f'{image_name}.*'))
        mask_path = glob(os.path.join(self.masks_dir, f"{image_name}.*"))

        image = Image.open(image_path[0])
        mask = Image.open(mask_path[0])   # convert to one-hot

        assert image.size == mask.size, f"Image and mask should be the same size, but are {image.size}, and {mask.size}"

        image = self._preprocess(image)
        # mask = self._preprocess(mask)
        mask = np.array(mask)[..., np.newaxis].transpose((2, 0, 1))
        # mask = pre_process_mask(mask, classes=self.n_classes)

        if self.augmentation:
            # todo
            sample = self.augmentation(image=image, mask=mask)

        response = {
            "image": torch.from_numpy(image).type(torch.FloatTensor),
            "mask": torch.from_numpy(mask).type(torch.FloatTensor)
        }
        return response

    @classmethod
    def _preprocess(cls, image):
        # width, height = image.size

        image_array = np.array(image)

        if len(image_array.shape) == 2:
            image_array = image_array[..., np.newaxis]  # (height, width) -> (height, width, 1)

        # HWC -> CHW
        image_trans = image_array.transpose((2, 0, 1))

        # todo normalize
        if image_trans.max() > 1:
            image_trans = image_trans / 255

        return image_trans
