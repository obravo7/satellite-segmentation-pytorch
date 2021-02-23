import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torch

from data_utils.preprocessing import preprocess


class InferenceDataset(Dataset):
    """Basic Pytorch datatset"""
    def __init__(self, image_dir, masks_dir, n_classes, augmentation=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.n_classes = n_classes
        self.augmentations = augmentation

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

        image = preprocess(image)

        mask = np.array(mask)[..., np.newaxis].transpose((2, 0, 1))

        n_labels = np.unique(mask)
        assert len(n_labels) == self.n_classes, f"image has too many labels: {image_path[0]}"

        mask = mask.transpose((2, 0, 1))

        if self.augmentations:
            augment = self.augmentations(image=image, mask=mask)
            image = augment['image']
            mask = augment['mask']
            response = {
                "image": image,
                "mask": mask
            }
        else:
            response = {
                "image": torch.from_numpy(image).type(torch.FloatTensor),
                "mask": torch.from_numpy(mask).type(torch.FloatTensor)
            }

        return response
