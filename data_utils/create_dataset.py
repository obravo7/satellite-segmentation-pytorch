"""script used to create base datset"""

import os
import glob
import numpy as np
from PIL import Image

from data_utils.common import load_json, create_mask, tile_image
from data_utils import colors

# set base dirs
data_path = 'data/colors'
mask_path = 'data/masks'
raw_path = 'data/raw'
color_mask_path = 'data/colors'
mask_save_path = 'data/train/masks'
image_save_path = 'data/train/images'
color_save_path = 'data/train/color'


colors_from_hex = {
    "#ff0000": colors.hex_to_rgb('#ff0000'),
    "#0037ff": colors.hex_to_rgb('#0037ff'),
    '#f900ff': colors.hex_to_rgb('#f900ff')
}


train_labels = {
    "#ff0000": (1, 1, 1),  # houses
    "#0037ff": (2, 2, 2),  # buildings
    '#f900ff': (3, 3, 3)   # Sheds/Garages
}


def create_base_dataset():

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    json_paths = glob.glob('annotations/*.json')
    for json_path in json_paths:

        annotation_data = load_json(json_path)
        label_data = annotation_data['labels']

        # create mask data
        height, width = annotation_data['height'], annotation_data['width']
        mask = np.ones((height, width, 3), dtype=np.uint8) * 255
        train_mask = np.zeros((height, width), dtype=np.uint8)  # background is a class, category 0

        mask = create_mask(mask, label_data=label_data, pixel_labels=colors_from_hex)
        train_mask = create_mask(train_mask, label_data=label_data, pixel_labels=train_labels)

        file_name = os.path.basename(json_path).split('.')[0]
        Image.fromarray(mask, mode="RGB").save(os.path.join(data_path, f"{file_name}.png"))

        print(f"{train_mask.shape}")
        Image.fromarray(train_mask, mode="L").save(os.path.join(mask_path, f"{file_name}.png"))


def create_train_dataset():

    mask_image_list = glob.glob(os.path.join(mask_path, '*.png'))

    # There are 9 tiles that are left out as evaluation (testing)
    # diff = list(set(raw_base_names) - set(mask_base_names))

    i = 1
    for mask_file_path in mask_image_list:
        file_name = os.path.basename(mask_file_path)
        img_file_path = os.path.join(raw_path, file_name)
        color_msk_file_path = os.path.join(color_mask_path, file_name)

        img = np.array(Image.open(img_file_path))
        mask = np.array(Image.open(mask_file_path))
        color = np.array(Image.open(color_msk_file_path))

        tile_image(image=img,
                   save_path=os.path.join(image_save_path, file_name.split('.')[0])
                   )
        tile_image(image=mask,
                   save_path=os.path.join(mask_save_path, file_name.split('.')[0])
                   )
        tile_image(image=color,
                   save_path=os.path.join(color_save_path, file_name.split('.')[0])
                   )
        print(f'files completed: \t{i}/{len(mask_image_list)}...', end='\r')
        i += 1
