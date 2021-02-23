"""script used to create base datset"""

import os
import glob
import numpy as np
from PIL import Image

from data_utils import common, colors, augment

# data paths
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
    json_paths = glob.glob('data/annotations/*.json')
    for json_path in json_paths:

        annotation_data = common.load_json(json_path)
        label_data = annotation_data['labels']

        # create mask data
        height, width = annotation_data['height'], annotation_data['width']
        mask = np.ones((height, width, 3), dtype=np.uint8) * 255
        train_mask = np.zeros((height, width), dtype=np.uint8)  # background is a class, category 0

        mask = common.create_mask(mask, label_data=label_data, pixel_labels=colors_from_hex)
        train_mask = common.create_mask(train_mask, label_data=label_data, pixel_labels=train_labels)

        file_name = os.path.basename(json_path).split('.')[0]
        Image.fromarray(mask, mode="RGB").save(os.path.join(data_path, f"{file_name}.png"))

        print(f"{train_mask.shape}")
        Image.fromarray(train_mask, mode="L").save(os.path.join(mask_path, f"{file_name}.png"))


def create_train_dataset():

    mask_image_list = glob.glob(os.path.join(mask_path, '*.png'))

    # There are 9 tiles that are left out as evaluation (testing)
    # diff = list(set(raw_base_names) - set(mask_base_names))
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)
    os.makedirs(color_save_path, exist_ok=True)

    i = 1
    for mask_file_path in mask_image_list:
        file_name = os.path.basename(mask_file_path)
        img_file_path = os.path.join(raw_path, file_name)
        color_msk_file_path = os.path.join(color_mask_path, file_name)

        img = np.array(Image.open(img_file_path))
        mask = np.array(Image.open(mask_file_path))
        color = np.array(Image.open(color_msk_file_path))

        common.tile_image(image=img,
                          save_path=os.path.join(image_save_path, file_name.split('.')[0])
                          )
        common.tile_image(image=mask,
                          save_path=os.path.join(mask_save_path, file_name.split('.')[0])
                          )
        common.tile_image(image=color,
                          save_path=os.path.join(color_save_path, file_name.split('.')[0])
                          )
        print(f'files completed: \t{i}/{len(mask_image_list)}...', end='\r')
        i += 1


def augment_train_data():
    i = 1
    base_names = [os.path.basename(fn) for fn in glob.glob(os.path.join(mask_save_path, "*.png"))]
    for fn in base_names:
        msk_fp = os.path.join(mask_save_path, fn)
        img_fp = os.path.join(image_save_path, fn)
        color_fp = os.path.join(color_save_path, fn)

        msk = np.array(Image.open(msk_fp))
        img = np.array(Image.open(img_fp))
        color = np.array(Image.open(color_fp))

        # total blur
        Image.fromarray(augment.totalBlur(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-blur.png'))
        Image.fromarray(msk, mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-blur.png'))
        Image.fromarray(color, mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-blur.png'))

        # distort elastic
        Image.fromarray(augment.distort_elastic_cv2(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-distort.png'))
        Image.fromarray(msk, mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-distort.png'))
        Image.fromarray(color, mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-distort.png'))

        # mirror
        Image.fromarray(augment.mirror(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-mirror.png'))
        Image.fromarray(augment.mirror(msk), mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-mirror.png'))
        Image.fromarray(augment.mirror(color), mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-mirror.png'))

        # rotation invariance
        Image.fromarray(augment.rotation_invariance(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-rt-inv.png'))
        Image.fromarray(augment.rotation_invariance(msk), mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-rt-inv.png'))
        Image.fromarray(augment.rotation_invariance(color), mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-rt-inv.png'))

        # gaussian blur
        Image.fromarray(augment.gaussBlur(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-gauss.png'))
        Image.fromarray(msk, mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-gauss.png'))
        Image.fromarray(color, mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-gauss.png'))

        # distort elastic + rotation
        rot = 90
        Image.fromarray(augment.distort_elastic_cv2(augment.rotate(img, rot)), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-distort-rt.png'))
        Image.fromarray(augment.rotate(msk, rot), mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-distort-rt.png'))
        Image.fromarray(augment.rotate(color, rot), mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-distort-rt.png'))

        # gaussian blur + rot
        rot = 270
        Image.fromarray(augment.gaussBlur(augment.rotate(img, rot)), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-gauss-rot.png'))
        Image.fromarray(augment.rotate(msk, rot), mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-gauss-rot.png'))
        Image.fromarray(augment.rotate(color, rot), mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-gauss-rot.png'))

        # HSV
        Image.fromarray(augment.to_hsv(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-hsv.png'))
        Image.fromarray(msk, mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-hsv.png'))
        Image.fromarray(color, mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-hsv.png'))

        # increase brightness
        Image.fromarray(augment.increase_brightness(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-bright.png'))
        Image.fromarray(msk, mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-bright.png'))
        Image.fromarray(color, mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-bright.png'))

        # median blur
        Image.fromarray(augment.medBlur(img), mode='RGB').save(
            os.path.join(image_save_path, f'{fn.split(".")[0]}-med-blur.png'))
        Image.fromarray(msk, mode='L').save(
            os.path.join(mask_save_path, f'{fn.split(".")[0]}-med-blur.png'))
        Image.fromarray(color, mode='RGB').save(
            os.path.join(color_save_path, f'{fn.split(".")[0]}-med-blur.png'))

        # crop and resize
        # img_cr, mask_cr = augment.random_crop_and_resize(img, msk)
        # img_cr.save(os.path.join(image_save_path, f'{fn.split(".")[0]}-crop-resize.png'))
        # mask_cr.save(os.path.join(mask_save_path, f'{fn.split(".")[0]}-crop-resize.png'))
        # Image.fromarray(color, mode='RGB').save(
        #     os.path.join(color_save_path, f'{fn.split(".")[0]}-crop-resize.png'))

