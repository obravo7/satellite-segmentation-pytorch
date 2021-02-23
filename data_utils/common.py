import numpy as np
import cv2
from PIL import Image
import json
from typing import Any


def create_mask(mask: np.ndarray, label_data: dict, pixel_labels: dict) -> np.ndarray:

    for label in label_data:

        hex_label = label['color']
        # category = label['name']
        color = pixel_labels[hex_label]

        for vector_points in label['annotations']:

            x_values = [i for i in vector_points['segmentation'][::2]]
            y_values = [i for i in vector_points['segmentation'][1::2]]
            contours = np.array([(x, y) for x, y in zip(x_values, y_values)])
            mask = cv2.drawContours(mask, [contours.astype(int)], -1, color, -1)

    return mask


def tile_image(image: np.ndarray, save_path, size=512) -> None:
    if len(image.shape) == 2:
        image = image[..., np.newaxis]  # (h, w) -> (h, w, 1)
    height, width, channels = image.shape

    h_stride, h_diff = divmod(height, size)
    w_stride, w_diff = divmod(width, size)

    h_stride = h_stride + 1 if h_diff > 0 else h_stride
    w_stride = w_stride + 1 if w_diff > 0 else w_diff

    for h in range(h_stride):
        for w in range(w_stride):
            tile = image[
                   size * h: size + (size * h),  # tile height of image
                   size * w: size + (size * w),  # tile width of image
                   :]

            if tile.shape[0] != size and tile.shape[1] != size:
                # adjust both height and width
                # shift according to height and width difference
                tile = image[
                       (size * h) - (size - h_diff): (size + (size * h)) - (size - h_diff),
                       (size * w) - (size - w_diff):(size + (size * w)) - (size - w_diff),
                       :]

            elif tile.shape[1] != size:
                # adjust width
                tile = image[
                       size * h: size + (size * h),
                       # shift back according to w_diff
                       (size * w) - (size - w_diff):(size + (size * w)) - (size - w_diff),
                       :]

            elif tile.shape[0] != size:  # height

                # adjust height
                tile = image[
                       # shift back according to height difference
                       (size * h) - (size - h_diff): (size + (size * h)) - (size - h_diff),
                       size * w: size + (size * w),  # stride width of image
                       :]

            else:
                # normal sequence; do nothing
                pass

            if channels == 1:
                Image.fromarray(tile.reshape((size, size)), mode='L').save(f'{save_path}-{h}_{w}.png')
            else:
                Image.fromarray(tile, mode='RGB').save(f'{save_path}-{h}_{w}.png')


def load_json(json_path: str):
    with open(json_path, 'rb') as f:
        data = json.load(f)
    return data


class EasyDict(dict):
    # adopted from https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/util.py

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
