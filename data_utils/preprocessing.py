import numpy as np


def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img


def preprocess(img):

    image_array = np.array(img)
    if len(image_array.shape) == 2:
        image_array = image_array[..., np.newaxis]  # (height, width) -> (height, width, 1)

    # HWC -> CHW
    image_trans = image_array.transpose((2, 0, 1))

    if image_trans.max() > 1:
        image_trans = image_trans / 255

    return image_trans
