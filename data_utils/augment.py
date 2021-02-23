import cv2
import random
import numpy as np
from PIL import Image
from typing import Tuple


def gaussBlur(image, filter_size=15):
    """
     gaussian blur
    """
    blur = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    return blur


def medBlur(image, filter_size=5):
    blur = cv2.medianBlur(image, filter_size)
    return blur


def bilateralBlur(image):
    blur = cv2.bilateralFilter(image, 9, 75, 75)
    return blur


def totalBlur(image, filter_size=11):
    mb = medBlur(image, filter_size)
    total = bilateralBlur(mb)
    return total


def distort_elastic_cv2(image, alpha=40, sigma=3, random_state=None):
    """
        Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape_size = image.shape[:2]

    # Downscale the random grid and then upsizing post filter
    # improve performance

    grid_scale = 4
    alpha //= grid_scale
    sigma //= grid_scale
    grid_shape = (shape_size[0] // grid_scale, shape_size[1] // grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    rand_y = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, shape_size[::-1])
        rand_y = cv2.resize(rand_y, shape_size[::-1])

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y,
                              borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

    return distorted_img


def rotation_invariance(img):
    """
    rotate and shift images randomly
    """
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (rows, cols))
    return dst


def speckle_noise(img):
    """
    add multiplicative speckle noise
    used for radar images
    """
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    noisy = img * (gauss / (len(gauss) - 0.50 * len(gauss)))

    return noisy


def salt_pepper_noise(img, prob):
    """
    salt and pepper noise
    prob: probability of noise
    """
    output = np.zeros(img.shape, np.uint8)
    thresh = 1 - prob

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thresh:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def mirror(img):
    """
    horizontal mirror of image
    """
    mirror = cv2.flip(img, +1)
    return mirror


def rotate(img, angle):
    """
    rotate image 90 degrees
    """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (h, w))
    return rotated


def to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def random_crop_and_resize(img, mask, size=(512, 512)) -> Tuple[Image.Image, Image.Image]:
    height, width = 256, 256
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    img = Image.fromarray(img, "RGB").resize(size, Image.ANTIALIAS)
    mask = Image.fromarray(mask, 'L').resize(size, Image.ANTIALIAS)
    return img, mask

