import numpy as np
from PIL import Image
import cv2
import os

import torch
import torch.nn.functional as F
from torchvision import transforms

from data_utils.preprocessing import preprocess
from data_utils import colors
from models.unet import UNet

colors_from_hex = {
    "0": (255, 255, 255),  # background
    "1": colors.hex_to_rgb('#ff0000'),
    "2": colors.hex_to_rgb('#0037ff'),
    "3": colors.hex_to_rgb('#f900ff')
}

hex_labels = {
    "0": 'none',
    "1": '#ff0000',
    "2": '#0037ff',
    "3": '#f900ff'

}

category_labels = {
    "0": 'none',
    "1": 'Houses',
    "2": 'Buildings',
    "3": 'Sheds/Garages'

}


def predict_on_image(net, src_img, device, thresh=0.1):
    net.eval()

    img = torch.from_numpy(preprocess(src_img))  # hack

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        out = net(img)  # tensor: [1, n_classes, height, width]

        if net.n_classes > 1:
            probs = F.softmax(out, dim=1)
        else:
            probs = torch.sigmoid(out)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()  # (n_classes, height, width)

    return mask > thresh


def decode_seg_map(image) -> np.ndarray:
    """decode generated segmentation map into 3 channel RGB image."""

    h, w, n_labels = image.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label in range(0, n_labels):
        idx = np.where(image[:, :, label].astype(int) == 1)
        rgb_mask[idx] = colors_from_hex[str(label)]

    return rgb_mask


def prediction_to_json(image_path, chkp_path, net=None) -> dict:
    """
    {'filename':file_name,
    {'labels': [{'name': label_name, 'annotations': [{'id':some_unique_integer_id, 'segmentation':[x,y,x,y,x,y....]}
                                             ....] }
        ....]
        }
    """
    file_name = os.path.basename(image_path)
    annotation = {'filename': file_name, 'labels': []}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not net:
        net = UNet(n_channels=3, n_classes=4)

        net.to(device=device)
        net.load_state_dict(
            torch.load(chkp_path, map_location=device)
        )

    img = Image.open(image_path, )

    msk = predict_on_image(net=net, device=device, src_img=img)
    msk = msk.transpose((1, 2, 0))

    h, w, n_labels = msk.shape
    rgb_mask = np.ones((h, w, 3), dtype=np.uint8)
    annotation['height'] = h
    annotation['width'] = w

    for label in range(1, n_labels):
        color = hex_labels[str(label)]
        category = category_labels[str(label)]
        c_label = {'color': color, 'name': category, 'annotations': []}

        label_mask = msk[:, :, label].astype(int)
        contours, hierarchy = cv2.findContours(label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            vector_points = []
            for x, y in contour.reshape((len(contour), 2)):
                vector_points += [x, y]

            c_label['annotations'].append({'segmentation': vector_points})

        idx = np.where(msk[:, :, label].astype(int) == 1)
        rgb_mask[idx] = colors_from_hex[str(label)]

        annotation['labels'].append(c_label)

    return annotation
