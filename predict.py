import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from data_utils.preprocessing import preprocess
from data_utils import colors

colors_from_hex = {
    "0": (255, 255, 255),  # background
    "1": colors.hex_to_rgb('#ff0000'),
    "2": colors.hex_to_rgb('#0037ff'),
    "3": colors.hex_to_rgb('#f900ff')
}


def predict_on_image(net, src_img, device, thresh=0.5):
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


def convert_to_json():
    pass
