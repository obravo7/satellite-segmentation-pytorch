import torch

from models.unet import UNet
from training import training_loop
from data_utils.common import EasyDict


def setup_run_arguments():
    args = EasyDict()
    args.epochs = 100
    args.batch = 4
    args.val_percent = 0.2
    args.n_classes = 4
    args.n_channels = 3
    args.num_workers = 8

    args.learning_rate = 0.001
    args.weight_decay = 1e-8
    args.momentum = 0.9
    args.save_cp = True
    args.loss = "CrossEntropy"

    args.checkpoint_path = 'checkpoints/'
    args.image_dir = 'data/train/images'
    args.mask_dir = 'data/train/masks'

    args.from_pretrained = False

    return args


def train():
    args = setup_run_arguments()

    # args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing UNet-model using: {device}")

    net = UNet(n_channels=args.n_channels, n_classes=args.n_classes, bilinear=True)

    if args.from_pretrained:
        net.load_state_dict(torch.load(args.from_pretrained, map_location=device))

    net.to(device=device)

    training_loop.run(network=net,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.learning_rate,
                      device=device,
                      n_classes=args.n_classes,
                      val_percent=args.val_percent,
                      image_dir=args.image_dir,
                      mask_dir=args.mask_dir,
                      checkpoint_path=args.checkpoint_path,
                      loss=args.loss,
                      num_workers=args.num_workers
                      )


if __name__ == "__main__":
    train()
