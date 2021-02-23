from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from training.loss import MultiClassCriterion, dice_coeff
from training.dataset import InferenceDataset


def run(
        image_dir,
        mask_dir,
        network,
        device,
        n_classes,
        checkpoint_path,
        epochs=50,
        batch_size=4,
        val_percent=0.1,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-8,
        save_cp=True,
        loss="CrossEntropy",
        num_workers=8,
):

    dataset = InferenceDataset(image_dir=image_dir, masks_dir=mask_dir, n_classes=n_classes)
    n_val = int(len(dataset) * val_percent)  # validation set size
    n_train = len(dataset) - n_val   # training set size

    train_data, val_data = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True
                              )

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True
                            )

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min' if network.n_classes > 1 else 'max', patience=2)

    if network.n_classes > 1:
        # loss function: Categorical cross entropy
        print(f'[INFO] using {loss} loss...')
        # criterion = nn.CrossEntropyLoss()
        criterion = MultiClassCriterion(loss_type=loss)
    else:
        # binary cross entropy, where only two classes exist, including the background
        print('[INFO] using binary cross entropy...')
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        network.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit='image') as progress_bar:

            for batch in train_loader:
                image = batch['image']
                target = batch['mask']

                image = image.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if network.n_classes == 1 else torch.long
                target = target.to(device=device, dtype=mask_type).squeeze(1)

                mask_pred = network(image)

                # prediction should be a FloadTensor of shape (batch, n_classes, h, w)
                # target should be a LongTensor of shape (batch, h, w)
                loss = criterion(mask_pred, target=target)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                # update progress bar
                progress_bar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(network.parameters(), 0.1)
                optimizer.step()

                progress_bar.update(image.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in network.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    validation_score = eval_net(network, val_loader, device)
                    scheduler.step(validation_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if network.n_classes > 1:
                        writer.add_scalar('Loss/test', validation_score, global_step)
                    else:
                        writer.add_scalar('Dice/test', validation_score, global_step)

                    writer.add_images('images', image, global_step)
                    if network.n_classes == 1:
                        writer.add_images('masks/true', mask_pred, global_step)
                        writer.add_images('mask/pred', torch.sigmoid(mask_pred) > 0.5, global_step)

            if save_cp:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(network.state_dict(),
                           os.path.join(checkpoint_path, f'CP_epoch{epoch + 1}.pth')
                           )

        writer.close()

    torch.save(network.state_dict(),
               os.path.join(checkpoint_path, f'CP_epochs-{epochs}-final.pth')
               )


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks.squeeze(1)).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val
