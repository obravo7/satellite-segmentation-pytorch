import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiClassCriterion(nn.Module):
    def __init__(self, loss_type='CrossEntropy', **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            self.criterion = LovaszSoftmax(**kwargs)
        elif loss_type == 'SoftIOU':
            self.criterion = SoftIoULoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss


class FocalLoss(nn.Module):
    """
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class LovaszSoftmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss
      logits: [B, C, H, W] class logits at each prediction (between -\infty and \infty)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      ignore_index: void class labels
      only_present: average only on classes present in ground truth
    """
    def __init__(self, ignore_index=None, only_present=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        total_loss = 0
        batch_size = logits.shape[0]
        for prb, lbl in zip(probas, labels):
            total_loss += lovasz_softmax_flat(prb, lbl, self.ignore_index, self.only_present)
        return total_loss / batch_size


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(prb, lbl, ignore_index, only_present):
    """
    Multi-class Lovasz-Softmax loss
      prb: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      lbl: [P] Tensor, ground truth labels (between 0 and C - 1)
      ignore_index: void class labels
      only_present: average only on classes present in ground truth
    """
    C = prb.shape[0]
    prb = prb.permute(1, 2, 0).contiguous().view(-1, C)  # H * W, C
    lbl = lbl.view(-1)  # H * W
    if ignore_index is not None:
        mask = lbl != ignore_index
        if mask.sum() == 0:
            return torch.mean(prb * 0)
        prb = prb[mask]
        lbl = lbl[mask]

    total_loss = 0
    cnt = 0
    for c in range(C):
        fg = (lbl == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - prb[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        total_loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        cnt += 1
    return total_loss / cnt


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, logit, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(logit)

        pred = F.softmax(logit, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()
