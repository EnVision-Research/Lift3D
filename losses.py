import math
import torch
import torchvision
from torch import nn
from torch import autograd
from torch.nn import functional as F

EPS = 1e-7


def loss_color(color, gt, mask):
    mask_sum = mask.sum() + 1e-5
    color_error = (color - gt) * mask
    return (
        F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum")
        / mask_sum
    )


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        h_delta = torch.cat([x[:, :, :, 1:, :], x[:, :, :, -1:, :]], 3)
        w_delta = torch.cat([x[:, :, :, :, 1:], x[:, :, :, :, -1:]], 4)

        temp = (h_delta - x) ** 2 + (w_delta - x) ** 2
        temp = torch.sqrt(temp + 1e-6).mean()

        return temp


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


def mask_iou(lhs_mask, rhs_mask):
    r"""Compute the Intersection over Union of two segmentation masks.
    Args:
        lhs_mask (torch.FloatTensor):
            A segmentation mask, of shape
            :math:`(\text{batch_size}, \text{height}, \text{width})`.
        rhs_mask (torch.FloatTensor):
            A segmentation mask, of shape
            :math:`(\text{batch_size}, \text{height}, \text{width})`.
    Returns:
        (torch.FloatTensor): The IoU loss, as a torch scalar.
    """
    batch_size, height, width = lhs_mask.shape
    assert rhs_mask.shape == lhs_mask.shape
    sil_mul = lhs_mask * rhs_mask
    sil_add = lhs_mask + rhs_mask
    iou_up = torch.sum(sil_mul.reshape(batch_size, -1), dim=1)
    iou_down = torch.sum((sil_add - sil_mul).reshape(batch_size, -1), dim=1)
    iou_neg = iou_up / (iou_down + 1e-10)
    mask_loss = 1.0 - torch.mean(iou_neg)
    return mask_loss


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer("mean_rgb", mean_rgb)
        self.register_buffer("std_rgb", std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        # out = x/2 + 0.5
        out = (x - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1, im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1 - f2) ** 2
            if conf_sigma is not None:
                loss = loss / (2 * conf_sigma**2 + EPS) + (conf_sigma + EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm // h, wm // w
                mask0 = nn.functional.avg_pool2d(
                    mask, kernel_size=(sh, sw), stride=(sh, sw)
                ).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


def viewpoints_loss(viewpoint_pred, viewpoint_target):
    loss = F.smooth_l1_loss(viewpoint_pred, viewpoint_target)

    return loss


def eikonal_loss(eikonal_term, sdf=None, beta=100):
    if eikonal_term == None:
        eikonal_loss = 0
    else:
        eikonal_loss = ((eikonal_term.norm(dim=-1) - 1) ** 2).mean()

    if sdf == None:
        minimal_surface_loss = torch.tensor(0.0, device=eikonal_term.device)
    else:
        minimal_surface_loss = torch.exp(-beta * torch.abs(sdf)).mean()

    return eikonal_loss, minimal_surface_loss


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )

    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def d_r1_loss_scale(real_pred, real_img, scaler):
    grad_real = autograd.grad(
        outputs=scaler.scale(real_pred.sum()), inputs=real_img, create_graph=True
    )
    # grad_real, = autograd.grad(outputs=scaler.scale(real_pred.sum()),
    #                            inputs=real_img,
    #                            create_graph=True)

    inv_scale = 1.0 / scaler.get_scale()
    grad_params = [p * inv_scale for p in grad_real]

    return grad_params


def smooth_l1_loss(pred, target):
    loss = F.smooth_l1_loss(pred, target)

    return loss


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents,
        create_graph=True,
        only_inputs=True,
    )[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths
