# --- Imports --- #
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage.metrics import structural_similarity as compare_ssim

from torchvision.utils import save_image


def to_psnr(clearImg_recons, gt):
    mse = F.mse_loss(clearImg_recons, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(clearImg_recons, gt):
    clearImg_recons_list = torch.split(clearImg_recons, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    clearImg_recons_list_np = [clearImg_recons_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(clearImg_recons_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(clearImg_recons_list))]
    ssim_list = [compare_ssim(clearImg_recons_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=2) for ind in range(len(clearImg_recons_list))]

    return ssim_list


def validation(net_dehaze, val_data_loader, device, save_tag=False):
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():

            clear, hazy, image_name = val_data
            clear = clear.to(device)
            hazy = hazy.to(device)
            F11, F22, Dehaze_Img = net_dehaze(hazy)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(Dehaze_Img, clear))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(Dehaze_Img, clear))

        # --- Save image --- #
        if save_tag:
            save_image(Dehaze_Img, image_name)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(clearImg_recons, image_name):
    clearImg_recons_images = torch.split(clearImg_recons, 1, dim=0)
    batch_num = len(clearImg_recons_images)

    for ind in range(batch_num):
        utils.save_image(clearImg_recons_images[ind], './results/dehazy_Img/{}'.format(image_name[ind][:-3] + 'png'))



