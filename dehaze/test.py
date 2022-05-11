# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from evaluation import *
from dehaze_model_v2 import *
from utils import *
from options import args

import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############################################################
val_batch_size = args.val_batch_size
val_data_dir = args.val_data_dir
crop_size = args.crop_size
############################################################

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\n'.format(val_batch_size))


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=0)


# --- Define the network --- #
net_dehaze = dehaze_network(args)


# --- Multi-GPU --- #
net_dehaze = net_dehaze.to(device)
net_dehaze = nn.DataParallel(net_dehaze, device_ids=device_ids)

# --- Load the network weight --- #
net_dehaze.load_state_dict(torch.load('checkpoint/dehaze_best_model.pth', map_location='cuda:0'))

# --- Use the evaluation model in testing --- #
net_dehaze.eval()
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation(net_dehaze, val_data_loader, device, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))


