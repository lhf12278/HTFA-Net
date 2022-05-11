import numpy as np
import cv2
import torch
import matplotlib.image as mpimg


def get_airlight(Real_image_path):

    hzimg = mpimg.imread(Real_image_path)
    airlight = np.zeros(hzimg.shape)
    kernel = np.ones((15, 15), np.uint8)

    for i in range(3):
        a = hzimg[:, :, i]
        img = cv2.erode(a, kernel, iterations=1)
        airlight[:, :, i] = np.amax(img)

    return airlight


def get_synhazeImage(Clear_img, airlight, transMap1):

    transMap1 = transMap1.cuda()
    synhazeImage = torch.zeros(Clear_img.shape)

    synhazeImage[:, 0] = Clear_img[:, 0] * transMap1[:, 0] + (torch.ones(transMap1[:, 0].shape).cuda() - transMap1[:, 0]) * airlight[:, 0]
    synhazeImage[:, 1] = Clear_img[:, 1] * transMap1[:, 1] + (torch.ones(transMap1[:, 1].shape).cuda() - transMap1[:, 1]) * airlight[:, 1]
    synhazeImage[:, 2] = Clear_img[:, 2] * transMap1[:, 2] + (torch.ones(transMap1[:, 2].shape).cuda() - transMap1[:, 2]) * airlight[:, 2]
    synhazeImage[synhazeImage < 0.0] = 0.0
    synhazeImage[synhazeImage > 1.0] = 1.0

    return synhazeImage
