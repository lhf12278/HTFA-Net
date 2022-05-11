import torchvision
import torch.optim
from PIL import Image

import glob
from torchvision.transforms import Compose, ToTensor
import model
from Syn_haze import *


def dehaze_image(Real_image_path, Cleal_image_path1, index):
    trans = Compose([ToTensor()])
    ################################################################
    data_hazy_Real = Image.open(Real_image_path).convert('RGB')
    data_Real_haze1 = (np.asarray(data_hazy_Real)/255.0)

    data_Real_haze = torch.from_numpy(data_Real_haze1).float()
    data_Real_haze = data_Real_haze.permute(2, 0, 1)
    data_Real_haze = data_Real_haze.cuda().unsqueeze(0)

    data_Clear = Image.open(Cleal_image_path1)
    data_Clear = trans(data_Clear).cuda().unsqueeze(0)

    trans_map_net = model.Trans_fineNet().cuda()
    # trans_map_net = nn.DataParallel(trans_map_net)
    trans_map_net.load_state_dict(torch.load('snapshots/transmap_fine.pth'))
    trans_map_net.eval()

    transMap = trans_map_net(data_Real_haze)
    transMap1 = torch.cat([transMap] * 3, 1)
    Airlight = get_airlight(Real_image_path)

    Airlight1 = (np.asarray(Airlight) / 255.0)
    Airlight0 = torch.from_numpy(Airlight1).float()
    Airlight2 = Airlight0.permute(2, 0, 1)
    Airlight3 = Airlight2.cuda().unsqueeze(0)

    Hyn_Haze_img = get_synhazeImage(data_Clear, Airlight3, transMap1)
    Hyn_Haze_img = Hyn_Haze_img.cuda()

    torchvision.utils.save_image(torch.cat((data_Real_haze, transMap1, Airlight3, Hyn_Haze_img, data_Clear), 0), "test_data/" + Real_image_path.split("/")[-1])


if __name__ == '__main__':
    Real_image_list = glob.glob("test_data/RealHaze_image/*")
    Cleal_imagel_list = glob.glob("test_data/Clear_image/*")

    for index in range(len(Real_image_list)):
        Realhaze_image = Real_image_list[index]
        Clear_image = Cleal_imagel_list[index]

        dehaze_image(Realhaze_image, Clear_image, index)
        print("Test done!")

