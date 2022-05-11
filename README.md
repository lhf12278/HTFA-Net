
# Haze transfer and feature aggregation network for real-world single image dehazing

This package contains the source code which is associated with the following paper:

Huafeng Li, Jirui Gao, Yafei Zhang, Minghong Xie, Zhengtao Yu, “Haze transfer and feature aggregation network for real-world single image dehazing.”

Edited by Jirui Gao

Usage of this code is free for research purposes only. 

Thank you.

# Requirements:
    CUDA  10.0
    Python  3.7.7
    Pytorch  1.5.0
    torchvision  0.6.0
    numpy  1.17.4

# Get Started
## 1.Install:
    download the code
    git clone https://github.com/lhf12278/HTFA-Net.git
    cd HTFA-Net
    
## 2.Datasets
- Download the synthetic hazy datasets that were generated by ourselves  through the links below:
*HTset*:[Baidu Pan](https://pan.baidu.com/s/1-JpjboWNf0dhwuId2vslPQ)（password：36sv）

## 3.If you dont want to download the synthetic hazy datasets, you can also generate by yourself. Put your Real Images into '''test_data/RealHaze_image''' and put Clear Images into '''test_data\Clear_image''':
    cd HTFA-Net/Generating_HazeImages
    python test.py

## 4.Dehazing:
### Dehazing the hazy images that were generated by ourselves.
        cd HTFA-Net/dehaze
        python test.py

### Dehazing the real hazy images: Replace the '''dehaze/checkpoint/dehaze_best_model.pth''' file with the '''dehaze_best_model.pth''' file in '''dehaze/checkpoint/real/'''.
        cd HTFA-Net/dehaze
        python test.py

# Contact:
    Don't hesitate to contact me if you meet any problems when using this code.

    Jirui Gao
    Faculty of Information Engineering and Automation
    Kunming University of Science and Technology                                                           
    Email: 2945266153@qq.com

# Acknowledgements
Our code is based on https://proteus1991.github.io/GridDehazeNet,https://github.com/zhilin007/FFA-Net and https://github.com/HUSTSYJ/DA_dahazing.