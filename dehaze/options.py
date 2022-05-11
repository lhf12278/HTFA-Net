import argparse

parser = argparse.ArgumentParser(description=None)

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for HTFA-Net')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=1, type=int)
parser.add_argument('-num_epochs', help='the epochs', default=70, type=int)
parser.add_argument('-num_workers', help='the works', default=10, type=int) 
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)

# model paramaters
parser.add_argument('--input_channels', type=int, default=3, help='the input channels')
parser.add_argument('--out_channels', type=int, default=3, help='the out_channels')
parser.add_argument('--G', type=int, default=32, help='the growth rate')
parser.add_argument('--G0', type=int, default=64, help='local and global feature fusion layers 64filter')
parser.add_argument('--kernel_size', type=int, default=3, help='the kernel_size')
parser.add_argument('--C', type=int, default=3, help='the number of conv layer in RDB 3')
parser.add_argument('--D', type=int, default=7, help='RDB number 7')
parser.add_argument('--D_dehaze', type=int, default=7, help='RDB number 7')
parser.add_argument('--channels', type=int, default=64, help='the channels of Attention')

# The pathes of some files to save.
parser.add_argument('--train_data_dir', type=str, default='./data/train/')
parser.add_argument('--val_data_dir', type=str, default='./data/test/')
parser.add_argument('--log_path', type=str, default='logs/', help='the dir of log to save')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
