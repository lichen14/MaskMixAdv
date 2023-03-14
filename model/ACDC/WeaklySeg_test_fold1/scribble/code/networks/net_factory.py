# from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import *
from torch import nn
import torch.nn.functional as F

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_mae":
        net = UNet_MAE(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_mae_cct":
        net = UNet_MAE_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_mae_cct_0109":   #feature dropout
        net = UNet_MAE_CCT_0109(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_mae_cct_0110":   #feature masked dropout
        net = UNet_MAE_CCT_0110(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_mae_cct_0111":   #feature masked dropout
        net = UNet_MAE_CCT_0111(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_mae_cct_0117":   #feature masked 2 branch
        net = UNet_MAE_CCT_0117(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    else:
        net = None
    return net

# def get_fc_discriminator(num_classes, ndf=64):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=8, stride=2, padding=1),    #每次尺寸缩小2倍
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=8, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=8, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 8, kernel_size=8, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 8, ndf * 12, kernel_size=6, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 12, 1, kernel_size=6, stride=2, padding=1),
#     )
def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),    #每次尺寸缩小2倍
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)