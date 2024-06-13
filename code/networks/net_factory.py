from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3, UNet_UAPS, UNet_UAPSv2
from networks.TFCNet import TFCNetv2


def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()

    elif net_type == "tfcnet_v2":
        net = TFCNetv2(in_chns=in_chns, class_num=class_num).cuda()

    return net
