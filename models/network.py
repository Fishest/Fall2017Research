import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.wieght.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

    return norm_layer

def define_G(
    input_nc,
    output_nc,
    ngf,
    which_model_net_G,
    norm='batch',
    use_dropout=False,
    gpu_ids=[],
):
    netG = None
    use_gpu = True if gpu_ids else False
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.available())

    # if which_model_net_G == 'resnet_9blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpuids)
    #
    if which_model_net_G == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_net_G == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise("Generator model name %s not recognized" % which_model_net_G)

    if gpu_ids:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG



class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, depth,
        ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[],
    ):
        super(UNetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        assert(input_nc == output_nc)

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)



class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, use_dropout=False, norm_layer=nn.BatchNorm2d):
        super(UnetSkipConnectionBlock, self).__init__()
        

if __name__ == "__main__":
    define_G()
