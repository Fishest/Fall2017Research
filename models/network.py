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
        for i in range(depth - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)



class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=False, outermost=False,
        innermost=False, use_dropout=False, norm_layer=nn.BatchNorm2d,
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downReLU = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(downReLU)

        upReLU = nn.LeakyReLU(True)
        upnorm = norm_layer(upReLU)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [upReLU, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downReLU, downconv]
            up = [upReLU, upconv]
            model = down + [submodule] + up
        else:
            upconv = nn.ConvTranspose2d(inner * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downReLU, downconv, downnorm]
            up = [upReLU, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)





if __name__ == "__main__":
    define_G()
