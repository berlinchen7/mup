import torch
from torchvision import transforms, datasets
from mup.shape import set_base_shapes
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from mup.layer import MuReadout
from functools import partial
from mup.init import (kaiming_normal_, kaiming_uniform_, normal_,
                         trunc_normal_, uniform_, xavier_normal_,
                         xavier_uniform_)
from torch.nn.modules.conv import _ConvNd
from einops import rearrange, repeat
from examples.SSMs.ssm import SSM, set_base_shapes_custom, BASEWIDTH


samplers = {
    'default': lambda x: x,
    'const_uniform': partial(uniform_, a=-0.1, b=0.1),
    'const_normal': partial(normal_, std=0.1),
    'const_trunc_normal': partial(trunc_normal_, std=0.1, a=-0.2, b=0.2),
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_fan_in_uniform': partial(kaiming_uniform_, mode='fan_in'),
    'kaiming_fan_in_normal': partial(kaiming_normal_, mode='fan_in'),
    'kaiming_fan_out_uniform': partial(kaiming_uniform_, mode='fan_out'),
    'kaiming_fan_out_normal': partial(kaiming_normal_, mode='fan_out')
}

def init_model(model, sampler):
    for param in model.parameters():
        if len(param.shape) >= 2:
            sampler(param)
    return model

init_methods = {
    k: partial(init_model, sampler=s) for k, s in samplers.items()
}


def get_train_loader(batch_size, num_workers=0, shuffle=False, train=True, download=False):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.CenterCrop(32 // 8),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: torch.reshape(x, (3, x.size(1)*x.size(2)))),
        ])
    trainset = datasets.CIFAR10(root='dataset', train=train,
                                download=download, transform=transform)
    # trainset.to(torch.device('mps'))
    # trainset.targets.to(torch.device('mps'))

    return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)


def _generate_depthwise1dCNN(width, bias=True, mup=True, batchnorm=False, device='cpu'):
    mods = [
        nn.Conv1d(
            in_channels=3,
            out_channels=width,
            bias=bias,
            kernel_size=4,
            groups=3,
            padding=4 - 1,
        ),
        nn.ReLU(inplace=True),

        nn.Conv1d(
            in_channels=width,
            out_channels=width,
            bias=bias,
            kernel_size=4,
            groups=width,
            padding=4 - 1,
        ),
        nn.ReLU(inplace=True),

        nn.Flatten(), 
        # nn.Linear(width, width, bias=bias, device=device),
        # nn.ReLU(inplace=True),
    ]
    if mup:
        # print('mup')
        mods.append(MuReadout(width*1030, 10, bias=bias, readout_zero_init=False, device=device))
    else:
        # print('not mup')
        mods.append(nn.Linear(width*1030, 10, bias=bias, device=device))
    return nn.Sequential(*mods)

def generate_depthwise1dCNN(width, bias=True, mup=True, readout_zero_init=True, batchnorm=False, init='default', bias_zero_init=False, base_width=8):
    if not mup:
        model = _generate_depthwise1dCNN(width, bias, mup, batchnorm)
        # set base shapes to model's own shapes, so we get SP
        return set_base_shapes(model, None)
    # it's important we make `model` first, because of random seed
    model = _generate_depthwise1dCNN(width, bias, mup, batchnorm)
    base_model = _generate_depthwise1dCNN(base_width, bias, mup, batchnorm, device='meta')
    set_base_shapes(model, base_model)
    init_methods[init](model)
    if readout_zero_init:
        readout = list(model.modules())[-1]
        readout.weight.data.zero_()
        if readout.bias is not None:
            readout.bias.data.zero_()
    # if bias_zero_init:
    #     for module in model.modules():
    #         if isinstance(module, (nn.Linear, _ConvNd)) and module.bias is not None:
    #             module.bias.data.zero_()
    return model

def generate_SSM_models(widths, 
                        num_input_channels=3, 
                        d_state=16, # TODO
                        mup=True, 
                        readout_zero_init=False,
                        hyperparam_mode='mup_fullalign',
                        **kernel_args,
                        ):
    base_width = BASEWIDTH

    def gen(w):
        def f():
            base_model = SSM(num_input_channels=num_input_channels,
                        d_model=base_width, d_state=d_state, hyperparam_mode=hyperparam_mode,
                        mup=mup, readout_zero_init=readout_zero_init, **kernel_args)
            model = SSM(num_input_channels=num_input_channels,
                        d_model=w, d_state=d_state, hyperparam_mode=hyperparam_mode,
                        mup=mup, readout_zero_init=readout_zero_init, **kernel_args)
            set_base_shapes_custom(model, base_model)
            if readout_zero_init:
                readout = list(model.modules())[-1]
                readout.weight.data.zero_()
                if readout.bias is not None:
                    readout.bias.data.zero_()
            return model #.to(torch.device("mps"))
        return f
    return {w: gen(w) for w in widths}



def get_lazy_models(arch, widths, mup=True, init='kaiming_fan_in_normal', readout_zero_init=True, batchnorm=True, base_width=None):
    '''if mup is False, then `init`, `readout_zero_init`, `base_width` don't matter.'''
    if  arch == 'depthwise_1dcnn':
        base_width = base_width or 3
        generate = generate_depthwise1dCNN
    def gen(w):
        def f():
            model = generate(w, mup=mup, init=init, readout_zero_init=readout_zero_init, batchnorm=batchnorm, base_width=base_width)
            return model
        return f
    return {w: gen(w) for w in widths}


if __name__ == '__main__':

    torch.set_default_device('cuda:0')
    hyperparam_mode='mf_fullalign'

    from mup.coord_check import get_coord_data, plot_coord_data
    # models = get_lazy_models('depthwise_1dcnn', [3, 6, 12, 24, 48, 96, 192, 384], mup=True,)
    models = generate_SSM_models(range(500, 12000, 500),
                                 mup=True, learn_A=False, A_scale=0.1,
                                 selective=True,
                                 readout_zero_init=False,
                                 cuda=True,
                                 hyperparam_mode=hyperparam_mode,) 

    dataloader = get_train_loader(batch_size=2, num_workers=0, shuffle=False, train=True, download=True) # by default, batch_size=20
    df = get_coord_data(models, dataloader, nseeds=1, nsteps=7, lr=0.1, optimizer='adam', cuda=True, hyperparam_mode=hyperparam_mode) # by default nseeds=5

    exp_name = 'ssm_mu1_runb29_mf_fullalign_nseeds1BatchSize1WithEx1'
    import os
    os.makedirs('/n/fs/scratch/bc2188/mup/examples/SSMs/coord_checks', exist_ok=True)
    df.to_pickle(f"/n/fs/scratch/bc2188/mup/examples/SSMs/coord_checks/{exp_name}.pkl")  
    png_filename = f'/n/fs/scratch/bc2188/mup/examples/SSMs/coord_checks/{exp_name}.png'
    import numpy as np
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    plot_coord_data(df.dropna(), save_to=png_filename)
    # If you are in jupyter notebook, you can also do
    #   `plt.show()`
    # to show the plot