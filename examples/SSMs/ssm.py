"""Minimal version of SSM with extra options and features stripped out, for testing purposes.

    
d_model is Chanels which is D
d_state is hidden dim which is N. 

"""

import math
from copy import copy
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.optim import SGD, Adam
from torch.nn import Linear, RMSNorm
from torch.nn.modules.conv import _ConvNd
from einops import rearrange, repeat
from mup.layer import MuReadout
from mup.shape import rescale_linear_bias
from mup.infshape import InfShape, InfDim
BASEWIDTH = 3

def get_train_loader(batch_size, num_workers=0, shuffle=False, train=True, download=False):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: torch.reshape(x, (3, x.size(1)*x.size(2)))),
        ])
    trainset = datasets.CIFAR10(root='dataset', train=train,
                                download=download, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers)

    
class NonSelectiveSSMKernel(nn.Module):
    def __init__(self, 
                 d_model, 
                 N=64, 
                 dt_min=0.001, 
                 dt_max=0.1, 
                 learn_A=True,
                 A_scale=1.0,
                 ):
        super().__init__()
        self.d_model = d_model # D, or the num of channels.
        self.N = N

        if learn_A:
            self.A = nn.Parameter(torch.diag(torch.rand(N)*A_scale))
        else:
            self.register_buffer('A', torch.diag(torch.rand(N)*A_scale), persistent=True)
        self.B = nn.Parameter(torch.rand(N)*1) 
        self.C = nn.Parameter(torch.rand(N)*1)

    def _compute_A_mat(self, A_n, L):
        A_mat = torch.zeros(L, L)
        for i in range(L):
            v = torch.ones(L - i) * (A_n**i)
            A_mat += torch.diag(v, -i)
        return A_mat

    def forward(self, L, u=None):
        """
        If u is None then compute the materialized K matrix,
        which transforms a flattened input vector of shape (L*d_model,)
        to the output vector of the same shape, where the flattened vector
        takes the form (x_1^1, ..., x_1^L, x_2^1, ..., x_2^L, ..., x_d^1, ..., x_d^L).

        Otherwise, compute y using the recursive defn of SSM. In this case u.size() = (B, H, L)
        """
        if u is not None:
            # return self.B*self.C*u
            batch_size = u.size(0)
            h = torch.zeros(batch_size, self.N, self.d_model)
            y = torch.zeros(u.size()) # B, H, L
            for l in range(L):
                first_term = torch.einsum('ij,bjd->bid', self.A, h)
                second_term = torch.einsum('i,bd->bid', self.B, u[:, :, l])
                h = first_term + second_term
                y[:, :, l] = torch.einsum('i,bid->bd', self.C, h)
            return y
        else:
            K = torch.zeros(L*self.d_model, L*self.d_model)
            for k in range(self.N):
                # Make the lower triangular matrix
                A_mat = self._compute_A_mat(self.A[k, k], L)
                K += self.B[k]*self.C[k]*torch.kron(torch.eye(self.d_model), A_mat) # NOTE: this will fail with even moderately sized d_model (e.g., 100),
                                                                                    # since torch.kron creates a matrix too large to be materialized.
            return K

    def __repr__(self):
        return f'NonSelectiveSSMKernel(num_A_param={np.prod(self.A.size())}, num_B_param={np.prod(self.B.size())}, num_C_param={np.prod(self.C.size())})'


class SelectiveSSMKernel(nn.Module):
    def __init__(self, 
                 d_model, 
                 N=64,
                 dt_min=0.001, 
                 dt_max=0.1,
                 learn_A=False,
                 A_scale=0.1,
                 cuda=False, # Deprecated
                 device=None,
                 hyperparam_mode='mup_fullalign',
                 dtype=torch.float,
                 d_model_base=None,
                 init_eps=0.1,
                 norm='rms',
                 ):
        super().__init__()
        self.d_model = d_model # D, or the num of channels.
        self.N = N
        self.d_model_base = d_model_base
        if d_model_base is not None:
            self.width_mult = int(d_model/ d_model_base)
        else:
            self.width_mult = d_model/BASEWIDTH
        self.hyperparam_mode = hyperparam_mode
    
        if device is not None:
            self.device = device
        else:
            self.device = torch.get_default_device()

        if learn_A:
            self.A = nn.Parameter(torch.rand(N)*A_scale + init_eps) # nn.Parameter(torch.diag(torch.rand(N)*A_scale + 0.1))
        else:
            A_scale_mu_adjustment = 1.0
            self.register_buffer('A', (torch.rand(N)*A_scale)*A_scale_mu_adjustment, persistent=True) 
            # self.register_buffer('A', torch.zeros(N)*1.0, persistent=True) # NOTE used for debugging purposes


        if 'mup' in self.hyperparam_mode:
            BC_scale_adjustment = (self.width_mult)**(-.5)
            # BC_scale_adjustment = 1.0
            Delta_scale = (1 / self.width_mult)**.5
        elif 'sp' in self.hyperparam_mode:
            BC_scale_adjustment = (self.width_mult)**(-.5)
            Delta_scale = (1 / self.width_mult)**.5
        elif 'ntk' in self.hyperparam_mode:
            BC_scale_adjustment = 1.0
            Delta_scale =  1.0
        elif 'mf' in self.hyperparam_mode:
            BC_scale_adjustment = 1.0
            Delta_scale =  1.0
        elif 'umup' in self.hyperparam_mode:
            BC_scale_adjustment = 1.0
            Delta_scale =  1.0
        self.B = nn.Parameter((torch.randn(N, d_model) + init_eps) * BC_scale_adjustment)
        self.C = nn.Parameter((torch.randn(N, d_model) + init_eps) * BC_scale_adjustment)
        self.D = nn.Parameter(torch.ones(d_model))

        self.Delta = nn.Parameter((torch.randn(d_model, d_model)) * Delta_scale) # NOTE not shifting randn here
        # self.Delta = nn.Parameter((torch.ones(d_model, d_model))) # TODO Set to ones temperary as a debugging measure, as opposed to above

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min/d_model and dt_max/d_model
        dt_min, dt_max = 0.001*Delta_scale, 0.1*Delta_scale
        dt_init_floor=1e-4
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        # self.dt_bias = nn.Parameter(inv_dt) # TODO delta bias set to a buffer
        self.register_buffer('dt_bias', inv_dt, persistent=True)

        if norm == 'rms':
            self.norm = RMSNorm(self.d_model)


    def forward(self, L, u=None):
        if u is not None:
            batch_size = u.size(0)
            # """ TODO temporary commented out for debugging purposes
            # For selective state space: B and C are now a function of u: 
            Bu = torch.einsum('nd,bdl->bnl', self.B, u) # size (B, N, L)
            Cu = torch.einsum('nd,bdl->bnl', self.C, u)
            # """
            # stub = u.new_ones(u.size()) # TODO Temporary
            # Bu = torch.einsum('nd,bdl->bnl', self.B, stub) # size (B, N, L)
            # Cu = torch.einsum('nd,bdl->bnl', self.C, stub)
     
            # Apply multiplier:
            Delta_multiplier = 1.0
            if 'mup' in self.hyperparam_mode:
                Bu = Bu * (self.width_mult**-.5) # TODO Commented out for now
                Cu = Cu * (self.width_mult**-.5)
                # Bu = Bu * (self.width_mult**-.75) # NOTE Used for testing
                # Cu = Cu * (self.width_mult**-.75) # NOTE Used for testing
            elif 'sp' in self.hyperparam_mode:
                pass
            elif 'ntk' in self.hyperparam_mode:
                Bu = Bu * (self.width_mult**-.5)
                Cu = Cu * (self.width_mult**-.5)
                Delta_multiplier *= (self.width_mult**-.5)
            elif 'mf' in self.hyperparam_mode:
                Bu = Bu * (self.width_mult**-1.0)
                Cu = Cu * (self.width_mult**-1.0)
                Delta_multiplier *= (self.width_mult**-.5)
            elif 'umup' in self.hyperparam_mode:
                Bu = Bu * (self.d_model**-.5)
                Cu = Cu * (self.d_model**-.5)
                Delta_multiplier *= (self.d_model**-.5)

            """ TODO temporary commented out for debugging purposes
            delta = torch.einsum('ed,bdl->bel', self.Delta, u)*Delta_multiplier
            shifted_delta = delta
            # print(u.size(), self.width_mult, self.d_model_base)
            # raise RuntimeError
            # shifted_delta = delta + self.dt_bias[..., None] # TODO Uncomment to apply delta bias
            shifted_delta = F.softplus(shifted_delta)
            """

            shifted_delta = self.Delta.new_ones(Bu.size(0), self.d_model, Bu.size(2)) # TODO temporary stub

            A = repeat(self.A, 'n -> d n', d=self.d_model)
            deltaA = torch.einsum('bdl,dn->bdln', shifted_delta, A) # torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
            DeltaB_u = torch.einsum('bdl,bnl,bdl->bdln', shifted_delta, Bu, u) 
            # DeltaB_u = torch.einsum('bdl,bnl,bdl->bdln', shifted_delta, Bu, stub) # TODO Use for debugging purposes

            h = torch.zeros(batch_size, self.d_model, self.N, device=self.device)
            y = torch.zeros(u.size(), device=self.device) # B, H, L
            for l in range(L):
                first_term = deltaA[:, :, l, :] * h # equivalent to torch.einsum('bdn,bdn->bnd', deltaA[:, :, l, :], h) 
                h = first_term + DeltaB_u[:, :, l, :]
                y[:, :, l] = torch.einsum('bn,bdn->bd', Cu[:, :, l], h)

            out = y + u * rearrange(self.D, "d -> d 1") # out.size() is (B, d_model, L)
            out = rearrange(out, "B D L -> B L D")
            out = self.norm(out)
            out = rearrange(out, "B L D -> B D L")

            # print(f"y.size is {y.size()}; torch.mean(y, dim=(1, 2),  keepdim=True)).size is {torch.mean(y, dim=(1, 2),  keepdim=True).size()}")
            # layer_norm = nn.LayerNorm([y.size(1), y.size(2)] , elementwise_affine=False)
            # out = layer_norm(y)
            # print(f"\t self.d_model = {self.d_model} \t\t torch.var(y) = {torch.sqrt(torch.var(y, dim=(1, 2), unbiased=False)[0])}")
            # out = y #(y - torch.mean(y, dim=(1, 2), keepdim=True)) / torch.sqrt(torch.var(y, dim=(1, 2), keepdim=True, unbiased=False)) # TODO For debugging purposes
            return out
        else:
            raise NotImplementedError

    def __repr__(self):
        return f'SelectiveSSMKernel(num_A_param={np.prod(self.A.size())}, num_B_param={np.prod(self.B.size())}, num_C_param={np.prod(self.C.size())})'

class CustomSSMUpProject(nn.Module):
    def __init__(self, fan_in, fan_out, width_mult, hyperparam_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width_mult = width_mult
        self.hyperparam_mode = hyperparam_mode
        if 'mup' in hyperparam_mode:
            self.up_project = nn.Parameter(torch.randn(fan_out, fan_in) * (width_mult**(-.5)))
        elif 'sp' in hyperparam_mode:
            self.up_project = nn.Parameter(torch.randn(fan_out, fan_in))
        elif 'ntk' in hyperparam_mode:
            self.up_project = nn.Parameter(torch.randn(fan_out, fan_in))
        elif 'mf' in hyperparam_mode:
            self.up_project = nn.Parameter(torch.randn(fan_out, fan_in))
        else:
            raise ValueError(f'hyperparam_mode = {hyperparam_mode} not recognized.')
                    
    def forward(self, x):
        if 'mup' in self.hyperparam_mode:
            return torch.einsum('di,bli->bld', self.up_project, x)*(self.width_mult**(.5))
        elif 'sp' in self.hyperparam_mode:
            return torch.einsum('di,bli->bld', self.up_project, x)
        elif 'ntk' in self.hyperparam_mode:
            return torch.einsum('di,bli->bld', self.up_project, x)
        elif 'mf' in self.hyperparam_mode:
            return torch.einsum('di,bli->bld', self.up_project, x)

class CustomSSMReadout(nn.Module):
    def __init__(self, fan_in, fan_out, width_mult, hyperparam_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width_mult = width_mult
        self.hyperparam_mode = hyperparam_mode
        if 'mup' in hyperparam_mode:
            self.down_project = nn.Parameter((torch.randn(fan_out, fan_in)*(width_mult**(-.5))))
        elif 'sp' in hyperparam_mode:
            self.down_project = nn.Parameter((torch.randn(fan_out, fan_in)*(width_mult**(-.5))))
        elif 'ntk' in hyperparam_mode:
            self.down_project = nn.Parameter((torch.randn(fan_out, fan_in)))
        elif 'mf' in hyperparam_mode:
            self.down_project = nn.Parameter((torch.randn(fan_out, fan_in)))
        else:
            raise ValueError(f'hyperparam_mode = {hyperparam_mode} not recognized.')
                    
    def forward(self, x):
        if 'mup' in self.hyperparam_mode:
            return torch.einsum('id,bd->bi', self.down_project, x)*(self.width_mult**(-.5))
        elif 'sp' in self.hyperparam_mode:
            return torch.einsum('id,bd->bi', self.down_project, x)
        elif 'ntk' in self.hyperparam_mode:
            return torch.einsum('id,bd->bi', self.down_project, x)*(self.width_mult**(-.5))
        elif 'mf' in self.hyperparam_mode:
            return torch.einsum('id,bd->bi', self.down_project, x)*(self.width_mult**(-1.0))

class SSM(nn.Module):
    def __init__(self, 
                 num_input_channels, 
                 d_model, 
                 d_state=64, 
                 transposed=True, 
                 hyperparam_mode='mup_fullalign',     # Currently supports: {mup, sp, ntk, mf}_{full, no}align
                 mup=True,                            # Deprecated
                 use_kernel=False,                    # Whether to compute SSM with materialized K matrix or the recursive definition of SSM
                 selective=False,
                 readout_zero_init=False,             # Deprecated
                 **kernel_args,
                 ):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        width_mult = float(d_model/BASEWIDTH)
        self.hyperparam_mode=hyperparam_mode

        self.up_project = CustomSSMUpProject(num_input_channels, d_model, width_mult, hyperparam_mode=hyperparam_mode)

        # Initialize SSM Kernel:
        self.use_kernel = use_kernel
        if selective:
            self.kernel = SelectiveSSMKernel(self.h, N=self.n, hyperparam_mode=hyperparam_mode, **kernel_args)
        else:
            self.kernel = NonSelectiveSSMKernel(self.h, N=self.n, **kernel_args)

        self.down_project = CustomSSMReadout(d_model, 10, width_mult, hyperparam_mode=hyperparam_mode)


    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        u = u.transpose(-1, -2)
        u = self.up_project(u)
        u = u.transpose(-1, -2)
        if self.use_kernel:
            K = self.kernel(L=L) # K is a (L*d_model) by (L*d_model) matrix
            u = u.transpose(-1, -2) # shape (B L H)
            u = rearrange(u, 'b l h -> b (l h)')
            u = torch.einsum('ij,bj->bi', K, u) 
            y = rearrange(u, 'b (l h) -> b h l', l=L)
        else:
            y = self.kernel(L=L, u=u)

        if not self.transposed: y = y.transpose(-1, -2)

        y = torch.sum(y, dim=2) / L # Average along L dimesion; note
                                    # that if we don't divide by L,
                                    # we may induce a large activation,
                                    # which may destabilize training by
                                    # having explosive gradients.
        y = self.down_project(y)
        return y


def process_param_groups(params, **kwargs):
    """
    If params is a list of tensors, then require lr in kwargs
    and create dict of form:
        {'params': [tensor1, tensor2, ...],
         'lr': lr,
         'weight_decay': 0}
    If params is a iterable of dictionaries already,
    then just set the 'lr' and 'weight_decay' key
    of each of the dicts.
     
    """
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for param_group in param_groups:
        if 'lr' not in param_group:
            param_group['lr'] = kwargs['lr']
        if 'weight_decay' not in param_group:
            param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
    return param_groups

# TODO: Will need to update MuSGD for correct µ scaling behavior.
def MuSGD(params, impl=SGD, decoupled_wd=False, model_names=None, ssm_force_multiply=1, L=None, **kwargs):
    '''SGD with μP scaling.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
     
    Inputs:
        impl: the specific SGD-like optimizer implementation from torch.optim or
            elsewhere 
        decoupled_wd: if True, skips the mup scaling for weight decay, which should
            be used for optimizer implementations that decouple weight decay from
            learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
    Outputs:
        An instance of `impl` with refined parameter groups, each of which has the correctly
        scaled learning rate according to mup.
    '''
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k:v for k, v in param_group.items() if k not in ['params', 'model_names']}
            new_g['params'] = []
            return new_g
        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        vector_like_p = defaultdict(new_group) # key is width mult
        matrix_like_p = defaultdict(new_group) # key is fan_in/out ratio

        ssm_like_p = defaultdict(new_group) # for selective ssm
        upproj_like_p = defaultdict(new_group) # for debugging ssm
        downproj_like_p = defaultdict(new_group) # for debugging ssm

        fixed_p = new_group()
        for i, p in enumerate(param_group['params']):
            assert hasattr(p, 'infshape'), (
                f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `mup.set_base_shapes` on the model?')
            if model_names is not None and ('kernel' in model_names[i] or 'B' in model_names[i] or 'C' in model_names[i]):
                ssm_like_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and ('up_project' in model_names[i]):
                upproj_like_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and ('down_project' in model_names[i]):
                downproj_like_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if p.infshape.ninf() == 1:
                # elif 'down_project' in model_names[i]:
                #     ssm_like_p[p.infshape.width_mult()]['params'].append(p)
                vector_like_p[p.infshape.width_mult()]['params'].append(p)
            elif p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.fanin_fanout_mult_ratio()]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('more than 2 inf dimensions')
            else:
                fixed_p['params'].append(p)

        for width_mult, group in vector_like_p.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] *= width_mult
            if not decoupled_wd:
                group['weight_decay'] /= width_mult
        for shape_ratio, group in matrix_like_p.items():
            group['lr'] /= shape_ratio
            if not decoupled_wd:
                group['weight_decay'] *= shape_ratio
        # Do multiplication for ssm_like_p for now:
        for width_mult, group in ssm_like_p.items():
            # print(group, width_mult)
            assert L is not None
            # print(f"kernel width_mul is {width_mult}")
            group['lr'] = group['lr'] * width_mult # * ssm_force_multiply #/ L#/ (width_mult**.5) # (width_mult**2) # (width_mult)
            assert 'lr' in group
        for width_mult, group in upproj_like_p.items():
            # print(f"DEBUG A: width is {width_mult}")
            assert L is not None
            # print(f"upproj width_mul is {width_mult}")
            group['lr'] = group['lr'] * width_mult #/ (L)# width_mult * group['lr'] / (L*4*50) 
        for width_mult, group in downproj_like_p.items():
            # print(f"DEBUG A: width is {width_mult}")
            assert L is not None
            # print(f"downproj width_mul is {width_mult}")
            group['lr'] *= width_mult # width_mult * group['lr'] / (L*4*50)
        new_param_groups.extend(list(matrix_like_p.values()) + \
                                list(vector_like_p.values()) + \
                                list(ssm_like_p.values()) + \
                                list(upproj_like_p.values()) + \
                                list(downproj_like_p.values()) + \
                                [fixed_p])
    return impl(new_param_groups, **kwargs)

def AdamSSM(params, impl=Adam, hyperparam_mode='mup_fullalign', decoupled_wd=False, model_names=None, L=None, **kwargs):
    '''Adam with custom scaling of LR.

    Note for this to work properly, your model needs to have its base shapes set
    already using `mup.set_base_shapes`.
    
    Inputs:
        impl: the specific Adam-like optimizer implementation from torch.optim or
            elsewhere 
        hyperparam_mode: LR scaling mode. Currently supports: 'mup_fullalign', 'sp_fullalign'
        decoupled_wd: if True, skips the mup scaling for weight decay, which should
            be used for optimizer implementations that decouple weight decay from
            learning rate. See https://github.com/microsoft/mup/issues/1 for a use case.
        model_names: [n for n, _ in model.named_parameters()]
        L: input sequence length for the SSM.
    Outputs:
        An instance of `impl` with refined parameter groups, each of which has the correctly
        scaled learning rate according to mup.
    '''
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k:v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g

        BC_ssm_p = defaultdict(new_group)
        upproj_p = defaultdict(new_group)
        downproj_p = defaultdict(new_group)
        delta_p = defaultdict(new_group)
        delta_bias_p = defaultdict(new_group)
        skip_p = defaultdict(new_group)

        for i, p in enumerate(param_group['params']):
            # print(model_names[i])
            assert hasattr(p, 'infshape'), (
                f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `mup.set_base_shapes` on the model?')
            # print(model_names[i])
            if model_names is not None and ('B' in model_names[i] or 'C' in model_names[i]):
                # print("'B' and 'C' detected")
                BC_ssm_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and (('up_project' in model_names[i]) or ('embed' in model_names[i])):
                # print("embedd detected")
                upproj_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and (('down_project' in model_names[i]) or ('decode' in model_names[i])):
                # print("decode detected")
                downproj_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and ('Delta' in model_names[i]):
                # print("Delta detected")
                delta_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and 'D' in model_names[i]:
                # print("D detected")
                skip_p[p.infshape.width_mult()]['params'].append(p)
                continue
            if model_names is not None and ('dt_bias' in model_names[i]):
                delta_bias_p[p.infshape.width_mult()]['params'].append(p)
                continue

        if hyperparam_mode == 'mup_fullalign':
            for width_mult, group in upproj_p.items():
                group['lr'] /= width_mult**.5 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] /= width_mult**.5 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult  # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] /= width_mult**.5 # Readout
        elif hyperparam_mode == 'umup_fullalign': # TODO will need to change so actually follow u-map
            for width_mult, group in upproj_p.items():
                group['lr'] /= width_mult**.5 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] /= width_mult**.5 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult  # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] /= width_mult**.5 # Readout
        elif hyperparam_mode == 'mup_noalign':
            for width_mult, group in upproj_p.items():
                group['lr'] /= width_mult**.5 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] *= 1.0 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult**.5  # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] *= 1.0 # Readout
        elif hyperparam_mode == 'sp_fullalign':
            for width_mult, group in upproj_p.items():
                group['lr'] *= 1.0 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] /= width_mult # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult  # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] /= width_mult # Readout
        elif hyperparam_mode == 'sp_noalign':
            for width_mult, group in upproj_p.items():
                group['lr'] *= 1.0 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] /= width_mult**.5 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult**.5  # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] /= width_mult**.5 # Readout
        elif hyperparam_mode == 'ntk_fullalign':
            for width_mult, group in upproj_p.items():
                group['lr'] *= 1.0 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] /= width_mult**.5 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult**.5 # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] /= width_mult**.5 # Readout
        elif hyperparam_mode == 'ntk_noalign':
            for width_mult, group in upproj_p.items():
                group['lr'] *= 1.0 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] *= 1.0 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] *= 1.0 # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] *= 1.0 # Readout
        elif hyperparam_mode == 'mf_fullalign':
            for width_mult, group in upproj_p.items():
                group['lr'] *= 1.0 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] *= 1.0 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] /= width_mult**.5 # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] *= 1.0 # Readout
        elif hyperparam_mode == 'mf_noalign':
            for width_mult, group in upproj_p.items():
                group['lr'] *= 1.0 # Embedding
            for width_mult, group in BC_ssm_p.items():
                group['lr'] *= width_mult**.5 # Readout
            for width_mult, group in delta_p.items():
                group['lr'] *= 1.0 # Hidden
            for width_mult, group in delta_bias_p.items():
                group['lr'] = 0.0 #*= 1.0 # TODO learning rate for delta bias set to 0
            for width_mult, group in skip_p.items():
                group['lr'] /= width_mult # Entry-wise
            for width_mult, group in downproj_p.items():
                group['lr'] *= width_mult**.5 # Readout
        else:
            raise ValueError(f'AdamSSM: hyperparam_mode = {hyperparam_mode} is not valid.')

        new_param_groups.extend(list(BC_ssm_p.values()) + \
                                list(upproj_p.values()) + \
                                list(downproj_p.values()) + \
                                list(delta_p.values()) + \
                                list(delta_bias_p.values()) + \
                                list(skip_p.values()))
    return impl(new_param_groups, **kwargs)

def zip_infshape(base_dims, dims, fin_if_same=True):
    infshape = []
    for bd, d in zip(base_dims, dims):
        if isinstance(bd, InfDim):
            # retain bd's base_dim but overwrite dim
            infdim = copy(bd)
            infdim.dim = d
            infshape.append(infdim)
        elif isinstance(bd, int):
            if bd == d and fin_if_same:
                infshape.append(InfDim(None, d))
            else:
                infshape.append(InfDim(bd, d))
        else:
            raise ValueError(f'unhandled base_dim type: {type(bd)}')
    return InfShape(infshape)

def _dataparallel_hack(base_shapes, shapes):
    '''Fix module name discrepancy caused by (Distributed)DataParallel module.

    The parameters of a (Distributed)DataParallel module all have names that
    start with 'module'. This causes a mismatch from non-DataParallel modules.
    This function tries to match `base_shapes` to `shapes`: if the latter starts
    with 'module', then make the former too; likewise if not.
    '''
    if all(k.startswith('module.') for k in shapes) and \
        all(not k.startswith('module.') for k in base_shapes):
        return {'module.' + k: v for k, v in base_shapes.items()}, shapes
    if all(not k.startswith('module.') for k in shapes) and \
        all(k.startswith('module.') for k in base_shapes):
        return {k.strip('module.'): v for k, v in base_shapes.items()}, shapes
    return base_shapes, shapes

def _zip_infshape_dict(base_shapes, shapes):
    '''make a dict of `InfShape` from two dicts of shapes.
    Inputs:
        base_shapes: dict of base shapes or InfShape objects
        shapes: dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    '''
    base_shapes, shapes = _dataparallel_hack(base_shapes, shapes)
    basenames = set(base_shapes.keys())
    names = set(shapes.keys())
    assert basenames == names, (
        f'`base_shapes` has extra names {basenames - names}. '
        f'`shapes` has extra names {names - basenames}.'
    )
    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes


def rescale_ssm(kernel):
    """
    Currently, recaling of SSM's parameters is being implemented in the
    class definition, but may need to be migrated here for consistent formatting.
    """
    pass

def set_base_shapes_custom(model, base, rescale_params=True, delta=None, savefile=None, do_assert=True):
    '''Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Inputs:
        model: nn.Module instance
        base: The base model.
            Can be nn.Module, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
        rescale_params:
            assuming the model is initialized using the default pytorch init (or
            He initialization etc that scale the same way with fanin): If True
            (default), rescales parameters to have the correct (μP) variances.
        do_assert: 
    Output:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    '''
    from mup.shape import get_shapes, apply_infshapes, assert_hidden_size_inf, _extract_shapes

    if base is None:
        base = model
    base_shapes = _extract_shapes(base) # {name: param.shape for name, param in base.named_parameters()} 

    shapes = get_shapes(model) #{name: param.shape for name, param in model.named_parameters()}
    infshapes = _zip_infshape_dict(base_shapes, shapes)

    apply_infshapes(model, infshapes) # just attaches infshapes to the params of the model
    if do_assert:
        assert_hidden_size_inf(model)
    if rescale_params:
        for name, module in model.named_modules():
            if isinstance(module, MuReadout):
                module._rescale_parameters()
            elif isinstance(module, (Linear, _ConvNd)):
                rescale_linear_bias(module)
            elif isinstance(module, (NonSelectiveSSMKernel, SelectiveSSMKernel)):
                rescale_ssm(module)
                # none-ize the base_dim of infdim in S4DKernel so that the learning rate stays constant from epoch to epoch
                # module.B.infshape[1].base_dim = None
                # module.C.infshape[0].base_dim = None
    return model


def simple_train(model, base_model, num_steps):
    """Barebone training code for testing purposes."""

    train_loader = get_train_loader(1,) #, download=True)
    batch = next(iter(train_loader))
    (u, target) = batch

    set_base_shapes_custom(model, base_model, do_assert=False)

    model_names = []
    for n, _ in model.named_parameters():
        model_names.append(n)

    model = model.train()
    model = model.cuda(torch.get_default_device())
    u, target = u.cuda(torch.get_default_device()), target.cuda(torch.get_default_device())

    d_model = model.h
    optimizer =  AdamSSM(model.parameters(), 
                      model_names=model_names,
                      lr=0.1, 
                      L=1024,
                      ssm_force_multiply=1/d_model,
                      )# SGD(model.parameters(), lr=1.0, momentum=0)


    for i in range(num_steps):
        optimizer.zero_grad()
        out = model(u)

        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

    return torch.sum(torch.abs(model.kernel.D.grad)) 


if __name__ == "__main__":
    import numpy as np
    import statistics

    torch.set_default_device('cuda:6')

    d_state = 16
    A_scale = 0.1
    use_kernel = False
    selective = True
    seeds = 5
    num_steps = 3

    d_models = range(500, 10000, 500)

    base_sd = SSM(num_input_channels=3, d_model=3, d_state=d_state, 
                  mup=True, use_kernel=use_kernel, A_scale=A_scale, 
                  selective=selective, readout_zero_init=False, learn_A=False, cuda=True)
    for d_model in d_models:
        quantity = []
        print(f"-------------{d_model}:")
        for seed in range(seeds):
            torch.manual_seed(seed)
            sd = SSM(num_input_channels=3, d_model=d_model, d_state=d_state, 
                    mup=True, use_kernel=use_kernel, A_scale=A_scale, 
                    selective=selective, readout_zero_init=False, learn_A=False, cuda=True)
            quantity.append(simple_train(sd, base_sd, num_steps=num_steps).item())
        print(quantity)
        print(statistics.median(quantity))
        # print(f"{d_model} \t {quantity}")
