"""Minimal version of SSM with extra options and features stripped out, for testing purposes.
    Source: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

    
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
from torch.optim import SGD
from torch.nn import Linear
from torch.nn.modules.conv import _ConvNd
from einops import rearrange, repeat
from mup.layer import MuReadout
from mup.shape import rescale_linear_bias
from mup.infshape import InfShape, InfDim


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
        # self.B = nn.Parameter(torch.rand(N)*1) # TODO
        # self.C = nn.Parameter(torch.rand(N)*1) # TODO
        self.B = nn.Parameter(torch.ones(N)*1)
        self.C = nn.Parameter(torch.ones(N)*1) 

    def _compute_A_mat(self, A_n, L):
        A_mat = torch.zeros(L, L)
        for i in range(L):
            # print(f"DEBUG: i is {i}")
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


    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def __repr__(self):
        return f'NonSelectiveSSMKernel(num_A_param={np.prod(self.A.size())}, num_B_param={np.prod(self.B.size())}, num_C_param={np.prod(self.C.size())})'


class SelectiveSSMKernel(nn.Module):
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
        # For selective SSM, B is now a dual of hidden vector:
        scale = 1 / (self.d_model*self.N)**.5
        self.B = nn.Parameter(torch.randn(d_model, N) * scale)
        self.C = nn.Parameter(torch.randn(d_model, N) * scale)
        # self.B = nn.Parameter(torch.ones(d_model, N)/ d_model)
        # self.C = nn.Parameter(torch.ones(d_model, N)/ d_model)
        
        # self.B = nn.Linear(d_model, N, bias=False)
        # self.C = nn.Linear(d_model, N, bias=False)

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
            batch_size = u.size(0)
            # For selective state space: B and C are now a function of u: 
            Bu = torch.einsum('dn,bdl->bnl', self.B, u) # size (B, N, L)
            Cu = torch.einsum('dn,bdl->bnl', self.C, u)
            # Bu = rearrange(
            #         'b l n -> b n l',
            #         self.B(rearrange('b d l -> b l d', u))
            # )
            # Cu = rearrange(
            #         'b l n -> b n l',
            #         self.C(rearrange('b d l -> b l d', u))
            # )

            h = torch.zeros(batch_size, self.N, self.d_model)
            y = torch.zeros(u.size()) # B, H, L
            for l in range(L):
                first_term = torch.einsum('nn,bnd->bnd', self.A, h) # NOTE: this assumes self.A is diagonal
                second_term = torch.einsum('bn,bd->bnd', Bu[:, :, l], u[:, :, l])
                h = first_term + second_term
                y[:, :, l] = torch.einsum('bn,bnd->bd', Cu[:, :, l], h)
            return y
        else:
            raise NotImplementedError
            K = torch.zeros(L*self.d_model, L*self.d_model)
            for k in range(self.N):
                # Make the lower triangular matrix
                A_mat = self._compute_A_mat(self.A[k, k], L)
                K += self.B[k]*self.C[k]*torch.kron(torch.eye(self.d_model), A_mat) # NOTE: this will fail with even moderately sized d_model (e.g., 100),
                                                                                    # since torch.kron creates a matrix too large to be materialized.
            return K


    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def __repr__(self):
        return f'SelectiveSSMKernel(num_A_param={np.prod(self.A.size())}, num_B_param={np.prod(self.B.size())}, num_C_param={np.prod(self.C.size())})'



class SSM(nn.Module):
    def __init__(self, 
                 num_input_channels, 
                 d_model, 
                 d_state=64, 
                 transposed=True, 
                 mup=True, 
                 use_kernel=False,
                 selective=False,
                 readout_zero_init=False,
                 **kernel_args,
                 ):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.up_project = nn.Linear(num_input_channels, d_model)

        # Initialize SSM Kernel:
        self.use_kernel = use_kernel # Whether to compute SSM with materialized K matrix or the recursive definition of SSM
        if selective:
            self.kernel = SelectiveSSMKernel(self.h, N=self.n, **kernel_args)
        else:
            self.kernel = NonSelectiveSSMKernel(self.h, N=self.n, **kernel_args)

        if mup:
            self.down_project = MuReadout(d_model, 10, bias=True, readout_zero_init=readout_zero_init)
        else:
            self.down_project = nn.Linear(d_model, 10, bias=True)


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
            # print(y.size())
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

    # if 'model_names' in kwargs:
    #     assert len(param_groups) == 1
    #     param_groups[0]['model_names'] = kwargs['model_names']

    return param_groups


def MuSGD(params, impl=SGD, decoupled_wd=False, model_names=None, ssm_force_multiply=1, **kwargs):
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

        fixed_p = new_group()
        for i, p in enumerate(param_group['params']):
            assert hasattr(p, 'infshape'), (
                f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `mup.set_base_shapes` on the model?')
            if model_names is not None and ('kernel' in model_names[i] or 'B' in model_names[i] or 'C' in model_names[i]):
                ssm_like_p[p.infshape.width_mult()]['params'].append(p)
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
            group['lr'] = group['lr'] * ssm_force_multiply#/ (width_mult**.5) # (width_mult**2) # (width_mult)
        new_param_groups.extend(list(matrix_like_p.values()) + \
                                list(vector_like_p.values()) + \
                                list(ssm_like_p.values()) + \
                                [fixed_p])
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
    # TODO rescale Selective kernel's B and C by 1/d_model. (currently it's being implemented in the
    # class definition, and need to be migrated here for consistent formatting)
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
    train_loader = get_train_loader(1)

    batch = next(iter(train_loader))
    (u, target) = batch

    set_base_shapes_custom(model, base_model, do_assert=False)

    # model.kernel.A = model.kernel.A*0.1
    # L = 2
    # svd_k = torch.svd(model.kernel(L))
    # cumsum = 0
    # for l in range(L):
    #     cumsum += model.kernel.A**l
    # print(f"model.A is {model.kernel.A} \n and model.B is \n {model.kernel.B} \n and model.C is \n {model.kernel.C}")
    # print(f"model.kernel is \n{model.kernel(L)}")
    # print(f"svd of model.kernel is \n {svd_k[:]}")
    # print(f"Theoretical spectral norm is {model.kernel.B*model.kernel.C*cumsum}")
    # print(f"UV* is \n {svd_k[0]@ svd_k[2].T}")
    # print(f"L = {L}\t norm_2(kernel) is {torch.linalg.norm(model.kernel(L), ord=2)}")
    # L = 2
    # print(model.kernel(L))

    # return

    model_names = []
    for n, _ in model.named_parameters():
        model_names.append(n)

    optimizer =  MuSGD(model.parameters(), 
                      model_names=model_names,
                      lr=1.0, 
                      momentum=0,
                      )# SGD(model.parameters(), lr=1.0, momentum=0)

    for i in range(num_steps):
        optimizer.zero_grad()
        out = model(u)

        loss = F.cross_entropy(out, target)
        loss.backward()
        # model.down_project.weight.grad = torch.zeros(model.down_project.weight.grad.size())
        # model.down_project.bias.grad = torch.zeros(model.down_project.bias.grad.size())

        # model.up_project.weight.grad = torch.zeros(model.up_project.weight.grad.size())
        # model.up_project.bias.grad = torch.zeros(model.up_project.bias.grad.size())

        # model.kernel.B.grad = torch.zeros(model.kernel.B.grad.size())
        # model.kernel.C.grad = torch.zeros(model.kernel.C.grad.size())

        # print(torch.sum(torch.abs(out)))

        # print( torch.sum(torch.abs(model.down_project.weight)))
        optimizer.step()

        # print(torch.sum(torch.abs(model.kernel.B.grad)))
        # print(f"kernel.A is {model.kernel.A.item()}")

        # for name, p in model.named_parameters():
        #     print(name, p)


    # print(torch.sum(model.kernel.B.grad))
    # print(model.kernel.B.grad.size())
    # print(torch.sum(torch.abs(model.down_project.weight.grad)))

    return torch.sum(torch.abs(model.kernel.B.grad))

    # for name, p in model.named_parameters():
    #     print(f"name is {name}")

"""
if __name__ == "__main__":
    import numpy as np
    import statistics

    d_state = 1
    A_scale = 0.0
    use_kernel = False
    selective = False
    seeds = 10
    num_steps = 2

    d_models = range(100, 1000, 100)# [1000, 2000, 3000, 4000, 5000]#[10, 100, 1000, 2000, 3000, 4000, 5000]

    base_sd = SSM(num_input_channels=3, d_model=3, d_state=d_state, 
                  mup=True, use_kernel=use_kernel, A_scale=A_scale, 
                  selective=selective, readout_zero_init=False, learn_A=False)
    for d_model in d_models:
        quantity = []
        for seed in range(seeds):
            torch.manual_seed(seed)
            sd = SSM(num_input_channels=3, d_model=d_model, d_state=d_state, 
                    mup=True, use_kernel=use_kernel, A_scale=A_scale, 
                    selective=selective, readout_zero_init=False, learn_A=False)
            # if d_model != 2000:
            #     continue
            quantity.append(simple_train(sd, base_sd, num_steps=num_steps).item())
        # print(quantity)
        print(f"{d_model} \t {statistics.mean(quantity)}")
        
"""
        

if __name__ == "__main__":

    get_l1_norm = lambda x: torch.abs(x).mean(dtype=torch.float32)

    seeds = 1#100
    B, L = 1, 1
    d_models = [100, 500, 900, 1300, 1700, 2100, 2500, 3300, 4100, 4900,]#[100, 1000, 5000, 10000, 15000, 20000, 25000]
    d_models = d_models + list(range(5700, 35000, 800))

    # d_models = [10]
    # d_models = [100, 300, 500, 700, 900]# 1000, 5000, 10000, 15000]
    # d_models = [5, 1000, 5000, 10000, 15000]


    d_state = 1
    num_steps = 40


    steps_inp = []
    quant_inp = []
    dmodel_inp = []

    # train_loader = get_train_loader(1)

    # batch = next(iter(train_loader))
    # (u, target) = batch


    base_model = SelectiveSSMKernel(d_model=3, 
                                        N=d_state,
                                        learn_A=False,
                                        A_scale=0.0,
                                        )

    for seed in tqdm(range(seeds)):
        torch.manual_seed(seed)
        for d_model in d_models:
            # print(d_model)
        
            model = SelectiveSSMKernel(d_model=d_model, 
                                        N=d_state,
                                        learn_A=False,
                                        A_scale=0.0,
                                        )
            set_base_shapes_custom(model, base_model, do_assert=False)
            # input = torch.rand(B, d_model, L)
            input = torch.ones(B, d_model, L)*1.0
            target = torch.ones(B, d_model, L)*5.01

            model_names = []
            for n, _ in model.named_parameters():
                model_names.append(n)
            # print(model_names)

            optimizer =  MuSGD(model.parameters(), 
                            model_names=model_names,
                            lr=0.1, 
                            momentum=0,
                            ssm_force_multiply=1/d_model
                        )

            # optimizer = MuSGD(model.parameters(), lr=1.0, momentum=0) #SGD(model.parameters(), lr=0.001, momentum=0)

            for i in range(num_steps):
                optimizer.zero_grad()

                # K = model(L=L)
                # K.retain_grad()
                # u = rearrange(input, 'b l h -> b (l h)')
                # u = torch.einsum('ij,bj->bi', K, u)
                # out = rearrange(u, 'b (l h) -> b h l', l=L)


                out = model(L, input)
                # out.retain_grad()

                # loss = nn.MSELoss(reduction='mean')
                # output = loss(out, target) #torch.sum(out)/d_model#
                output = torch.sum((out - target)**2) / d_model
                output.backward()
                optimizer.step()

                if i == 39:
                    # print(f"d_model is {d_model} \t\t B.grad is {model.B.grad.item()} \t\t output is {get_l1_norm(out).item()} ")
                    print(f"d_model is {d_model} \t\t |B.grad|*/ d_model**.5 is {torch.linalg.matrix_norm(model.B.grad.T, ord=2) / d_model**.5} \t\t output is {get_l1_norm(out).item()} ")
                    # print(f"d_model is {d_model} \t\t \t\t output is {get_l1_norm(out).item()} ")
                #     print(f"output is {output}; get_l1_norm(out) is {get_l1_norm(out)}")
                # print(K)
                # print(K.grad)

                steps_inp.append(i)
                quant_inp.append((get_l1_norm(out)).item())
                dmodel_inp.append(d_model)
                # print(f"B.grad is {model.B.grad}")
                # for name, p in model.named_parameters():
                #     print(name, p)

    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({"steps": steps_inp, "quantity": quant_inp, "d_model": dmodel_inp})
    fig = sns.lineplot(df, x="d_model", y="quantity", hue="steps").get_figure()
    fig.savefig("out3.png") 


