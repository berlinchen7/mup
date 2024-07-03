"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.
    Source: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py

    
d_model is Chanels which is D
d_state is hidden dim which is N. 

"""

import math
from copy import copy
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







class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X
    
class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        self.d_model = d_model # D, or the num of channels.
        self.N = N

        # self.A = torch.diag(torch.rand(N))
        # self.B = torch.rand(N, d_model)
        # self.C = torch.rand(d_model, N)

        # self.register_parameter("kernA", nn.Parameter(self.A))
        # self.register_parameter("kernB", nn.Parameter(self.B))
        # self.register_parameter("kernC", nn.Parameter(self.C))

        self.A = nn.Parameter(torch.diag(torch.rand(N)))
        self.B = nn.Parameter(torch.rand(N, d_model))
        self.C = nn.Parameter(torch.rand(d_model, N))

        # log_dt = torch.rand(H) * (
        #     math.log(dt_max) - math.log(dt_min)
        # ) + math.log(dt_min)

        # C = torch.randn(H, N // 2, dtype=torch.cfloat) #/ math.sqrt(N)
        # self.C = nn.Parameter(torch.view_as_real(C))
        # self.register("log_dt", log_dt, lr)

        # log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        # A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        # self.register("log_A_real", log_A_real, lr)
        # self.register("A_imag", A_imag, lr)

    def forward(self, L, u=None):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        if u is not None:
            h = torch.zeros(self.N, B)
            y = torch.zeros(u.size())
            for l in range(L):
                h = self.A @ h + self.B @ u[:, :, l].T
                y[:, :, l] = (self.C @ h).T
                return y


        # # Materialize parameters
        # dt = torch.exp(self.log_dt) # (H)
        # C = torch.view_as_complex(self.C) # (H N)
        # A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # # Vandermonde multiplication
        # dtA = A * dt.unsqueeze(-1)  # (H N)
        # K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        # C = C * (torch.exp(dtA)-1.) / A
        # K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        # def apply_along_axis(function, x, axis: int = 0):
        #     return torch.stack([
        #         function(x_i) for x_i in torch.unbind(x, dim=axis)
        #     ], dim=axis)
        # A_power = self.A.unsqueeze(0)
        # A_power = A_power.expand(L, self.H, self.H)
        # A_power 

        #TODO
        # max_A = torch.max(torch.max(self.A))
        # A_normalized = self.A / max_A

        #TODO
        # A =  torch.diag(torch.rand(self.N)) #.to(torch.device("mps"))
        # B = torch.rand(self.N, self.d_model) #.to(torch.device("mps"))
        # C = torch.rand(self.d_model, self.N) #.to(torch.device("mps"))


        K = torch.stack([
            self.C @ torch.matrix_power(self.A, i) @ self.B for i in range(L)
        ], dim=0)

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
    
    # def __str__(self):
    #     return f'S4DKernel(num_A_param={np.prod(self.A.size())}, num_B_param={np.prod(self.B.size())}, num_C_param={np.prod(self.C.size())})'
    
    def __repr__(self):
        return f'S4DKernel(num_A_param={np.prod(self.A.size())}, num_B_param={np.prod(self.B.size())}, num_C_param={np.prod(self.C.size())})'

class SSM(nn.Module):
    def __init__(self, num_input_channels, d_model, d_state=64, dropout=0.0, transposed=True, mup=True, 
                 kernel=False, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        # self.D = nn.Parameter(torch.randn(self.h))

        self.up_project = nn.Linear(num_input_channels, d_model)

        self.kernel = kernel
        # SSM Kernel
        # if kernel:
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args) #TODO
        # else:
            # self.A = nn.Parameter(torch.diag(torch.rand(d_state)))
            # self.B = nn.Parameter(torch.rand(d_state, d_model))
            # self.C = nn.Parameter(torch.rand(d_model, d_state))


        # # Pointwise
        # self.activation = nn.GELU()
        # # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        # dropout_fn = DropoutNd
        # self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # # position-wise output transform to mix features
        # self.output_linear = nn.Sequential(
        #     nn.Conv1d(self.h, 2*self.h, kernel_size=1),
        #     nn.GLU(dim=-2),
        # )

        # self.penulti_linear = nn.Linear(d_model*1024, d_model*1, bias=True) # TODO


        if mup:
            # self.down_project = MuReadout(d_model, 10, bias=True, readout_zero_init=True, device='cpu')
            self.down_project = MuReadout(d_model, 10, bias=True, readout_zero_init=True)# self.down_project = MuReadout(d_model*1024, 10, bias=True, readout_zero_init=True, device='cpu')
        else:
            self.down_project = nn.Linear(d_model, 10, bias=True)

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        B = u.size(0)


        # K = torch.stack([
        #     self.C @ torch.matrix_power(self.A, i) @ self.B for i in range(L)
        # ], dim=0)

        u = u.transpose(-1, -2)
        u = self.up_project(u)
        u = u.transpose(-1, -2)
        if self.kernel:
            K = self.kernel(L=L) # (H L) # TODO
            y = torch.einsum('lih,bhl->bil', K, u) # TODO
        else:
            y = self.kernel(L=L, u=u)


        # # Convolution
        # k_f = torch.fft.rfft(k, n=2*L) # (H L)
        # u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        # y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # # Compute D term in state space equation - essentially a skip connection
        # y = y + u * self.D.unsqueeze(-1)

        # y = self.dropout(self.activation(y))
        # y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)



        # print(y.size())
        # y = y.contiguous().view(-1, y.size(1)*y.size(2))  #NOTE

        y = torch.sum(y, dim=2)

        # y = y.contiguous()
        # print(y.size())
        # y = self.penulti_linear(y) # TODO

        # # idx = torch.randperm(y.shape[0])
        # # y = y[idx].view(y.size())
        # p = y.detach().flatten()
        # # eps = torch.finfo(torch.float).tiny
        # p = p - p.min()
        # p = p[p.nonzero().detach()]
        # # p = p/p.max()
        # p = p / torch.sum(p)
        # # p = nn.Softmax()(p)
        # print(p[:3])
        # print(f"torch.sum(p) is {torch.sum(p)};torch.norm(p) is {torch.norm(p)} ")
        # print(torch.sum(-p*torch.log2(p)))

        # y = y.transpose(-1, -2)
        y = self.down_project(y)
        # y = y.transpose(-1, -2)
        # print(f"y[:, -1, :] has shape {y[:, -1, :]}")
        return y

        # return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified



def process_param_groups(params, **kwargs):
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for param_group in param_groups:
        if 'lr' not in param_group:
            param_group['lr'] = kwargs['lr']
        if 'weight_decay' not in param_group:
            param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
    return param_groups


def MuSGD(params, impl=SGD, decoupled_wd=False, **kwargs):
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
            new_g = {k:v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g
        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        vector_like_p = defaultdict(new_group) # key is width mult
        matrix_like_p = defaultdict(new_group) # key is fan_in/out ratio
        fixed_p = new_group()
        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (
                f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `mup.set_base_shapes` on the model?')
            if p.infshape.ninf() == 1:
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
        new_param_groups.extend(list(matrix_like_p.values()) + \
                                list(vector_like_p.values()) + [fixed_p])
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
    pass # nothing needs to be done

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
            elif isinstance(module, S4DKernel):
                rescale_ssm(module)
                # none-ize the base_dim of infdim in S4DKernel so that the learning rate stays constant from epoch to epoch
                module.B.infshape[1].base_dim = None
                module.C.infshape[0].base_dim = None
    return model



def simple_train(model, base_model):
    train_loader = get_train_loader(1)

    batch = next(iter(train_loader))
    (u, target) = batch

    set_base_shapes_custom(model, base_model)

    optimizer = MuSGD(model.parameters(), lr=0.001, momentum=0) #SGD(model.parameters(), lr=0.001, momentum=0)
    optimizer.zero_grad()

    out = model(u)

    loss = F.cross_entropy(out, target)
    loss.backward()
    optimizer.step()

    print(torch.sum(model.B.grad))
    print(torch.sum(model.down_project.weight.grad))

    for name, p in model.named_parameters():
        print(f"name is {name}")


if __name__ == "__main__":
    import numpy as np

    B, H, L = 1, 3, 10
    d_state = 16

    print('hellow')

    ns = [100]#, 100, 1000, 2000, 3000, 4000, 5000]

    base_sd = SSM(num_input_channels=3, d_model=5, d_state=d_state, mup=True)

    for n in ns:
        sd = SSM(num_input_channels=3, d_model=n, d_state=d_state, mup=True)
        # model_parameters = filter(lambda p: p.requires_grad, sd.parameters())

        # params = sum([np.prod(p.size()) for p in model_parameters])

        # u = torch.rand(B, H, L)
        # print(f"n={n} \t torch.norm(x)/ torch.norm(y) is {torch.norm(u)/ torch.norm(sd(u))}".expandtabs(10))

        simple_train(sd, base_sd)

        