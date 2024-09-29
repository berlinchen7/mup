import torch
from torch.nn import Linear
import torch.nn.functional as F

class ScaledLinear(Linear):
    def __init__(self, 
                 fan_in, 
                 fan_out,  
                 mode,
                 width_mult,
                 bias=False,
                 hyperparam_mode='mup_fullalign',
                 eps=0.0,
                 **kwargs):

        super().__init__(fan_in, fan_out, bias=bias, **kwargs)

        self.hyperparam_mode = hyperparam_mode
        self.width_mult = width_mult
        self.mode = mode
        self.use_bias = bias

        valid_hyperparam_mode = ['mup_fullalign', 'mup_noalign', 
                                 'umup_fullalign',
                                 'sp_fullalign', 'sp_noalign',
                                 'ntk_fullalign', 'ntk_noalign',
                                 'mf_fullalign', 'mf_noalign']
        assert mode in ['HIDDEN', 'EMBEDDING', 'READOUT'] #, 'ENTRYWISE']
        assert hyperparam_mode in valid_hyperparam_mode

        if mode == 'EMBEDDING':
            if 'umup' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1.0
            elif 'mup' in self.hyperparam_mode:
                self.init_stddev = (1 / self.width_mult)**.5
                self.multiplier = self.width_mult**.5
            elif 'sp' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1.0
            elif 'ntk' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1.0
            elif 'mf' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1.0
        elif mode == 'HIDDEN':
            if 'umup' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1 / (fan_in**.5)
                self.bias_init_stddev = 1.0
                self.bias_multiplier = 1 / fan_in
            elif 'mup' in self.hyperparam_mode:
                self.init_stddev = (1 / self.width_mult)**.5
                self.multiplier = 1.0
                self.bias_init_stddev = (1 / self.width_mult)**.5
                self.bias_multiplier = (1 / self.width_mult)**.5
            elif 'sp' in self.hyperparam_mode:
                self.init_stddev = (1 / self.width_mult)**.5
                self.multiplier = 1.0
                self.bias_init_stddev = (1 / self.width_mult)**.5
                self.bias_multiplier = 1.0
            elif 'ntk' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = (1 / self.width_mult)**.5
                self.bias_init_stddev = 1.0
                self.bias_multiplier = (1 / self.width_mult)**.5
            elif 'mf' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = (1 / self.width_mult)**.5
                self.bias_init_stddev = 1.0
                self.bias_multiplier = 1 / self.width_mult
        elif mode == 'READOUT':
            if 'umup' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1 / fan_in
            elif 'mup' in self.hyperparam_mode:
                # breakpoint()
                self.init_stddev = (1 / self.width_mult)**.5
                self.multiplier = (1 / self.width_mult)**.5
            elif 'sp' in self.hyperparam_mode:
                self.init_stddev = (1 / self.width_mult)**.5
                self.multiplier = 1.0
            elif 'ntk' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = (1 / self.width_mult)**.5
            elif 'mf' in self.hyperparam_mode:
                self.init_stddev = 1.0
                self.multiplier = 1 / self.width_mult
        else:
            raise ValueError(f"mode {mode} is not valid!")
        
        with torch.no_grad():
            w = (torch.randn(fan_out, fan_in) + eps) * self.init_stddev
            self.weight.copy_(w)
            if self.use_bias:
                if mode == 'EMBEDDING':
                    b = (torch.randn(fan_out)) * self.init_stddev
                    self.bias_multiplier = self.multiplier
                elif mode == 'HIDDEN':
                    b = (torch.randn(fan_out)) * self.bias_init_stddev
                    assert hasattr(self, "bias_multiplier")
                elif mode == 'READOUT': # TODO: wrong here
                    b = (torch.randn(fan_out)) * 1.0
                    self.bias_multiplier = 1.0
                self.bias.copy_(b)
                
    
    # def reset_parameters(self) -> None:
    #     if self.readout_zero_init:
    #         self.weight.data[:] = 0
    #         if self.bias is not None:
    #             self.bias.data[:] = 0
    #     else:
    #         super().reset_parameters()

    # def width_mult(self):
    #     assert hasattr(self.weight, 'infshape'), (
    #         'Please call set_base_shapes(...). If using torch.nn.DataParallel, '
    #         'switch to distributed training with '
    #         'torch.nn.parallel.DistributedDataParallel instead'
    #     )
    #     return self.weight.infshape.width_mult()

    # def _rescale_parameters(self):
    #     '''Rescale parameters to convert SP initialization to μP initialization.

    #     Warning: This method is NOT idempotent and should be called only once
    #     unless you know what you are doing.
    #     '''
    #     if hasattr(self, '_has_rescaled_params') and self._has_rescaled_params:
    #         raise RuntimeError(
    #             "`_rescale_parameters` has been called once before already. "
    #             "Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n"
    #             "If you called `set_base_shapes` on a model loaded from a checkpoint, "
    #             "or just want to re-set the base shapes of an existing model, "
    #             "make sure to set the flag `rescale_params=False`.\n"
    #             "To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call.")
    #     if self.bias is not None:
    #         self.bias.data *= self.width_mult()**0.5
    #     self.weight.data *= self.width_mult()**0.5
    #     self._has_rescaled_params = True
                    
    def forward(self, input):
        # return super().forward(
        #     self.output_mult * x)
        if self.use_bias:
            return F.linear(input, 
                            self.weight*self.multiplier,
                            self.bias*self.bias_multiplier,)
        else:
            return F.linear(input, 
                            self.weight*self.multiplier,)