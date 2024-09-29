# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

# from mamba_ssm.models.config_mamba import MambaConfig
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.modules.mha import MHA
# from mamba_ssm.modules.mlp import GatedMLP
from examples.SSMs.ssm import SelectiveSSMKernel
from examples.SSMs.block import Block
# from mamba_ssm.utils.generation import GenerationMixin
# from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf


# try:
#     from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate=0,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,

    hyperparam_mode='mup_fullalign',
    d_model_base=16,
):
    # if ssm_cfg is None:
    #     ssm_cfg = {}
    # if attn_layer_idx is None:
    #     attn_layer_idx = []
    # if attn_cfg is None:
    #     attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # if layer_idx not in attn_layer_idx:
    #     # Create a copy of the config to modify
    #     ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
    #     ssm_layer = ssm_cfg.pop("layer", "Mamba1")
    #     if ssm_layer not in ["Mamba1", "Mamba2"]:
    #         raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
    #     mixer_cls = partial(
    #         Mamba2 if ssm_layer == "Mamba2" else Mamba,
    #         layer_idx=layer_idx,
    #         **ssm_cfg,
    #         **factory_kwargs
    #     )
    # else:
    #     mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

    ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
    ssm_cfg['hyperparam_mode'] = hyperparam_mode
    ssm_cfg['d_model_base'] = d_model_base
    mixer_cls = partial(SelectiveSSMKernel, **ssm_cfg,  **factory_kwargs) # Might modify to add 1dCNN along with layer_index
    norm_cls = nn.Identity

    # norm_cls = partial(
    #     nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    # )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        raise ValueError
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=False,
        residual_in_fp32=False,
    )

    block.layer_idx = layer_idx
    return block


# # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
# def _init_weights(
#     module,
#     n_layer,
#     initializer_range=0.02,  # Now only used for embedding layer.
#     rescale_prenorm_residual=True,
#     n_residuals_per_layer=1,  # Change to 2 if we have MLP
# ):
#     if isinstance(module, nn.Linear):
#         if module.bias is not None:
#             if not getattr(module.bias, "_no_reinit", False):
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         nn.init.normal_(module.weight, std=initializer_range)

#     if rescale_prenorm_residual:
#         # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
#         #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
#         #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
#         #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
#         #
#         # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
#         for name, p in module.named_parameters():
#             if name in ["out_proj.weight", "fc2.weight"]:
#                 # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
#                 # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
#                 # We need to reinit p since this code could be called multiple times
#                 # Having just p *= scale would repeatedly scale it down
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 with torch.no_grad():
#                     p /= math.sqrt(n_residuals_per_layer * n_layer)

class MixerModelEmbedding(nn.Module):
    def __init__(self, fan_in, fan_out, width_mult, hyperparam_mode, *args, **kwargs):
        """In this case, fan_in is vocab_size and fan_out is the embedding dimension,
        which in mamba is d_model"""
        super().__init__(*args, **kwargs)
        self.width_mult = width_mult
        self.vocab_size = fan_in
        self.hyperparam_mode = hyperparam_mode
        # NOTE: we initialize embed_w to be of shape (fan_in, fan_out), whereas
        # we initialize decode_w in MixerModelDecoder to be of shape (fan_out, fan_in).
        # The reason is to accommodate what F.embedding expects
        if 'umap' in hyperparam_mode:
            self.embed_w = nn.Parameter(torch.randn(fan_in, fan_out))
        elif 'mup' in hyperparam_mode:
            self.embed_w = nn.Parameter(torch.randn(fan_in, fan_out) * (width_mult**(-.5)))
        elif 'sp' in hyperparam_mode:
            self.embed_w = nn.Parameter(torch.randn(fan_in, fan_out))
        elif 'ntk' in hyperparam_mode:
            self.embed_w = nn.Parameter(torch.randn(fan_in, fan_out))
        elif 'mf' in hyperparam_mode:
            self.embed_w = nn.Parameter(torch.randn(fan_in, fan_out))
        else:
            raise ValueError(f'hyperparam_mode = {hyperparam_mode} not recognized.')
                    
    def forward(self, x):
        """
        x is of the form (sequence_length, batch_size).
        First convert to one-hot, before applying self.embed_w,
        along with hyperparam_mode specific multipliers.

        output should have the form (batch_size, embedding_dim, sequence_length)
        """
        # TODO: Stud to effectively decrease token size:
        # breakpoint()
        with torch.no_grad():
            MAX_TOKEN_SIZE = 20
            self.embed_w[MAX_TOKEN_SIZE:, :] = 0.
        
        x_onehot = F.one_hot(x, num_classes=self.vocab_size).to(self.embed_w.dtype) # size (seq_length, batch_size, vocab_size)
        embeded = torch.einsum('lbv,vd->bdl', x_onehot, self.embed_w)


        # embeded = F.embedding(x, self.embed_w) # size (seq_length, batch_size, embedding_dim)
        # embeded = einops.rearrange(embeded, 'l b e -> b e l')
        # breakpoint()

        if 'umap' in self.hyperparam_mode:
            return embeded
        elif 'mup' in self.hyperparam_mode:
            return embeded*(self.width_mult**(.5))
        elif 'sp' in self.hyperparam_mode:
            return embeded
        elif 'ntk' in self.hyperparam_mode:
            return embeded
        elif 'mf' in self.hyperparam_mode:
            return embeded
        else:
            raise ValueError

class MixerModelDecoder(nn.Module):
    def __init__(self, fan_in, fan_out, width_mult, hyperparam_mode, *args, **kwargs):
        """In this case, fan_in is embedding dimension and fan_out is vocab_size"""
        super().__init__(*args, **kwargs)
        self.width_mult = width_mult
        self.hyperparam_mode = hyperparam_mode
        self.fan_in = fan_in # Used for u-mup
        if 'umup' in hyperparam_mode:
            self.decode_w = nn.Parameter((torch.randn(fan_out, fan_in)))
        elif 'mup' in hyperparam_mode:
            self.decode_w = nn.Parameter((torch.randn(fan_out, fan_in)))# *(width_mult**(-.5))) #TODO comment out for now
            # self.decode_w = nn.Parameter((torch.rand(fan_out, fan_in)*2) - 1)
        elif 'sp' in hyperparam_mode:
            self.decode_w = nn.Parameter((torch.randn(fan_out, fan_in)*(width_mult**(-.5))))
        elif 'ntk' in hyperparam_mode:
            self.decode_w = nn.Parameter((torch.randn(fan_out, fan_in)))
        elif 'mf' in hyperparam_mode:
            self.decode_w = nn.Parameter((torch.randn(fan_out, fan_in)))
        else:
            raise ValueError(f'hyperparam_mode = {hyperparam_mode} not recognized.')
                    
    def forward(self, x):
        """x has shape (batch_size, d_model, sequence_length)
        output has shape (sequence_length, batch_size, vocab_size)
        """
        x_t = einops.rearrange(x, "b h l -> b l h")
        if 'umup' in self.hyperparam_mode:
            return torch.einsum('vh,blh->lbv', self.decode_w, x_t)*(self.fan_in**(-1.0))
        elif 'mup' in self.hyperparam_mode:
            return torch.einsum('vh,blh->lbv', self.decode_w, x_t)*(self.width_mult**(-.5))/ 2**1.5 # TODO Added the division by constant arbitrarily
        elif 'sp' in self.hyperparam_mode:
            return torch.einsum('vh,blh->lbv', self.decode_w, x_t) / 2**1.5 # TODO Added the division by constant arbitrarily
        elif 'ntk' in self.hyperparam_mode:
            return torch.einsum('vh,blh->lbv', self.decode_w, x_t)*(self.width_mult**(-.5))
        elif 'mf' in self.hyperparam_mode:
            return torch.einsum('vh,blh->lbv', self.decode_w, x_t)*(self.width_mult**(-1.0))
        else:
            raise ValueError


class MixerModelWithSimpleHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_intermediate: int = 0,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,

        hyperparam_mode='mup_fullalign',
        d_model_base=5,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # if ssm_cfg is not None and 'd_model_base' in ssm_cfg:
        #     d_model_base = ssm_cfg['d_model_base']
        # else:
        #     d_model_base = 16
        self.width_mult = int(d_model/d_model_base)
        # print(self.width_mult)
        self.hyperparam_mode = hyperparam_mode

        self.embedding = MixerModelEmbedding(vocab_size, d_model, self.width_mult, self.hyperparam_mode)

        # # We change the order of residual and layer norm:
        # # Instead of LN -> Attn / MLP -> Add, we do:
        # # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # # This is for performance reason: we can fuse add + layer_norm.
        # self.fused_add_norm = fused_add_norm
        # if self.fused_add_norm:
        #     if layer_norm_fn is None or rms_norm_fn is None:
        #         raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,

                    hyperparam_mode=hyperparam_mode,
                    d_model_base=d_model_base,
                )
                for i in range(n_layer)
            ]
        )

        # self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
        #     d_model, eps=norm_epsilon, **factory_kwargs
        # )

        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=n_layer,
        #         **(initializer_cfg if initializer_cfg is not None else {}),
        #         n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
        #     )
        # )
        self.decoder = MixerModelDecoder(d_model, vocab_size, self.width_mult, hyperparam_mode=hyperparam_mode)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        # breakpoint()
        hidden_states = self.embedding(input_ids)
        # breakpoint()
        residual = None
        for li, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, None, inference_params=inference_params, **mixer_kwargs # Note: can change residual to None to get rid of residuals
            )
        # if (not hasattr(self, 'fused_add_norm')) or (not self.fused_add_norm):
        #     residual = (hidden_states + residual) if residual is not None else hidden_states
        #     hidden_states = residual # NOTE: Got rid of norm_f here
        # else:
        #     raise ValueError
        #     # Set prenorm=False here since we don't need the residual
        #     hidden_states = layer_norm_fn(
        #         hidden_states,
        #         self.norm_f.weight,
        #         self.norm_f.bias,
        #         eps=self.norm_f.eps,
        #         residual=residual,
        #         prenorm=False,
        #         residual_in_fp32=self.residual_in_fp32,
        #         is_rms_norm=isinstance(self.norm_f, RMSNorm)
        #     )
        # breakpoint()
        output = self.decoder(hidden_states)
        return output
        # return F.log_softmax(output, dim=-1)


# class MambaLMHeadModel(nn.Module, GenerationMixin):

#     def __init__(
#         self,
#         config: MambaConfig,
#         initializer_cfg=None,
#         device=None,
#         dtype=None,
#     ) -> None:
#         self.config = config
#         d_model = config.d_model
#         n_layer = config.n_layer
#         d_intermediate = config.d_intermediate
#         vocab_size = config.vocab_size
#         ssm_cfg = config.ssm_cfg
#         attn_layer_idx = config.attn_layer_idx
#         attn_cfg = config.attn_cfg
#         rms_norm = config.rms_norm
#         residual_in_fp32 = config.residual_in_fp32
#         fused_add_norm = config.fused_add_norm
#         pad_vocab_size_multiple = config.pad_vocab_size_multiple
#         factory_kwargs = {"device": device, "dtype": dtype}

#         super().__init__()
#         if vocab_size % pad_vocab_size_multiple != 0:
#             vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
#         self.backbone = MixerModel(
#             d_model=d_model,
#             n_layer=n_layer,
#             d_intermediate=d_intermediate,
#             vocab_size=vocab_size,
#             ssm_cfg=ssm_cfg,
#             attn_layer_idx=attn_layer_idx,
#             attn_cfg=attn_cfg,
#             rms_norm=rms_norm,
#             initializer_cfg=initializer_cfg,
#             fused_add_norm=fused_add_norm,
#             residual_in_fp32=residual_in_fp32,
#             **factory_kwargs,
#         )
#         self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

#         # Initialize weights and apply final processing
#         self.apply(
#             partial(
#                 _init_weights,
#                 n_layer=n_layer,
#                 **(initializer_cfg if initializer_cfg is not None else {}),
#             )
#         )
#         self.tie_weights()

#     def tie_weights(self):
#         if self.config.tie_embeddings:
#             self.lm_head.weight = self.backbone.embedding.weight

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

#     def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
#         """
#         "position_ids" is just to be compatible with Transformer generation. We don't use it.
#         num_last_tokens: if > 0, only return the logits for the last n tokens
#         """
#         hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
#         if num_last_tokens > 0:
#             hidden_states = hidden_states[:, -num_last_tokens:]
#         lm_logits = self.lm_head(hidden_states)
#         CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
#         return CausalLMOutput(logits=lm_logits)

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
#         config_data = load_config_hf(pretrained_model_name)
#         config = MambaConfig(**config_data)
#         model = cls(config, device=device, dtype=dtype, **kwargs)
#         model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
#         return model

#     def save_pretrained(self, save_directory):
#         """
#         Minimal implementation of save_pretrained for MambaLMHeadModel.
#         Save the model and its configuration file to a directory.
#         """
#         # Ensure save_directory exists
#         os.makedirs(save_directory, exist_ok=True)

#         # Save the model's state_dict
#         model_path = os.path.join(save_directory, 'pytorch_model.bin')
#         torch.save(self.state_dict(), model_path)

#         # Save the configuration of the model
#         config_path = os.path.join(save_directory, 'config.json')
#         with open(config_path, 'w') as f:
#             json.dump(self.config.__dict__, f, indent=4)





if __name__ == '__main__':
    # create_block(d_model = 16,
    #             d_intermediate=0,
    #             ssm_cfg=None,
    #             # attn_layer_idx=None,
    #             # attn_cfg=None,
    #             # norm_epsilon=1e-5,
    #             # rms_norm=False,
    #             # residual_in_fp32=False,
    #             # fused_add_norm=False,
    #             layer_idx=None,
    #             device=None,
    #             dtype=None)
    m = MixerModelWithSimpleHead(
        d_model = 16,
        n_layer = 2,
        vocab_size = 4,
        d_intermediate = 0,
        ssm_cfg=None,
        device=None,
        dtype=None,
        hyperparam_mode='mup_fullallign')
    print([n for n, _ in m.named_parameters()])