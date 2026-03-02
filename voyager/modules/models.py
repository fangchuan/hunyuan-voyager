from typing import Any, List, Tuple, Optional, Union, Dict

import torch
import numpy as np
import torch.nn as nn
import torch.utils
import torch.utils.checkpoint
from loguru import logger
from einops import rearrange
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate, ckpt_wrapper
from .token_refiner import SingleTokenRefiner
from voyager.modules.posemb_layers import get_nd_rotary_pos_embed


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if condition_type == "token_replace":
            img_mod1, token_replace_img_mod1 = self.img_mod(vec, condition_type=condition_type,
                                                            token_replace_vec=token_replace_vec)
            (img_mod1_shift,
             img_mod1_scale,
             img_mod1_gate,
             img_mod2_shift,
             img_mod2_scale,
             img_mod2_gate) = img_mod1.chunk(6, dim=-1)
            (tr_img_mod1_shift,
             tr_img_mod1_scale,
             tr_img_mod1_gate,
             tr_img_mod2_shift,
             tr_img_mod2_scale,
             tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        if condition_type == "token_replace":
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale, condition_type=condition_type,
                tr_shift=tr_img_mod1_shift, tr_scale=tr_img_mod1_scale,
                frist_frame_token_num=frist_frame_token_num
            )
        else:
            img_modulated = modulate(
                img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
            )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(
                img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )

        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

        # Calculate the img bloks.
        if condition_type == "token_replace":
            img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate, condition_type=condition_type,
                                   tr_gate=tr_img_mod1_gate, frist_frame_token_num=frist_frame_token_num)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale,
                        condition_type=condition_type, tr_shift=tr_img_mod2_shift,
                        tr_scale=tr_img_mod2_scale, frist_frame_token_num=frist_frame_token_num
                    )
                ),
                gate=img_mod2_gate, condition_type=condition_type,
                tr_gate=tr_img_mod2_gate, frist_frame_token_num=frist_frame_token_num
            )
        else:
            img = img + \
                apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
            img = img + apply_gate(
                self.img_mlp(
                    modulate(
                        self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                    )
                ),
                gate=img_mod2_gate,
            )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn),
                               gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True,
                          eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,
    ) -> torch.Tensor:
        if condition_type == "token_replace":
            mod, tr_mod = self.modulation(vec,
                                          condition_type=condition_type,
                                          token_replace_vec=token_replace_vec)
            (mod_shift,
             mod_scale,
             mod_gate) = mod.chunk(3, dim=-1)
            (tr_mod_shift,
             tr_mod_scale,
             tr_mod_gate) = tr_mod.chunk(3, dim=-1)
        else:
            mod_shift, mod_scale, mod_gate = self.modulation(
                vec).chunk(3, dim=-1)
        if condition_type == "token_replace":
            x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale, condition_type=condition_type,
                             tr_shift=tr_mod_shift, tr_scale=tr_mod_scale, frist_frame_token_num=frist_frame_token_num)
        else:
            x_mod = modulate(self.pre_norm(
                x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D",
                            K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(
                img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        if condition_type == "token_replace":
            output = x + apply_gate(output, gate=mod_gate, condition_type=condition_type,
                                    tr_gate=tr_mod_gate, frist_frame_token_num=frist_frame_token_num)
            return output
        else:
            return x + apply_gate(output, gate=mod_gate)


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.i2v_condition_type = args.i2v_condition_type

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        # Gradient checkpoint.
        self.gradient_checkpoint = args.gradient_checkpoint
        self.gradient_checkpoint_layers = args.gradient_checkpoint_layers
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= mm_double_blocks_depth + mm_single_blocks_depth, \
                f"Gradient checkpoint layers must be less or equal than the depth of the model. " \
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and " \
                f"depth={mm_double_blocks_depth + mm_single_blocks_depth}."

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # context block
        self.use_context_block = args.use_context_block
        if self.use_context_block:
            self.condition_in = PatchEmbed(
                self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
            )

            self.context_block1 = MMDoubleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                **factory_kwargs,
            )

            self.context_block2 = MMSingleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                **factory_kwargs,
            )

            self.zero_linear1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.zero_linear2 = nn.Linear(self.hidden_size, self.hidden_size)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()
            
    ###############################################
    # 20250308 pftq: Riflex workaround to fix 192-frame-limit bug, credit to Kijai for finding it in ComfyUI
    # and thu-ml for making it
    # https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
    @staticmethod
    def get_1d_rotary_pos_embed_riflex(
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        use_real=False,
        k: Optional[int] = None,
        L_test: Optional[int] = None,
    ):
        """
        RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
        index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
        data type.

        Args:
            dim (`int`): Dimension of the frequency tensor.
            pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
            theta (`float`, *optional*, defaults to 10000.0):
                Scaling factor for frequency computation. Defaults to 10000.0.
            use_real (`bool`, *optional*):
                If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
            L_test (`int`, *optional*, defaults to None): the number of frames for inference
        Returns:
            `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
        """
        assert dim % 2 == 0

        if isinstance(pos, int):
            pos = torch.arange(pos)
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)  # type: ignore  # [S]

        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=pos.device)
                    [: (dim // 2)].float() / dim)
        )  # [D/2]

        # === Riflex modification start ===
        # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
        # Empirical observations show that a few videos may exhibit repetition in the tail frames.
        # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
        if k is not None:
            freqs[k-1] = 0.9 * 2 * torch.pi / L_test
        # === Riflex modification end ===

        freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
        if use_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
            return freqs_cos, freqs_sin
        else:
            # lumina
            freqs_cis = torch.polar(torch.ones_like(
                freqs), freqs)  # complex64     # [S, D/2]
            return freqs_cis

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2  # B, C, F, H, W -> F, H, W

        VAE_VERSION = "884-16c-hy"  # Placeholder, should be set according to actual VAE used
        # Compute latent sizes based on VAE type
        if "884" in VAE_VERSION:
            latents_size = [(video_length - 1) // 4 +
                            1, height // 8, width // 8]
        elif "888" in VAE_VERSION:
            latents_size = [(video_length - 1) // 8 +
                            1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        # Compute rope sizes
        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            assert all(
                s % self.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size[idx]
                          for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)
                                ) + rope_sizes  # Pad time axis

        # 20250316 pftq: Add RIFLEx logic for > 192 frames
        L_test = rope_sizes[0]  # Latent frames
        L_train = 25  # Training length from HunyuanVideo
        actual_num_frames = video_length  # Use input video_length directly

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list or [
            head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(
            rope_dim_list) == head_dim, "sum(rope_dim_list) must equal head_dim"

        rope_theta = 256  # theta used for RoPE
        if actual_num_frames > 192:
            k = 2+((actual_num_frames + 3) // (4 * L_train))
            k = max(4, min(8, k))
            logger.debug(f"actual_num_frames = {actual_num_frames} > 192, RIFLEx applied with k = {k}")

            # Compute positional grids for RIFLEx
            axes_grids = [torch.arange(
                size, device=self.device, dtype=torch.float32) for size in rope_sizes]
            grid = torch.meshgrid(*axes_grids, indexing="ij")
            grid = torch.stack(grid, dim=0)  # [3, t, h, w]
            pos = grid.reshape(3, -1).t()  # [t * h * w, 3]

            # Apply RIFLEx to temporal dimension
            freqs = []
            for i in range(3):
                if i == 0:  # Temporal with RIFLEx
                    freqs_cos, freqs_sin = self.get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=rope_theta,
                        use_real=True,
                        k=k,
                        L_test=L_test
                    )
                else:  # Spatial with default RoPE
                    freqs_cos, freqs_sin = self.get_1d_rotary_pos_embed_riflex(
                        rope_dim_list[i],
                        pos[:, i],
                        theta=rope_theta,
                        use_real=True,
                        k=None,
                        L_test=None
                    )
                freqs.append((freqs_cos, freqs_sin))
                logger.debug(f"freq[{i}] shape: {freqs_cos.shape}, device: {freqs_cos.device}")

            freqs_cos = torch.cat([f[0] for f in freqs], dim=1)
            freqs_sin = torch.cat([f[1] for f in freqs], dim=1)
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")
        else:
            # 20250316 pftq: Original code for <= 192 frames
            logger.debug(f"actual_num_frames = {actual_num_frames} <= 192, using original RoPE")
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim_list,
                rope_sizes,
                theta=rope_theta,
                use_real=True,
                theta_rescale_factor=1,
            )
            logger.debug(f"freqs_cos shape: {freqs_cos.shape}, device: {freqs_cos.device}")

        return freqs_cos, freqs_sin
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        # Text embedding for modulation.
        text_states_2: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        freqs_cos_cond: Optional[torch.Tensor] = None,
        freqs_sin_cond: Optional[torch.Tensor] = None,
        # Guidance for modulation, should be cfg_scale x 1000.
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        print(f"img shape: {x.shape}, t: {t.shape}, text_states: {text_states.shape if text_states is not None else None}, text_mask: {text_mask.shape if text_mask is not None else None}")
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)
        print(f"time_in output shape: {vec.shape}")

        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None

        # text modulation
        vec_2 = self.vector_in(text_states_2)
        print(f"vector_in output shape: {vec_2.shape}")
        vec = vec + vec_2
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = token_replace_vec + vec_2

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image, condition and text.
        if self.use_context_block:
            condition = img.clone()
            height = (condition.shape[-2] - 2) // 2
            condition = condition[..., -height:, :]  # depth
            print(f"only use the depth frame as condition, condition shape: {condition.shape}")
            condition = self.condition_in(condition)
            print(f"condition_in output shape: {condition.shape}")

        img = self.img_in(img)
        print(f"img_in shape: {img.shape}")
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(
                txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q
        print(f"cu_seqlens_q: {cu_seqlens_q}, max_seqlen_q: {max_seqlen_q}")
        if self.use_context_block:
            cond_seq_len = condition.shape[1]
            cu_seqlens_q_cond = get_cu_seqlens(text_mask, cond_seq_len)
            cu_seqlens_kv_cond = cu_seqlens_q_cond
            max_seqlen_q_cond = cond_seq_len + txt_seq_len
            max_seqlen_kv_cond = max_seqlen_q_cond

            # ---------------------------- Context Block ------------------------------
            context_block_args = [
                condition,
                txt,
                vec,
                cu_seqlens_q_cond,
                cu_seqlens_kv_cond,
                max_seqlen_q_cond,
                max_seqlen_kv_cond,
                (freqs_cos_cond, freqs_sin_cond),
                # (freqs_cos, freqs_sin),
                self.i2v_condition_type,
                token_replace_vec,
                frist_frame_token_num,
            ]
            condition1, txt1 = self.context_block1(*context_block_args)
            print(f"context_block1 output shape: condition1: {condition1.shape}, txt1: {txt1.shape}")

            condition2 = torch.cat((condition1, txt1), 1)
            context_block_args = [
                condition2,
                vec,
                txt_seq_len,
                cu_seqlens_q_cond,
                cu_seqlens_kv_cond,
                max_seqlen_q_cond,
                max_seqlen_kv_cond,
                (freqs_cos_cond, freqs_sin_cond),
                # (freqs_cos, freqs_sin),
                self.i2v_condition_type,
                token_replace_vec,
                frist_frame_token_num,
            ]
            condition2 = self.context_block2(*context_block_args)
            print(f"context_block2 output shape: condition2: {condition2.shape}")
            condition1 = self.zero_linear1(condition1)
            condition2 = self.zero_linear2(condition2)
            print(f"after zero_linear, condition1 shape: {condition1.shape}, condition2 shape: {condition2.shape}")

            condition2 = torch.cat(
                (torch.zeros_like(img)[:, :-condition1.shape[1]], condition2), dim=1)
            condition1 = torch.cat(
                (torch.zeros_like(img)[:, :-condition1.shape[1]], condition1), dim=1)

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for layer_num, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
                self.i2v_condition_type,
                token_replace_vec,
                frist_frame_token_num,
            ]

            if self.training and self.gradient_checkpoint and \
                    (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                logger.info('gradient checkpointing in double blocks...')
                img, txt = torch.utils.checkpoint.checkpoint(
                    ckpt_wrapper(block), *double_block_args, use_reentrant=False)
                if self.use_context_block:
                    img += condition1
            else:
                img, txt = block(*double_block_args)
                if self.use_context_block:
                    img += condition1

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)

        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                ]

                if self.training and self.gradient_checkpoint and \
                        (self.gradient_checkpoint_layers == -1 or \
                        layer_num + len(self.double_blocks) < self.gradient_checkpoint_layers):
                    logger.info('gradient checkpointing in single blocks...')
                    x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(
                        block), *single_block_args, use_reentrant=False)
                    if self.use_context_block:
                        x += condition2
                else:
                    x = block(*single_block_args)
                    if self.use_context_block:
                        x += condition2

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        # (N, T, patch_size ** 2 * out_channels)
        img = self.final_layer(img, vec)
        print(f"after final_layer, img shape: {img.shape}")

        img = self.unpatchify(img, tt, th, tw)
        print(f"after unpatchify, img shape: {img.shape}")
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

    def set_input_tensor(self, input_tensor):
        pass

#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################


HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
    "HYVideo-S/2": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
    },
}
