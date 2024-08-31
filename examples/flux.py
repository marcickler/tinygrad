import re
from dataclasses import dataclass

from tinygrad import Tensor, nn
from tinygrad.dtype import dtypes
import math

from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
from extra.models.clip import FrozenClosedClipEmbedder
from glob import iglob

def randn_like(x:Tensor) -> Tensor: return Tensor.randn(*x.shape, dtype=x.dtype).to(device=x.device)

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
  q, k = apply_rope(q, k, pe)

  x = Tensor.scaled_dot_product_attention(q, k, v)
  x = x.rearrange(x, "B H L D -> B L (H D)")

  return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
  assert dim % 2 == 0
  scale = Tensor.arange(0, dim, 2).float() / dim
  omega = 1.0 / (theta**scale)
  out = pos.unsqueeze(-1) * omega
  out = Tensor.stack([out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1)
  out = out.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
  return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(xk.dtype)


@dataclass
class AutoEncoderParams:
  resolution: int
  in_channels: int
  ch: int
  out_ch: int
  ch_mult: list[int]
  num_res_blocks: int
  z_channels: int
  scale_factor: float
  shift_factor: float

def swish(x: Tensor) -> Tensor:
  return x * x.sigmoid()

class AttnBlock:
  def __init__(self, in_channels: int):
    super().__init__()
    self.in_channels = in_channels

    self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

  def attention(self, h_: Tensor) -> Tensor:
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    b, c, h, w = q.shape
    q = q.reshape(b, 1, h * w, c)
    k = k.reshape(b, 1, h * w, c)
    v = v.reshape(b, 1, h * w, c)
    h_ = q.scaled_dot_product_attention(k, v)

    return h_.reshape(b, c, h, w)

  def __call__(self, x: Tensor) -> Tensor:
    return x + self.proj_out(self.attention(x))

class ResnetBlock:
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels

    self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if self.in_channels != self.out_channels:
      self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

  def __call__(self, x):
    h = x
    h = self.norm1(h)
    h = swish(h)
    h = self.conv1(h)

    h = self.norm2(h)
    h = swish(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      x = self.nin_shortcut(x)

    return x + h

class Downsample:
  def __init__(self, in_channels: int):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

  def __call__(self, x: Tensor):
    pad = ((0,0),(0,0), (0, 1), (0, 1))
    x = x.pad(pad, value=0)
    x = self.conv(x)
    return x

class Upsample:
  def __init__(self, in_channels: int):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

  def __call__(self, x: Tensor):
    x = x.interpolate(size=(x.shape[2]*2, x.shape[3]*2), mode="nearest")
    x = self.conv(x)
    return x

class Encoder:
  def __init__(
    self,
    resolution: int,
    in_channels: int,
    ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    z_channels: int,
  ):
    super().__init__()
    self.ch = ch
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels
    self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

    curr_res = resolution
    in_ch_mult = (1,) + tuple(ch_mult)
    self.in_ch_mult = in_ch_mult
    self.down_blocks = []
    self.down_attn = []
    self.downsample_layers = []
    block_in = self.ch
    for i_level in range(self.num_resolutions):
      block = []
      attn = []
      block_in = ch * in_ch_mult[i_level]
      block_out = ch * ch_mult[i_level]
      for _ in range(self.num_res_blocks):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
        block_in = block_out
      self.down_blocks.append(block)
      self.down_attn.append(attn)
      if i_level != self.num_resolutions - 1:
        self.downsample_layers.append(Downsample(block_in))
        curr_res = curr_res // 2
      else:
        self.downsample_layers.append(None)

    self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
    self.mid_attn_1 = AttnBlock(block_in)
    self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

    # end
    self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

  def __call__(self, x: Tensor) -> Tensor:
    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down_blocks[i_level][i_block](hs[-1])
        if len(self.down_attn[i_level]) > 0:
          h = self.down_attn[i_level][i_block](h)
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(self.downsample_layers[i_level](hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid_block_1(h)
    h = self.mid_attn_1(h)
    h = self.mid_block_2(h)
    # end
    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    return h

class Decoder:
  def __init__(
    self,
    ch: int,
    out_ch: int,
    ch_mult: list[int],
    num_res_blocks: int,
    in_channels: int,
    resolution: int,
    z_channels: int,
  ):
    super().__init__()
    self.ch = ch
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels
    self.ffactor = 2 ** (self.num_resolutions - 1)

    # compute in_ch_mult, block_in and curr_res at lowest res
    block_in = ch * ch_mult[self.num_resolutions - 1]
    curr_res = resolution // 2 ** (self.num_resolutions - 1)
    self.z_shape = (1, z_channels, curr_res, curr_res)

    # z to block_in
    self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

    # middle
    self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
    self.mid_attn_1 = AttnBlock(block_in)
    self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

    # upsampling
    self.up_blocks = []
    self.up_attn = []
    self.upsample_layers = []
    for i_level in reversed(range(self.num_resolutions)):
      block = []
      attn = []
      block_out = ch * ch_mult[i_level]
      for _ in range(self.num_res_blocks + 1):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
        block_in = block_out
      self.up_blocks.insert(0, block)
      self.up_attn.insert(0, attn)
      if i_level != 0:
        self.upsample_layers.insert(0, Upsample(block_in))
        curr_res = curr_res * 2
      else:
        self.upsample_layers.insert(0, None)

    self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

  def __call__(self, z: Tensor) -> Tensor:
    h = self.conv_in(z)

    h = self.mid_block_1(h)
    h = self.mid_attn_1(h)
    h = self.mid_block_2(h)

    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = self.up_blocks[i_level][i_block](h)
        if len(self.up_attn[i_level]) > 0:
          h = self.up_attn[i_level][i_block](h)
      if i_level != 0:
        h = self.upsample_layers[i_level](h)

    h = self.norm_out(h)
    h = swish(h)
    h = self.conv_out(h)
    return h

class DiagonalGaussian:
  def __init__(self, sample: bool = True, chunk_dim: int = 1):
    super().__init__()
    self.sample = sample
    self.chunk_dim = chunk_dim

  def __call__(self, z: Tensor) -> Tensor:
    mean, logvar = z.chunk(2, dim=self.chunk_dim)
    if self.sample:
      std = (0.5 * logvar).exp()
      return mean + std * randn_like(mean)
    else:
      return mean

class AutoEncoder:
  def __init__(self, params: AutoEncoderParams):
    super().__init__()
    self.encoder = Encoder(
      resolution=params.resolution,
      in_channels=params.in_channels,
      ch=params.ch,
      ch_mult=params.ch_mult,
      num_res_blocks=params.num_res_blocks,
      z_channels=params.z_channels,
    )
    self.decoder = Decoder(
      resolution=params.resolution,
      in_channels=params.in_channels,
      ch=params.ch,
      out_ch=params.out_ch,
      ch_mult=params.ch_mult,
      num_res_blocks=params.num_res_blocks,
      z_channels=params.z_channels,
    )
    self.reg = DiagonalGaussian()

    self.scale_factor = params.scale_factor
    self.shift_factor = params.shift_factor

  def encode(self, x: Tensor) -> Tensor:
    z = self.reg(self.encoder(x))
    z = self.scale_factor * (z - self.shift_factor)
    return z

  def decode(self, z: Tensor) -> Tensor:
    z = z / self.scale_factor + self.shift_factor
    return self.decoder(z)

  def __call__(self, x: Tensor) -> Tensor:
    return self.decode(self.encode(x))


class HFEmbedder:
  def __init__(self, version: str, max_length: int, **hf_kwargs):
    super().__init__()
    self.is_clip = version.startswith("openai")
    self.max_length = max_length
    self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

    if self.is_clip:
      self.tokenizer = FrozenClosedClipEmbedder(version, max_length=max_length)
      self.hf_module = FrozenClosedClipEmbedder(version, **hf_kwargs)
    else:
      pass
      # self.tokenizer = T5Encoder(version, max_length=max_length)
      # self.hf_module = T5Encoder(version, **hf_kwargs)

    self.hf_module.eval().requires_grad_(False)

  def __call__(self, text: list[str]) -> Tensor:
    batch_encoding = self.tokenizer.tokenize(
      text,
      truncation=True,
      max_length=self.max_length,
      return_length=False,
      return_overflowing_tokens=False,
      padding="max_length",
      return_tensors="pt",
    )

    outputs = self.hf_module(
      input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
      attention_mask=None,
      output_hidden_states=False,
    )
    return outputs[self.output_key]


class EmbedND:
  def __init__(self, dim: int, theta: int, axes_dim: list[int]):
    super().__init__()
    self.dim = dim
    self.theta = theta
    self.axes_dim = axes_dim

  def __call__(self, ids: Tensor) -> Tensor:
    n_axes = ids.shape[-1]
    emb = Tensor.cat(
      [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
      dim=-3,
    )

    return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
  t = time_factor * t
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half) / half).exp().to(t.device)

  args = t[:, None].float() * freqs[None]
  embedding = Tensor.cat([args.cos(), args.sin()], dim=-1)
  if dim % 2:
    embedding = Tensor.cat([embedding, Tensor.zeros_like(embedding[:, :1])], dim=-1)
  if t.is_floating_point():
    embedding = embedding.to(t)
  return embedding


class MLPEmbedder:
  def __init__(self, in_dim: int, hidden_dim: int):
    super().__init__()
    self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
    self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

  def __call__(self, x: Tensor) -> Tensor:
    return self.out_layer((self.in_layer(x)).silu())


class QKNorm:
  def __init__(self, dim: int):
    super().__init__()
    self.query_norm = nn.RMSNorm(dim)
    self.key_norm = nn.RMSNorm(dim)

  def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    q = self.query_norm(q)
    k = self.key_norm(k)
    return q.to(v), k.to(v)


class SelfAttention:
  def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.norm = QKNorm(head_dim)
    self.proj = nn.Linear(dim, dim)

  def __call__(self, x: Tensor, pe: Tensor) -> Tensor:
    qkv = self.qkv(x)
    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)
    x = attention(q, k, v, pe=pe)
    x = self.proj(x)
    return x


@dataclass
class ModulationOut:
  shift: Tensor
  scale: Tensor
  gate: Tensor


class Modulation:
  def __init__(self, dim: int, double: bool):
    super().__init__()
    self.is_double = double
    self.multiplier = 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

  def __call__(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
    out = self.lin(Tensor.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

    return (
      ModulationOut(*out[:3]),
      ModulationOut(*out[3:]) if self.is_double else None,
    )


class DoubleStreamBlock:
  def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
    super().__init__()

    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.img_mod = Modulation(hidden_size, double=True)
    self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True),Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

    self.txt_mod = Modulation(hidden_size, double=True)
    self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp = [nn.Linear(hidden_size, mlp_hidden_dim, bias=True),Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size, bias=True)]

  def __call__(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    # run actual attention
    q = Tensor.cat((txt_q, img_q), dim=2)
    k = Tensor.cat((txt_k, img_k), dim=2)
    v = Tensor.cat((txt_v, img_v), dim=2)

    attn = attention(q, k, v, pe=pe)
    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * ((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift).sequential(self.img_mlp)

    # calculate the txt bloks
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * ((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift).sequential(self.txt_mlp)
    return img, txt


class SingleStreamBlock:
  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qk_scale: float | None = None,
  ):
    super().__init__()
    self.hidden_dim = hidden_size
    self.num_heads = num_heads
    head_dim = hidden_size // num_heads
    self.scale = qk_scale or head_dim**-0.5

    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
    self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

    self.norm = QKNorm(head_dim)

    self.hidden_size = hidden_size
    self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    self.mlp_act = Tensor.gelu
    self.modulation = Modulation(hidden_size, double=False)

  def __call__(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = Tensor.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)

    # compute attention
    attn = attention(q, k, v, pe=pe)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(Tensor.cat((attn, self.mlp_act(mlp)), 2))
    return x + mod.gate * output


class LastLayer:
  def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
    super().__init__()
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = [Tensor.silu, nn.Linear(hidden_size, 2 * hidden_size, bias=True)]

  def __call__(self, x: Tensor, vec: Tensor) -> Tensor:
    shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=1)
    x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
    x = self.linear(x)
    return x


@dataclass
class FluxParams:
  in_channels: int
  vec_in_dim: int
  context_in_dim: int
  hidden_size: int
  mlp_ratio: float
  num_heads: int
  depth: int
  depth_single_blocks: int
  axes_dim: list[int]
  theta: int
  qkv_bias: bool
  guidance_embed: bool


class Flux:
  """
  Transformer model for flow matching on sequences.
  """

  def __init__(self, params: FluxParams):
    super().__init__()

    self.params = params
    self.in_channels = params.in_channels
    self.out_channels = self.in_channels
    if params.hidden_size % params.num_heads != 0:
      raise ValueError(
        f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
      )
    pe_dim = params.hidden_size // params.num_heads
    if sum(params.axes_dim) != pe_dim:
      raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
    self.hidden_size = params.hidden_size
    self.num_heads = params.num_heads
    self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
    self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
    self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
    self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
    self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else lambda x: x
    self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

    self.double_blocks = [
        DoubleStreamBlock(
          self.hidden_size,
          self.num_heads,
          mlp_ratio=params.mlp_ratio,
          qkv_bias=params.qkv_bias,
        )
        for _ in range(params.depth)
      ]

    self.single_blocks = [
        SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
        for _ in range(params.depth_single_blocks)
      ]

    self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

  def __call__(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor | None = None,
  ) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3:
      raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256))
    if self.params.guidance_embed:
      if guidance is None:
        raise ValueError("Didn't get guidance strength for guidance distilled model.")
      vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + self.vector_in(y)
    txt = self.txt_in(txt)

    ids = Tensor.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    for block in self.double_blocks:
      img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

    img = Tensor.cat((txt, img), 1)
    for block in self.single_blocks:
      img = block(img, vec=vec, pe=pe)
    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


import os

@dataclass
class ModelSpec:
  params: FluxParams
  ae_params: AutoEncoderParams
  ckpt_path: str | None
  ae_path: str | None
  repo_id: str | None
  repo_flow: str | None
  repo_ae: str | None


configs = {
  "flux-dev": ModelSpec(
    repo_id="black-forest-labs/FLUX.1-dev",
    repo_flow="flux1-dev.safetensors",
    repo_ae="ae.safetensors",
    ckpt_path=os.getenv("FLUX_DEV"),
    params=FluxParams(
      in_channels=64,
      vec_in_dim=768,
      context_in_dim=4096,
      hidden_size=3072,
      mlp_ratio=4.0,
      num_heads=24,
      depth=19,
      depth_single_blocks=38,
      axes_dim=[16, 56, 56],
      theta=10_000,
      qkv_bias=True,
      guidance_embed=True,
    ),
    ae_path=os.getenv("AE"),
    ae_params=AutoEncoderParams(
      resolution=256,
      in_channels=3,
      ch=128,
      out_ch=3,
      ch_mult=[1, 2, 4, 4],
      num_res_blocks=2,
      z_channels=16,
      scale_factor=0.3611,
      shift_factor=0.1159,
    ),
  ),
  "flux-schnell": ModelSpec(
    repo_id="black-forest-labs/FLUX.1-schnell",
    repo_flow="flux1-schnell.safetensors",
    repo_ae="ae.safetensors",
    ckpt_path=os.getenv("FLUX_SCHNELL"),
    params=FluxParams(
      in_channels=64,
      vec_in_dim=768,
      context_in_dim=4096,
      hidden_size=3072,
      mlp_ratio=4.0,
      num_heads=24,
      depth=19,
      depth_single_blocks=38,
      axes_dim=[16, 56, 56],
      theta=10_000,
      qkv_bias=True,
      guidance_embed=False,
    ),
    ae_path=os.getenv("AE"),
    ae_params=AutoEncoderParams(
      resolution=256,
      in_channels=3,
      ch=128,
      out_ch=3,
      ch_mult=[1, 2, 4, 4],
      num_res_blocks=2,
      z_channels=16,
      scale_factor=0.3611,
      shift_factor=0.1159,
    ),
  ),
}


def vae_key_mapping(keys: list[str]):
  decoder_up_pattern = r'decoder\.up\.(\d+)\.block\.(\d+)\.'
  decoder_up_replacement = r'decoder.up_blocks.\1.\2.'
  encoder_down_pattern = r'encoder\.down\.(\d+)\.block\.(\d+)\.'
  encoder_down_replacement = r'encoder.down_blocks.\1.\2.'
  upsampler_pattern = r'decoder\.up\.(\d+)\.upsample\.'
  upsampler_replacement = r'decoder.upsample_layers.\1.'
  downsampler_pattern = r'encoder\.down\.(\d+)\.downsample\.'
  downsampler_replacement = r'encoder.downsample_layers.\1.'
  key_map = {}
  for key in keys:
    if "mid.attn" in key:
      key_map[key] = key.replace("mid.attn", "mid_attn")
    elif "mid.block" in key:
      key_map[key] = key.replace("mid.block", "mid_block")
    elif "upsample" in key:
      key_map[key] = re.sub(upsampler_pattern, upsampler_replacement, key)
    elif "downsample" in key:
      key_map[key] = re.sub(downsampler_pattern, downsampler_replacement, key)
    elif key.startswith("decoder.up"):
      key_map[key] = re.sub(decoder_up_pattern, decoder_up_replacement, key)
    elif key.startswith("encoder.down"):
      key_map[key] = re.sub(encoder_down_pattern, encoder_down_replacement, key)

    else:
      key_map[key] = key
  return key_map


def load_vae():
  ae_weights_url = 'https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors'
  # tiny_state_dict =get_state_dict(ae)
  weights_fn  = fetch(ae_weights_url, os.path.basename(str(ae_weights_url)))
  weights = safe_load(weights_fn)
  vae_key_map = vae_key_mapping(list(weights.keys()))
  new_state_dict = {}
  for k, v in weights.items():
    new_key = vae_key_map[k]
    new_state_dict[new_key] = v
  ae = AutoEncoder(configs["flux-schnell"].ae_params)
  load_state_dict(ae, new_state_dict, strict=True)
  return ae

def load_clip():
  clip_weights_url = 'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors'
  weights_fn = fetch(clip_weights_url, os.path.basename(str(clip_weights_url)))
  clip_state_dict = safe_load(weights_fn)
  for k in list(clip_state_dict.keys()):
    if k.startswith("text_model"):
      new_key = 'transformer.' + k
      clip_state_dict[new_key] = clip_state_dict.pop(k)
  clip = FrozenClosedClipEmbedder()
  load_state_dict(clip, clip_state_dict, strict=True)

def load_flux():
  flow_weights_url = 'https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors'
  flow_weights_fn = fetch(flow_weights_url, os.path.basename(str(flow_weights_url)))
  state_dict = safe_load(flow_weights_fn)
  for key in list(state_dict.keys()):
    if "scale" in key:
      new_key = key.replace("scale", "weight")
      state_dict[new_key] = state_dict.pop(key)
  flow_model = Flux(configs["flux-schnell"].params)
  load_state_dict(flow_model, state_dict, strict=True)


@dataclass
class SamplingOptions:
  prompt: str
  width: int
  height: int
  num_steps: int = 4
  guidance: float = 3.5
  seed: int | None = 42


def get_noise(
  num_samples: int,
  height: int,
  width: int,
  device: str,
  dtype: str,
  seed: int,
):
  Tensor.manual_seed(seed)
  return Tensor.randn(
    num_samples,
    16,
    2 * math.ceil(height / 16),
    2 * math.ceil(width / 16),
    dtype=dtype,
  ).to(device)

def run_inference(width, height, seed, prompt, num_steps, guidance, output_dir, device="cpu"):
  # allow for packing and conversion to latent space
  height = 16 * (height // 16)
  width = 16 * (width // 16)
  output_name = os.path.join(output_dir, "img_{idx}.jpg")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    idx = 0
  else:
    fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
    if len(fns) > 0:
      idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
    else:
      idx = 0
  x = get_noise(1, height, width, device, dtype=dtypes.bfloat16, seed=seed)
  print(x.realize())


if __name__ == "__main__":
  opts = SamplingOptions(prompt='Happy cat', width=384, height=384)
  run_inference(opts.width, opts.height, opts.seed, opts.prompt, opts.num_steps, opts.guidance, output_dir=os.getcwd())


