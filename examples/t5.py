import copy
from dataclasses import dataclass

from typing import Optional, Union, Tuple

from tinygrad import Tensor, nn
import math

from tinygrad.helpers import fetch
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, load_state_dict


class T5DenseActDense:
  def __init__(self, config):
    self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
    self.dropout = lambda x: x.dropout(config.dropout_rate)
    self.act = Tensor.gelu # gelu-new

  def __call__(self, hidden_states):
    hidden_states = self.wi(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states

class T5DenseGatedActDense:
  def __init__(self, config):
    self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
    self.dropout = lambda x: x.dropout(config.dropout_rate)
    self.act = Tensor.gelu # gelu-new

  def __call__(self, hidden_states):
    hidden_gelu = self.act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states

class T5LayerFF:
  def __init__(self, config):
    if config.is_gated_act:
      self.DenseReluDense = T5DenseGatedActDense(config)
    else:
      self.DenseReluDense = T5DenseActDense(config)

    self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = lambda x: x.dropout(config.dropout_rate)

  def __call__(self, hidden_states):
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    hidden_states = hidden_states + self.dropout(forwarded_states)
    return hidden_states

class T5Attention:
  def __init__(self, config, has_relative_attention_bias=False):
    self.is_decoder = config.is_decoder
    self.has_relative_attention_bias = has_relative_attention_bias
    self.relative_attention_num_buckets = config.relative_attention_num_buckets
    self.relative_attention_max_distance = config.relative_attention_max_distance
    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.dropout = config.dropout_rate
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    if self.has_relative_attention_bias:
      self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

  @staticmethod
  def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = Tensor.zeros_like(relative_position)
    if bidirectional:
      num_buckets //= 2
      relative_buckets += (relative_position > 0).astype(Tensor.int32) * num_buckets
      relative_position = Tensor.abs(relative_position)
    else:
      relative_position = -Tensor.minimum(relative_position, Tensor.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
      Tensor.log(relative_position.float() / max_exact)
      / math.log(max_distance / max_exact)
      * (num_buckets - max_exact)
    ).astype(Tensor.int32)
    relative_position_if_large = Tensor.minimum(
      relative_position_if_large, Tensor.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += Tensor.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets

  def compute_bias(self, query_length, key_length):
    context_position = Tensor.arange(query_length)[:, None]
    memory_position = Tensor.arange(key_length)[None, :]
    relative_position = memory_position - context_position
    relative_position_bucket = self._relative_position_bucket(
      relative_position,
      bidirectional=(not self.is_decoder),
      num_buckets=self.relative_attention_num_buckets,
      max_distance=self.relative_attention_max_distance,
    )
    values = self.relative_attention_bias(relative_position_bucket)
    values = values.permute(2, 0, 1).unsqueeze(0)
    return values

  def __call__(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
  ):
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
      assert len(past_key_value) == 2, "past_key_value should have 2 past states: keys and values"
      real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
      return states.reshape(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
      return states.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
      if key_value_states is None:
        hidden_states = shape(proj_layer(hidden_states))
      elif past_key_value is None:
        hidden_states = shape(proj_layer(key_value_states))

      if past_key_value is not None:
        if key_value_states is None:
          hidden_states = Tensor.cat([past_key_value, hidden_states], dim=2)
        elif past_key_value.shape[2] != key_value_states.shape[1]:
          hidden_states = shape(proj_layer(key_value_states))
        else:
          hidden_states = past_key_value
      return hidden_states

    query_states = shape(self.q(hidden_states))
    key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
    value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)

    scores = Tensor.matmul(query_states, key_states.transpose(2, 3))

    if position_bias is None:
      if not self.has_relative_attention_bias:
        position_bias = Tensor.zeros((1, self.n_heads, real_seq_length, key_length))
      else:
        position_bias = self.compute_bias(real_seq_length, key_length)

      if past_key_value is not None:
        position_bias = position_bias[:, :, -hidden_states.shape[1]:, :]

      if mask is not None:
        position_bias = position_bias + mask

    scores += position_bias
    attn_weights = scores.softmax().cast(scores.dtype)
    attn_weights = attn_weights.dropout(self.dropout)

    if layer_head_mask is not None:
      attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(Tensor.matmul(attn_weights, value_states))
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
      outputs = outputs + (attn_weights,)
    return outputs

class T5LayerSelfAttention:
  def __init__(self, config, has_relative_attention_bias=False):
    self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
    self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = lambda x: x.dropout(config.dropout_rate)

  def __call__(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
  ):
    normed_hidden_states = self.layer_norm(hidden_states)
    attention_output = self.SelfAttention(
      normed_hidden_states,
      mask=attention_mask,
      position_bias=position_bias,
      layer_head_mask=layer_head_mask,
      past_key_value=past_key_value,
      use_cache=use_cache,
      output_attentions=output_attentions,
    )
    hidden_states = hidden_states + self.dropout(attention_output[0])
    outputs = (hidden_states,) + attention_output[1:]
    return outputs

class T5LayerCrossAttention:
  def __init__(self, config):
    self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
    self.layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = lambda x: x.dropout(config.dropout_rate)

  def __call__(
    self,
    hidden_states,
    key_value_states,
    attention_mask=None,
    position_bias=None,
    layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    query_length=None,
    output_attentions=False,
  ):
    normed_hidden_states = self.layer_norm(hidden_states)
    attention_output = self.EncDecAttention(
      normed_hidden_states,
      mask=attention_mask,
      key_value_states=key_value_states,
      position_bias=position_bias,
      layer_head_mask=layer_head_mask,
      past_key_value=past_key_value,
      use_cache=use_cache,
      query_length=query_length,
      output_attentions=output_attentions,
    )
    layer_output = hidden_states + self.dropout(attention_output[0])
    outputs = (layer_output,) + attention_output[1:]
    return outputs


class T5Block:
  def __init__(self, config, has_relative_attention_bias=False):
    self.is_decoder = config.is_decoder
    self.layer = []
    self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
    if self.is_decoder:
      self.layer.append(T5LayerCrossAttention(config))
    self.layer.append(T5LayerFF(config))

  def __call__(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
    return_dict=True,
  ):
    if past_key_value is not None:
      if not self.is_decoder:
        print("Warning: `past_key_values` is passed to the encoder. Please make sure this is intended.")
      expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

      if len(past_key_value) != expected_num_past_key_values:
        raise ValueError(
          f"There should be {expected_num_past_key_values} past states. "
          f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
          f"Got {len(past_key_value)} past key / value states"
        )

      self_attn_past_key_value = past_key_value[:2]
      cross_attn_past_key_value = past_key_value[2:]
    else:
      self_attn_past_key_value, cross_attn_past_key_value = None, None

    self_attention_outputs = self.layer[0](
      hidden_states,
      attention_mask=attention_mask,
      position_bias=position_bias,
      layer_head_mask=layer_head_mask,
      past_key_value=self_attn_past_key_value,
      use_cache=use_cache,
      output_attentions=output_attentions,
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[2:]

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == dtypes.float16:
      clamp_value = Tensor.where(
        Tensor.isinf(hidden_states).any(),
        Tensor.full_like(hidden_states, Tensor.finfo(hidden_states.dtype).max - 1000),
        Tensor.full_like(hidden_states, Tensor.finfo(hidden_states.dtype).max)
      )
      hidden_states = Tensor.clip(hidden_states, -clamp_value, clamp_value)

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
      if present_key_value_state is not None:
        query_length = present_key_value_state[0].shape[2]
      else:
        query_length = None

      cross_attention_outputs = self.layer[1](
        hidden_states,
        key_value_states=encoder_hidden_states,
        attention_mask=encoder_attention_mask,
        position_bias=encoder_decoder_position_bias,
        layer_head_mask=cross_attn_layer_head_mask,
        past_key_value=cross_attn_past_key_value,
        query_length=query_length,
        use_cache=use_cache,
        output_attentions=output_attentions,
      )
      hidden_states = cross_attention_outputs[0]

      # clamp inf values to enable fp16 training
      if hidden_states.dtype == dtypes.float16:
        clamp_value = Tensor.where(
          Tensor.isinf(hidden_states).any(),
          Tensor.full_like(hidden_states, Tensor.finfo(hidden_states.dtype).max - 1000),
          Tensor.full_like(hidden_states, Tensor.finfo(hidden_states.dtype).max)
        )
        hidden_states = Tensor.clip(hidden_states, -clamp_value, clamp_value)

      if present_key_value_state is not None:
        present_key_value_state = present_key_value_state + cross_attention_outputs[1]

      attention_outputs = attention_outputs + cross_attention_outputs[2:]

    hidden_states = self.layer[-1](hidden_states)

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == dtypes.float16:
      clamp_value = Tensor.where(
        Tensor.isinf(hidden_states).any(),
        Tensor.full_like(hidden_states, Tensor.finfo(hidden_states.dtype).max - 1000),
        Tensor.full_like(hidden_states, Tensor.finfo(hidden_states.dtype).max)
      )
      hidden_states = Tensor.clip(hidden_states, -clamp_value, clamp_value)

    outputs = [hidden_states]

    if use_cache:
      outputs = outputs + [present_key_value_state] + attention_outputs
    else:
      outputs = outputs + attention_outputs

    return outputs


class T5Stack:
  def __init__(self, config, embed_tokens=None):
    self.config = config
    self.embed_tokens = embed_tokens
    self.is_decoder = config.is_decoder

    self.block = [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
    self.final_layer_norm = nn.RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = lambda x: x.dropout(config.dropout_rate)

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, new_embeddings):
    self.embed_tokens = new_embeddings

  def __call__(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    inputs_embeds=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
      err_msg_prefix = "decoder_" if self.is_decoder else ""
      raise ValueError(
        f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
      )
    if input_ids is not None:
      input_shape = input_ids.shape
      input_ids = input_ids.reshape(-1, input_shape[-1])
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.shape[:-1]
    else:
      err_msg_prefix = "decoder_" if self.is_decoder else ""
      raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
      if self.embed_tokens is None:
        raise ValueError("You have to initialize the model with valid token embeddings")
      inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

    if use_cache is True:
      if not self.is_decoder:
        raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

    if past_key_values is None:
      past_key_values = [None] * len(self.block)

    if attention_mask is None:
      attention_mask = Tensor.ones(batch_size, mask_seq_length)

    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

    if self.is_decoder and encoder_hidden_states is not None:
      encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
      encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
      if encoder_attention_mask is None:
        encoder_attention_mask = Tensor.ones(encoder_hidden_shape)
      encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
      encoder_extended_attention_mask = None

    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
      layer_head_mask = head_mask[i]
      cross_attn_layer_head_mask = cross_attn_head_mask[i]

      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_outputs = layer_module(
        hidden_states,
        attention_mask=extended_attention_mask,
        position_bias=position_bias,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        encoder_decoder_position_bias=encoder_decoder_position_bias,
        layer_head_mask=layer_head_mask,
        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
      )

      if use_cache is False:
        layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

      hidden_states, present_key_value_state = layer_outputs[:2]

      position_bias = layer_outputs[2]
      if self.is_decoder and encoder_hidden_states is not None:
        encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

      if use_cache:
        present_key_value_states = present_key_value_states + (present_key_value_state,)

      if output_attentions:
        all_attentions = all_attentions + (layer_outputs[3],)
        if self.is_decoder:
          all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(
        v
        for v in [
          hidden_states,
          present_key_value_states,
          all_hidden_states,
          all_attentions,
          all_cross_attentions,
        ]
        if v is not None
      )
    return {
      "last_hidden_state": hidden_states,
      "past_key_values": present_key_value_states,
      "hidden_states": all_hidden_states,
      "attentions": all_attentions,
      "cross_attentions": all_cross_attentions,
    }


class T5EncoderModel:
  def __init__(self, config):
    self.config = config
    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = T5Stack(encoder_config, self.shared)

  def __call__(
    self,
    input_ids: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
  ) -> Union[Tuple[Tensor], dict]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    encoder_outputs = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask,
      inputs_embeds=inputs_embeds,
      head_mask=head_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    if return_dict:
      return {
        "last_hidden_state": encoder_outputs[0],
        "hidden_states": encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        "attentions": encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      }
    return encoder_outputs


@dataclass
class t5_encoder_config:
  _name_or_path: str = "google/t5-v1_1-xxl"
  is_decoder: bool = False
  architectures: list[str] = ("T5ForConditionalGeneration",)
  classifier_dropout: float = 0.0
  d_ff: int = 10240
  d_kv: int = 64
  d_model: int = 4096
  decoder_start_token_id: int = 0
  dense_act_fn: str = "gelu_new"
  dropout_rate: float = 0.1
  eos_token_id: int = 1
  feed_forward_proj: str = "gated-gelu"
  initializer_factor: float = 1.0
  is_encoder_decoder: bool = True
  is_gated_act: bool = True
  layer_norm_epsilon: float = 1e-06
  model_type: str = "t5"
  num_decoder_layers: int = 24
  num_heads: int = 64
  num_layers: int = 24
  output_past: bool = True
  pad_token_id: int = 0
  relative_attention_max_distance: int = 128
  relative_attention_num_buckets: int = 32
  tie_word_embeddings: bool = False
  transformers_version: str = "4.44.2"
  use_cache: bool = True
  vocab_size: int = 32128

if __name__ == "__main__":
  t5_weights_url = 'https://huggingface.co/google/t5-v1_1-xxl/resolve/3db68a3ef122daf6e605701de53f766d671c19aa/model.safetensors'
  weights_fn = fetch(t5_weights_url)
  t5_state_dict = safe_load(weights_fn)
  t5_encoder_dict = {}
  for k, v in t5_state_dict.items():
    if k.startswith("encoder") or k == "shared.weight":
      t5_encoder_dict[k] = v
  # Using the same tensor for multiple embeds:
  t5_encoder_dict['encoder.embed_tokens.weight'] = t5_encoder_dict['shared.weight']

  t5 = T5EncoderModel(t5_encoder_config)
  load_state_dict(t5, t5_encoder_dict)
  # tiny_state_dict = get_state_dict(t5)
  # for k, v in tiny_state_dict.items():
  #   print(k, v.shape)
    # t5_keys = sorted(list(t5_encoder_dict.keys()))
  # tiny_keys = set(tiny_state_dict.keys())
  # print(len(t5_keys))
  # print(len(tiny_keys))
  # for k in t5_keys:
  #   if k not in tiny_keys:
  #     print(k)
  #   else:
  #     tiny_keys.remove(k)
  # print(tiny_keys)