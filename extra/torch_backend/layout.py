from __future__ import annotations

from collections.abc import Callable

from tinygrad import Tensor, UOp
from tinygrad.schedule.indexing import run_rangeify
from tinygrad.uop.ops import AxisType, Ops, UOp


MOVEMENT_OPS = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK, Ops.FLIP}


def affine_layout(expr: UOp, ndim: int):
  loops = sorted(
    [rng for rng in expr.ranges if rng.op is Ops.RANGE and rng.arg[1] is AxisType.LOOP],
    key=lambda r: r.arg[0],
  )
  if len(loops) < ndim:
    return None, None

  mapping: dict[UOp, UOp] = {}
  loop_vars: list[UOp] = []
  for rng in loops[:ndim]:
    extent = int(rng.src[0].vmax) + 1  # RANGE runs 0…extent-1
    var = UOp.variable(f"r{len(loop_vars)}", 0, extent - 1)
    mapping[rng] = var
    loop_vars.append(var)

  expr_vars = expr.substitute(mapping)
  zero = {var.arg[0]: 0 for var in loop_vars}
  base = expr_vars.sym_infer(zero)

  strides = []
  for var in loop_vars:
    name = var.arg[0]
    one = expr_vars.sym_infer({**zero, name: 1})
    two = expr_vars.sym_infer({**zero, name: 2})
    delta = one - base
    if two - base != 2 * delta:
      return None, None
    strides.append(delta)

  return tuple(strides), base

def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
  stride = 1
  out: list[int] = []
  for dim in reversed(shape):
    out.append(stride)
    stride *= dim
  return tuple(reversed(out))


def linear_to_multi(index: int, shape: tuple[int, ...]) -> list[int]:
  if not shape:
    return []
  strides = contiguous_strides(shape)
  remaining = index
  multi: list[int] = []
  for stride, dim in zip(strides, shape):
    if dim == 0:
      multi.append(0)
      continue
    val = remaining // stride
    multi.append(int(val))
    remaining -= int(val) * stride
  return multi


def _get_tensor_layout(t: Tensor, seen: set[int]) -> tuple[Tensor, tuple[tuple[int, ...], int]]:
  uop_id = id(t.uop)
  if uop_id in seen:
    raise RuntimeError("cycle detected while resolving tensor layout")
  seen.add(uop_id)

  chain: list[dict[str, object]] = []
  cur = t.uop
  while cur.op in MOVEMENT_OPS:
    src_uop = cur.src[0]
    in_shape = tuple(int(s) for s in src_uop.shape)
    out_shape = tuple(int(s) for s in cur.shape)
    entry: dict[str, object] = {"op": cur.op, "in_shape": in_shape, "out_shape": out_shape}
    if cur.op is Ops.RESHAPE:
      entry["reshape"] = tuple(int(s) for s in cur.marg)
    elif cur.op is Ops.PERMUTE:
      entry["permute"] = tuple(int(ax) for ax in cur.marg)
    elif cur.op is Ops.EXPAND:
      entry["expand"] = tuple(int(s) for s in cur.marg)
    elif cur.op is Ops.SHRINK:
      entry["shrink"] = tuple((int(b), int(e)) for b, e in cur.marg)
    elif cur.op is Ops.FLIP:
      entry["flip"] = tuple(bool(x) for x in cur.marg)
    chain.append(entry)
    cur = src_uop

  if chain:
    base_tensor = Tensor(cur)
    realized_base, (base_strides, base_offset) = _get_tensor_layout(base_tensor, seen)

    def compute_offset(idx: tuple[int, ...]) -> int:
      cur_idx = list(idx)
      for info in chain:
        op = info["op"]
        in_shape = info["in_shape"]  # type: ignore[index]
        out_shape = info["out_shape"]  # type: ignore[index]
        if op is Ops.FLIP:
          flip_mask = info["flip"]  # type: ignore[index]
          cur_idx = [(in_shape[i]-1 - cur_idx[i]) if flip_mask[i] else cur_idx[i] for i in range(len(in_shape))]
          continue
        if op is Ops.SHRINK:
          shrink = info["shrink"]  # type: ignore[index]
          cur_idx = [val + shrink[i][0] for i, val in enumerate(cur_idx)]
          continue
        if op is Ops.EXPAND:
          expand_in = in_shape
          expand_out = out_shape
          pad = len(expand_out) - len(expand_in)
          aligned_in = ((1,) * pad) + expand_in if pad > 0 else expand_in
          aligned_idx: list[int] = []
          for out_val, in_dim in zip(cur_idx, aligned_in):
            aligned_idx.append(0 if in_dim == 1 else out_val)
          cur_idx = aligned_idx[pad:] if pad > 0 else aligned_idx
          continue
        if op is Ops.PERMUTE:
          perm = info["permute"]  # type: ignore[index]
          next_idx = [0] * len(perm)
          for out_axis, in_axis in enumerate(perm):
            next_idx[in_axis] = cur_idx[out_axis]
          cur_idx = next_idx
          continue
        if op is Ops.RESHAPE:
          in_shape_rs = in_shape
          out_shape_rs = out_shape
          out_strides_rs = contiguous_strides(out_shape_rs)
          lin = sum(val * stride for val, stride in zip(cur_idx, out_strides_rs))
          cur_idx = linear_to_multi(lin, in_shape_rs)
          continue
        raise RuntimeError(f"unsupported movement op {op}")
      if len(cur_idx) != len(base_strides):
        raise RuntimeError("base stride rank mismatch")
      return int(base_offset) + sum(int(val) * int(stride) for val, stride in zip(cur_idx, base_strides))

    final_shape = tuple(int(s) for s in t.shape)
    if final_shape:
      zero_offset = compute_offset((0,) * len(final_shape))
    else:
      zero_offset = int(base_offset)
    final_strides: list[int] = []
    for axis in range(len(final_shape)):
      test_index = [0] * len(final_shape)
      test_index[axis] = 1
      stride_val = compute_offset(tuple(test_index)) - zero_offset
      final_strides.append(int(stride_val))

    view_tensor = realized_base
    for info in reversed(chain):
      op = info["op"]
      if op is Ops.RESHAPE:
        view_tensor = view_tensor.reshape(info["out_shape"])  # type: ignore[index]
      elif op is Ops.PERMUTE:
        view_tensor = view_tensor.permute(info["permute"])  # type: ignore[index]
      elif op is Ops.EXPAND:
        view_tensor = view_tensor.expand(info["out_shape"])  # type: ignore[index]
      elif op is Ops.SHRINK:
        view_tensor = view_tensor.shrink(info["shrink"])  # type: ignore[index]
      elif op is Ops.FLIP:
        view_tensor = view_tensor.flip(info["flip"])  # type: ignore[index]

    return view_tensor, (tuple(final_strides), int(zero_offset))

  dummy = Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device)
  consumer = dummy.assign(t)

  _, rctx = run_rangeify(UOp.sink(consumer.uop))
  base = t.uop.base
  if base not in rctx.range_map:
    raise RuntimeError(f"no range info for {base.op}")

  ranges, _ = rctx.range_map[base]
  expr = ranges[0]
  strides, offset = affine_layout(expr, len(t.shape))
  if strides is None:
    t = t.contiguous()
    return t, (contiguous_strides(t.shape), 0)

  return t, (strides, offset)


def get_tensor_layout(t: Tensor) -> tuple[Tensor, tuple[tuple[int, ...], int]]:
  return _get_tensor_layout(t, set())


if __name__ == "__main__":
  try:
    import torch
  except ImportError as exc:
    raise SystemExit("PyTorch is required to run this test harness.") from exc

  def tg_arange(numel: int) -> Tensor:
    return Tensor.arange(numel)

  def th_arange(numel: int):
    return torch.arange(numel, dtype=torch.float32)

  cases: list[tuple[str, Callable[[], Tensor], Callable[[], object]]] = [
    ("flat_12", lambda: tg_arange(12), lambda: th_arange(12)),
    ("reshape_3x4", lambda: tg_arange(12).reshape((3, 4)), lambda: th_arange(12).reshape(3, 4)),
    ("transpose_3x4", lambda: tg_arange(12).reshape((3, 4)).permute(1, 0), lambda: th_arange(12).reshape(3, 4).permute(1, 0)),
    ("reshape_2x3x4", lambda: tg_arange(24).reshape((2, 3, 4)), lambda: th_arange(24).reshape(2, 3, 4)),
    ("permute_3d", lambda: tg_arange(24).reshape((2, 3, 4)).permute(1, 2, 0), lambda: th_arange(24).reshape(2, 3, 4).permute(1, 2, 0)),
    ("slice_rows", lambda: tg_arange(20).reshape((5, 4))[1:], lambda: th_arange(20).reshape(5, 4)[1:]),
    ("slice_strided", lambda: tg_arange(60).reshape((3, 4, 5))[:, 1:, ::2], lambda: th_arange(60).reshape(3, 4, 5)[:, 1:, ::2]),
    ("expand_broadcast", lambda: tg_arange(12).reshape((1, 3, 4)).expand(2, 3, 4), lambda: th_arange(12).reshape(1, 3, 4).expand(2, 3, 4)),
    ("permute_then_slice", lambda: tg_arange(48).reshape((2, 4, 6)).permute(2, 0, 1)[:, :, 1:], lambda: th_arange(48).reshape(2, 4, 6).permute(2, 0, 1)[:, :, 1:]),
    # tinygrad's repeat is implemented via expand, so compare against PyTorch expand semantics here.
    ("repeat_tile", lambda: tg_arange(6).reshape((1, 3, 2)).repeat(2, 1, 1), lambda: th_arange(6).reshape(1, 3, 2).expand(2, 3, 2)),
    ("unsqueeze_middle", lambda: tg_arange(12).reshape((3, 4)).unsqueeze(1), lambda: th_arange(12).reshape(3, 4).unsqueeze(1)),
  ]

  failures: list[str] = []
  for name, tiny_builder, torch_builder in cases:
    tiny_tensor = tiny_builder()
    torch_tensor = torch_builder()
    tiny_shape = tuple(tiny_tensor.shape)
    torch_shape = tuple(torch_tensor.shape)
    if tiny_shape != torch_shape:
      failures.append(f"{name}: shape mismatch tinygrad {tiny_shape} vs torch {torch_shape}")
      continue
    realized, (tiny_strides, tiny_offset) = get_tensor_layout(tiny_tensor)
    torch_strides = tuple(int(s) for s in torch_tensor.stride())
    torch_offset = int(torch_tensor.storage_offset())
    if tuple(tiny_strides) != torch_strides or int(tiny_offset) != torch_offset:
      failures.append(
        f"{name}: tinygrad (strides={tiny_strides}, offset={int(tiny_offset)}) vs torch (strides={torch_strides}, offset={torch_offset})"
      )
    realized_shape = tuple(realized.shape)
    if realized_shape != torch_shape:
      failures.append(f"{name}: realized tinygrad shape {realized_shape} differs from torch {torch_shape}")

  if failures:
    raise AssertionError("layout mismatches detected:\n" + "\n".join(failures))

  print(f"all {len(cases)} layout checks passed")
