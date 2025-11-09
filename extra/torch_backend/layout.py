from tinygrad import Tensor, UOp
from tinygrad.schedule.indexing import run_rangeify
from tinygrad.uop.ops import AxisType

from tinygrad.uop.ops import PatternMatcher, UPat, UOp, Ops, graph_rewrite

from tinygrad.uop.ops import AxisType, Ops, UOp, PatternMatcher, UPat, graph_rewrite

from tinygrad.uop.ops import AxisType, Ops, UOp


def affine_layout(expr: UOp, ndim: int):
  loops = sorted(
    [rng for rng in expr.ranges if rng.op is Ops.RANGE and rng.arg[1] is AxisType.LOOP],
    key=lambda r: r.arg[0],
  )
  if len(loops) < ndim:
    raise RuntimeError(f"only {len(loops)} loop vars for {ndim} axes (probably realized or fancy indexed)")

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
      raise RuntimeError(f"non-affine access along {name}")
    strides.append(delta)

  return tuple(strides), base

def get_tensor_layout(t: Tensor):
  # Build a fake consumer that forces the reshape to be iterated.
  dummy = Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device)
  consumer = dummy.assign(t)

  _, rctx = run_rangeify(UOp.sink(consumer.uop))
  base = t.uop.base
  if base not in rctx.range_map:
    raise RuntimeError(f"no range info for {base.op}")

  ranges, _ = rctx.range_map[base]
  print(ranges)
  expr = ranges[0]          # the pointer expression for the storage axis
  try:
    strides, offset = affine_layout(expr, len(t.shape))
  except RuntimeError:
    return None
  print(strides, offset)
  return (strides, offset)


if __name__ == "__main__":
    a = Tensor.arange(12)
    # b = a.reshape((3, 4))
    # get_tensor_layout(b)
    # b.numpy()
    # c = a[1:]
    # get_tensor_layout(c)
    # c.numpy()
    d = a.repeat((2, 2))
    out = get_tensor_layout(d)
    # get_tensor_layout(d)
    d.numpy()
