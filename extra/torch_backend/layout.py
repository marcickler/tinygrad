from tinygrad import Tensor, UOp
from tinygrad.schedule.indexing import run_rangeify
from tinygrad.uop.ops import AxisType

def get_tensor_layout(t: Tensor):
  # Build a fake consumer that forces the reshape to be iterated.
  dummy = Tensor.zeros(*t.shape, dtype=t.dtype, device=t.device)
  consumer = dummy.assign(t)

  _, rctx = run_rangeify(UOp.sink(consumer.uop))
  base = t.uop.base
  if base not in rctx.range_map:
    raise RuntimeError(f"no range info for {base.op}")

  ranges, _ = rctx.range_map[base]
  for r in ranges:
    print(r)



if __name__ == "__main__":
    a = Tensor.arange(12)
    b = a.reshape((3, 4))
    rmap = get_tensor_layout(b)
    c = b.numpy()
