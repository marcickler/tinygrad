from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UPat
from tinygrad.uop.symbolic import split_uop

a = Tensor([11])
b = a.sign() * a * a.sign() * a
c = a.abs()**2
d = a.abs() * a.abs()
print(b.tolist())
print(c.tolist())
print(d.tolist())


