from tinygrad import Tensor
a = Tensor([11.])
b = Tensor([12.])
c = b * a / b
d = a * b / b
print(c.tolist())
print(d.tolist())
