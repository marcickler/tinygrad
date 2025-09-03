from tinygrad import Tensor
a = Tensor([11])
b = a.sign() * a.sign() * a * a
print(b.tolist())