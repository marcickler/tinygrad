from tinygrad import Tensor
x = Tensor([11])
print((x.abs()**2).tolist())
