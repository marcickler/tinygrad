from tinygrad import Tensor, dtypes
from tinygrad import nn
from tinygrad.nn.state import load_state_dict


class FakeModel:
    def __init__(self):
        self.a = nn.Linear(128, 128)

fake_state_dict = {"a.weight": Tensor.empty((128, 128), dtype=dtypes.float32).realize(),
                   "a.bias": Tensor.empty(128, dtype=dtypes.float32).realize()}

model = FakeModel()
_ = load_state_dict(model, fake_state_dict)