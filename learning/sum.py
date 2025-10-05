import torch
import numpy as np
from tinygrad import Tensor

N = 2048
torch_dt = torch.float32
torch_device = torch.accelerator.current_accelerator()

torch.manual_seed(0)
torch_a = (torch.rand(N, N, dtype=torch_dt) - 0.5).to(torch_device)
sum_torch = torch_a.sum()

tiny_a = Tensor(torch_a.cpu().numpy())
sum_tiny = tiny_a.sum()

atol, rtol = (1e-2, 1e-2) if torch_dt == torch.float16 else (1e-3, 1e-3)
np.testing.assert_allclose(sum_torch.numpy(force=True), sum_tiny.numpy(), atol=atol, rtol=rtol)
