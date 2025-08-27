import tempfile
from pathlib import Path
import argparse
from collections import namedtuple
from typing import Dict, Any

from PIL import Image
import numpy as np
from tinygrad import Device, GlobalCounters, dtypes, Tensor, TinyJit
from tinygrad.helpers import Timing, Context, getenv, fetch, colored, tqdm
from tinygrad.nn import Conv2d, GroupNorm
from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
from extra.models.clip import Closed, Tokenizer
from extra.models.unet import UNetModel
from extra.bench_log import BenchEvent, WallTimeEvent