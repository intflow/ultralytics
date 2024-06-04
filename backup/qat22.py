import torch
import yaml
import collections
from tqdm import tqdm
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import os
################################################################################
import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F  # 이 줄을 추가하여 F.interpolate 사용 에러 해결
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
import numpy as np
from torch.nn.utils.rnn import pad_sequence
# Pytorch Quantization
import pytorch_quantization

from pytorch_quantization import nn as quant_nn
from torch.quantization import QuantStub, DeQuantStub
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from absl import logging as quant_logging

quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
from pytorch_quantization import quant_modules
quant_modules.initialize()
from ultralytics import YOLO

model = YOLO("yolov8m.yaml") 

class QuantAdd(torch.nn.Module):
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            print(f"QAdd {self._input0_quantizer}  {self._input1_quantizer}")
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y

class QuantC2fChunk(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c
    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)

class QuantConcat(torch.nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim) 

class QuantUpsample(torch.nn.Module): 
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        
    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)
    
def bottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def concat_quant_forward(self, x):
    if hasattr(self, "concatop"):
        return self.concatop(x, self.d)
    return torch.cat(x, self.d)

def upsample_quant_forward(self, x):
    if hasattr(self, "upsampleop"):
        return self.upsampleop(x)
    return F.interpolate(x)

def c2f_qaunt_forward(self, x):
    if hasattr(self, "c2fchunkop"):
        y = list(self.c2fchunkop(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        
    else:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    for name, module in model.named_modules():
    if module.__class__.__name__ == "C2f":
        if not hasattr(module, "c2fchunkop"):
            print(f"Add C2fQuantChunk to {name}")
            module.c2fchunkop = QuantC2fChunk(module.c)
        module.__class__.forward = c2f_qaunt_forward

    if module.__class__.__name__ == "Bottleneck":
        if module.add:
            if not hasattr(module, "addop"):
                print(f"Add QuantAdd to {name}")
                module.addop = QuantAdd(module.add)
            module.__class__.forward = bottleneck_quant_forward
            
    if module.__class__.__name__ == "Concat":
        if not hasattr(module, "concatop"):
            print(f"Add QuantConcat to {name}")
            module.concatop = QuantConcat(module.d)
        module.__class__.forward = concat_quant_forward

    if module.__class__.__name__ == "Upsample":
        if not hasattr(module, "upsampleop"):
            print(f"Add QuantUpsample to {name}")
            module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
        module.__class__.forward = upsample_quant_forward
