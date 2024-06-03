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
from ultralytics import YOLO

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
    

print("4. Load Q/DQ model with calibration values for QAT")  
model=YOLO('yolov8m_qdq_cal.pt')
model.train(data="coco8.yaml", cfg="qat.yaml", epochs=20, batch=30, device=[0, 1, 2])
## Get 'yolov8m_qat.pt' manually from runs/train#/weights/last.pt


print("5. Convert .pt to .onnx")  
model=YOLO('yolov8m_qat.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

quant_modules.initialize()
quant_nn.TensorQuantizer.use_fb_fake_quant = True
dummy_input = torch.randn(1, 3, 640, 640, device='cuda')

input_names = [ "actual_input_1" ]
output_names = [ "output1" ]
torch.onnx.export( model.model, dummy_input, "yolov8m_qat.onnx",  opset_version=13, verbose=True)

quant_nn.TensorQuantizer.use_fb_fake_quant = False
print("sucessfully export yolov8  onnx!")
     # enable_onnx_checker needs to be disabled. See notes below.

# Load the exported ONNX model
onnx_model = YOLO('yolov8m_qat.onnx')

# Run inference
results = onnx_model('https://ultralytics.com/images/bus.jpg')
