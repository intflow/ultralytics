from ultralytics import YOLO

import pytorch_quantization
import torch
import yaml
import collections
from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
import numpy as np
from torch import nn
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
import pytorch_quantization
from torch.nn.utils.rnn import pad_sequence
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
    
    
    


    
model = YOLO("yolov8m.yaml") 
print(model)


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
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
class CocoDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # PIL 이미지를 PyTorch 텐서로 변환
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 일반적인 이미지넷 정규화
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = self.transform(image)  # 변환 적용
        
        # 여기서는 annotation을 dictionary로 반환하나, 실제 사용 시 필요에 따라 변환할 수 있습니다.
        target = {ann['category'] for ann in coco_annotations}
        return image, target

    def __len__(self):
        return len(self.ids)

# # 데이터셋과 데이터 로더 설정
# transform = transforms.Compose([
#     transforms.Resize((640, 640)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def collate_fn(batch):
#     filtered_batch = [item for item in batch if item[1]['boxes'].shape[0] > 0]
    
#     # 필터링된 배치에서 이미지와 타겟 분리
#     images = [item[0] for item in filtered_batch]
#     targets = [item[1] for item in filtered_batch]

#     # 이미지 텐서 스택
#     images = torch.stack(images, dim=0)

#     # targets에서 'boxes'와 'labels' 추출 및 텐서 변환
#     # boxes = [torch.tensor(t['boxes'], dtype=torch.float32) for t in targets]
#     # labels = [torch.tensor(t['labels'], dtype=torch.int64) for t in targets]
#     boxes = [t['boxes'].clone().detach().float() for t in targets]
#     labels = [t['labels'].clone().detach().long() for t in targets]
#     # pad_sequence로 모든 boxes와 labels를 동일한 길이로 패딩
#     boxes_padded = pad_sequence(boxes, batch_first=True, padding_value=0)
#     labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)  # 레이블 패딩 값으로 -1을 사용할 수 있습니다.

#     # 패딩된 boxes와 labels를 사용하여 targets 구조 재구성
#     targets_padded = [{'boxes': b, 'labels': l} for b, l in zip(boxes_padded, labels_padded)]

#     return images, targets_padded


data_path = "/usr/src/datasets/coco/labels"
batch_size = 16
train_data_dir = '/DL_data_super_hdd/coco_dataset/data/images/train2017'
val_data_dir = '/DL_data_super_hdd/coco_dataset/data/images/val2017/'
train_ann_file = '/DL_data_super_hdd/coco_dataset/annotations/instances_train2017.json'
val_ann_file = '/DL_data_super_hdd/coco_dataset/annotations/instances_val2017.json'



# Transform은 필요에 따라 정의합니다.
transform = None  # 예를 들어, torchvision.transforms.Compose([...]) 사용 가능

# Dataset 및 DataLoader 인스턴스 생성
coco_dataset = CocoDataset(
    root='/DL_data_super_hdd/coco_dataset/data/images/train2017',
    annotation='/DL_data_super_hdd/coco_dataset/annotations/instances_train2017.json',
    transform=transform
)
coco_dataloader = DataLoader(coco_dataset, batch_size=4, shuffle=True)
        

def cal_model(model, data_loader, device, num_batch=1024):
    num_batch = num_batch
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
    def collect_stats(model, data_loader, num_batches):
        """Feed data to the network and collect statistic"""
        # model.eval()
        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                imgs = datas[0].to(device, non_blocking=True).float() / 255.0
                # imgs = datas['img'].to(device, non_blocking=True).float() / 255.0
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
    with torch.no_grad():
        collect_stats(model, data_loader,100)
        compute_amax(model, method="mse", percentile=99.99)
    model.train(data="coco.yaml", epochs=3)  # train the model
    # quant_nn.TensorQuantizer.use_fb_fake_quant = True
    # quant_modules.initialize()
    # dummy_input = torch.randn(1, 3, 640, 640, device='cuda')

    # input_names = [ "actual_input_1" ]
    # output_names = [ "output1" ]

    # with pytorch_quantization.enable_onnx_export():
    #     # enable_onnx_checker needs to be disabled. See notes below.
    #     torch.onnx.export(
    #         model, dummy_input, "quant_resnet50.onnx", verbose=True, opset_version=13)
        
        
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_modules.initialize()
quant_nn.TensorQuantizer.use_fb_fake_quant = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



print("cal_model")
# 캘리브레이션 함수 실행
cal_model(model, coco_dataloader, device)
# model.train(data="coco.yaml", epochs=3)  # train the model
# print(model)
# # metrics = model.val()  # evaluate model performance on the validation set
# # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx") 