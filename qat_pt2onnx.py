import torch
import torch.nn.functional as F  # 이 줄을 추가하여 F.interpolate 사용 에러 해결
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
import collections
from tqdm import tqdm
import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib, quant_modules, tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor
from ultralytics import YOLO
import numpy as np
import sys
import os
from typing import List, Callable, Union, Dict
from copy import deepcopy
import onnx
import onnx_graphsurgeon as gs
from torch.nn.utils.rnn import pad_sequence
from absl import logging as quant_logging
from torchvision.datasets import CocoDetection

class QuantAdd(torch.nn.Module):
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
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
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())

    def forward(self, x):
        if self.size is None and self.scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        return F.interpolate(self._input_quantizer(x), size=self.size, scale_factor=self.scale_factor, mode=self.mode)

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
    if self.size is not None or self.scale_factor is not None:
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
    else:
        raise ValueError("either size or scale_factor should be defined")


def c2f_qaunt_forward(self, x):
    if hasattr(self, "c2fchunkop"):
        y = list(self.c2fchunkop(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    else:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def cal_model(model, data_loader, device, num_batch=1024, model_save_path='cal.pth'):
    num_batch = num_batch
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    print(f"Calibrator type for {name}: {type(module._calibrator)}")

                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
                    if torch.isnan(module._amax).any():
                        print(f"Warning: amax for {name} is NaN")
                print(f"{name:40}: {module}")
        model.to(device)

    def collect_stats(model, data_loader, device, num_batch=1024):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        ##model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                    print(f"Enabled calibrator for {name}")
                else:
                    module.disable()
                    print(f"Disabled quantizer for {name}")

        # Feed data to the network for collecting stats
        for i, datas in tqdm(enumerate(data_loader),  desc="Collect stats for calibrating"):
            if datas == None:
                continue
            imgs = datas[0].to(device, non_blocking=True)
            print(f"Feeding batch {i+1}/{num_batch} with shape {imgs.shape} to the model")
            model(imgs)
            if i >= num_batch:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_calib()
                    module.enable_quant()
                    print(f"Disabled calibrator and enabled quantizer for {name}")
                else:
                    module.enable()

    with torch.no_grad():
        collect_stats(model, data_loader, device, num_batch)
        compute_amax(model, method="mse")


    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
            print(f"Calibrator type for {name}: {type(module._calibrator)}")
            try:
                print(f"Calibrator min value for {name}: {module._calibrator.min_val}")
                print(f"Calibrator max value for {name}: {module._calibrator.max_val}")
            except AttributeError:
                print(f"Calibrator for {name} does not have min_val or max_val attributes")
    
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CustomCocoDataset, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        img, target = super(CustomCocoDataset, self).__getitem__(index)
        if self.transform:
            img = self.transform(img)
        boxes = torch.tensor([t['bbox'] for t in target], dtype=torch.float32)
        labels = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        return img, target

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    ##transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn(batch):
    filtered_batch = [item for item in batch if item[1]['boxes'].shape[0] > 0]
    if len(filtered_batch) == 0:
        return None
    images = [item[0] for item in filtered_batch]
    targets = [item[1] for item in filtered_batch]
    images = torch.stack(images, dim=0)
    boxes = [t['boxes'].clone().detach().float() for t in targets]
    labels = [t['labels'].clone().detach().long() for t in targets]
    boxes_padded = pad_sequence(boxes, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    targets_padded = [{'boxes': b, 'labels': l} for b, l in zip(boxes_padded, labels_padded)]
    return images, targets_padded

def graphsurgeon_model(input_model_path, output_model_path):
    onnx_model = onnx.load(input_model_path)
    graph = gs.import_onnx(onnx_model)
    nodes = graph.nodes
    mul_nodes = [node for node in nodes if node.op == "Mul" and node.i(0).op == "BatchNormalization" and node.i(1).op == "Sigmoid"]
    many_outputs_mul_nodes = []
    for node in mul_nodes:
        try:
            for i in range(99):
                node.o(i)
        except:
            if i > 1:
                mul_nodename_outnum = {"node": node, "out_num": i}
                many_outputs_mul_nodes.append(mul_nodename_outnum)

    for node_dict in many_outputs_mul_nodes:
        if node_dict["out_num"] == 2:
            if node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "QuantizeLinear":
                if node_dict["node"].o(1).o(0).o(0).op == "Concat":
                    concat_dq_out_name = node_dict["node"].o(1).o(0).outputs[0].name
                    for i, concat_input in enumerate(node_dict["node"].o(1).o(0).o(0).inputs):
                        if concat_input.name == concat_dq_out_name:
                            node_dict["node"].o(1).o(0).o(0).inputs[i] = node_dict["node"].o(0).o(0).outputs[0]
                else:
                    node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            elif node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "Concat":
                concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
                for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
                    if concat_input.name == concat_dq_out_name:
                        node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0]
        elif node_dict["out_num"] == 3:
            node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
        elif node_dict["out_num"] == 4:
            node_dict["node"].o(3).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]

    add_nodes = [node for node in nodes if node.op == "Add"]
    many_outputs_add_nodes = []
    for node in add_nodes:
        try:
            for i in range(99):
                node.o(i)
        except:
            if i > 1 and node.o().op == "QuantizeLinear":
                add_nodename_outnum = {"node": node, "out_num": i}
                many_outputs_add_nodes.append(add_nodename_outnum)

    for node_dict in many_outputs_add_nodes:
        if node_dict["node"].outputs[0].outputs[0].op == "QuantizeLinear" and node_dict["node"].outputs[0].outputs[1].op == "Concat":
            concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
            for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
                if concat_input.name == concat_dq_out_name:
                    node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0]

    modified_onnx_model = gs.export_onnx(graph)
    onnx.save(modified_onnx_model, output_model_path)

def modify_onnx_model(onnx_file_in_path, onnx_file_out_path):
    onnx_model = onnx.load(onnx_file_in_path)
    graph = onnx_model.graph
    for node in graph.node:
        if node.op_type == "Mul":
            pass
    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]
    if conv_nodes:
        last_conv = conv_nodes[-1]
    onnx.save(onnx_model, onnx_file_out_path)
    print("Modified ONNX model saved at:", onnx_file_out_path)
    onnx_model = YOLO(onnx_file_out_path)
    results = onnx_model('https://ultralytics.com/images/bus.jpg')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    ###############################################################################################################
    ###############################################################################################################
    print("1. Load target FP32 model")
    model = YOLO('yolov8m.pt')


    ###############################################################################################################
    ###############################################################################################################
    print("2. Add Q/DQ Layers to the YOLO model")
    for name, module in model.named_modules():
        if module.__class__.__name__ == "C2f":
            if not hasattr(module, "c2fchunkop"):
                module.c2fchunkop = QuantC2fChunk(module.c)
            module.__class__.forward = c2f_qaunt_forward
        if module.__class__.__name__ == "Bottleneck":
            if module.add:
                if not hasattr(module, "addop"):
                    module.addop = QuantAdd(module.add)
                module.__class__.forward = bottleneck_quant_forward
        if module.__class__.__name__ == "Concat":
            if not hasattr(module, "concatop"):
                module.concatop = QuantConcat(module.d)
            module.__class__.forward = concat_quant_forward
        if module.__class__.__name__ == "Upsample":
            if not hasattr(module, "upsampleop"):
                module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.__class__.forward = upsample_quant_forward

    ###############################################################################################################
    ###############################################################################################################
    print("3. Get Calibration values for the Q/DQ model")
    batch_size = 128
    train_data_dir = '/DL_data_super_hdd/coco_dataset/data/images/train2017'
    train_ann_file = '/DL_data_super_hdd/coco_dataset/annotations/instances_train2017.json'
    #val_data_dir = '/DL_data_super_hdd/coco_dataset/data/images/val2017/'
    #val_ann_file = '/DL_data_super_hdd/coco_dataset/annotations/instances_val2017.json'

    model.to(device)

    # Dataset and DataLoader setup
    dataset = CustomCocoDataset(root=train_data_dir, annFile=train_ann_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn, pin_memory=True)
    #dataset_test = CustomCocoDataset(root=val_data_dir, annFile=val_ann_file, transform=transform)
    #data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,collate_fn=collate_fn, pin_memory=True)

    cal_model(model, data_loader, device, num_batch=8, model_save_path='yolov8m_qdq_cal.pt')
    '''

    ###############################################################################################################
    ###############################################################################################################
    print("4. Load Q/DQ model with calibration values for QAT")
    model = YOLO('yolov8m_qdq_cal.pt')
    model.train(data="coco.yaml", cfg="qat.yaml", epochs=20, batch=32)
    

    ###############################################################################################################
    ###############################################################################################################
    print("5. Convert .pt to .onnx")
    model = YOLO('runs/detect/train17/weights/last.pt')
    model.to(device)
    quant_modules.initialize()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    dummy_input = torch.randn(1, 3, 640, 640, device='cuda')
    with pytorch_quantization.enable_onnx_export():
        torch.onnx.export(model.model, dummy_input, "yolov8m_qat.onnx", verbose=True, opset_version=13)
    quant_nn.TensorQuantizer.use_fb_fake_quant = False
    print("Successfully exported YOLOv8 ONNX!")

    input_model_path = "yolov8m_qat.onnx"
    opt1_model_path = "yolov8m_qat_opt.onnx"
    opt2_model_path = "yolov8m_qat_opt2.onnx"


    ###############################################################################################################
    ###############################################################################################################
    print("6. Remove redundant Q/DQ layer")
    graphsurgeon_model(input_model_path, opt1_model_path)
    print("Optimized ONNX model saved at:", opt1_model_path)


    ###############################################################################################################
    ###############################################################################################################
    print("7. Remove Conv Q/DQ in DFL block")
    modify_onnx_model(opt1_model_path, opt2_model_path)

if __name__ == "__main__":
    main()
