import onnx
import onnx_graphsurgeon as gs
from ultralytics import YOLO

def graphsurgeon_model(input_model_path, output_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(input_model_path)

    # Import the ONNX model to onnx_graphsurgeon
    graph = gs.import_onnx(onnx_model)

    # Process nodes to find specific patterns and modify the graph
    nodes = graph.nodes
    mul_nodes = [node for node in nodes if node.op == "Mul" and node.i(0).op == "BatchNormalization" and node.i(1).op == "Sigmoid"]
    many_outputs_mul_nodes = []
    for node in mul_nodes: # convolution mul node for silu activation.
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
                            node_dict["node"].o(1).o(0).o(0).inputs[i] = node_dict["node"].o(0).o(0).outputs[0] # concat 4개
                else:
                    node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0] # 그 외
            elif node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "Concat":
                concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
                for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
                    if concat_input.name == concat_dq_out_name:
                        node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0] # concat 4개
        elif node_dict["out_num"] == 3:
            node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
        elif node_dict["out_num"] == 4: # shape node not merged
            node_dict["node"].o(3).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]

    add_nodes = [node for node in nodes if node.op == "Add"]
    many_outputs_add_nodes = []
    for node in add_nodes: # convolution mul node for silu activation.
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
                    node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0] # concat 4개

    # Export the modified graph back to ONNX
    modified_onnx_model = gs.export_onnx(graph)

    # Save the modified ONNX model to the specified output path
    onnx.save(modified_onnx_model, output_model_path)

def modify_onnx_model(onnx_file_in_path, onnx_file_out_path):
    # ONNX 모델 불러오기
    onnx_model = onnx.load(onnx_file_in_path)
    graph = onnx_model.graph
    
    # 그래프 노드 조작
    for node in graph.node:
        if node.op_type == "Mul":
            # 여기서 특정 조건을 만족하는 노드를 찾아 수정
            pass
    # Convolution 노드 찾기
    conv_nodes = [node for node in graph.node if node.op_type == "Conv"]

    # 마지막 Convolution 노드의 입력을 수정
    if conv_nodes:
        last_conv = conv_nodes[-1]
        # 입력 노드 변경 예시 (구체적인 인덱스는 모델을 분석하여 결정해야 함)
        last_conv.input[0] = last_conv.input[0]  # 여기에 적절한 입력 노드 ID를 할당
        last_conv.input[1] = last_conv.input[1]  # 여기에 적절한 가중치 노드 ID를 할당
    
    # 수정된 ONNX 모델 저장
    onnx.save(onnx_model, onnx_file_out_path)
    print("Modified ONNX model saved at:", onnx_file_out_path)
    onnx_model = YOLO(onnx_file_out_path)
    results = onnx_model('https://ultralytics.com/images/bus.jpg')
    print(results)


# Example usage
input_model_path = "yolov8m_qat.onnx"
opt1_model_path = "yolov8m_qat_opt.onnx"
opt2_model_path = "yolov8m_qat_opt2.onnx"

print("6. Remove redundant Q/DQ layer")  
graphsurgeon_model(input_model_path, opt1_model_path)
print("Optimized ONNX model saved at:", opt1_model_path)

print("7. Remove Conv Q/DQ in DFL block")  
modify_onnx_model(opt1_model_path, opt2_model_path)

# from ultralytics import YOLO
# onnx_model = YOLO(output_model_path)
# results = onnx_model('https://ultralytics.com/images/bus.jpg')