import onnx
from ultralytics import YOLO
def modify_onnx_model(onnx_file_path):
    # ONNX 모델 불러오기
    onnx_model = onnx.load(onnx_file_path)
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
    modified_onnx_path = "yolov8m_qat_opt2.onnx"
    onnx.save(onnx_model, modified_onnx_path)
    print("Modified ONNX model saved at:", modified_onnx_path)
    onnx_model = YOLO(modified_onnx_path)
    results = onnx_model('https://ultralytics.com/images/bus.jpg')
    print(results)

onnx_file_path = 'yolov8m_qat_opt.onnx'
modify_onnx_model(onnx_file_path)