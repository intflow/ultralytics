from ultralytics import YOLO



# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch
# model = YOLO("/usr/src/ultralytics/runs/detect/train40/weights/last.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco.yaml")
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format