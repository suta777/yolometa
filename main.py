import torch
from ultralytics import YOLO
import ultralytics
model = YOLO("yolov8n.pt")

model.train(data = "data.yaml", epochs = 15, imgsz = 640, project = "/", name= "fish", exist_ok = True, device = 0)
