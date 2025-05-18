from ultralytics import YOLO

model = YOLO("yolov11n.pt")

results = model.train(
    data="datasets/juggling_balls/juggling_balls.yaml", epochs=100, imgsz=256
)

print(results)
