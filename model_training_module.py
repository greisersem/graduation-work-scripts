from ultralytics import YOLO

model = YOLO ('yolov8n.yaml')
model = YOLO('yolov8n.pt')
model.train(data='', epochs=50, imgsz=640, batch=16)
results = model.train(data='C:\Users\greis\Desktop\Работы уник\Диплом\Датасеты\merged_dataset\data.yaml', epochs=50, batch=16)
results = model.val()