from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    model = YOLO(r"F:\my_code\yolov8\ultralytics\cfg\models\v8\yolov8.yaml")
    model.train(
        data=r"F:\my_code\yolov8\TransmissionTower.yaml",
        task="detect",
        device="0",
        imgsz=256,
        epochs=200,
        patience=50,
        workers=8,
        batch=16,
        name="TransmissionTower_BGR_yolov8m_2m",
        amp=False,  # 禁用半精度训练
        pretrained=False,  # 禁用官方权重
    )

    # Resume training
    # model = YOLO(r"D:\code\yolov8-4b\runs\detect\train\weights\last.pt")
    # results = model.train(resume=True)

    # model = YOLO(r"D:\code\yolov8-4b\runs\detect\20240824\weights\best.pt")  # 模型文件路径
    # results = model(
    #     r"D:\code\detect\samples\1201lj0.tif",
    #     visualize=True)
