from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"F:\dfy_code\yolov8\ultralytics\cfg\models\v8\yolov8_ndsi.yaml")
    model.train(
        data=r"F:\dfy_code\yolov8\ultralytics\cfg\datasets\coco128.yaml",
        task="detect",
        device="0",
        imgsz=416,
        epochs=1,
        name="test_dataloader",
        amp=False,  # 禁用半精度训练
        pretrained=False  # 禁用官方权重
    )

    # Resume training
    # model = YOLO(r"D:\code\yolov8-4b\runs\detect\train\weights\last.pt")
    # results = model.train(resume=True)

    # model = YOLO(r"D:\code\yolov8-4b\runs\detect\20240824\weights\best.pt")  # 模型文件路径
    # results = model(
    #     r"D:\code\detect\samples\1201lj0.tif",
    #     visualize=True)
