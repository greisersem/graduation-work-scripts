import os
import argparse
from ultralytics import YOLO


def train_yolo(dataset_path, model_version="yolov8n.pt", epochs=50, batch=16, img_size=640):
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Не найден файл конфигурации: {data_yaml}")

    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    models_dir = os.path.join("models", dataset_name)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[INFO] Начинается обучение модели: {model_version}")
    print(f"[INFO] Датасет: {dataset_name}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов: {models_dir}")
    print("=" * 60 + "\n")

    model = YOLO(model_version)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=models_dir,
        name="train",
        exist_ok=True
    )

    best_model_path = os.path.join(models_dir, "train", "weights", "best.pt")

    print("\n" + "-" * 60)
    if os.path.exists(best_model_path):
        print(f"[OK] Обучение завершено.")
        print(f"[INFO] Лучший вес сохранён по пути:\n{best_model_path}")
    else:
        print("[WARNING] best.pt не найден. Проверьте лог Ultralytics.")
    print("-" * 60 + "\n")

    return best_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Удобный запуск обучения YOLO моделей")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Путь к папке с датасетом (должен содержать файл data.yaml)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO модель (например: yolov8n.pt, yolov8s.pt, yolov11n.pt)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Размер batch"
    )

    parser.add_argument(
        "--img",
        type=int,
        default=640,
        help="Размер изображения (imgsz)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_yolo(
        dataset_path=args.data,
        model_version=args.model,
        epochs=args.epochs,
        batch=args.batch,
        img_size=args.img
    )


if __name__ == "__main__":
    main()
