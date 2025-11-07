import os
import argparse
from ultralytics import YOLO
import csv


DATASET_PATH = "/media/user/Data/IndustrialSafety/Datasets/HardHatSkz"
MODELS_BASE_DIR = "/media/user/Data/IndustrialSafety/Models"
MODEL_VERSION = "yolov8n"
EPOCHS = 50
BATCH = 16
IMG_SIZE = 640


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение моделей")

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Путь к папке с датасетом (должен содержать файл data.yaml)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Модель (например: yolov8n.pt, yolov8s.pt, yolov11n.pt)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Количество эпох обучения"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Размер batch"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Размер изображения"
    )

    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Путь к папке с результатами обучения"
    )

    return parser.parse_args()


def train_yolo(dataset_path, model_version, epochs, batch, img_size, target_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Папка с датасетом не найдена: {dataset_path}")
    
    data_yaml = os.path.join(dataset_path, "data.yaml")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Не найден yaml файл: {data_yaml}")

    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    
    model_dir = os.path.join(
        target_dir, 
        dataset_name, 
        f"{model_version.replace('.pt', '')}_{epochs}epochs"
        )
    
    if os.path.exists(model_dir):
        while True:
            answer = input(f"[WARNING] Папка с таким названием уже существует: {model_dir}\nПродолжить обучение? (y/n): ").strip().lower()
            if answer == 'y':
                break
            elif answer == 'n':
                exit(0)
            else:
                print("Пожалуйста, введите только 'y' или 'n'.")
    else:
        os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[INFO] Обучение модели: {model_version}")
    print(f"[INFO] Датасет: {dataset_name}")
    print(f"[INFO] Конфигурация: {data_yaml}")
    print(f"[INFO] Сохранение результатов в {model_dir}")
    print("=" * 60 + "\n")

    _, model_ext = os.path.splitext(model_version)
    if model_ext == '':
        model = YOLO(model_version + ".pt")
    else:
        model = YOLO(model_version)

    # model.train(
    #     data=data_yaml,
    #     epochs=epochs,
    #     batch=batch,
    #     imgsz=img_size,
    #     project=model_dir,
    #     name="train",
    #     exist_ok=False
    # )

    trained_model_path = os.path.join(model_dir, "train", "weights", "best.pt")
    trained_model = YOLO(trained_model_path)

    result = trained_model.val(
        data=data_yaml, 
        split='test', 
        project=model_dir, 
        name="test",
        exist_ok = False
        )

    save_metrics_csv(result, model_dir)

    print("\n" + "-" * 60)
    if os.path.exists(trained_model_path):
        print(f"[OK] Обучение завершено.")
        print(f"[INFO] Модель сохранена по пути:\n{trained_model_path}")
    else:
        print("[WARNING] best.pt не найден. Проверьте лог Ultralytics.")
    print("-" * 60 + "\n")


def save_metrics_csv(test_result, model_dir):
    csv_file = os.path.join(model_dir, "test_metrics.csv") 
    
    csv_data = test_result.to_csv()

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(csv_data)
    

def main():
    args = parse_args()
    
    data = args.data if args.data else DATASET_PATH
    model_version = args.model if args.model else MODEL_VERSION
    epochs = args.epochs if args.epochs else EPOCHS
    batch = args.batch if args.batch else BATCH
    img_size = args.img_size if args.img_size else IMG_SIZE
    target_dir = args.target_path if args.target_path else MODELS_BASE_DIR

    train_yolo(
        dataset_path=data,
        model_version=model_version,
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        target_dir=target_dir
    )


if __name__ == "__main__":
    main()
