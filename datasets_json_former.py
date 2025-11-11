import os
import yaml
import json
import sys
import argparse


sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\greis\Desktop\Работы уник\Диплом\Датасеты"
OUTPUT_FILE = "datasets_info.json"
OUTPUT_CLASS_NAMES_FILE = "class_names.json"


def find_yaml_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() in ("data.yaml", "data.yml"):
                return os.path.join(root, f)
    return None


def find_obj_names_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == "obj.names":
                return os.path.join(root, f)
    return None


def find_obj_data_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == "obj.data":
                return os.path.join(root, f)
    return None


def load_obj_names(file_path):
    """Чтение obj.names файла (по одному классу на строку)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return names
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {file_path}: {e}")
        return None


def load_obj_data(file_path):
    """Парсинг obj.data файла для получения количества классов"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Ищем строку classes = X
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('classes'):
                parts = line.split('=')
                if len(parts) == 2:
                    return int(parts[1].strip())
        return None
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {file_path}: {e}")
        return None


def detect_structure(folder_path):
    subfolders = [d.lower() for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    # Проверка на формат Darknet YOLO
    obj_train_data_path = os.path.join(folder_path, "obj_train_data")
    obj_names_path = find_obj_names_file(folder_path)
    obj_data_path = find_obj_data_file(folder_path)
    
    if os.path.exists(obj_train_data_path) and (obj_names_path or obj_data_path):
        return "darknet"

    if any(x in subfolders for x in ["train", "val", "test"]):
        return "split"

    elif all(os.path.exists(os.path.join(folder_path, subdir)) for subdir in ["images", "labels"]):
        images_sub = os.listdir(os.path.join(folder_path, "images"))
        if any(x in images_sub for x in ["train", "val", "test"]):
            return "nested_split"
        return "flat"

    else:
        return "unknown"


def load_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать {file_path}: {e}")
        return None


def count_elements(folder_path, structure):
    labels_count = 0
    images_count = 0
    IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

    if structure == "split":
        for dir_name in os.listdir(folder_path):
            dir_path = os.path.join(folder_path, dir_name)
            if not os.path.isdir(dir_path):
                continue

            img_dir = os.path.join(folder_path, dir_name, "images")
            lbl_dir = os.path.join(folder_path, dir_name, "labels")

            if os.path.exists(img_dir):
                images_count += len([
                    f for f in os.listdir(img_dir)
                    if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                ])

            if os.path.exists(lbl_dir):
                labels_count += len([
                    f for f in os.listdir(lbl_dir)
                    if f.lower().endswith(".txt")
                ])

    elif structure == "flat":
        img_dir = os.path.join(folder_path, "images")
        lbl_dir = os.path.join(folder_path, "labels")

        images_count = len([
            f for f in os.listdir(img_dir)
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
        ]) if os.path.exists(img_dir) else 0

        labels_count = len([
            f for f in os.listdir(lbl_dir)
            if f.lower().endswith(".txt")
        ]) if os.path.exists(lbl_dir) else 0

    elif structure == "nested_split":
        for split in ["train", "val", "test"]:
            img_dir = os.path.join(folder_path, "images", split)
            lbl_dir = os.path.join(folder_path, "labels", split)

            if os.path.exists(img_dir):
                images_count += len([
                    f for f in os.listdir(img_dir)
                    if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
                ])

            if os.path.exists(lbl_dir):
                labels_count += len([
                    f for f in os.listdir(lbl_dir)
                    if f.lower().endswith(".txt")
                ])

    elif structure == "darknet":
        obj_train_data_path = os.path.join(folder_path, "obj_train_data")
        if os.path.exists(obj_train_data_path):
            files = os.listdir(obj_train_data_path)
            images_count = len([
                f for f in files
                if any(f.lower().endswith(ext) for ext in IMAGE_EXTS)
            ])
            labels_count = len([
                f for f in files
                if f.lower().endswith(".txt")
            ])
        else:
            return None

    else:
        return None

    if images_count == labels_count:
        return images_count
    else:
        print(f"[WARNING] В папке {folder_path} число изображений ({images_count}) "
              f"не совпадает с числом аннотаций ({labels_count})")
        return images_count, labels_count



def process_dataset(folder_path, folder_name):
    yaml_path = find_yaml_file(folder_path)
    
    names = None
    structure = detect_structure(folder_path)

    # Попытка загрузить из YAML (формат YOLOv8)
    if yaml_path:
        data = load_yaml(yaml_path)
        if data and "names" in data:
            names = data["names"]
            if isinstance(names, list):
                pass
            elif isinstance(names, dict):
                names = [v for k, v in sorted(names.items())]
            else:
                print(f"[ERROR] Поле 'names' в {yaml_path} имеет неверный формат")
                return None

    # Если не нашли YAML, пробуем формат Darknet
    if not names and structure == "darknet":
        obj_names_path = find_obj_names_file(folder_path)
        if obj_names_path:
            names = load_obj_names(obj_names_path)
            if not names:
                print(f"[WARNING] Не удалось загрузить классы из {obj_names_path} — пропуск")
                return None
        else:
            print(f"[WARNING] В папке {folder_name} не найден obj.names — пропуск")
            return None

    # Если ничего не нашли
    if not names:
        print(f"[WARNING] В папке {folder_name} не найден data.yaml или obj.names — пропуск")
        return None

    elements_count = count_elements(folder_path, structure)

    return {
        "classes": {name: idx for idx, name in enumerate(names)},
        "structure": structure,
        "elements_count": elements_count
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обработка датасетов и создание JSON файлов с информацией о классах и структуре"
    )

    parser.add_argument(
        "--datasets-path",
        type=str,
        default=None,
        help="Путь к папке с датасетами (если не указан, используется значение BASE_DIR)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Путь для сохранения выходных JSON файлов (если не указан, используется директория с датасетами)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    datasets_dir = args.datasets_path if args.datasets_path else BASE_DIR
    
    if args.output_path:
        output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(output_dir, OUTPUT_CLASS_NAMES_FILE)
    else:
        output_file = os.path.join(datasets_dir, OUTPUT_FILE)
        output_class_names_file = os.path.join(datasets_dir, OUTPUT_CLASS_NAMES_FILE)

    datasets_info = {}
    class_names = {}

    if not os.path.exists(datasets_dir):
        print(f"[ERROR] Папка '{datasets_dir}' не найдена.")
        return

    for folder_name in os.listdir(datasets_dir):
        folder_path = os.path.join(datasets_dir, folder_name)
        if os.path.isdir(folder_path):
            info = process_dataset(folder_path, folder_name)
            if info:
                datasets_info[folder_name] = info
                for class_name in info["classes"]:
                    class_names[class_name] = class_name

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datasets_info, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {output_file}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")

    try:
        with open(output_class_names_file, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {output_class_names_file}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")


if __name__ == "__main__":
    main()
