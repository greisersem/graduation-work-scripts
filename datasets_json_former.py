import os
import yaml
import json
import sys


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


def detect_structure(folder_path):
    subfolders = [d.lower() for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

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

    if not yaml_path:
        print(f"[WARNING] В папке {folder_name} не найден data.yaml — пропуск")
        return None

    data = load_yaml(yaml_path)
    if (not data) or ("names" not in data):
        print(f"[WARNING] В {yaml_path} отсутствует ключ 'names' — пропуск")
        return None

    names = data["names"]
    if isinstance(names, list):
        pass
    elif isinstance(names, dict):
        names = [v for k, v in sorted(names.items())]
    else:
        print(f"[ERROR] Поле 'names' в {yaml_path} имеет неверный формат")
        return None

    structure = detect_structure(folder_path)
    elements_count = count_elements(folder_path, structure)

    return {
        "classes": {name: idx for idx, name in enumerate(names)},
        "structure": structure,
        "elements_count": elements_count
    }


def main():
    datasets_info = {}
    class_names = {}

    if not os.path.exists(BASE_DIR):
        print(f"[ERROR] Папка '{BASE_DIR}' не найдена.")
        return

    for folder_name in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder_name)
        if os.path.isdir(folder_path):
            info = process_dataset(folder_path, folder_name)
            if info:
                datasets_info[folder_name] = info
                for class_name in info["classes"]:
                    class_names[class_name] = class_name

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(datasets_info, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {OUTPUT_FILE}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")

    try:
        with open(OUTPUT_CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=4)
        print(f"[OK] Информация успешно сохранена в {OUTPUT_CLASS_NAMES_FILE}")
    except Exception as e:
        print(f"[ERROR] Не удалось записать JSON: {e}")


if __name__ == "__main__":
    main()
