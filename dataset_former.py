import os
import json
import shutil
import random
import argparse
from tqdm import tqdm


BASE_DIR = r"C:\Users\greis\Desktop\Работы уник\Диплом\Датасеты"
JSON_FILE = "datasets_info.json"
CLASS_NAMES_FILE = "class_names.json"
OUTPUT_DATASET_NAME = "merged_dataset"
OUTPUT_DIR = os.path.join(r"C:\Users\greis\Desktop\Работы уник\Диплом\Датасеты", OUTPUT_DATASET_NAME)
SELECTED_CLASSES = ["helmet", "gloves", "vest"]
TRAIN_PART = 0.8  # 80%
VAL_PART = 0.1    # 10%
TEST_PART = 0.1   # 10%
RANDOM_SEED = random.seed(12345)


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def find_dataset_paths(dataset_path, structure):
    paths = []
    if structure == "split":
        for subset in ["train", "val", "test"]:
            subdir = os.path.join(dataset_path, subset)
            if os.path.exists(os.path.join(subdir, "images")) and os.path.exists(os.path.join(subdir, "labels")):
                paths.append((os.path.join(subdir, "images"), os.path.join(subdir, "labels")))
    elif structure == "flat":
        if os.path.exists(os.path.join(dataset_path, "images")) and os.path.exists(os.path.join(dataset_path, "labels")):
            paths.append((os.path.join(dataset_path, "images"), os.path.join(dataset_path, "labels")))
    elif structure == "nested_split":
        for subset in ["train", "val", "test"]:
            img_dir = os.path.join(dataset_path, "images", subset)
            lbl_dir = os.path.join(dataset_path, "labels", subset)
            if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                paths.append((img_dir, lbl_dir))
    elif structure == "darknet":
        obj_train_data_path = os.path.join(dataset_path, "obj_train_data")
        if os.path.exists(obj_train_data_path):
            # Для Darknet формата изображения и аннотации в одной папке
            paths.append((obj_train_data_path, obj_train_data_path))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Объединение и фильтрация датасетов по выбранным классам"
    )
    parser.add_argument(
        "--source-path",
        type=str,
        default=None,
        help="Путь к исходным датасетам (если не указан, используется значение BASE_DIR)"
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Путь к новому создаваемому датасету (если не указан, используется значение OUTPUT_DIR)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Имена классов, разделенные запятыми (если не указаны, используется значение SELECTED_CLASSES)"
    )
    parser.add_argument(
        "--datasets-info-path",
        type=str,
        default=None,
        help="Путь к JSON файлам с информацией о датасетах (если не указан, используется source-path)"
    )
    return parser.parse_args()


def filter_label_file(src_label_path, dst_label_path, class_map, class_names_map, selected_classes):
    id_to_normalized = {}
    for name, idx in class_map.items():
        normalized = class_names_map.get(name, name)
        id_to_normalized[idx] = normalized

    new_id_map = {cls: i for i, cls in enumerate(selected_classes)}

    filtered_lines = []

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        normalized_name = id_to_normalized.get(class_id)
        if normalized_name in selected_classes:
            new_id = new_id_map[normalized_name]
            parts[0] = str(new_id)
            filtered_lines.append(" ".join(parts) + "\n")

    if filtered_lines:
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)
        return True
    return False


def main():
    args = parse_args()
    
    # Определяем пути
    source_dir = args.source_path if args.source_path else BASE_DIR
    target_dir = args.target_path if args.target_path else OUTPUT_DIR
    
    # Определяем выбранные классы
    if args.classes:
        selected_classes = [cls.strip() for cls in args.classes.split(",")]
    else:
        selected_classes = SELECTED_CLASSES
    
    # Определяем путь к JSON файлам
    if args.datasets_info_path:
        info_dir = args.datasets_info_path
    else:
        info_dir = source_dir
    
    json_file = os.path.join(info_dir, JSON_FILE)
    class_names_file = os.path.join(info_dir, CLASS_NAMES_FILE)
    
    with open(json_file, "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    with open(class_names_file, "r", encoding="utf-8") as f:
        class_names_map = json.load(f)

    for split in ["train", "valid", "test"]:
        safe_mkdir(os.path.join(target_dir, split, "images"))
        safe_mkdir(os.path.join(target_dir, split, "labels"))

    matching_datasets = []
    output_dataset_name = os.path.basename(target_dir)
    for dataset_name, info in datasets_info.items():
        if dataset_name == output_dataset_name:
            continue
        normalized_classes = []
        
        for cls in info["classes"].keys():
            normalized_classes.append(class_names_map.get(cls, cls))
        if all(cls in normalized_classes for cls in selected_classes):
            matching_datasets.append((dataset_name, info))

    if not matching_datasets:
        print("[ERROR] Ни один датасет не содержит все выбранные классы.")
        return

    print(f"[INFO] Найдено {len(matching_datasets)} подходящих датасета:")
    for name, _ in matching_datasets:
        print(f"   - {name}")

    total_labels = 0
    for dataset_name, info in matching_datasets:
        dataset_path = os.path.join(source_dir, dataset_name)
        for _, labels_path in find_dataset_paths(dataset_path, info["structure"]):
            total_labels += len([f for f in os.listdir(labels_path) if f.endswith(".txt")])

    image_counter = 0

    with tqdm(total=total_labels, desc="Обработка датасетов", unit="файл") as pbar:
        for dataset_name, info in matching_datasets:
            dataset_path = os.path.join(source_dir, dataset_name)
            for images_path, labels_path in find_dataset_paths(dataset_path, info["structure"]):

                pairs = []
                for label_file in os.listdir(labels_path):
                    if not label_file.endswith(".txt"):
                        continue
                    image_name = os.path.splitext(label_file)[0]
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = os.path.join(images_path, image_name + ext)
                        if os.path.exists(candidate):
                            pairs.append((candidate, os.path.join(labels_path, label_file)))
                            break

                if not pairs:
                    continue

                random.shuffle(pairs)
                n = len(pairs)
                train_split = pairs[:int(n * TRAIN_PART)]
                val_split = pairs[int(n * TRAIN_PART):int(n * (TRAIN_PART + VAL_PART))]
                test_split = pairs[int(n * (VAL_PART + TRAIN_PART)):]

                splits_data = {"train": train_split, "valid": val_split, "test": test_split}

                for split_name, split_pairs in splits_data.items():
                    for image_src, label_src in split_pairs:
                        image_ext = os.path.splitext(image_src)[1]
                        image_dst = os.path.join(target_dir, split_name, "images", f"{dataset_name}_{image_counter}{image_ext}")
                        label_dst = os.path.join(target_dir, split_name, "labels", f"{dataset_name}_{image_counter}.txt")

                        ok = filter_label_file(label_src, label_dst, info["classes"], class_names_map, selected_classes)
                        if ok:
                            shutil.copy2(image_src, image_dst)
                            image_counter += 1
                        pbar.update(1)
    print(f"\n[DEBUG] Всего label-файлов: {total_labels}")
    print(f"[DEBUG] Отфильтровано и скопировано: {image_counter}")
    print(f"[DEBUG] Процент используемых файлов: {image_counter / total_labels * 100:.2f}%")

    print(f"\n[OK] Скопировано {image_counter} изображений с фильтрованными аннотациями.")

    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: ./train/images\n")
        f.write("val: ./valid/images\n")
        f.write("test: ./test/images\n\n")
        f.write(f"nc: {len(selected_classes)}\n")
        f.write(f"names: {selected_classes}\n")

    print(f"[OK] Итоговый YAML создан: {yaml_path}")


if __name__ == "__main__":
    main()
