import os
import json
import shutil
import random
from tqdm import tqdm


BASE_DIR = r"C:\Users\greis\Desktop\Работы уник\Диплом\Датасеты"
JSON_FILE = "datasets_info.json"
OUTPUT_DIR = r"C:\Users\greis\Desktop\Работы уник\Диплом\Датасеты\merged_dataset"
SELECTED_CLASSES = ["helmet", "gloves", "vest"]
TRAIN_PART = 0.8  # 80%
VAL_PART = 0.1    # 10%
TEST_PART = 0.1   # 10%


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
    return paths

def filter_label_file(src_label_path, dst_label_path, class_map, selected_classes):
    selected_ids = [idx for cls, idx in class_map.items() if cls in selected_classes]

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if class_id in selected_ids:
            filtered_lines.append(line)

    if filtered_lines:
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)
        return True
    return False


def main():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    for split in ["train", "val", "test"]:
        safe_mkdir(os.path.join(OUTPUT_DIR, split, "images"))
        safe_mkdir(os.path.join(OUTPUT_DIR, split, "labels"))

    matching_datasets = []
    for dataset_name, info in datasets_info.items():
        classes = list(info["classes"].keys())
        if all(cls in classes for cls in SELECTED_CLASSES):
            matching_datasets.append((dataset_name, info))

    if not matching_datasets:
        print("[ERROR] Ни один датасет не содержит все выбранные классы.")
        return

    print(f"[INFO] Найдено {len(matching_datasets)} подходящих датасетов:")
    for name, _ in matching_datasets:
        print(f"   - {name}")

    total_labels = 0
    for dataset_name, info in matching_datasets:
        dataset_path = os.path.join(BASE_DIR, dataset_name)
        for _, labels_path in find_dataset_paths(dataset_path, info["structure"]):
            total_labels += len([f for f in os.listdir(labels_path) if f.endswith(".txt")])

    image_counter = 0
    splits = ["train", "val", "test"]

    with tqdm(total=total_labels, desc="Обработка датасетов", unit="файл") as pbar:
        for dataset_name, info in matching_datasets:
            dataset_path = os.path.join(BASE_DIR, dataset_name)
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
                test_split = pairs[int(n * VAL_PART):]

                splits_data = {"train": train_split, "val": val_split, "test": test_split}

                for split_name, split_pairs in splits_data.items():
                    for image_src, label_src in split_pairs:
                        image_ext = os.path.splitext(image_src)[1]
                        image_dst = os.path.join(OUTPUT_DIR, split_name, "images", f"{dataset_name}_{image_counter}{image_ext}")
                        label_dst = os.path.join(OUTPUT_DIR, split_name, "labels", f"{dataset_name}_{image_counter}.txt")

                        ok = filter_label_file(label_src, label_dst, info["classes"], SELECTED_CLASSES)
                        if ok:
                            shutil.copy2(image_src, image_dst)
                            image_counter += 1
                        pbar.update(1)

    print(f"\n[OK] Скопировано {image_counter} изображений с фильтрованными аннотациями.")

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: ./train/images\n")
        f.write("val: ./val/images\n")
        f.write("test: ./test/images\n\n")
        f.write(f"nc: {len(SELECTED_CLASSES)}\n")
        f.write(f"names: {SELECTED_CLASSES}\n")

    print(f"[OK] Итоговый YAML создан: {yaml_path}")


if __name__ == "__main__":
    main()
