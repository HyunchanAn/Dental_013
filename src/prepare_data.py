import os
import cv2
import glob
from tqdm import tqdm


def prepare_dataset():
    """
    Parses YOLO format annotations from the Kaggle dataset,
    crops the bounding box regions, and saves them into a classification folder structure.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    # Assuming Kaggle dataset extracts into something like:
    # data/raw/YOLO/YOLO/train/images/, data/raw/YOLO/YOLO/train/labels/
    yolo_dir = os.path.join(raw_dir, "YOLO", "YOLO")

    # Class mapping based on actual Kaggle dataset's data.yaml
    class_mapping = {
        1: "Crown",
        2: "Filling",
        3: "Implant",
        9: "RCT",
        25: "Other",  # post - core
        18: "Other",  # abutment
        # Note: 'Natural' might need to be generated separately if we want to classify healthy teeth.
    }

    splits = ["train", "valid"]

    for split in splits:
        img_dir = os.path.join(yolo_dir, split, "images")
        lbl_dir = os.path.join(yolo_dir, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Skipping {split} split, directories not found.")
            continue

        print(f"Processing {split} split...")
        img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(
            os.path.join(img_dir, "*.png")
        )

        for img_path in tqdm(img_paths):
            basename = os.path.basename(img_path)
            stem = os.path.splitext(basename)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            if not os.path.exists(lbl_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Un-normalize YOLO format
                x_center = cx * w
                y_center = cy * h
                box_w = bw * w
                box_h = bh * h

                x1 = max(0, int(x_center - box_w / 2))
                y1 = max(0, int(y_center - box_h / 2))
                x2 = min(w, int(x_center + box_w / 2))
                y2 = min(h, int(y_center + box_h / 2))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                cls_name = class_mapping.get(cls_id, "Other")

                # Determine target folder (map 'valid' to 'val' for torchvision compat)
                out_split = "val" if split == "valid" else split
                out_folder = os.path.join(processed_dir, out_split, cls_name)
                os.makedirs(out_folder, exist_ok=True)

                out_path = os.path.join(out_folder, f"{stem}_{i}.jpg")
                cv2.imwrite(out_path, crop)


if __name__ == "__main__":
    prepare_dataset()
