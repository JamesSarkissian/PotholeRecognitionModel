"""
train.py - Teach a model to DRAW A BOX around every pothole in a picture.

How to run it:
    python train.py                          # uses ~/Downloads/Pothole
    python train.py --data /path/to/Pothole  # or point it somewhere else
    python train.py --epochs 20              # shorter run, just to try it out

The old version of this project only answered "is there a pothole somewhere
in this picture, yes or no?". This version is an OBJECT DETECTOR: for every
pothole it finds, it gives back a box (x1, y1, x2, y2) and a confidence score.

Which detector? We use YOLO ("You Only Look Once") from the Ultralytics
package. YOLO is the most popular real-time detector: it's accurate, fast
enough for live webcam video on a laptop, and training it is one function
call. The TensorFlow Object Detection API can do the same job, but it is
old, hard to install on a Mac, and full of config files. YOLO can still be
exported to TensorFlow at the end (see --export-tf below).

What this script does, step by step:
    1. CONVERT the data. Your Roboflow export lists boxes in CSV files as
       pixel corners (xmin, ymin, xmax, ymax). YOLO wants one small .txt file
       per image with each box as (center x, center y, width, height), all
       written as fractions of the image size. We write those files into
       data/pothole_yolo/ inside this project folder. Your original
       ~/Downloads/Pothole folder is never changed.
    2. TRAIN. We start from "yolo11n.pt", a small YOLO model already trained
       on 118,000 everyday photos (people, cars, dogs...). It already knows
       how to find objects in general; we only teach it what a pothole looks
       like. This is called transfer learning, like the old MobileNetV2 model.
    3. TEST the finished model on the test images it never saw, and print
       how accurate its boxes are.
    4. SAVE the best model as pothole_detector.pt next to this script, where
       live.py looks for it.
"""

import argparse
import csv
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Settings. These are the "knobs" you can turn.
# ---------------------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"  # n = nano, the smallest/fastest. Try "yolo11s.pt" for
                           # a bigger model: more accurate but ~3x slower.
IMAGE_SIZE = 640           # images are resized to 640x640 for the model
EPOCHS = 60                # one epoch = one full pass over every training image
PATIENCE = 15              # stop early if 15 epochs in a row don't improve
BATCH_SIZE = 16            # images per update. Lower to 8 if you run out of memory

HERE = os.path.dirname(os.path.abspath(__file__))  # this project's folder
YOLO_DATA_DIR = os.path.join(HERE, "data", "pothole_yolo")
MODEL_PATH = os.path.join(HERE, "pothole_detector.pt")

# Roboflow calls the validation folder "valid"; YOLO calls it "val"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


# ---------------------------------------------------------------------------
# Step 1: convert the Roboflow CSV boxes into YOLO's format
# ---------------------------------------------------------------------------
def convert_split(data_dir, roboflow_split, yolo_split):
    """Copy one split's images and write a YOLO label file for each one.

    The CSV has one row per BOX, so an image with 3 potholes is on 3 rows.
    Clear-road images (the 300 extra ones) are on one row with the class and
    box columns left blank: they get an EMPTY label file, which tells YOLO
    "there is nothing to find here". Those empty examples teach the model not
    to draw boxes on manholes, speed bumps and normal road.
    """
    src_dir = os.path.join(data_dir, roboflow_split)
    img_out = os.path.join(YOLO_DATA_DIR, "images", yolo_split)
    lbl_out = os.path.join(YOLO_DATA_DIR, "labels", yolo_split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    boxes = {}  # filename -> list of YOLO label lines
    with open(os.path.join(src_dir, "_annotations.csv"), newline="") as f:
        for row in csv.DictReader(f):
            name = row["filename"]
            boxes.setdefault(name, [])  # make sure clear-road images get an entry
            if row["class"] != "pothole":
                continue  # blank row = clear-road image, no box to add

            w, h = float(row["width"]), float(row["height"])
            # Keep the corners inside the picture, just in case
            x1 = min(max(float(row["xmin"]), 0), w)
            y1 = min(max(float(row["ymin"]), 0), h)
            x2 = min(max(float(row["xmax"]), 0), w)
            y2 = min(max(float(row["ymax"]), 0), h)
            if x2 <= x1 or y2 <= y1:
                continue  # skip broken boxes with no area

            # Corners in pixels -> center + size as fractions (0 to 1)
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # The leading 0 is the class number: 0 = pothole (our only class)
            boxes[name].append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    for name, lines in boxes.items():
        shutil.copy2(os.path.join(src_dir, name), os.path.join(img_out, name))
        label_name = os.path.splitext(name)[0] + ".txt"
        with open(os.path.join(lbl_out, label_name), "w") as f:
            f.write("\n".join(lines))

    n_boxes = sum(len(lines) for lines in boxes.values())
    n_empty = sum(1 for lines in boxes.values() if not lines)
    print(f"  {yolo_split:5s}: {len(boxes)} images, {n_boxes} pothole boxes, "
          f"{n_empty} clear-road images")


def prepare_data(data_dir):
    """Build data/pothole_yolo/ and the data.yaml file that tells YOLO where it is."""
    for split in SPLITS:
        if not os.path.exists(os.path.join(data_dir, split, "_annotations.csv")):
            sys.exit(f"Could not find {split}/_annotations.csv inside {data_dir}.\n"
                     "Use --data to point at your Pothole folder.")

    # Start fresh each time so old files never sneak in
    if os.path.exists(YOLO_DATA_DIR):
        shutil.rmtree(YOLO_DATA_DIR)

    print(f"Converting {data_dir} to YOLO format...")
    for roboflow_split, yolo_split in SPLITS.items():
        convert_split(data_dir, roboflow_split, yolo_split)

    # data.yaml is a tiny settings file: where the images are, and class names.
    # YOLO finds each image's label by swapping "images" for "labels" in the path.
    yaml_path = os.path.join(YOLO_DATA_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DATA_DIR}\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
                "names:\n"
                "  0: pothole\n")
    return yaml_path


# ---------------------------------------------------------------------------
# Pick the fastest hardware available
# ---------------------------------------------------------------------------
def pick_device():
    """Use the Mac's graphics chip ("mps") if there is one, else the CPU."""
    import torch
    if torch.cuda.is_available():
        return "0"    # an NVIDIA GPU (e.g. Google Colab)
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon Macs (M1, M2, M3...)
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train the pothole box detector.")
    parser.add_argument("--data", default=os.path.expanduser("~/Downloads/Pothole"),
                        help="folder with train/, valid/ and test/ (default: ~/Downloads/Pothole)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--model", default=BASE_MODEL,
                        help="starting model: yolo11n.pt (fast) or yolo11s.pt (more accurate)")
    parser.add_argument("--device", default=None,
                        help="mps, cpu, or 0 for an NVIDIA GPU (default: pick automatically)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--export-tf", action="store_true",
                        help="also save a TensorFlow SavedModel copy (slow, optional)")
    args = parser.parse_args()

    yaml_path = prepare_data(os.path.expanduser(args.data))

    # Imported here so the data step above works even before ultralytics is set up
    from ultralytics import YOLO

    device = args.device or pick_device()
    print(f"\nTraining on: {device}  (mps = your Mac's graphics chip)\n")

    # Step 2: train. The first run downloads yolo11n.pt (about 5 MB).
    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=args.batch,
        device=device,
        project=os.path.join(HERE, "runs"),  # charts and checkpoints go here
        name="pothole",
        exist_ok=True,   # overwrite the previous run instead of making pothole2, 3...
        # Augmentation: YOLO randomly flips, recolors, scales and stitches
        # images together while training, so 675 photos feel like many more.
        # The defaults are good; fliplr=0.5 means "mirror half the images".
        fliplr=0.5,
        plots=True,      # saves charts like results.png in runs/pothole/
    )

    # YOLO keeps the checkpoint that scored best on the validation images
    best = os.path.join(HERE, "runs", "pothole", "weights", "best.pt")
    shutil.copy2(best, MODEL_PATH)

    # Step 3: test on images the model has never seen
    print("\nTesting on the test images...")
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=yaml_path, split="test", imgsz=IMAGE_SIZE, device=device,
                        project=os.path.join(HERE, "runs"), name="pothole_test",
                        exist_ok=True)

    # What these numbers mean:
    #   precision - of the boxes it drew, how many really were potholes
    #   recall    - of the real potholes, how many it found
    #   mAP50     - overall score: a box counts as right if it overlaps the
    #               true box by at least 50%. Higher is better, 1.0 is perfect.
    print("\n========== Test results ==========")
    print(f"Precision: {metrics.box.mp:.1%}")
    print(f"Recall:    {metrics.box.mr:.1%}")
    print(f"mAP50:     {metrics.box.map50:.1%}")
    print("==================================")
    print(f"Saved the model to {MODEL_PATH}")
    print("Charts and example predictions are in runs/pothole/ and runs/pothole_test/")

    # Step 4 (optional): a TensorFlow copy, for phones (TFLite) or TF-only apps.
    # live.py doesn't need this.
    if args.export_tf:
        print("\nExporting a TensorFlow SavedModel (this can take a few minutes)...")
        model.export(format="saved_model", imgsz=IMAGE_SIZE)

    print("\nNext: run 'python live.py' to see boxes on your webcam.")


if __name__ == "__main__":
    main()
