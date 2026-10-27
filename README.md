# Pothole Recognition Model

An object detector that finds potholes in road images and **draws a box around each one**,
plus a script that does it live on webcam video or a recorded clip.

## Files

| File | What it does |
| --- | --- |
| `train.py` | Converts the dataset, trains the detector, tests it, saves `pothole_detector.pt` |
| `live.py` | Opens a webcam or video file and draws a box around every pothole it sees |
| `requirements.txt` | The Python packages this project needs |

## Setup (one time)

Open Terminal, go to this project folder, and turn on your Python environment:

```bash
cd path/to/PotholeRecognitionModel
source ~/tf_env/bin/activate      # or wherever your tf_env lives
pip install -r requirements.txt   # installs ultralytics (YOLO) and PyTorch, a few minutes
```

You only need `pip install` once (and again whenever `requirements.txt` changes).

## Train

```bash
python train.py                          # reads ~/Downloads/Pothole by default
python train.py --data /path/to/Pothole  # or point it somewhere else
python train.py --epochs 10              # quick trial run first, if you like
```

The dataset folder needs `train/`, `valid/` and `test/`, each with images and an
`_annotations.csv` (Roboflow's "TensorFlow Object Detection CSV" export). Rows with
a blank `class` column are clear-road images with no potholes.

On a MacBook Air (M-series chip) expect very roughly 30 to 90 minutes for the full 60
epochs. It stops early on its own if it stops improving. Training prints a table each
epoch; the column to watch is `mAP50`, which should climb.

At the end it prints **precision**, **recall** and **mAP50** on the test images and
saves `pothole_detector.pt`. Charts and sample pictures with predicted boxes are saved
in `runs/pothole/` and `runs/pothole_test/`. Open `runs/pothole_test/val_batch0_pred.jpg`
to see what it drew.

If training crashes with an error mentioning `mps`, run it on the CPU instead
(slower, but always works): `python train.py --device cpu`

## Scan live video

```bash
python live.py                 # webcam
python live.py drive.mp4       # recorded video
python live.py --conf 0.5      # only show boxes it is at least 50% sure about
```

Press `q` to quit. The first time, macOS will ask for camera permission for your terminal.

## How it works, in short

1. **Detector, not classifier.** The old version only said "pothole somewhere: yes/no".
   An object detector outputs a list of boxes, each with a confidence score. Because it
   has to point at the pothole, it learns what potholes actually look like instead of
   guessing from the whole scene, which also makes it more accurate.
2. **YOLO with transfer learning.** We start from `yolo11n`, a small, fast detector
   already trained on 118,000 everyday photos, and teach it one new thing: potholes.
   It uses all 1,740 boxes you have, plus the 300 clear-road photos as "nothing here"
   examples so it doesn't draw boxes on manholes or speed bumps.
3. **Live scanning** is a loop: grab a frame, ask the model for boxes, draw them, repeat.

### Why YOLO instead of TensorFlow's own detector?

TensorFlow's Object Detection API is the official TF way to do this, but it is old,
painful to install on a Mac, and needs long config files. YOLO (from the Ultralytics
package, built on PyTorch) trains with one function call, runs in real time on a laptop,
and is one of the most accurate small detectors available. If you need a TensorFlow
version later (for example TensorFlow Lite on a phone), run `python train.py --export-tf`
and it will also save a TensorFlow SavedModel.

Note: Ultralytics is free for personal and open-source projects (AGPL-3.0 license).

## Making it more accurate

- **Bigger model:** `python train.py --model yolo11s.pt` (slower, usually more accurate).
- **More data:** more labelled pothole photos, especially from your own camera angle,
  help more than anything else.
- **Tune the threshold:** in `live.py`, raise `--conf` if it draws boxes on things that
  aren't potholes, lower it if it misses potholes.
