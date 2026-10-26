# Pothole Recognition Model

A TensorFlow model that looks at road images and decides **pothole** or **clear road**, plus a script that runs it on live video from a webcam or a recorded clip.

## Files

| File | What it does |
| --- | --- |
| `train.py` | Loads the dataset, trains the model, tests it, saves `pothole_classifier.keras` |
| `live.py` | Opens a webcam or video file and labels each frame live |
| `requirements.txt` | The Python packages this project needs |

## Setup (one time)

```bash
python3 -m venv venv           # make an isolated Python environment
source venv/bin/activate       # turn it on (do this in every new terminal)
pip install -r requirements.txt
```

## Train

```bash
python train.py                          # reads ~/Downloads/Pothole by default
python train.py --data /path/to/Pothole  # or point it somewhere else
```

The dataset folder needs `train/`, `valid/` and `test/`, each with images and an
`_annotations.csv` (Roboflow's "TensorFlow Object Detection CSV" export). Rows with
a blank `class` column are clear-road images with no potholes.

Expect roughly 20 to 60 minutes on a laptop CPU, or a few minutes on a Google Colab GPU.
At the end it prints accuracy, precision and recall on the test images.

## Scan live video

```bash
python live.py                 # webcam
python live.py drive.mp4       # recorded video
```

Press `q` to quit. The first time, macOS will ask for camera permission for your terminal.

## How it works, in short

1. **Transfer learning.** `train.py` borrows MobileNetV2, a network Google trained on
   1.4 million photos, and trains only a small new layer on top that answers
   "pothole?". That's why it learns from under 1,000 images in minutes.
2. **Two training stages.** First only the new layer learns. Then the top of
   MobileNetV2 is unfrozen and fine-tuned with a tiny learning rate.
3. **Live scanning** is a loop: grab a frame, resize to 224x224, ask the model,
   draw the answer, repeat. Scores are averaged over the last few frames so the
   label doesn't flicker.

## Next step

This version says *whether* a frame has a pothole. The dataset already has boxes
drawn around every pothole, so the next version can be an object detector that
shows *where* each pothole is.
