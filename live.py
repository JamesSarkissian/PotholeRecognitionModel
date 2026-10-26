"""
live.py - Scan live video for potholes using the model from train.py.

How to run it:
    python live.py                  # use your webcam
    python live.py road_video.mp4   # use a recorded video instead
Press q in the video window to quit.

How it works: live video is just a fast loop over still pictures. Each pass
of the loop grabs one frame, shrinks it to 224x224, asks the model "pothole?",
and writes the answer on the frame before showing it.
"""

import sys
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "pothole_classifier.keras"
IMAGE_SIZE = 224     # must match train.py
THRESHOLD = 0.5      # score above this = pothole. Raise it for fewer false alarms
SMOOTHING = 8        # average the last 8 scores so the label doesn't flicker
EVERY_NTH_FRAME = 2  # only run the model on every 2nd frame to keep video smooth


def main():
    # A number means a camera (0 = built-in webcam); anything else is a file path
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(source) if source.isdigit() else source

    model = tf.keras.models.load_model(MODEL_PATH)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")

    recent_scores = deque(maxlen=SMOOTHING)  # a list that keeps only the newest 8
    frame_count = 0
    last_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # video ended or camera disconnected

        if frame_count % EVERY_NTH_FRAME == 0:
            # OpenCV stores colors as BGR; the model was trained on RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
            # The model expects a batch, so add a batch dimension: (1, 224, 224, 3)
            batch = np.expand_dims(small, axis=0).astype("float32")
            # Calling the model directly is faster than model.predict for one image
            score = float(model(batch, training=False)[0][0])
            recent_scores.append(score)
        frame_count += 1

        avg = sum(recent_scores) / len(recent_scores)
        if avg > THRESHOLD:
            label, color = f"POTHOLE {avg:.0%}", (0, 0, 255)  # red (BGR order)
        else:
            label, color = f"clear road {1 - avg:.0%}", (0, 200, 0)  # green

        # Frames per second, so you can see how fast it runs on your machine
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f"{fps:.0f} fps", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Pothole scanner (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
