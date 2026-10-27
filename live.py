"""
live.py - Draw boxes around potholes in live video, using the model from train.py.

How to run it:
    python live.py                  # use your webcam
    python live.py road_video.mp4   # use a recorded video instead
    python live.py --conf 0.5       # only show boxes the model is 50%+ sure about
Press q in the video window to quit.

How it works: live video is just a fast loop over still pictures. Each pass
of the loop grabs one frame, asks the detector "where are the potholes?",
and draws a box plus a confidence score around each one it finds.
"""

import argparse
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pothole_detector.pt")  # made by train.py
IMAGE_SIZE = 640       # must match train.py
CONFIDENCE = 0.35      # hide boxes the model is less sure about than this.
                       # Raise it for fewer false alarms, lower it to catch more.
BOX_COLOR = (0, 0, 255)  # red. OpenCV writes colors as (Blue, Green, Red)


def pick_device():
    """Use the Mac's graphics chip ("mps") if there is one, else the CPU."""
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def draw_boxes(frame, result):
    """Draw each detected pothole's box and score on the frame. Returns the count."""
    count = 0
    # result.boxes holds every box the model found in this frame
    for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # pixel corners
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

        # A filled label above the box, e.g. "pothole 87%"
        label = f"pothole {score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(y1, th + 10)  # keep the label on screen if the box touches the top
        cv2.rectangle(frame, (x1, top - th - 10), (x1 + tw + 8, top), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 4, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Draw boxes around potholes in live video.")
    parser.add_argument("source", nargs="?", default="0",
                        help="camera number (0 = built-in webcam) or a video file path")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"minimum confidence to show a box (default {CONFIDENCE})")
    args = parser.parse_args()

    # A number means a camera; anything else is a file path
    source = int(args.source) if args.source.isdigit() else args.source

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"No trained model found at {MODEL_PATH}.\n"
                 "Run 'python train.py' first to create it, then run live.py again.")

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    device = pick_device()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if isinstance(source, int):
            sys.exit("Could not open the camera. On a Mac, allow camera access for "
                     "Terminal (or VS Code) in System Settings > Privacy & Security > Camera.")
        sys.exit(f"Could not open video file: {source}")

    last_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # video ended or camera disconnected

        # Ask the model where the potholes are. YOLO handles resizing and
        # color order itself, so we can hand it the raw OpenCV frame.
        # verbose=False stops it printing a line for every single frame.
        result = model.predict(frame, imgsz=IMAGE_SIZE, conf=args.conf,
                               device=device, verbose=False)[0]
        count = draw_boxes(frame, result)

        # Frames per second, so you can see how fast it runs on your machine
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        status = f"{count} pothole{'s' if count != 1 else ''}" if count else "clear road"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    BOX_COLOR if count else (0, 200, 0), 2)
        cv2.putText(frame, f"{fps:.0f} fps", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imshow("Pothole detector (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
